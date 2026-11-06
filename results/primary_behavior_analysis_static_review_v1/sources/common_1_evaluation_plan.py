"""CPU-only, multi-phase evaluation planning. Frozen scoring modules are untouched.

The phase spec is an explicit scientific authority, not an inference of authority
from a directory name or an old summary. All its source hashes are checked before
large checkpoints are hashed. One plan may contain several models, but each
registry has one model and one held-out family. A baseline is shared only when
the actual inference inputs and execution policy match.
"""
import datetime as dt
import hashlib
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "artifacts"
OPERATIONAL = {"output_dir", "device", "resume_from", "max_steps", "deadline_utc",
               "deadline_margin_seconds", "save_every_steps", "keep_step_checkpoints"}
ROLES = {"primary", "supplement_A", "supplement_B", "extended"}
CORPORA = {"target": "finemath", "c4": "c4", "wikitext": "wikitext103"}
PARENT_TIMERS = ("setup_seconds", "training_seconds", "validation_seconds", "checkpoint_seconds")


def training_execution(cfg, summary):
    """Map recorded training facts only; never fill missing facts from this process."""
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Missing training execution provenance")
    for key in ("torch_version", "cuda_version", "gpu", "device"):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise ValueError("Missing training execution provenance: " + key)
    if not re.fullmatch(r"cuda(?::[0-9]+)?", provenance["device"]):
        raise ValueError("Training execution device must be CUDA for this frozen evaluation policy")
    backend = cfg.get("attention_backend", "auto")
    expected = {"precision": cfg["precision"], "attention_implementation": cfg.get("attention_implementation", "sdpa"),
        "attention_backend_policy": backend, "deterministic_algorithms": cfg["deterministic_algorithms"],
        "deterministic_warn_only": False, "allow_tf32_matmul": cfg.get("allow_tf32", True),
        "allow_tf32_cudnn": cfg.get("allow_tf32", True), "cudnn_deterministic": True,
        "cudnn_benchmark": False, "cublas_workspace_config": ":4096:8", "model_weight_dtype": "torch.float32"}
    if cfg.get("evaluate_teacher_kl") or cfg.get("replay_objective") == "kl" or "teacher_weight_dtype" in provenance:
        expected["teacher_weight_dtype"] = "torch.float32"
    for key, value in expected.items():
        if key not in provenance or type(provenance[key]) is not type(value) or provenance[key] != value:
            raise ValueError("Missing or mismatched training execution provenance: " + key)
    backends = provenance.get("sdpa_backends_enabled")
    backend_expected = {"flash": backend in ("auto", "flash"), "memory_efficient": backend == "auto",
                        "math": backend in ("auto", "math"), "cudnn": backend == "auto"}
    if not isinstance(backends, dict):
        raise ValueError("Missing training execution provenance: sdpa_backends_enabled")
    for key, value in backend_expected.items():
        if type(backends.get(key)) is not bool or backends[key] != value:
            raise ValueError("Missing or mismatched training execution provenance: sdpa_backends_enabled." + key)
    # Validation emits its own dtype/precision. Reject contradictory observations.
    for point in summary.get("validation_trajectory", []):
        for metrics in point["metrics"].values():
            for key in ("model_weight_dtype", "precision"):
                if key in metrics and metrics[key] != provenance[key]:
                    raise ValueError("Training validation execution differs from provenance: " + key)
    return {"torch_version": provenance["torch_version"], "cuda_version": provenance["cuda_version"],
        "gpu_name": provenance["gpu"], "precision": provenance["precision"], "device_type": "cuda",
        "effective_precision": "cuda_bfloat16_autocast" if provenance["precision"] == "bf16" else "float32",
        "attention_implementation": provenance["attention_implementation"],
        "attention_backend": provenance["attention_backend_policy"],
        "deterministic_algorithms": provenance["deterministic_algorithms"],
        "deterministic_warn_only": provenance["deterministic_warn_only"],
        "allow_tf32_matmul": provenance["allow_tf32_matmul"], "allow_tf32_cudnn": provenance["allow_tf32_cudnn"],
        "cudnn_deterministic": provenance["cudnn_deterministic"], "cudnn_benchmark": provenance["cudnn_benchmark"],
        "CUBLAS_WORKSPACE_CONFIG": provenance["cublas_workspace_config"],
        "flash_sdp_enabled": backends["flash"], "mem_efficient_sdp_enabled": backends["memory_efficient"],
        "math_sdp_enabled": backends["math"], "cudnn_sdp_enabled": backends["cudnn"],
        "model_weight_dtype": provenance["model_weight_dtype"]}


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def absolute(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def safe(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Unsafe path component or job ID: {value!r}")
    return value


class Hashes:
    """Avoid re-reading shared assets; reject a change during one plan build."""
    def __init__(self):
        self.cache = {}

    def digest(self, path):
        path = absolute(path)
        before = path.stat()
        signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if str(path) in self.cache:
            old_signature, value = self.cache[str(path)]
            if signature != old_signature:
                raise ValueError(f"Source changed during plan construction: {path}")
            return value
        value = sha(path)
        after = path.stat()
        if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f"Source changed while hashing: {path}")
        self.cache[str(path)] = (signature, value)
        return value

    def proof(self, proof):
        path = absolute(proof["path"])
        expected = proof["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or self.digest(path) != expected:
            raise ValueError(f"Frozen source SHA mismatch: {path}")
        return str(path), expected


def subset(actual, expected, label):
    """The spec can bind protocol fields without imposing a new protocol schema."""
    for key, value in expected.items():
        if isinstance(value, dict):
            if not isinstance(actual.get(key), dict):
                raise ValueError(f"Frozen protocol field mismatch: {label}.{key}")
            subset(actual[key], value, f"{label}.{key}")
        elif actual.get(key) != value:
            raise ValueError(f"Frozen protocol field mismatch: {label}.{key}")


def endpoint(summary, kind):
    if kind == "final":
        point = summary["counters"]
        checkpoint = summary["checkpoint_path"]
    else:
        best = summary["best_target_validation"]
        matches = [point for point in summary["validation_trajectory"] if point["step"] == best["step"]]
        if len(matches) != 1:
            raise ValueError("Best checkpoint must match exactly one validation point")
        point, checkpoint = matches[0], best["checkpoint_path"]
        for key in ("target_prediction_tokens", "replay_prediction_tokens"):
            if key in best and best[key] != point[key]:
                raise ValueError(f"Best checkpoint {key} disagrees with validation point")
        if point["metrics"]["target"]["nll"] != best["nll"]:
            raise ValueError("Best checkpoint NLL disagrees with validation point")
        selected = min(summary["validation_trajectory"], key=lambda p: p["metrics"]["target"]["nll"])
        if selected["step"] != best["step"]:
            raise ValueError("Best checkpoint was not the first minimum of target validation NLL")
    exposure = {"step": point["step"],
                "target_prediction_tokens_at_checkpoint": point["target_prediction_tokens"],
                "replay_prediction_tokens_at_checkpoint": point["replay_prediction_tokens"]}
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in exposure.values()):
        raise ValueError("Checkpoint step/exposure must be nonnegative integer counters")
    if exposure["target_prediction_tokens_at_checkpoint"] > summary["counters"]["target_prediction_tokens"]:
        raise ValueError("Best exposure exceeds completed training exposure")
    return str(absolute(checkpoint)), exposure


def data_proofs(phase, hashes):
    family = phase["data_family"]
    if family not in ("main_v3", "extended_v1"):
        raise ValueError("Explicit main_v3 or extended_v1 held-out family required")
    if (phase["role"] == "extended") != (family == "extended_v1"):
        raise ValueError("Extended role and held-out family must agree; main v3 cannot evaluate extended training")
    base = DATA_ROOT / ("evaluation_v3" if family == "main_v3" else "data_extended")
    train = DATA_ROOT / ("data/finemath_train.pt" if family == "main_v3" else "data_extended/finemath_train.pt")
    inputs, paths = {}, {}
    path, value = hashes.proof(phase["training_data"])
    if absolute(path) != train.resolve():
        raise ValueError("Frozen training data does not belong to the declared data family")
    inputs[path] = value
    paths["training"] = path
    for split in ("validation", "test"):
        if set(phase[f"{split}_data"]) != set(CORPORA):
            raise ValueError("Three explicit validation/test corpus proofs required")
        paths[split] = {}
        for corpus, stem in CORPORA.items():
            path, value = hashes.proof(phase[f"{split}_data"][corpus])
            if absolute(path) != (base / f"{stem}_{split}.pt").resolve():
                raise ValueError(f"Wrong {split} data family or corpus: {corpus}")
            inputs[path], paths[split][corpus] = value, path
            meta = absolute(path).with_suffix(".json")
            if meta.exists():
                inputs[str(meta)] = hashes.digest(meta)
    return inputs, paths


def qualify(spec_path, hashes):
    spec_path = absolute(spec_path)
    spec = read(spec_path)
    if spec.get("schema_version") != 2 or spec.get("science_frozen") is not True:
        raise ValueError("An explicitly scientifically frozen phase-spec v2 is required")
    phases = spec.get("phases", [])
    if not phases or len({p["name"] for p in phases}) != len(phases):
        raise ValueError("Nonempty uniquely named phases required")
    qualified, frozen, ids, model_tags = [], [], set(), {}
    common = {str(spec_path): hashes.digest(spec_path)}
    for phase in phases:
        safe(phase["name"]); safe(phase["model_tag"])
        if phase.get("role") not in ROLES or phase.get("science_frozen") is not True:
            raise ValueError("Each phase needs an explicit scientific freeze and supported study role")
        regpath, regsha = hashes.proof(phase["registry"])
        protocolpath, protocolsha = hashes.proof(phase["protocol"])
        protocol = read(protocolpath)
        if not phase.get("protocol_expect"):
            raise ValueError("Explicit frozen protocol field expectations required")
        subset(protocol, phase["protocol_expect"], phase["name"])
        if protocol.get("model_revision") != phase["model"]["revision"]:
            raise ValueError("Frozen protocol/model revision mismatch")
        phase_inputs = {regpath: regsha, protocolpath: protocolsha}
        for path, expected in phase.get("source_proof_files", {}).items():
            verified, value = hashes.proof({"path": path, "sha256": expected})
            phase_inputs[verified] = value
        data_inputs, paths = data_proofs(phase, hashes)
        phase_inputs.update(data_inputs)
        if "test_paths" in protocol and {k: str(absolute(v)) for k, v in protocol["test_paths"].items()} != paths["test"]:
            raise ValueError("Protocol test paths differ from frozen held-out proofs")
        if "validation_paths" in protocol and {k: str(absolute(v)) for k, v in protocol["validation_paths"].items()} != paths["validation"]:
            raise ValueError("Protocol validation paths differ from frozen data proofs")
        for name, value in protocol.get("validation_sha256", {}).items():
            if phase_inputs[paths["validation"][name]] != value:
                raise ValueError("Protocol validation SHA differs from frozen data proofs")
        for path, value in protocol.get("data_sha256", {}).items():
            if str(absolute(path)) in phase_inputs and phase_inputs[str(absolute(path))] != value:
                raise ValueError("Protocol data SHA differs from frozen data proofs")
        rows = read(regpath)
        if not isinstance(rows, list) or not rows:
            raise ValueError("Registry must be a nonempty explicit list")
        selection = phase.get("selection")
        if selection:
            selectionpath, selectionsha = hashes.proof(selection["record"])
            phase_inputs[selectionpath] = selectionsha
            selectionrecord = read(selectionpath)
            subset(selectionrecord, selection["record_expect"], phase["name"] + ".selection")
            included = selection["include_run_ids"]
            if not included or len(included) != len(set(included)) or set(included) - {r["id"] for r in rows}:
                raise ValueError("Selection requires unique explicit IDs present in the source registry")
            if set(selectionrecord[selection["run_ids_field"]]) != set(included):
                raise ValueError("Phase subset differs from frozen validation-only selection record")
            rows = [r for r in rows if r["id"] in included]
        kinds = phase.get("checkpoints", spec.get("checkpoints", ["final", "best"]))
        if not kinds or len(kinds) != len(set(kinds)) or set(kinds) - {"final", "best"}:
            raise ValueError("Unique final/best checkpoint kinds required")
        registry_model = None
        for row in rows:
            safe(row["id"])
            logical_id = (phase["name"], row["id"])
            if logical_id in ids:
                raise ValueError("Duplicate phase/run identity")
            ids.add(logical_id)
            is_alias = row.get("execution_kind") == "alias"
            if (row.get("phase") != phase["expected_phase"] or
                    str(row.get("status", "")).startswith("provisional") or row.get("exclusion_reason") or
                    (not is_alias and row.get("active") is not True)):
                raise ValueError(f"Inactive, provisional, or wrong-phase registry row: {row['id']}")
            if phase["expected_phase"] in ("lr", "lambda", "pilot", "diagnostic"):
                raise ValueError("Calibration and diagnostic registries cannot become held-out evaluations")
            if is_alias and (row.get("counted_as_new_training") is not False or not row.get("alias_of")):
                raise ValueError("Alias must explicitly identify its physical training and zero new training")
            actual_id = row.get("actual_run_id") or (row.get("alias_of") if is_alias else row["id"])
            safe(actual_id)
            configpath, summarypath = str(absolute(row["config_path"])), str(absolute(row["summary_path"]))
            try:
                cfg, summary = read(configpath), read(summarypath)
            except FileNotFoundError as error:
                raise ValueError(f"Incomplete registry: missing config or summary for {row['id']}") from error
            expected_config = row.get("config_sha256") or phase.get("config_sha256", {}).get(row["id"])
            if not expected_config or hashes.digest(configpath) != expected_config:
                raise ValueError(f"Missing or mismatched frozen config SHA: {row['id']}")
            changed = [key for key in cfg if key not in OPERATIONAL and summary.get("config", {}).get(key) != cfg[key]]
            if summary.get("status") != "completed" or changed:
                raise ValueError(f"Incomplete training or scientific config mismatch: {row['id']} {changed}")
            if cfg.get("run_label", actual_id) != actual_id:
                raise ValueError("Logical alias or actual_run_id disagrees with physical config identity")
            training_seed = row.get("training_seed", row.get("seed"))
            if training_seed != cfg["seed"]:
                raise ValueError("Actual training seed differs from registry")
            for rowkey, cfgkey in (("ratio", "replay_ratio"), ("weight", "replay_weight"), ("learning_rate", "learning_rate")):
                if row.get(rowkey) != cfg.get(cfgkey):
                    raise ValueError(f"Registry/config {rowkey} mismatch")
            if summary["counters"]["target_prediction_tokens"] != cfg["target_prediction_tokens"]:
                raise ValueError("Incomplete prescribed target prediction-token exposure")
            ratio = Fraction(str(cfg["replay_ratio"]))
            if summary["counters"]["replay_prediction_tokens"] != cfg["target_prediction_tokens"] * ratio.numerator // ratio.denominator:
                raise ValueError("Incomplete prescribed replay prediction-token exposure")
            if (cfg.get("deterministic_algorithms") is not True or cfg.get("attention_implementation", "sdpa") != "sdpa"
                    or cfg.get("attention_backend", "auto") not in ("auto", "flash", "math")
                    or cfg.get("precision") not in ("bf16", "fp32")):
                raise ValueError("Unsupported or non-strict frozen training policy")
            execution = training_execution(cfg, summary)
            model = {"path": str(absolute(cfg["model_name_or_path"])), "revision": cfg["model_revision"]}
            expected_model = {"path": str(absolute(phase["model"]["path"])), "revision": phase["model"]["revision"]}
            if model != expected_model or (registry_model is not None and model != registry_model):
                raise ValueError("Different models cannot be mixed inside a phase registry")
            registry_model = model
            if phase["model_tag"] in model_tags and model_tags[phase["model_tag"]] != model:
                raise ValueError("A model tag cannot name different initial models")
            model_tags[phase["model_tag"]] = model
            if str(absolute(cfg["target_path"])) != paths["training"]:
                raise ValueError("Training data family differs from frozen phase")
            if {k: str(absolute(v)) for k, v in cfg["validation_paths"].items()} != paths["validation"]:
                raise ValueError("Training validation paths differ from phase data family")
            if {k: str(absolute(v)) for k, v in row.get("test_paths", {}).items()} != paths["test"]:
                raise ValueError("Registry held-out paths differ from phase data family")
            # Validate all endpoint metadata before touching any checkpoint bytes.
            endpoints = {kind: endpoint(summary, kind) for kind in kinds}
            qualified.append({"phase": phase, "row": row, "cfg": cfg, "summary": summary,
                "actual_run_id": actual_id, "training_seed": training_seed, "is_training_alias": is_alias,
                "training_execution": execution,
                "config_path": configpath, "summary_path": summarypath, "paths": paths,
                "source_sha256": {**phase_inputs, configpath: expected_config, summarypath: hashes.digest(summarypath)},
                "endpoints": endpoints})
        frozen.append({"name": phase["name"], "role": phase["role"], "model_tag": phase["model_tag"],
                       "expected_phase": phase["expected_phase"], "data_family": phase["data_family"],
                       "registry": regpath, "registry_sha256": regsha,
                       "protocol": protocolpath, "protocol_sha256": protocolsha, "rows": len(rows),
                       "checkpoints": kinds, "selection": selection})
    return spec, qualified, frozen, common


def policy_for(item, mode, behavior):
    cfg, phase = item["cfg"], item["phase"]
    backend = cfg.get("attention_backend", "auto")
    batch = cfg["evaluation_batch_size"] if mode == "nll" else phase.get("behavior", behavior)["batch_size"]
    context = 512 if mode == "nll" else phase.get("behavior", behavior)["max_length"]
    if not isinstance(batch, int) or batch < 1 or not isinstance(context, int) or context < 2:
        raise ValueError("Positive explicit evaluation batch/context limit required")
    execution = training_execution(cfg, item["summary"])
    return {"model": str(absolute(cfg["model_name_or_path"])), "revision": cfg["model_revision"],
        "batch_size": batch, "max_length": context, "device": "cuda:0", **execution}


def command(policy, mode, data_root, paths):
    script = "run_evaluate.py" if mode == "nll" else "evaluate_tasks.py"
    argv = [sys.executable, str(ROOT / "scripts" / script), "--model", policy["model"],
        "--revision", policy["revision"], "--device", "cuda:0", "--batch-size", str(policy["batch_size"]),
        "--precision", policy["precision"], "--attention-backend", policy["attention_backend"],
        "--deterministic-algorithms", "--cudnn-deterministic", "--no-cudnn-benchmark",
        "--allow-tf32-matmul" if policy["allow_tf32_matmul"] else "--no-allow-tf32-matmul",
        "--allow-tf32-cudnn" if policy["allow_tf32_cudnn"] else "--no-allow-tf32-cudnn"]
    if mode == "nll":
        argv += ["--teacher-kl", "--include-chunks"]
        for corpus, path in sorted(paths.items()):
            argv += ["--corpus", f"{corpus}={path}"]
    else:
        argv += ["--max-length", str(policy["max_length"]), "--data-root", str(data_root),
                 "--tasks", "piqa", "hellaswag"]
    return argv


def outputs(output, mode):
    path = Path(output)
    values = [str(path)]
    if mode == "behavior":
        values += [str(path.with_name(path.stem + f"_{task}.jsonl")) for task in ("piqa", "hellaswag")]
    return values + [str(path.with_name(path.stem + "_execution.json"))]


def workload(group, mode):
    if mode == "behavior":
        return {"questions": sum(x["items"] for x in group["tasks"].values()),
                "choices": sum(x["choices"] for x in group["tasks"].values()),
                "tokenizer_forward_work": "context+continuation tokens depend on model tokenizer; no inference estimate invented"}
    counts = {}
    for name, path in group["test_paths"].items():
        meta = read(Path(path).with_suffix(".json"))
        stats = meta.get("statistics", {})
        counts[name] = {"prediction_tokens": meta.get("prediction_tokens", stats.get("prediction_tokens")),
                       "raw_input_tokens": meta.get("raw_input_tokens", stats.get("raw_input_tokens")),
                       "chunks": meta.get("chunks", stats.get("chunks"))}
    return {"corpora": counts, "teacher_kl": True,
            "forward_multiplier": {"initial": 1, "adapted": 2},
            "note": "Teacher forward is nested in evaluation wall time; never add it again"}


def equivalence_proofs(spec, hashes):
    """Consume complete-state proofs bound to both current serialized checkpoints.

    The evaluator loads model_state only. Config/optimizer serialization may
    therefore differ even when the inference inputs are bitwise identical.
    No equality is inferred from banks, method names, validation metrics, or steps.
    """
    known = {}
    for proof in spec.get("model_equivalence_proofs", []):
        path, proofsha = hashes.proof(proof)
        record = read(path)
        if (record.get("schema_version") != 1 or
                record.get("algorithm") != "sha256_sorted_key_dtype_shape_contiguous_uint8_v1"):
            raise ValueError("Unsupported full model_state equivalence proof algorithm")
        sourcepath, sourcesha = hashes.proof(record["algorithm_source"])
        for index, comparison in enumerate(record["comparisons"]):
            if comparison.get("bitwise_model_state_equal") is not True:
                continue
            sides = [comparison["left"], comparison["right"]]
            fingerprints = [s["model_state_fingerprint"] for s in sides]
            if (fingerprints[0] != fingerprints[1] or
                    not re.fullmatch(r"[0-9a-f]{64}", fingerprints[0]["sha256_model_state"]) or
                    not isinstance(fingerprints[0]["state_elements"], int) or fingerprints[0]["state_elements"] < 1):
                raise ValueError("Full model_state equivalence fingerprint mismatch")
            inputs = {path: proofsha, sourcepath: sourcesha}
            for side in sides:
                cppath, cpsha = hashes.proof({"path": side["checkpoint"], "sha256": side["checkpoint_sha256"]})
                inputs[cppath] = cpsha
            for side in sides:
                digest = side["checkpoint_sha256"]
                if digest in known and known[digest]["fingerprint"] != fingerprints[0]:
                    raise ValueError("One checkpoint has contradictory model_state proofs")
                evidence = {"proof": path, "proof_sha256": proofsha, "comparison_index": index,
                            "algorithm": record["algorithm"], "fingerprint": fingerprints[0],
                            "left_checkpoint_sha256": sides[0]["checkpoint_sha256"],
                            "right_checkpoint_sha256": sides[1]["checkpoint_sha256"]}
                known[digest] = {"fingerprint": fingerprints[0], "input_sha256": inputs, "evidence": evidence}
    return known


def build(spec_path, output_root, manifest_path, mode, expected_seconds, hashes=None):
    if mode not in ("nll", "behavior") or not math.isfinite(expected_seconds) or expected_seconds <= 0:
        raise ValueError("Supported mode and positive scheduling allowance required")
    hashes = hashes or Hashes()
    spec, qualified, phases, common = qualify(spec_path, hashes)
    kinds = spec.get("checkpoints", ["final", "best"])
    if not kinds or len(kinds) != len(set(kinds)) or set(kinds) - {"final", "best"}:
        raise ValueError("Unique final/best checkpoint kinds required")
    output_root, manifest_path = absolute(output_root), absolute(manifest_path)
    source_files = ["scripts/evaluation_plan.py", "scripts/run_frozen_evaluation.py",
                    "src/replay_study/evaluate.py", "src/replay_study/execution.py",
                    "scripts/run_evaluate.py" if mode == "nll" else "scripts/evaluate_tasks.py"]
    tasks, task_inputs, behavior = {}, {}, spec.get("behavior", {})
    data_root = absolute(behavior.get("data_root", "artifacts/behavioral"))
    if mode == "behavior":
        # Import the existing CPU manifest validator, never the model/scorer.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from make_behavioral_queue import task_inputs as validate_tasks
        for key, path in (("data_manifest_sha256", data_root / "data_manifest.json"),
                          ("harness_manifest_sha256", data_root / "sources/harness_manifest.json")):
            if behavior.get(key) != hashes.digest(path):
                raise ValueError("Explicit frozen behavior data/harness SHA required")
        task_inputs, tasks = validate_tasks(data_root)
        task_inputs = {str(absolute(k)): v for k, v in task_inputs.items()}
        for name, task in tasks.items():
            rows = [json.loads(line) for line in Path(task["artifact"]).open() if line.strip()]
            task["choices"] = sum(len(row["choices"]) for row in rows)
        source_files += ["src/replay_study/tasks.py", "scripts/make_behavioral_queue.py"]
    scorer_inputs = {str(ROOT / path): hashes.digest(ROOT / path) for path in source_files}
    common.update(scorer_inputs)
    equal_states = equivalence_proofs(spec, hashes)
    prior_groups, prior_inferences = {}, {}
    for proof in spec.get("prior_plans", []):
        priorpath, priorsha = hashes.proof(proof)
        prior = read(priorpath)
        if prior.get("schema_version") != 2 or prior.get("mode") != mode:
            raise ValueError("Only an explicitly SHA-bound prior plan of the same scorer kind can be reused")
        common[priorpath] = priorsha
        for key, group in prior["baseline_groups"].items():
            if group.get("initial_origin"):
                continue  # bind the original plan directly; do not create alias chains
            origin = {"manifest": priorpath, "manifest_sha256": priorsha,
                      "job_id": group["initial_job_id"], "output": group["initial_output"]}
            prior_groups.setdefault(key, origin)
        for oldjob in prior["jobs"]:
            if oldjob["kind"] == "adapted":
                prior_inferences.setdefault((oldjob["baseline_group"], oldjob["weights_identity"]),
                    {"manifest": priorpath, "manifest_sha256": priorsha, "job_id": oldjob["id"], "output": oldjob["output"]})
    groups, records, jobs, physical_science = {}, [], [], {}
    for item in qualified:
        phase, row, cfg, summary = (item[k] for k in ("phase", "row", "cfg", "summary"))
        policy = policy_for(item, mode, behavior)
        model_dir = Path(policy["model"])
        weights = sorted([*model_dir.glob("*.safetensors"), *model_dir.glob("pytorch_model*.bin")])
        if not weights:
            raise ValueError("Pinned local base-model weights required")
        assets = sorted(set(weights + list(model_dir.glob("*.json")) + list(model_dir.glob("*.model")) + list(model_dir.glob("*.txt"))))
        model_inputs = {str(path): hashes.digest(path) for path in assets}
        evaluation_inputs = task_inputs if mode == "behavior" else {
            path: hashes.digest(path) for path in item["paths"]["test"].values()}
        key = identity({"mode": mode, "policy": policy, "model_assets": model_inputs,
                        "evaluation_inputs": evaluation_inputs, "scorer": scorer_inputs})
        # The spec hash is an authority, not an inference input. Do not make
        # otherwise identical initial baselines differ by logical phase.
        if key not in groups:
            group_id = f"{phase['model_tag']}_{key[:20]}"
            initial_id = f"{mode}_{group_id}_initial"
            initial_output = str(output_root / phase["model_tag"] / "baselines" / f"{key}.json")
            group = {"id": key, "model_tag": phase["model_tag"], "inference_policy": policy,
                     "base_model_assets_sha256": model_inputs, "evaluation_input_sha256": evaluation_inputs,
                     "test_paths": item["paths"]["test"] if mode == "nll" else {}, "tasks": tasks,
                     "initial_job_id": initial_id, "initial_output": initial_output}
            group["work_per_inference"] = workload(group, mode)
            if key in prior_groups:
                group["initial_origin"] = prior_groups[key]
                group["initial_job_id"] = prior_groups[key]["job_id"]
                group["initial_output"] = prior_groups[key]["output"]
            groups[key] = group
            if not group.get("initial_origin"):
                jobs.append({"id": initial_id, "kind": "initial", "baseline_group": key,
                             "output": initial_output, "argv": command(policy, mode, data_root, group["test_paths"]) + ["--output", initial_output],
                             "input_sha256": {**model_inputs, **evaluation_inputs}, "record_ids": []})
        group = groups[key]
        science = {k: v for k, v in cfg.items() if k not in OPERATIONAL}
        physical_key = (item["actual_run_id"], policy["model"], policy["revision"])
        if physical_key in physical_science and physical_science[physical_key] != science:
            raise ValueError("Same actual_run_id refers to different scientific training configs")
        physical_science[physical_key] = science
        for kind in item["endpoints"]:
            checkpoint, exposure = item["endpoints"][kind]
            digest = hashes.digest(checkpoint)
            equivalence = equal_states.get(digest)
            state_id = ("model_state:" + equivalence["fingerprint"]["sha256_model_state"]
                        if equivalence else "checkpoint:" + digest)
            inference_key = identity({"baseline_group": key, "weights_identity": state_id})
            origin = prior_inferences.get((key, state_id))
            job_id = f"{mode}_{phase['model_tag']}_{inference_key[:24]}"
            if origin:
                job_id = f"{mode}_{phase['model_tag']}_reuse_{inference_key[:16]}_{hashes.digest(spec_path)[:12]}"
            record_id = f"{phase['name']}__{row['id']}__{kind}"
            output = str(output_root / phase["model_tag"] / phase["name"] / row["id"] / f"{kind}.json")
            stat = Path(checkpoint).stat()
            record = {"id": record_id, "run_id": row["id"], "actual_run_id": item["actual_run_id"],
                "phase": row["phase"], "phase_name": phase["name"], "role": phase["role"], "data_family": phase["data_family"],
                "model_tag": phase["model_tag"], "kind": kind, "group": row["group"],
                "training_seed": item["training_seed"], "selection_seed": row.get("selection_seed"),
                "generation_seed": row.get("generation_seed"), "ratio": row["ratio"], "weight": row["weight"],
                "learning_rate": row["learning_rate"], "checkpoint": checkpoint, "checkpoint_sha256": digest,
                "weights_identity": state_id,
                "source_model_equivalence": equivalence["evidence"] if equivalence else None,
                "checkpoint_bytes": stat.st_size, "checkpoint_mtime_ns": stat.st_mtime_ns,
                "baseline_group": key, "initial_job_id": group["initial_job_id"], "inference_job_id": job_id,
                "output": output, "is_training_alias": item["is_training_alias"], **exposure,
                "full_training_committed_path": {"target_prediction_tokens": summary["counters"]["target_prediction_tokens"],
                    "replay_prediction_tokens": summary["counters"]["replay_prediction_tokens"],
                    "parent_wall_seconds": {k: summary["counters"].get(k) for k in PARENT_TIMERS},
                    "note": "Full training cost; best checkpoint exposure is separate. Aliases incur no new training."},
                "config_path": item["config_path"], "summary_path": item["summary_path"],
                "input_sha256": {**item["source_sha256"], checkpoint: digest,
                                 **(equivalence["input_sha256"] if equivalence else {})}}
            existing = next((job for job in jobs if job["id"] == job_id), None)
            if existing is None:
                argv = command(policy, mode, data_root, group["test_paths"]) + ["--checkpoint", checkpoint]
                if mode == "behavior":
                    argv += ["--baseline", group["initial_output"]]
                jobs.append({"id": job_id, "kind": "reuse" if origin else "adapted", "baseline_group": key,
                    "actual_run_id": item["actual_run_id"], "checkpoint": checkpoint, "checkpoint_sha256": digest,
                    "weights_identity": state_id, "source_model_equivalence": record["source_model_equivalence"],
                    "output": origin["output"] if origin else output, "argv": None if origin else argv + ["--output", output],
                    "inference_origin": origin, "input_sha256": {
                        **model_inputs, **evaluation_inputs, **record["input_sha256"]}, "record_ids": [record_id],
                    "depends_on": [origin["job_id"]] if origin else [group["initial_job_id"]]})
                record["inference_alias_of"] = origin if origin else None
            else:
                record["inference_alias_of"] = existing["record_ids"][0]
                existing["record_ids"].append(record_id)
                existing["input_sha256"].update(record["input_sha256"])
            records.append(record)
    manifest = {"schema_version": 2, "mode": mode, "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase_spec": str(absolute(spec_path)), "phase_spec_sha256": hashes.digest(spec_path),
        "output_root": str(output_root), "phases": phases, "baseline_groups": groups,
        "common_input_sha256": common, "jobs": jobs, "records": records,
        "scorer_input_sha256": scorer_inputs,
        "checkpoint_selection": "Prescribed-exposure final and target-validation best; no held-out or task selection",
        "behavior_metrics": {"primary": "acc_norm", "auxiliary": "acc", "transitions": "CC/CW/WC/WW independently for both scores"} if mode == "behavior" else None,
        "analysis_strata": ["role", "phase_name", "data_family", "model_tag", "baseline_group", "kind", "group", "ratio", "weight", "learning_rate"],
        "conditional_bank_replicates": "selection_seed varies at fixed training_seed; never call them independent training seeds",
        "analysis_policy": {"bootstrap_replicates": 5000, "bootstrap_seed": 20261106, "ci_level": 0.95,
            "question_intervals": "Conditional 95% Wilson and paired-example percentile; IID/exchangeability may fail for same-source activities/contexts",
            "training_variability": "Individual trained seeds and mean/sample SD ddof=1; questions/final/best are not training replicates"},
        "inference_performed_by_generator": False, "execution_requires_root_scheduling": True}
    counts = {"logical_runs": len(qualified), "physical_training_runs": len(physical_science),
        "logical_checkpoint_endpoints": len(records), "adapted_inference_jobs": sum(j["kind"] == "adapted" for j in jobs),
        "initial_baselines": sum(j["kind"] == "initial" for j in jobs),
        "reused_initial_baselines": sum(bool(g.get("initial_origin")) for g in groups.values()),
        "reuse_materialization_jobs": sum(j["kind"] == "reuse" for j in jobs),
        "total_inference_jobs": sum(j["kind"] != "reuse" for j in jobs), "total_queue_jobs": len(jobs),
        "avoided_duplicate_inferences": len(records) - sum(j["kind"] == "adapted" for j in jobs)}
    manifest["workload"] = {**counts, "by_baseline_group": {key: {
        "model_tag": group["model_tag"], "policy": group["inference_policy"],
        "adapted_inferences": sum(j["kind"] == "adapted" and j["baseline_group"] == key for j in jobs),
        "initial_inferences": int(not bool(group.get("initial_origin"))),
        "per_inference": group["work_per_inference"]} for key, group in groups.items()},
        "scheduler_allowance_seconds": len(jobs) * expected_seconds,
        "scheduler_allowance_per_job_seconds": expected_seconds,
        "cost_formula": "sum_g(initial_count_g * measured_initial_seconds_g + adapted_count_g * measured_adapted_seconds_g); add separately measured CPU verification/materialization wall time",
        "estimate_available": False, "allowance_is_measured_cost_estimate": False,
        "wall_seconds_are_not_gpu_seconds": True}
    digest = hashlib.sha256(encoded(manifest)).hexdigest()
    queue = []
    for job in jobs:
        aliases = [r for r in records if r["inference_job_id"] == job["id"]]
        job_outputs = [] if job["kind"] == "reuse" else outputs(job["output"], mode)
        for record in aliases:
            job_outputs += outputs(record["output"], mode)
        item = {"id": job["id"], "argv": [sys.executable, str(ROOT / "scripts/run_frozen_evaluation.py"),
            "--manifest", str(manifest_path), "--manifest-sha256", digest, "--job-id", job["id"], "--device", "cuda:0"],
            "outputs": sorted(set(job_outputs)), "expected_seconds": expected_seconds,
            "env": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}}
        if job.get("depends_on"):
            item["depends_on"] = job["depends_on"]
        queue.append(item)
    return queue, manifest
