#!/usr/bin/env python3
"""Freeze behavioral jobs from a complete active formal registry; never infer."""
import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS_REVISION = "d6de81643928d653435c431bae19945d41d32520"
OPERATIONAL = {"output_dir", "device", "resume_from", "max_steps", "deadline_utc",
               "deadline_margin_seconds", "save_every_steps", "keep_step_checkpoints"}
TASKS = ("piqa", "hellaswag")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def encoded(payload):
    return (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def authority_paths(status):
    values = [status.get("active_registry")]
    for key in ("active_registries", "active_extended_registries", "active_supplementary_registries"):
        value = status.get(key, [])
        values.extend(value.values() if isinstance(value, dict) else value)
    return {(Path(value) if Path(value).is_absolute() else ROOT / value).resolve()
            for value in values if isinstance(value, str)}


def qualify(registry, protocol_status):
    status = read(protocol_status)
    if Path(registry).resolve() not in authority_paths(status):
        raise ValueError("Registry is not an explicitly active registry in protocol-status authority")
    rows = read(registry)
    if not rows or any(row.get("active") is not True or row.get("phase") != "formal" or
                       str(row.get("status", "")).startswith("provisional") or row.get("exclusion_reason") for row in rows):
        raise ValueError("Every registry row must be active formal and non-provisional")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate run IDs in registry")
    qualified, shared = [], None
    for row in rows:
        try:
            cfg, summary = read(row["config_path"]), read(row["summary_path"])
        except FileNotFoundError as error:
            raise ValueError(f"Incomplete registry: missing config or summary for {row['id']}") from error
        changed = [key for key in cfg if key not in OPERATIONAL and summary["config"].get(key) != cfg[key]]
        if changed or summary.get("status") != "completed":
            raise ValueError(f"Incomplete run or scientific config mismatch: {row['id']} {changed}")
        if cfg.get("run_label", row["id"]) != row["id"] or cfg["seed"] != row["seed"]:
            raise ValueError(f"Registry identity differs from trained run: {row['id']}")
        for row_key, cfg_key in (("ratio", "replay_ratio"), ("weight", "replay_weight"), ("learning_rate", "learning_rate")):
            if row.get(row_key) != cfg.get(cfg_key):
                raise ValueError(f"Registry/config {row_key} mismatch: {row['id']}")
        if summary["counters"]["target_prediction_tokens"] != cfg["target_prediction_tokens"]:
            raise ValueError(f"Run did not finish its prescribed target exposure: {row['id']}")
        if cfg.get("deterministic_algorithms") is not True or cfg.get("attention_implementation", "sdpa") != "sdpa":
            raise ValueError("Frozen strict SDPA training policy required")
        if cfg.get("attention_backend", "auto") not in ("auto", "flash", "math") or cfg["precision"] not in ("bf16", "fp32"):
            raise ValueError("Unsupported frozen attention backend or precision")
        policy = {"model": cfg["model_name_or_path"], "revision": cfg["model_revision"],
                  "precision": cfg["precision"], "attention_backend": cfg.get("attention_backend", "auto"),
                  "allow_tf32_matmul": cfg.get("allow_tf32", True),
                  "allow_tf32_cudnn": cfg.get("allow_tf32", True), "deterministic_algorithms": True,
                  "cudnn_deterministic": True, "cudnn_benchmark": False,
                  "training_evaluation_batch_size": cfg["evaluation_batch_size"]}
        if shared is None:
            shared = policy
        elif shared != policy:
            raise ValueError("Different models or inference policies cannot be mixed in one behavioral queue")
        if row.get("model") and Path(policy["model"]).name != row["model"] + "-final":
            raise ValueError("Registry model tag differs from model path")
        qualified.append((row, cfg, summary))
    return qualified, shared


def task_inputs(data_root):
    data_root = Path(data_root)
    top_path = data_root / "data_manifest.json"
    top = read(top_path)
    harness_path = data_root / "sources/harness_manifest.json"
    harness = read(harness_path)
    if top["harness_revision"] != HARNESS_REVISION or harness["revision"] != HARNESS_REVISION:
        raise ValueError("Pinned data/harness revision mismatch")
    inputs = {str(top_path): sha(top_path), str(harness_path): sha(harness_path)}
    tasks = {}
    for task in TASKS:
        path = data_root / f"{task}_validation.jsonl"
        meta_path = path.with_suffix(".json")
        meta = read(meta_path)
        if (meta["harness_revision"] != HARNESS_REVISION or meta["artifact_sha256"] != sha(path)
                or meta["artifact_sha256"] != top["tasks"][task]["artifact_sha256"]
                or meta.get("fewshot") != 0 or meta.get("apply_chat_template") is not False):
            raise ValueError(f"Task artifact differs from pinned zero-shot protocol: {task}")
        count = sum(1 for line in path.open() if line.strip())
        if count != meta["items"]:
            raise ValueError("Task item count differs from manifest")
        inputs[str(path)], inputs[str(meta_path)] = meta["artifact_sha256"], sha(meta_path)
        tasks[task] = {"artifact": str(path), "sha256": meta["artifact_sha256"], "items": count,
                       "revision": meta["revision"], "ordered_example_fingerprint": meta["ordered_example_fingerprint"],
                       "license": meta["license"]}
    for source in harness["files"]:
        path = Path(source["path"])
        if not path.is_absolute():
            path = ROOT / path
        if sha(path) != source["sha256"]:
            raise ValueError(f"Pinned harness source bytes differ: {path}")
        inputs[str(path)] = source["sha256"]
    return inputs, tasks


def endpoint(summary, kind):
    if kind == "final":
        counters = summary["counters"]
        return summary["checkpoint_path"], counters["step"], counters["target_prediction_tokens"], counters["replay_prediction_tokens"]
    best = summary["best_target_validation"]
    points = [row for row in summary["validation_trajectory"] if row["step"] == best["step"]]
    if len(points) != 1:
        raise ValueError("Best checkpoint must match exactly one validation point")
    point = points[0]
    if best.get("target_prediction_tokens", point["target_prediction_tokens"]) != point["target_prediction_tokens"]:
        raise ValueError("Best checkpoint target exposure disagrees with validation point")
    if "metrics" in point and best["nll"] != point["metrics"]["target"]["nll"]:
        raise ValueError("Best checkpoint NLL disagrees with target validation point")
    return best["checkpoint_path"], best["step"], point["target_prediction_tokens"], point["replay_prediction_tokens"]


def build(registry, output_root, model_tag, checkpoints, expected_seconds, manifest_path,
          data_root=ROOT / "artifacts/behavioral", protocol_status=ROOT / "docs/protocol_status.json",
          batch_size=16, max_length=2048, initial_manifest=None):
    registry, output_root, manifest_path = Path(registry).resolve(), Path(output_root).resolve(), Path(manifest_path).resolve()
    data_root, protocol_status = Path(data_root).resolve(), Path(protocol_status).resolve()
    if initial_manifest is not None:
        initial_manifest = Path(initial_manifest).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", model_tag):
        raise ValueError("model-tag must be a safe single path component")
    if set(checkpoints) - {"final", "best"} or len(set(checkpoints)) != len(checkpoints) or not checkpoints:
        raise ValueError("Unique final/best checkpoint kinds required")
    if batch_size < 1 or max_length < 2 or not math.isfinite(expected_seconds) or expected_seconds <= 0:
        raise ValueError("Positive batch/context/expected-seconds required")
    qualified, shared = qualify(registry, protocol_status)  # qualify every row before hashing checkpoints
    inputs, tasks = task_inputs(data_root)
    for path in (ROOT / "scripts/evaluate_tasks.py", ROOT / "scripts/run_frozen_behavioral.py",
                 ROOT / "scripts/analyze_behavioral_results.py", ROOT / "src/replay_study/behavioral_analysis.py",
                 ROOT / "src/replay_study/tasks.py", ROOT / "src/replay_study/evaluate.py",
                 ROOT / "src/replay_study/execution.py"):
        inputs[str(path)] = sha(path)
    model_dir = Path(shared["model"])
    weights = sorted([*model_dir.glob("*.safetensors"), *model_dir.glob("pytorch_model*.bin")])
    if not weights:
        raise ValueError("Pinned local base-model weight files are required for SHA-bound behavioral jobs")
    model_assets = sorted(set(weights + list(model_dir.glob("*.json")) + list(model_dir.glob("*.model"))
                              + list(model_dir.glob("*.txt"))))
    for path in model_assets:
        inputs[str(path)] = sha(path)
    output_root = Path(output_root) / model_tag
    baseline = output_root / "initial.json"
    shared.update(batch_size=batch_size, max_length=max_length, tasks=list(TASKS),
                  primary_metric="acc_norm", auxiliary_metric="acc", fewshot=0,
                  apply_chat_template=False, CUBLAS_WORKSPACE_CONFIG=":4096:8",
                  attention_implementation="sdpa", deterministic_warn_only=False,
                  flash_sdp_enabled=shared["attention_backend"] in ("auto", "flash"),
                  mem_efficient_sdp_enabled=shared["attention_backend"] == "auto",
                  math_sdp_enabled=shared["attention_backend"] in ("auto", "math"),
                  cudnn_sdp_enabled=shared["attention_backend"] == "auto",
                  device_type="cuda", model_weight_dtype="torch.float32",
                  effective_precision="cuda_bfloat16_autocast" if shared["precision"] == "bf16" else "float32")
    base = [sys.executable, str(ROOT / "scripts/evaluate_tasks.py"), "--model", shared["model"],
            "--revision", shared["revision"], "--device", "cuda:0", "--batch-size", str(batch_size),
            "--precision", shared["precision"], "--max-length", str(max_length),
            "--data-root", str(data_root), "--tasks", *TASKS,
            "--attention-backend", shared["attention_backend"], "--deterministic-algorithms",
            "--allow-tf32-matmul" if shared["allow_tf32_matmul"] else "--no-allow-tf32-matmul",
            "--allow-tf32-cudnn" if shared["allow_tf32_cudnn"] else "--no-allow-tf32-cudnn",
            "--cudnn-deterministic", "--no-cudnn-benchmark"]
    initial_id = f"behavior_{model_tag}_initial"
    jobs = [{"id": initial_id, "kind": "initial", "phase": "initial", "run_id": None,
             "step": 0, "target_prediction_tokens_at_checkpoint": 0,
             "replay_prediction_tokens_at_checkpoint": 0, "output": str(baseline),
             "argv": base + ["--output", str(baseline)], "input_sha256": {}}]
    initial_origin = None
    if initial_manifest is not None:
        prior = read(initial_manifest)
        comparable = lambda policy: {k:v for k,v in policy.items() if k != "training_evaluation_batch_size"}
        if comparable(prior["inference_policy"]) != comparable(shared) or prior["common_input_sha256"] != inputs:
            raise ValueError("Reused initial plan must have identical model/revision, behavior policy, task and scorer SHA")
        origins = [job for job in prior["jobs"] if job["kind"] == "initial"]
        if len(origins) != 1 or origins[0].get("external_initial"):
            raise ValueError("Reuse the original initial-model plan directly, without chained aliases")
        initial_origin = {"manifest": str(initial_manifest), "manifest_sha256": sha(initial_manifest)}
        jobs[0] = {**origins[0], "external_initial": True,
                   "origin_manifest_sha256": initial_origin["manifest_sha256"]}
        baseline, initial_id = Path(jobs[0]["output"]), jobs[0]["id"]
        inputs[str(initial_manifest)] = initial_origin["manifest_sha256"]
    for row, cfg, summary in qualified:
        for kind in checkpoints:
            checkpoint, step, target, replay = endpoint(summary, kind)
            path = Path(checkpoint)
            stat = path.stat()
            output = output_root / row["id"] / f"{kind}.json"
            record = {"id": f"behavior_{model_tag}_{row['id']}_{kind}", "kind": kind, "phase": row["phase"],
                      "run_id": row["id"], "group": row["group"], "seed": row["seed"],
                      "ratio": row["ratio"], "weight": row["weight"], "learning_rate": row["learning_rate"],
                      "checkpoint": checkpoint, "checkpoint_sha256": sha(checkpoint),
                      "checkpoint_bytes": stat.st_size, "checkpoint_mtime_ns": stat.st_mtime_ns,
                      "step": step, "target_prediction_tokens_at_checkpoint": target,
                      "replay_prediction_tokens_at_checkpoint": replay, "output": str(output),
                      "summary_path": row["summary_path"], "summary_sha256": sha(row["summary_path"]),
                      "config_path": row["config_path"], "config_sha256": sha(row["config_path"]),
                      "argv": base + ["--checkpoint", checkpoint, "--baseline", str(baseline), "--output", str(output)]}
            record["input_sha256"] = {checkpoint: record["checkpoint_sha256"],
                                      row["summary_path"]: record["summary_sha256"],
                                      row["config_path"]: record["config_sha256"]}
            jobs.append(record)
    manifest = {"frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "registry": str(registry), "registry_sha256": sha(registry),
                "protocol_status": str(protocol_status), "protocol_status_sha256": sha(protocol_status),
                "model_tag": model_tag, "output_root": str(output_root), "inference_policy": shared,
                "harness_revision": HARNESS_REVISION, "tasks": tasks, "common_input_sha256": inputs,
                "initial_origin": initial_origin,
                "base_model_assets_sha256": {str(path): inputs[str(path)] for path in model_assets},
                "checkpoint_selection": "final prescribed exposure and target-validation best; never behavior-score selection",
                "transitions": "CC/CW/WC/WW separately for acc_norm and acc; compare every adapted checkpoint to initial model",
                "analysis_policy": {"bootstrap_replicates": 5000, "bootstrap_seed": 20261106,
                                    "ci_level": 0.95, "binomial_interval": "Wilson",
                                    "paired_example_bootstrap": "paired percentile via multinomial transition counts",
                                    "training_seed_summary": "mean and sample SD ddof=1; no question pseudo-replicates",
                                    "paired_seed_comparisons": [["M", "B4"], ["C", "B4"], ["B2", "B1"], ["B4", "B3"], ["D", "B4"]]},
                "inference_performed_by_generator": False, "execution_requires_root_scheduling": True,
                "queue_generator_sha256": sha(__file__), "jobs": jobs}
    manifest_sha = hashlib.sha256(encoded(manifest)).hexdigest()
    queue = []
    for job in jobs:
        if job.get("external_initial"):
            continue  # existing initial job remains an explicit external DAG dependency
        outputs = [job["output"], *[str(Path(job["output"]).with_name(Path(job["output"]).stem + f"_{task}.jsonl")) for task in TASKS],
                   str(Path(job["output"]).with_name(Path(job["output"]).stem + "_execution.json"))]
        item = {"id": job["id"], "argv": [sys.executable, str(ROOT / "scripts/run_frozen_behavioral.py"),
                "--manifest", str(manifest_path), "--manifest-sha256", manifest_sha,
                "--record-id", job["id"], "--device", "cuda:0"], "expected_seconds": expected_seconds,
                "outputs": outputs, "env": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}}
        if job["kind"] != "initial":
            item["depends_on"] = [initial_id]
        queue.append(item)
    return queue, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--registry", type=Path, help="Legacy single active primary registry")
    source.add_argument("--phase-spec", type=Path, help="SHA-bound v2 multi-phase authority; no inference")
    parser.add_argument("--output-root", type=Path, required=True, help="Model tag is appended as a separate directory")
    parser.add_argument("--model-tag", help="Required only for the legacy single-registry interface")
    parser.add_argument("--checkpoints", nargs="+", choices=("final", "best"), default=["final", "best"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--expected-seconds", type=float, default=300)
    parser.add_argument("--data-root", type=Path, default=ROOT / "artifacts/behavioral")
    parser.add_argument("--protocol-status", type=Path, default=ROOT / "docs/protocol_status.json")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--initial-manifest", type=Path,
                        help="Reuse a previously frozen identical model baseline plan; merge DAGs so its job runs once")
    args = parser.parse_args()
    if args.queue.resolve() == args.manifest.resolve():
        parser.error("queue and manifest must use distinct output paths")
    for path in (args.queue, args.manifest):
        if path.exists():
            raise FileExistsError(f"Use new frozen output paths: {path}")
    if args.phase_spec:
        if args.initial_manifest:
            parser.error("v2 plans share baselines within the combined plan; do not pass --initial-manifest")
        from evaluation_plan import build as build_phases
        queue, manifest = build_phases(args.phase_spec, args.output_root, args.manifest, "behavior", args.expected_seconds)
    else:
        if not args.model_tag:
            parser.error("--model-tag is required with --registry")
        queue, manifest = build(args.registry, args.output_root, args.model_tag, args.checkpoints,
                                args.expected_seconds, args.manifest, args.data_root, args.protocol_status,
                                args.batch_size, args.max_length, args.initial_manifest)
    for path, payload in ((args.manifest, manifest), (args.queue, queue)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded(payload))
    print(json.dumps({"jobs": len(queue), "model_tag": args.model_tag, "inference_performed": False,
                      "workload": manifest.get("workload")}))


if __name__ == "__main__":
    main()
