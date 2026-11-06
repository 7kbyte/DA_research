"""CPU-only multi-phase freeze/DAG tests. Checkpoint files contain fixture bytes."""
import copy
import hashlib
import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluation_plan as plan
import run_frozen_evaluation as runner
from schedule_jobs import validate_queue


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plan.encoded(value)); return path


def proof(path):
    return {"path": str(path), "sha256": plan.sha(path)}


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "DATA_ROOT", tmp_path / "artifacts")
    model = tmp_path / "model"; model.mkdir()
    (model / "model.safetensors").write_bytes(b"initial weights")
    taskroot = tmp_path / "behavior"
    import make_behavioral_queue as behavior
    top = {"harness_revision": behavior.HARNESS_REVISION, "tasks": {}}
    for task, nchoices in [("piqa", 2), ("hellaswag", 4)]:
        path = taskroot / f"{task}_validation.jsonl"; path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"task": task, "item_id": task + ":0", "fingerprint": task + "-fixture",
            "context": "fixture context", "choices": ["a"] * nchoices, "gold": 0}) + "\n")
        meta = {"harness_revision": behavior.HARNESS_REVISION, "artifact_sha256": plan.sha(path),
                "fewshot": 0, "apply_chat_template": False, "items": 1, "revision": "fixture",
                "ordered_example_fingerprint": task, "license": {"license": "fixture"}}
        write(path.with_suffix(".json"), meta); top["tasks"][task] = meta
    write(taskroot / "data_manifest.json", top)
    official = tmp_path / "official"; official.write_text("pinned official task")
    write(taskroot / "sources/harness_manifest.json", {"revision": behavior.HARNESS_REVISION,
          "files": [{"path": str(official), "sha256": plan.sha(official)}]})
    spec = {"schema_version": 2, "science_frozen": True, "checkpoints": ["final", "best"], "phases": [],
            "behavior": {"data_root": str(taskroot), "batch_size": 16, "max_length": 2048,
                "data_manifest_sha256": plan.sha(taskroot / "data_manifest.json"),
                "harness_manifest_sha256": plan.sha(taskroot / "sources/harness_manifest.json")}}
    return spec, model


def add_phase(tmp, spec, model, name="main", role="primary", family="main_v3", batch=16, same_best=True):
    train = plan.DATA_ROOT / ("data/finemath_train.pt" if family == "main_v3" else "data_extended/finemath_train.pt")
    train.parent.mkdir(parents=True, exist_ok=True); train.write_bytes(b"training")
    base = plan.DATA_ROOT / ("evaluation_v3" if family == "main_v3" else "data_extended")
    splits = {}
    for split in ("validation", "test"):
        splits[split] = {}
        for corpus, stem in plan.CORPORA.items():
            path = base / f"{stem}_{split}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{family}:{corpus}:{split}".encode())
            write(path.with_suffix(".json"), {"chunks": 1, "raw_input_tokens": 512, "prediction_tokens": 511})
            splits[split][corpus] = proof(path)
    runid = f"{name}_B4_s11"
    cp = tmp / name / "final.pt"; cp.parent.mkdir(); cp.write_bytes(f"{name} final".encode())
    best = cp if same_best else tmp / name / "best.pt"
    if not same_best: best.write_bytes(f"{name} best".encode())
    cfg = {"model_name_or_path": str(model), "model_revision": "revision", "precision": "bf16",
        "attention_backend": "auto", "attention_implementation": "sdpa", "allow_tf32": True,
        "deterministic_algorithms": True, "evaluation_batch_size": batch, "run_label": runid,
        "seed": 11, "replay_ratio": .05, "replay_weight": .5, "learning_rate": 3e-5,
        "target_prediction_tokens": 100, "target_path": str(train),
        "validation_paths": {k: v["path"] for k, v in splits["validation"].items()},
        "device": "cuda:0", "output_dir": "original"}
    summary = {"status": "completed", "config": {**cfg, "device": "cuda:3", "resume_from": "old.pt"},
        "provenance": {"torch_version": "fixture", "cuda_version": "fixture", "gpu": "CPU-fixture-no-GPU-used",
            "device": "cuda:3", "precision": "bf16", "attention_implementation": "sdpa", "attention_backend_policy": "auto",
            "deterministic_algorithms": True, "deterministic_warn_only": False, "allow_tf32_matmul": True,
            "allow_tf32_cudnn": True, "cudnn_deterministic": True, "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8", "model_weight_dtype": "torch.float32", "teacher_weight_dtype": "torch.float32",
            "sdpa_backends_enabled": {"flash": True, "math": True, "memory_efficient": True, "cudnn": True}},
        "checkpoint_path": str(cp), "counters": {"step": 4, "target_prediction_tokens": 100,
            "replay_prediction_tokens": 5, "training_seconds": 9},
        "best_target_validation": {"step": 2, "nll": 1.5, "checkpoint_path": str(best), "target_prediction_tokens": 50},
        "validation_trajectory": [{"step": 2, "target_prediction_tokens": 50, "replay_prediction_tokens": 2,
                                   "metrics": {"target": {"nll": 1.5}}}]}
    configpath = write(tmp / name / "config.json", cfg)
    summarypath = write(tmp / name / "summary.json", summary)
    row = {"id": runid, "active": True, "phase": "formal" if role in ("primary", "extended") else name,
           "group": "B4", "seed": 11, "ratio": .05, "weight": .5, "learning_rate": 3e-5,
           "config_path": str(configpath), "config_sha256": plan.sha(configpath), "summary_path": str(summarypath),
           "test_paths": {k: v["path"] for k, v in splits["test"].items()}}
    regpath = write(tmp / name / "registry.json", [row])
    protocolpath = write(tmp / name / "protocol.json", {"model_revision": "revision", "frozen": True,
            "test_paths": row["test_paths"], "validation_paths": cfg["validation_paths"]})
    phase = {"name": name, "role": role, "science_frozen": True, "expected_phase": row["phase"],
        "data_family": family, "model_tag": "pythia-fixture", "model": {"path": str(model), "revision": "revision"},
        "registry": proof(regpath), "protocol": proof(protocolpath), "protocol_expect": {"frozen": True},
        "training_data": proof(train), "validation_data": splits["validation"], "test_data": splits["test"]}
    spec["phases"].append(phase)
    return phase, row, cfg, summary


def build(tmp, spec, mode="nll", filename="spec.json", output="output"):
    path = write(tmp / filename, spec)
    return plan.build(path, tmp / output, tmp / f"{output}_manifest.json", mode, 30)


def test_model_policy_data_baselines_and_final_best_aliases(tmp_path, monkeypatch):
    spec, model = fixture(tmp_path, monkeypatch)
    add_phase(tmp_path, spec, model)
    add_phase(tmp_path, spec, model, "extended", "extended", "extended_v1")
    queue, manifest = build(tmp_path, spec)
    validate_queue(queue)
    assert len(queue) == 4 and len(manifest["baseline_groups"]) == 2
    assert manifest["workload"]["logical_checkpoint_endpoints"] == 4
    assert manifest["workload"]["adapted_inference_jobs"] == 2
    best = manifest["records"][1]
    assert best["step"] == 2 and best["target_prediction_tokens_at_checkpoint"] == 50
    assert best["full_training_committed_path"]["target_prediction_tokens"] == 100
    assert best["inference_alias_of"] == manifest["records"][0]["id"]
    assert {r["role"] for r in manifest["records"]} == {"primary", "extended"}
    behaviorq, behaviorm = build(tmp_path, spec, "behavior", output="behavior-output")
    assert len(behaviorq) == 3 and len(behaviorm["baseline_groups"]) == 1
    assert behaviorm["behavior_metrics"]["primary"] == "acc_norm"
    assert next(iter(behaviorm["baseline_groups"].values()))["work_per_inference"]["choices"] == 6


def test_batch_difference_needs_matched_nll_baselines_behavior_can_share_explicit_batch(tmp_path, monkeypatch):
    spec, model = fixture(tmp_path, monkeypatch)
    add_phase(tmp_path, spec, model)
    phase, row, cfg, summary = add_phase(tmp_path, spec, model, "suppA", "supplement_A", batch=8)
    nq, nm = build(tmp_path, spec)
    assert len(nm["baseline_groups"]) == 2
    bq, bm = build(tmp_path, spec, "behavior", output="bo")
    assert len(bm["baseline_groups"]) == 1
    phase["behavior"] = {"batch_size": 8, "max_length": 2048}
    bq, bm = build(tmp_path, spec, "behavior", output="bo2")
    assert len(bm["baseline_groups"]) == 2


@pytest.mark.parametrize("change", ["incomplete", "config_science", "config_sha", "data_family", "model", "provisional", "protocol", "freeze", "best_exposure", "source"])
def test_refuses_incomplete_wrong_freeze_and_source_mutations_before_checkpoint_hash(tmp_path, monkeypatch, change):
    spec, model = fixture(tmp_path, monkeypatch)
    phase, row, cfg, summary = add_phase(tmp_path, spec, model)
    if change == "incomplete": summary["status"] = "deadline_interrupted"
    elif change == "config_science": summary["config"]["learning_rate"] = 1e-3
    elif change == "config_sha": cfg["learning_rate"] = 1e-3; write(row["config_path"], cfg)
    elif change == "data_family": phase["data_family"] = "extended_v1"
    elif change == "model": phase["model"]["revision"] = "different"
    elif change == "provisional": row["status"] = "provisional_noise"
    elif change == "protocol": phase["protocol_expect"]["frozen"] = False
    elif change == "freeze": phase["science_frozen"] = False
    elif change == "best_exposure": summary["best_target_validation"]["target_prediction_tokens"] = 51
    elif change == "source": Path(phase["test_data"]["target"]["path"]).write_bytes(b"changed")
    write(row["summary_path"], summary)
    write(phase["registry"]["path"], [row]); phase["registry"] = proof(phase["registry"]["path"])
    Path(summary["checkpoint_path"]).unlink()  # eligibility must fail before this missing checkpoint is touched
    with pytest.raises(ValueError): build(tmp_path, spec)


def test_alias_actual_id_and_training_seed_are_not_relabelled_as_primary(tmp_path, monkeypatch):
    spec, model = fixture(tmp_path, monkeypatch)
    main, row, cfg, summary = add_phase(tmp_path, spec, model)
    alias = {**row, "id": "suppA_alias", "phase": "suppA", "active": False,
             "execution_kind": "alias", "alias_of": row["id"], "counted_as_new_training": False}
    reg = write(tmp_path / "aliases.json", [alias])
    phase = {**main, "name": "suppA", "role": "supplement_A", "expected_phase": "suppA", "registry": proof(reg)}
    spec["phases"].append(phase)
    queue, manifest = build(tmp_path, spec)
    assert len(queue) == 2
    assert manifest["workload"]["physical_training_runs"] == 1
    assert manifest["records"][2]["role"] == "supplement_A"
    assert manifest["records"][2]["actual_run_id"] == row["id"]


def test_selection_subset_requires_sha_bound_validation_record_and_phase_final_only(tmp_path, monkeypatch):
    spec, model = fixture(tmp_path, monkeypatch)
    phase, row, cfg, summary = add_phase(tmp_path, spec, model, "suppA", "supplement_A")
    phase["checkpoints"] = ["final"]
    selected = write(tmp_path / "selection.json", {"basis": "complete_validation_grid", "run_ids": [row["id"]]})
    phase["selection"] = {"record": proof(selected), "record_expect": {"basis": "complete_validation_grid"},
                          "run_ids_field": "run_ids", "include_run_ids": [row["id"]]}
    queue, manifest = build(tmp_path, spec)
    assert len(queue) == 2 and len(manifest["records"]) == 1
    phase["selection"]["include_run_ids"] = ["other_run"]
    with pytest.raises(ValueError, match="Selection"): build(tmp_path, spec)


def test_runtime_rechecks_inputs_and_materializes_alias_without_forward(tmp_path, monkeypatch):
    spec, model = fixture(tmp_path, monkeypatch)
    phase, row, cfg, summary = add_phase(tmp_path, spec, model)
    queue, manifest = build(tmp_path, spec)
    path = write(tmp_path / "output_manifest.json", manifest); digest = plan.sha(path)
    initial, job = manifest["jobs"]
    runner.verified_job(path, digest, job["id"], require_baseline=False)
    write(job["output"], {"model": str(model), "revision": "revision", "checkpoint": job["checkpoint"], "corpora": {}})
    attestation = {"manifest_sha256": digest, "job_id": job["id"], "checkpoint_sha256": job["checkpoint_sha256"],
                   "verified_frozen_inputs": True, "inference_subprocess_wall_seconds": 1,
                   "output_sha256": {job["output"]: plan.sha(job["output"])}}
    runner.materialize_aliases(manifest, job, attestation)
    alias = manifest["records"][1]
    assert runner.read(alias["output"])["inference_reuse"]["additional_model_forward_calls"] == 0
    assert runner.read(runner.attestation_path(alias["output"]))["inference_subprocess_wall_seconds"] == 0
    Path(job["checkpoint"]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="input SHA mismatch"):
        runner.verified_job(path, digest, job["id"], require_baseline=False)


def test_prior_plan_baseline_and_checkpoint_reuse_flat_master_dag(tmp_path, monkeypatch):
    spec, model = fixture(tmp_path, monkeypatch)
    add_phase(tmp_path, spec, model)
    oldqueue, oldmanifest = build(tmp_path, spec, "behavior")
    oldpath = write(tmp_path / "prior_manifest.json", oldmanifest)
    spec["prior_plans"] = [proof(oldpath)]
    queue, manifest = build(tmp_path, spec, "behavior", filename="new_spec.json", output="new")
    validate_queue(oldqueue + queue)
    assert len(queue) == 1 and manifest["jobs"][0]["kind"] == "reuse"
    assert manifest["workload"]["total_inference_jobs"] == 0
    assert manifest["workload"]["reused_initial_baselines"] == 1
    assert queue[0]["depends_on"] == [oldmanifest["jobs"][1]["id"]]


def test_full_model_state_proof_deduplicates_different_checkpoint_bytes_and_actual_runs(tmp_path, monkeypatch):
    spec, model = fixture(tmp_path, monkeypatch)
    left = add_phase(tmp_path, spec, model)
    right = add_phase(tmp_path, spec, model, "suppA", "supplement_A")
    _, baseline = build(tmp_path, spec)
    assert baseline["workload"]["adapted_inference_jobs"] == 2
    source = tmp_path / "fingerprint_algorithm.py"; source.write_text("fixture proof algorithm")
    fingerprint = {"sha256_model_state": "a" * 64, "state_elements": 8}
    comparisons = [{"bitwise_model_state_equal": True,
        "left": {"checkpoint": left[3]["checkpoint_path"], "checkpoint_sha256": plan.sha(left[3]["checkpoint_path"]),
                 "model_state_fingerprint": fingerprint},
        "right": {"checkpoint": right[3]["checkpoint_path"], "checkpoint_sha256": plan.sha(right[3]["checkpoint_path"]),
                  "model_state_fingerprint": fingerprint}}]
    proofs = write(tmp_path / "equivalence.json", {"schema_version": 1,
        "algorithm": "sha256_sorted_key_dtype_shape_contiguous_uint8_v1", "algorithm_source": proof(source),
        "comparisons": comparisons})
    spec["model_equivalence_proofs"] = [proof(proofs)]
    queue, manifest = build(tmp_path, spec)
    assert len(queue) == 2 and manifest["workload"]["physical_training_runs"] == 2
    assert len({r["checkpoint_sha256"] for r in manifest["records"]}) == 2
    assert len({r["weights_identity"] for r in manifest["records"]}) == 1
    assert all(r["source_model_equivalence"] for r in manifest["records"])
    Path(right[3]["checkpoint_path"]).write_bytes(b"different again")
    with pytest.raises(ValueError, match="Frozen source SHA mismatch"): build(tmp_path, spec)


def test_v2_analysis_stratifies_phases_and_uses_integer_paired_counts_for_aliases(tmp_path, monkeypatch):
    spec, model = fixture(tmp_path, monkeypatch)
    add_phase(tmp_path, spec, model)
    add_phase(tmp_path, spec, model, "extended", "extended", "extended_v1")
    queue, manifest = build(tmp_path, spec, "behavior")
    path = write(tmp_path / "output_manifest.json", manifest); digest = plan.sha(path)
    sys.path.insert(0, str(ROOT / "src"))
    from replay_study.tasks import choice_predictions
    from analyze_behavioral_plan import collect
    for job in manifest["jobs"]:
        group = manifest["baseline_groups"][job["baseline_group"]]
        policy = group["inference_policy"]
        actual = {**policy, "attention_backend_requested": policy["attention_backend"],
                  "gpu_name": "CPU-fixture-no-GPU-used", "torch_version": "fixture", "cuda_version": "fixture"}
        value = {"model": policy["model"], "revision": policy["revision"], "precision": policy["precision"],
                 "checkpoint": job.get("checkpoint"), "execution_provenance": actual, "complete": True,
                 "project_primary_metric": "acc_norm", "project_auxiliary_metric": "acc", "evaluation_role": "formal", "tasks": {}}
        output = Path(job["output"]); output.parent.mkdir(parents=True, exist_ok=True); paths = [output]
        for name, task in group["tasks"].items():
            prepared = json.loads(Path(task["artifact"]).read_text())
            scores = [-2.] * len(prepared["choices"]); scores[0 if job["kind"] == "initial" else 1] = 0.
            pred = {**prepared, **choice_predictions(scores, prepared["choices"], prepared["gold"])}
            prediction_path = output.with_name(output.stem + f"_{name}.jsonl"); prediction_path.write_text(json.dumps(pred) + "\n")
            paths.append(prediction_path)
            value["tasks"][name] = {"items": 1, "complete_split": True, "batch_size": policy["batch_size"],
                "max_length": policy["max_length"], "data_sha256": task["sha256"], "dataset_revision": task["revision"],
                "predictions_path": str(prediction_path), "predictions_sha256": plan.sha(prediction_path),
                "acc": pred["correct_acc"], "acc_norm": pred["correct_acc_norm"]}
        write(output, value)
        attestation = {"manifest_sha256": digest, "job_id": job["id"], "checkpoint_sha256": job.get("checkpoint_sha256"),
            "weights_identity": job.get("weights_identity"), "verified_frozen_inputs": True,
            "inference_subprocess_wall_seconds": 1., "output_sha256": {str(p): plan.sha(p) for p in paths}}
        write(runner.attestation_path(output), attestation)
        runner.materialize_aliases(manifest, job, attestation)
    bundle = collect(path)
    assert len(bundle["endpoints"]) == 16
    assert {r["role"] for r in bundle["seed_aggregates"]} == {"primary", "extended"}
    assert all(r["n_training_seeds"] == 1 for r in bundle["seed_aggregates"] if r["quantity"] == "accuracy")
    assert all(r["mean"] is None for r in bundle["seed_aggregates"] if r["quantity"] == "conditional_acquisition_rate")
    assert all((r["CW"], r["WC"], r["accuracy_delta"]) == (1, 0, -1.) for r in bundle["endpoints"])
    assert all(r["target_prediction_tokens_at_checkpoint"] == 50 for r in bundle["endpoints"] if r["checkpoint_kind"] == "best")
    assert bundle["metadata"]["model_inference_performed_by_analysis"] is False


@pytest.mark.parametrize("field", ["torch_version", "cuda_version", "gpu", "device", "precision", "attention_implementation",
    "attention_backend_policy", "deterministic_algorithms", "deterministic_warn_only", "allow_tf32_matmul", "allow_tf32_cudnn",
    "cudnn_deterministic", "cudnn_benchmark", "cublas_workspace_config", "model_weight_dtype", "sdpa_backends_enabled.flash",
    "sdpa_backends_enabled.math", "sdpa_backends_enabled.memory_efficient", "sdpa_backends_enabled.cudnn"])
def test_missing_training_provenance_rejected_without_current_environment_fallback(tmp_path, monkeypatch, field):
    spec, model = fixture(tmp_path, monkeypatch)
    phase, row, cfg, summary = add_phase(tmp_path, spec, model)
    obj = summary["provenance"]
    parts = field.split(".")
    for name in parts[:-1]: obj = obj[name]
    del obj[parts[-1]]
    write(row["summary_path"], summary)
    Path(summary["checkpoint_path"]).unlink()
    with pytest.raises(ValueError, match="training execution provenance"):
        build(tmp_path, spec)


@pytest.mark.parametrize("field,wrong", [("device", "cpu"), ("precision", "fp32"),
    ("attention_implementation", "eager"), ("attention_backend_policy", "math"),
    ("deterministic_algorithms", False), ("deterministic_warn_only", True),
    ("allow_tf32_matmul", False), ("allow_tf32_cudnn", False), ("cudnn_deterministic", False), ("cudnn_benchmark", True),
    ("cublas_workspace_config", ":16:8"), ("model_weight_dtype", "torch.bfloat16"), ("teacher_weight_dtype", "torch.bfloat16"),
    ("sdpa_backends_enabled.flash", False), ("sdpa_backends_enabled.math", False),
    ("sdpa_backends_enabled.memory_efficient", False), ("sdpa_backends_enabled.cudnn", False)])
def test_observed_training_flags_must_match_prescribed_policy(tmp_path, monkeypatch, field, wrong):
    spec, model = fixture(tmp_path, monkeypatch)
    phase, row, cfg, summary = add_phase(tmp_path, spec, model)
    obj = summary["provenance"]; parts = field.split(".")
    for name in parts[:-1]: obj = obj[name]
    obj[parts[-1]] = wrong
    write(row["summary_path"], summary)
    Path(summary["checkpoint_path"]).unlink()
    with pytest.raises(ValueError, match="(training execution|Training execution)"):
        build(tmp_path, spec)


@pytest.mark.parametrize("field", ["torch_version", "cuda_version", "gpu"])
def test_recorded_training_versions_and_gpu_define_baseline_identity(tmp_path, monkeypatch, field):
    spec, model = fixture(tmp_path, monkeypatch)
    add_phase(tmp_path, spec, model)
    phase, row, cfg, summary = add_phase(tmp_path, spec, model, "suppA", "supplement_A")
    summary["provenance"][field] = "other-recorded-environment"
    write(row["summary_path"], summary)
    queue, manifest = build(tmp_path, spec, "behavior")
    assert len(manifest["baseline_groups"]) == 2
    mapped = "gpu_name" if field == "gpu" else field
    assert {g["inference_policy"][mapped] for g in manifest["baseline_groups"].values()} == {
        "other-recorded-environment", "CPU-fixture-no-GPU-used" if field == "gpu" else "fixture"}


@pytest.mark.parametrize("field", ["torch_version", "cuda_version", "gpu_name"])
@pytest.mark.parametrize("mutation", ["missing", "mismatch"])
def test_scorer_result_must_match_recorded_training_versions_and_gpu(tmp_path, monkeypatch, field, mutation):
    spec, model = fixture(tmp_path, monkeypatch)
    add_phase(tmp_path, spec, model)
    queue, manifest = build(tmp_path, spec)
    job = manifest["jobs"][0]
    policy = manifest["baseline_groups"][job["baseline_group"]]["inference_policy"]
    actual = {**policy, "attention_backend_requested": policy["attention_backend"]}
    if mutation == "missing": del actual[field]
    else: actual[field] = "different-runtime"
    write(job["output"], {"model": policy["model"], "revision": policy["revision"], "precision": policy["precision"],
        "checkpoint": None, "execution_provenance": actual, "corpora": {}})
    with pytest.raises(ValueError, match="Scorer execution differs from frozen " + field):
        runner.validate_result(manifest, job)


def test_teacher_dtype_cannot_be_missing_when_teacher_was_required(tmp_path, monkeypatch):
    spec, model = fixture(tmp_path, monkeypatch)
    phase, row, cfg, summary = add_phase(tmp_path, spec, model)
    cfg["evaluate_teacher_kl"] = True
    summary["config"]["evaluate_teacher_kl"] = True
    del summary["provenance"]["teacher_weight_dtype"]
    with pytest.raises(ValueError, match="teacher_weight_dtype"):
        plan.training_execution(cfg, summary)


@pytest.mark.parametrize("field", ["torch_version", "cuda_version", "gpu_name"])
def test_worker_preflight_rejects_version_or_gpu_difference_before_scoring(field):
    observed = {"torch_version": "fixture", "cuda_version": "fixture-cuda", "gpu_name": "fixture-gpu"}
    fake = SimpleNamespace(__version__=observed["torch_version"], version=SimpleNamespace(cuda=observed["cuda_version"]),
                           cuda=SimpleNamespace(get_device_name=lambda _: observed["gpu_name"]))
    assert runner.verify_environment(observed, fake) == observed
    with pytest.raises(ValueError, match=field):
        runner.verify_environment({**observed, field: "other"}, fake)
