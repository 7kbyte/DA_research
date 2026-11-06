"""Temporary tiny checkpoints only; never touch actual retraining outputs."""
import copy
import importlib.util
import json
from pathlib import Path
import time
from datetime import datetime, timezone

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qa_postrun", ROOT / "scripts/verify_reproduction_qa_postrun.py")
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


def put(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    workspace = tmp_path / "archive/research"; workspace.mkdir(parents=True)
    algorithm = workspace / "scripts/audit_formal_runs.py"; algorithm.parent.mkdir()
    algorithm.write_bytes((ROOT / "scripts/audit_formal_runs.py").read_bytes())
    fp = qa.load_algorithm(qa.proof(algorithm))
    namespace = "tiny_qa"; state_path = workspace / "logs/state.json"
    registry, jobs, originals, final_runs, best_runs = [], [], [], [], []
    metadata_root = tmp_path / "accepted_best"
    provenance = {
        "torch_version": "synthetic", "cuda_version": "synthetic", "gpu": "synthetic",
        "attention_backend_policy": "auto", "attention_implementation": "sdpa", "precision": "bf16",
        "deterministic_algorithms": True, "deterministic_warn_only": False,
        "cublas_workspace_config": ":4096:8", "allow_tf32": True,
        "allow_tf32_matmul": True, "allow_tf32_cudnn": True,
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "sdpa_backends_enabled": dict.fromkeys(("flash", "math", "memory_efficient", "cudnn"), True),
        "model_weight_dtype": "torch.float32", "teacher_weight_dtype": "torch.float32",
    }
    payloads = {}
    for index, group in enumerate(("B1", "B2", "B3", "B4")):
        origin = "old_" + group; run_id = namespace + "__" + origin
        old_dir = tmp_path / "originals" / origin; old_dir.mkdir(parents=True)
        output = tmp_path / "scratch/runs" / run_id
        config = {
            "run_label": run_id, "output_dir": str(output), "resume_from": None,
            "target_prediction_tokens": 70, "replay_ratio": .1, "seed": 11,
            "deterministic_algorithms": True, "precision": "bf16", "attention_backend": "auto",
            "attention_implementation": "sdpa", "allow_tf32": True,
        }
        source_config = {**config, "run_label": origin, "output_dir": str(old_dir)}
        source_path = put(workspace / "configs" / (origin + ".json"), source_config)
        config_path = put(workspace / "configs" / (run_id + ".json"), config)
        summary_copy = workspace / "results" / (run_id + ".json")
        registry.append({"id": run_id, "group": group, "config_path": str(config_path),
                         "config_sha256": qa.sha(config_path), "output_dir": str(output), "summary_path": str(summary_copy)})
        jobs.append({"id": run_id, "argv": ["/usr/bin/python", str(workspace / "scripts/make_experiments.py"),
                     "run", "--config", str(config_path), "--summary-copy", str(summary_copy)],
                     "expected_seconds": 600, "outputs": [str(output / "summary.json"), str(summary_copy)]})
        full = {"step": 7, "target_prediction_tokens": 70, "replay_prediction_tokens": 7}
        best_counters = {"step": index + 2, "target_prediction_tokens": (index + 2) * 10,
                         "replay_prediction_tokens": index + 2}
        final_state = {"a.weight": torch.arange(5, dtype=torch.float32) + index,
                       "z.last": torch.tensor([index + .25], dtype=torch.float32)}
        best_state = {k: v + .5 for k, v in final_state.items()}
        old_final, old_best = old_dir / "latest.pt", old_dir / "checkpoint_best.pt"
        selection = {"nll": 2. + index / 10, "step": best_counters["step"],
                     "target_prediction_tokens": best_counters["target_prediction_tokens"], "checkpoint_path": str(old_best)}
        fm = {"config": source_config, "provenance": provenance, "counters": full}
        bm = {"config": source_config, "provenance": provenance, "counters": best_counters,
              "selection": selection, "checkpoint_kind": "model_only_best_target_validation"}
        torch.save({**fm, "model_state": final_state}, old_final)
        torch.save({**bm, "model_state": best_state}, old_best)
        member = "metadata/" + origin + ".pt"; metadata = metadata_root / member
        metadata.parent.mkdir(parents=True, exist_ok=True); torch.save(bm, metadata)
        final_runs.append({"condition_run_id": origin, "original_checkpoint_metadata": fm,
                           "summary_counters": full, "source_checkpoint_path": str(old_final),
                           "source_checkpoint_sha256": qa.sha(old_final), "model_fingerprint": fp(old_final)})
        best_runs.append({"run_id": origin, "kind": "best", "source_config_sha256": qa.sha(source_path),
                          "source_checkpoint_path": str(old_best), "source_checkpoint_sha256": qa.sha(old_best),
                          "model_fingerprint": fp(old_best), "metadata_member": member, "metadata_sha256": qa.sha(metadata),
                          "actual_exposure": {"checkpoint_step": best_counters["step"],
                              "target_prediction_tokens": best_counters["target_prediction_tokens"],
                              "replay_prediction_tokens": best_counters["replay_prediction_tokens"],
                              "full_training_target_prediction_tokens": 70, "full_training_replay_prediction_tokens": 7}})
        originals.append({"reproduction_run_id": run_id, "reproduction_of": origin, "group": group,
                          "relocated_config": qa.proof(config_path), "source_config": qa.proof(source_path),
                          "expected_final_model_state_fingerprint": fp(old_final),
                          "required_original_execution_provenance": provenance,
                          "planned_final_checkpoint": str(output / "checkpoints/latest.pt")})
        payloads[run_id] = (final_state, best_state)
    registry_path = put(workspace / "registry.json", registry)
    queue_path = put(workspace / "queue.json", jobs)
    final_manifest = put(tmp_path / "final_manifest.json", {"logical_runs": final_runs})
    best_manifest = put(tmp_path / "best_manifest.json", {"status": "all_unique_weights_roundtrip_verified", "logical_runs": best_runs})
    durable = put(tmp_path / "durable_receipt.json", {"manifest_sha256": qa.sha(best_manifest)})
    acceptance = put(tmp_path / "acceptance.json", {"manifest_sha256": qa.sha(best_manifest),
        "receipt_path": str(durable), "receipt_sha256": qa.sha(durable),
        "receipt": {"status": "passed", "all_original_weight_bytes_compared": True}})
    relocation = put(tmp_path / "relocation.json", {"new_workspace": str(workspace), "new_scratch": str(tmp_path / "scratch")})
    identity = put(tmp_path / "identity.json", {"relocation_manifest": qa.proof(relocation), "scientific_core_files": [
        {"restored_path": str(algorithm), "sha256": qa.sha(algorithm)}]})
    plan = put(tmp_path / "plan.json", {"namespace": namespace,
        "registry": qa.proof(registry_path), "queue": {**qa.proof(queue_path), "jobs": jobs},
        "scientific_identity_audit": qa.proof(identity), "planned_state": str(state_path),
        "planned_gpus": [0, 1, 2, 3], "qa_deadline_utc": "2099-01-01T00:00:00Z",
        "future_full_run_comparison": {"expected_runs": originals, "source_weight_manifest_sha256": qa.sha(final_manifest)}})
    expected = qa.prepare(plan, qa.sha(plan), final_manifest, best_manifest, qa.sha(best_manifest),
                          metadata_root, acceptance, algorithm, qa.sha(algorithm))
    expected_path = put(tmp_path / "expected.json", expected)

    def complete():
        records = {}
        for index, run in enumerate(expected["runs"]):
            cfg = qa.read(run["config"]["path"]); output = Path(run["output_dir"])
            output.mkdir(parents=True); (output / "checkpoints").mkdir()
            best = run["endpoints"][1]; final = run["endpoints"][0]
            final_state, best_state = payloads[run["run_id"]]
            common = {"config": cfg, "provenance": provenance}
            torch.save({**common, "model_state": final_state, "counters": final["counters"],
                        "optimizer_state": {}, "best_target_validation": best["selection"]}, final["checkpoint"])
            torch.save({**common, "model_state": best_state, "counters": best["counters"],
                        "selection": best["selection"], "checkpoint_kind": best["checkpoint_kind"]}, best["checkpoint"])
            summary = {**common, "status": "completed", "counters": run["full_counters"],
                       "best_target_validation": best["selection"], "checkpoint_path": final["checkpoint"]}
            put(run["internal_summary"], summary); put(run["summary_copy"], summary)
            job = jobs[index]
            records[run["run_id"]] = {"status": "completed", "returncode": 0, "missing_outputs": [],
                "definition_sha256": qa.hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
                "argv": job["argv"], "pid": 2**31 - 1, "gpu": str(index), "started_at_epoch": time.time() - 120,
                "ended_at_utc": datetime.fromtimestamp(time.time() + 120, timezone.utc).isoformat()}
        put(state_path, {"jobs": records, "active_jobs": [], "scheduler_pid": 2**31 - 1,
                         "deadline_utc": "2099-01-01T00:00:00Z"})

    return {"expected": expected, "path": expected_path, "complete": complete, "state": state_path,
            "out": tmp_path / "postrun.json", "algorithm": algorithm}


def run(prepared):
    return qa.verify_to_receipt(prepared["path"], qa.sha(prepared["path"]), prepared["out"])


def edit_checkpoint(prepared, kind, mutate):
    endpoint = next(e for e in prepared["expected"]["runs"][0]["endpoints"] if e["kind"] == kind)
    payload = torch.load(endpoint["checkpoint"], map_location="cpu", weights_only=False)
    mutate(payload); torch.save(payload, endpoint["checkpoint"])


def test_all_eight_complete_full_states_pass_without_fixed_306_steps(prepared):
    prepared["complete"](); result = run(prepared)
    assert result["status"] == "passed_all_final_and_best_full_model_state_bytes"
    assert len(result["endpoint_proofs"]) == 8 and result["checkpoint_read_operations_count"] == 24
    assert {e["actual_exposure"]["step"] for e in result["endpoint_proofs"] if e["kind"] == "final"} == {7}
    exact_sha_bytes = sum(Path(e["checkpoint"]).stat().st_size for e in result["endpoint_proofs"])
    assert result["fresh_full_checkpoint_SHA_bytes"] == exact_sha_bytes
    assert result["serialized_checkpoint_read_bytes_total"] > exact_sha_bytes * 2
    assert not result["training_or_validation_trajectories_compared"]
    assert not result["optimizer_or_RNG_or_stream_states_compared"]


def test_no_actual_new_run_never_produces_passed(prepared):
    result = run(prepared)
    assert result["status"].startswith("failed_") and not result["endpoint_proofs"]
    assert result["serialized_checkpoint_read_bytes_total"] == 0


@pytest.mark.parametrize("mutation", ["incomplete", "returncode", "stop", "definition", "active", "live_scheduler", "live_group"])
def test_full_phase_gate_precedes_all_checkpoint_reads(prepared, mutation):
    prepared["complete"](); state = qa.read(prepared["state"]); record = next(iter(state["jobs"].values()))
    if mutation == "incomplete": record["status"] = "running"
    elif mutation == "returncode": record["returncode"] = 1
    elif mutation == "stop": record["stop_requested_at"] = 0
    elif mutation == "definition": record["definition_sha256"] = "changed"
    elif mutation == "active": state["active_jobs"] = [next(iter(state["jobs"]))]
    elif mutation == "live_scheduler": state["scheduler_pid"] = qa.os.getpid()
    else: record["pid"] = qa.os.getpgrp()
    put(prepared["state"], state); result = run(prepared)
    assert result["status"].startswith("failed_") and not result["checkpoint_read_operations"]


@pytest.mark.parametrize("kind", ["final", "best"])
def test_last_tensor_byte_change_fails_complete_weight_proof(prepared, kind):
    prepared["complete"]()
    edit_checkpoint(prepared, kind, lambda x: x["model_state"]["z.last"].view(torch.uint8).__setitem__(-1, 42))
    result = run(prepared)
    assert result["status"].startswith("failed_")
    assert "Full model_state bytes differ" in result["error"]["message"]


@pytest.mark.parametrize("kind,mutation", [("best", "selection"), ("best", "kind"), ("best", "exposure"),
                                            ("final", "config"), ("final", "policy"), ("best", "policy_missing")])
def test_checkpoint_metadata_cannot_be_faked_by_equal_weights(prepared, kind, mutation):
    prepared["complete"]()
    def change(x):
        if mutation == "selection": x["selection"]["step"] += 1
        elif mutation == "kind": x["checkpoint_kind"] = "final"
        elif mutation == "exposure": x["counters"]["replay_prediction_tokens"] -= 1
        elif mutation == "config": x["config"]["run_label"] = "old_B1"
        elif mutation == "policy": x["provenance"]["deterministic_warn_only"] = True
        else: del x["provenance"]["sdpa_backends_enabled"]["cudnn"]
    edit_checkpoint(prepared, kind, change); result = run(prepared)
    assert result["status"].startswith("failed_")


def test_original_file_or_symlink_cannot_be_substituted(prepared):
    prepared["complete"](); endpoint = prepared["expected"]["runs"][0]["endpoints"][0]
    p = Path(endpoint["checkpoint"]); p.unlink(); p.symlink_to(endpoint["original_checkpoint"])
    result = run(prepared)
    assert result["status"].startswith("failed_") and not result["checkpoint_read_operations"]


def test_original_hardlink_and_duplicate_endpoint_inode_rejected(prepared):
    prepared["complete"](); run = prepared["expected"]["runs"][0]
    endpoint = run["endpoints"][0]; p = Path(endpoint["checkpoint"])
    p.unlink(); qa.os.link(endpoint["original_checkpoint"], p)
    result = qa.verify_to_receipt(prepared["path"], qa.sha(prepared["path"]), prepared["out"])
    assert result["status"].startswith("failed_") and not result["checkpoint_read_operations"]
    p.unlink(); qa.os.link(run["endpoints"][1]["checkpoint"], p)
    result = qa.verify_to_receipt(prepared["path"], qa.sha(prepared["path"]), prepared["out"].with_name("duplicate_inode.json"))
    assert result["status"].startswith("failed_") and not result["checkpoint_read_operations"]


def test_summary_config_and_best_selection_must_match_before_read(prepared):
    prepared["complete"](); r = prepared["expected"]["runs"][0]
    x = qa.read(r["summary_copy"]); x["config"]["run_label"] = "original"
    put(r["summary_copy"], x); result = run(prepared)
    assert result["status"].startswith("failed_") and not result["checkpoint_read_operations"]


def test_missing_best_and_source_code_change_are_hard_failures(prepared):
    prepared["complete"](); Path(prepared["expected"]["runs"][-1]["endpoints"][1]["checkpoint"]).unlink()
    result = run(prepared)
    assert result["status"].startswith("failed_") and not result["checkpoint_read_operations"]
    prepared["out"] = prepared["out"].with_name("source_changed.json")
    prepared["algorithm"].write_text(prepared["algorithm"].read_text() + "\n# changed\n")
    result = run(prepared)
    assert result["status"].startswith("failed_") and not result["checkpoint_read_operations"]


def test_checkpoint_mutation_between_fingerprint_and_SHA_fails(prepared, monkeypatch):
    prepared["complete"](); original = qa.load_algorithm
    def changing(reference):
        function = original(reference)
        def fingerprint(stream):
            result = function(stream)
            p = Path(stream.raw.name); s = p.stat(); qa.os.utime(p, ns=(s.st_atime_ns, s.st_mtime_ns + 1000))
            return result
        return fingerprint
    monkeypatch.setattr(qa, "load_algorithm", changing)
    result = run(prepared)
    assert result["status"].startswith("failed_") and "changed during" in result["error"]["message"]


def test_failed_or_passed_receipt_cannot_be_overwritten(prepared):
    result = run(prepared); before = prepared["out"].read_bytes()
    with pytest.raises(FileExistsError): run(prepared)
    assert prepared["out"].read_bytes() == before


def test_expected_SHA_and_CPU_only_gate_fail_closed(prepared, monkeypatch):
    prepared["complete"]()
    result = qa.verify_to_receipt(prepared["path"], "0" * 64, prepared["out"])
    assert result["status"].startswith("failed_") and not result["checkpoint_read_operations"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    result = qa.verify_to_receipt(prepared["path"], qa.sha(prepared["path"]), prepared["out"].with_name("not_cpu.json"))
    assert result["status"].startswith("failed_") and not result["checkpoint_read_operations"]
