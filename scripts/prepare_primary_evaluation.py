#!/usr/bin/env python3
"""CPU-only primary endpoint freeze; no model initialization or scheduling."""
import argparse
import json
from evaluation_plan import ROOT, Hashes, absolute, build, encoded, read, sha


def prepare(prefix="primary_evaluation_v2"):
    if not prefix.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Safe output prefix required")
    registry = ROOT / "configs/formal_deterministic_registry.json"
    snapshot_path = ROOT / "artifacts/provenance/formal_deterministic_input_snapshot.json"
    protocol_path = ROOT / "docs/formal_deterministic_protocol.json"
    audit_path = ROOT / "results/formal_run_audit.json"
    snapshot, protocol, audit, rows = (read(p) for p in (snapshot_path, protocol_path, audit_path, registry))
    if audit.get("status") != "passed" or audit.get("completed_runs") != 36 or audit.get("model_weights_compared") is not True:
        raise ValueError("Complete passed strict primary audit with model comparison required")
    if len(rows) != 36 or any(r.get("active") is not True or r.get("phase") != "formal" for r in rows):
        raise ValueError("Complete active primary 36-run registry required")
    if sha(protocol_path) != snapshot["protocol"]["docs/formal_deterministic_protocol.json"]["sha256"]:
        raise ValueError("Primary protocol changed since scientific input freeze")
    proof = lambda path: {"path": str(path), "sha256": sha(path)}
    frozen_data = lambda path: {"path": str(absolute(path)), "sha256": snapshot["data"][str(absolute(path).relative_to(ROOT))]["sha256"]}
    config_proofs = {r["id"]: snapshot["configs"][str(absolute(r["config_path"]).relative_to(ROOT))]["sha256"] for r in rows}
    cfg = read(rows[0]["config_path"])
    phase = {"name": "primary", "role": "primary", "science_frozen": True, "expected_phase": "formal",
        "model_tag": "pythia-410m", "model": {"path": cfg["model_name_or_path"], "revision": cfg["model_revision"]},
        "data_family": "main_v3", "registry": proof(registry), "protocol": proof(protocol_path),
        "protocol_expect": {"active": True, "phase": "formal_main_strict_deterministic", "data_version": "v3",
            "model_revision": cfg["model_revision"], "shared_learning_rate": cfg["learning_rate"],
            "shared_lambda": protocol["shared_lambda"], "formal_target_prediction_tokens": 10_000_000,
            "frozen_for_formal_at_utc": protocol["frozen_for_formal_at_utc"]},
        "training_data": frozen_data(cfg["target_path"]),
        "validation_data": {k: frozen_data(v) for k, v in cfg["validation_paths"].items()},
        "test_data": {k: frozen_data(v) for k, v in rows[0]["test_paths"].items()}, "config_sha256": config_proofs,
        "source_proof_files": {str(snapshot_path): sha(snapshot_path), str(audit_path): sha(audit_path)}}
    eq_paths = [ROOT / "results/formal_model_equivalence_proofs.json", ROOT / "results/formal_best_model_equivalence_proofs.json"]
    for path in eq_paths:
        if read(path).get("status") != "passed":
            raise ValueError("Passed final and best equivalence proofs required")
    taskroot = ROOT / "artifacts/behavioral"
    scope_path = ROOT / "configs/behavioral_scope_execution_freeze_v2.json"
    scope = read(scope_path)
    if scope.get("state") != "root_frozen_before_first_real_score" or scope.get("real_task_scores_seen") is not False:
        raise ValueError("Pre-score root-frozen behavior scope/execution record required")
    phase["source_proof_files"][str(scope_path)] = sha(scope_path)
    spec = {"schema_version": 2, "science_frozen": True, "checkpoints": ["final", "best"], "phases": [phase],
        "model_equivalence_proofs": [proof(path) for path in eq_paths],
        "behavior": {"data_root": str(taskroot), "batch_size": scope["behavior_batch_size_by_model"]["pythia-410m"], "max_length": scope["max_length"],
            "data_manifest_sha256": sha(taskroot / "data_manifest.json"),
            "harness_manifest_sha256": sha(taskroot / "sources/harness_manifest.json")},
        "inference_launch_authorized_by_preparation": False,
        "subsequent_science_rule": "Supplementary methods/grids preregistered; LR/lambda selection remains validation-only; heldout/task results cannot change scientific conditions"}
    specpath = ROOT / f"configs/{prefix}_phase_spec.json"
    destinations = [specpath, ROOT / f"results/{prefix}_preparation.json"]
    for mode in ("nll", "behavior"):
        destinations += [ROOT / f"configs/{prefix}_{mode}_queue.json", ROOT / f"docs/{prefix}_{mode}_manifest.json"]
    for path in destinations:
        if path.exists():
            raise FileExistsError("Use a new output prefix: " + str(path))
    specpath.write_bytes(encoded(spec))
    hashes, evidence = Hashes(), {}
    for mode in ("nll", "behavior"):
        manifest_path = ROOT / f"docs/{prefix}_{mode}_manifest.json"
        queue_path = ROOT / f"configs/{prefix}_{mode}_queue.json"
        queue, manifest = build(specpath, ROOT / f"results/{prefix}/{mode}", manifest_path, mode,
                                180 if mode == "nll" else 300, hashes=hashes)
        from schedule_jobs import validate_queue
        validate_queue(queue)
        manifest_path.write_bytes(encoded(manifest)); queue_path.write_bytes(encoded(queue))
        evidence[mode] = {"queue": str(queue_path), "queue_sha256": sha(queue_path),
            "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "workload": manifest["workload"]}
    evidence.update(phase_spec=str(specpath), phase_spec_sha256=sha(specpath),
                    CPU_only=True, scorer_started=False, model_weights_deserialized=False, scheduler_started=False)
    destinations[1].write_bytes(encoded(evidence))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="primary_evaluation_v2")
    value = prepare(parser.parse_args().output_prefix)
    print(json.dumps({mode: value[mode]["workload"] for mode in ("nll", "behavior")}))


if __name__ == "__main__":
    main()
