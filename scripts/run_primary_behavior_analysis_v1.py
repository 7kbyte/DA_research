"""Independent operational closure/journal around the unmodified primary guard.

prepare never opens behavioral results. inspect-state reads attestations only.
Only run with exact explicit ROOT authority can invoke the original check/analyze.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/primary_evaluation_v2_execution_matched_behavior_manifest.json"
MANIFEST_SHA = "312d8e150030d489593e7e705699ec1da5f7d505e349ecd15a978eca2c2f7934"
QUEUE = ROOT / "configs/primary_evaluation_v2_execution_matched_behavior_queue.json"
QUEUE_SHA = "393f3df2f0213e2e4e4d2387109b750825c797b909f4b05d16cd3cb739f6aaa9"
REVIEW = ROOT / "results/primary_behavior_analysis_static_review_v1/receipt.json"
GUARD = ROOT / "scripts/guarded_behavior_analysis.py"
COUNTS = {"physical_jobs": 67, "logical_endpoints": 72, "proved_aliases": 6, "initial_baselines": 1}
JOURNAL = ROOT / "artifacts/provenance/primary_behavior_first_reads_v1" / (MANIFEST_SHA + ".json")


def require(value, message):
    if not value: raise ValueError(message)


def now(): return datetime.now(timezone.utc).isoformat()
def encoded(value): return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
def signature(path):
    s = Path(path).stat()
    return {k: getattr(s, "st_" + k) for k in ("dev", "ino", "size", "mtime_ns", "ctime_ns")}
def sha_bytes(raw): return hashlib.sha256(raw).hexdigest()
def canonical_proof(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {"path": str(path), "sha256": sha_bytes(raw)}


def small(path, expected=None):
    path = Path(path).resolve(strict=True); before = signature(path)
    require(before["size"] < 16 << 20 and path.suffix not in (".pt", ".safetensors", ".jsonl"), "Only explicit small metadata/code, never score predictions or weights")
    raw = path.read_bytes(); require(signature(path) == before, "Small source changed")
    proof = {"path": str(path), "sha256": sha_bytes(raw)}
    require(expected is None or proof["sha256"] == expected, "Bound source SHA mismatch: " + str(path))
    return proof, raw


def read_proof(item):
    proof, raw = small(item["path"], item["sha256"])
    return json.loads(raw)


def exclusive(path, value):
    return exclusive_bytes(path, encoded(value))


def exclusive_bytes(path, raw):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    return canonical_proof(path)


def verify_task_inputs(mapping):
    """Hash only manifest-declared prepared task inputs; never parse their rows."""
    for name, expected in mapping.items():
        path = Path(name); require(path.suffix == ".jsonl", "Only declared prepared task inputs")
        before = signature(path); digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""): digest.update(block)
        require(signature(path) == before and digest.hexdigest() == expected, "Prepared task input changed")


def definition_sha(job): return sha_bytes(json.dumps(job, sort_keys=True).encode())
def attestation_path(output):
    path = Path(output); return path.with_name(path.stem + "_execution.json")
def output_set(output, tasks):
    p = Path(output)
    return {str(p), *[str(p.with_name(p.stem + "_" + name + ".jsonl")) for name in tasks]}


def validate_scope(manifest, queue, counts=COUNTS):
    require(manifest.get("schema_version") == 2 and manifest.get("mode") == "behavior", "Only frozen v2 behavior scope")
    jobs, records = manifest["jobs"], manifest["records"]
    require(len(jobs) == counts["physical_jobs"] and len(records) == counts["logical_endpoints"] and len(queue) == len(jobs), "Full physical/logical/queue scope required")
    lookup = {j["id"]: j for j in jobs}; definitions = {j["id"]: j for j in queue}; by_id = {r["id"]: r for r in records}
    require(len(lookup) == len(jobs) and len(by_id) == len(records) and set(lookup) == set(definitions), "Unique full inference/logical IDs required")
    initial = [j for j in jobs if j["kind"] == "initial"]
    require(len(initial) == len(manifest["baseline_groups"]) == counts["initial_baselines"] and all(j["kind"] in ("initial", "adapted") for j in jobs), "Primary baseline closure only; no reuse/foreign phase")
    aliases = []
    for job in jobs:
        group = manifest["baseline_groups"][job["baseline_group"]]
        if job["kind"] == "initial":
            require(group["initial_job_id"] == job["id"] and group["initial_output"] == job["output"] and not group.get("initial_origin"), "Initial baseline must be in this full closure")
        attached = [r for r in records if r["inference_job_id"] == job["id"]]
        require((not attached) == (job["kind"] == "initial"), "Physical/adapted endpoint coverage differs")
        canonical = [r for r in attached if r["output"] == job["output"]]
        require(job["kind"] == "initial" or len(canonical) == 1, "One real canonical endpoint required")
        expected_outputs = output_set(job["output"], group["tasks"]) | {str(attestation_path(job["output"]))}
        for row in attached:
            require(row["baseline_group"] == job["baseline_group"] and row["initial_job_id"] == group["initial_job_id"], "Endpoint baseline identity differs")
            if row["output"] != job["output"]:
                require(row.get("inference_alias_of") == canonical[0]["id"] and row.get("source_model_equivalence"), "Unproved logical alias")
                aliases.append((row, canonical[0], job))
                expected_outputs |= output_set(row["output"], group["tasks"]) | {str(attestation_path(row["output"]))}
        require(set(definitions[job["id"]]["outputs"]) == expected_outputs, "Frozen complete summary/task/alias output closure differs")
    require(len(aliases) == counts["proved_aliases"], "Expected complete proved alias set differs")
    return aliases


def verify_equivalence(alias, canonical, sources):
    e = alias["source_model_equivalence"]
    item = {"path": e["proof"], "sha256": e["proof_sha256"]}; p = read_proof(item); sources[item["path"]] = item["sha256"]
    require(p.get("schema_version") == 1 and p.get("status") == "passed" and p.get("algorithm") == "sha256_sorted_key_dtype_shape_contiguous_uint8_v1", "Complete-state proof schema/status differs")
    c = p["comparisons"][e["comparison_index"]]; fp = e["fingerprint"]
    require(c["bitwise_model_state_equal"] is True and c["left"]["model_state_fingerprint"] == c["right"]["model_state_fingerprint"] == fp, "Alias complete model state is not proved equal")
    expected_sides = {(alias["checkpoint"], alias["checkpoint_sha256"]), (canonical["checkpoint"], canonical["checkpoint_sha256"])}
    require({(side["checkpoint"], side["checkpoint_sha256"]) for side in (c["left"], c["right"])} == expected_sides, "Proof is not bound to both exact checkpoint files")
    require({e["left_checkpoint_sha256"], e["right_checkpoint_sha256"]} == {s[1] for s in expected_sides}
            and alias["weights_identity"] == canonical["weights_identity"] == "model_state:" + fp["sha256_model_state"], "Complete-state identity/checkpoint SHA differs")
    require(alias["kind"] == canonical["kind"] and alias["training_seed"] == canonical["training_seed"], "Cannot alias different endpoint/seed")
    for key in ("algorithm_source", "source_audit", "source_registry", "source_snapshot", "producer_source"):
        if not p.get(key): continue
        evidence = p[key]; path = Path(evidence["path"])
        if not path.is_absolute(): path = ROOT / path
        q, _ = small(path, evidence["sha256"]); sources[q["path"]] = q["sha256"]


def prepare(output, analysis_output):
    manifest_proof, raw = small(MANIFEST, MANIFEST_SHA); manifest = json.loads(raw)
    queue_proof, raw = small(QUEUE, QUEUE_SHA); queue = json.loads(raw)
    review_proof, raw = small(REVIEW); review = json.loads(raw)
    sources = {str(Path(__file__).resolve()): small(__file__)[0]["sha256"]}
    for item in review["source_proofs"].values():
        p, _ = small(item["path"], item["sha256"]); sources[p["path"]] = p["sha256"]
    for path, digest in manifest["common_input_sha256"].items(): small(path, digest); sources[path] = digest
    aliases = validate_scope(manifest, queue)
    require(all(r["role"] == "primary" and r["phase_name"] == "primary" and r["model_tag"] == "pythia-410m" and r["data_family"] == "main_v3" for r in manifest["records"]), "Supplemental/foreign behavior phase forbidden")
    require(Counter(r["kind"] for r in manifest["records"]) == {"final": 36, "best": 36}, "Full final+best72 required")
    require(all(alias["group"] == "D" and canonical["group"] == "B4" for alias, canonical, _ in aliases), "Only six bound D/B4 aliases are frozen")
    for alias, canonical, _ in aliases: verify_equivalence(alias, canonical, sources)
    for definition in queue:
        expected = ["/usr/bin/python", str(ROOT / "scripts/run_frozen_evaluation.py"), "--manifest", str(MANIFEST), "--manifest-sha256", MANIFEST_SHA, "--job-id", definition["id"], "--device", "cuda:0"]
        require(definition["argv"] == expected, "Frozen scheduler dispatcher argv differs")
    plan = {"schema_version": 1, "scope": "primary_complete_behavior_67physical_72logical_v1", "prepared_at_utc": now(),
            "manifest": manifest_proof, "frozen_queue": queue_proof, "static_review": review_proof, "counts": COUNTS,
            "source_sha256": sources, "guard": str(GUARD), "journal_path": str(JOURNAL), "analysis_output": str(Path(analysis_output).resolve()),
            "frozen_definitions": queue, "analysis_policy": manifest["analysis_policy"], "behavior_metrics": manifest["behavior_metrics"],
            "analysis_task_inputs_sha256": {task["artifact"]: task["sha256"] for group in manifest["baseline_groups"].values() for task in group["tasks"].values()},
            "first_read_rule": "After exact true ROOT authority, full state/attestation closure, successful original check; exclusive per-manifest journal then unchanged original analyze",
            "checkpoint_payload_verification": "Stored SHA/model-state proof identity only; no analysis-stage checkpoint payload reads",
            "prepared_only": True, "scores_or_predictions_opened": False, "original_check_or_analyze_called": False}
    require(not Path(plan["analysis_output"]).exists() and not JOURNAL.exists(), "Analysis/journal already exists")
    plan_proof = exclusive(output, plan)
    authority = {"schema_version": 1, "authorized_by": "root", "allow_score_analysis": False,
                 "plan": plan_proof, "manifest": manifest_proof, "manifest_sha256": MANIFEST_SHA, "frozen_queue": queue_proof,
                 "state_proof": None, "source_sha256": sources, "expected_counts": COUNTS,
                 "expected_inference_jobs": 67, "expected_logical_endpoints": 72, "journal_path": str(JOURNAL),
                 "allow_partial": False, "analysis_output": plan["analysis_output"],
                 "note": "Disabled template. After complete inspect-state proof root must issue a separate exact true authority and supply its SHA; never rewrite this template."}
    exclusive(Path(output).with_name(Path(output).stem + "_ROOT_FALSE_TEMPLATE.json"), authority)
    return plan_proof


def load_plan(item):
    plan = read_proof(item)
    require(plan.get("scope") == "primary_complete_behavior_67physical_72logical_v1" and plan["counts"] == COUNTS
            and plan["manifest"] == {"path": str(MANIFEST), "sha256": MANIFEST_SHA}
            and plan["frozen_queue"] == {"path": str(QUEUE), "sha256": QUEUE_SHA}
            and plan["journal_path"] == str(JOURNAL), "Only this exact full primary scope is allowed")
    for path, digest in plan["source_sha256"].items(): small(path, digest)
    read_proof(plan["static_review"])
    manifest = read_proof(plan["manifest"])
    validate_scope(manifest, read_proof(plan["frozen_queue"]))
    task_inputs = {task["artifact"]: task["sha256"] for group in manifest["baseline_groups"].values() for task in group["tasks"].values()}
    require(plan["analysis_task_inputs_sha256"] == task_inputs, "Task input mapping differs from frozen plan")
    verify_task_inputs(task_inputs)
    require(plan["frozen_definitions"] == read_proof(plan["frozen_queue"]), "Plan definitions differ")
    return plan


def state_records(definitions, actual_queue, state):
    jobs = {j["id"]: j for j in actual_queue}
    require(len(jobs) == len(actual_queue), "Duplicate actual queue ID")
    result = {}
    for definition in definitions:
        key = definition["id"]
        require(jobs.get(key) == definition, "Actual scheduler definition differs: " + key)
        s = state.get("jobs", {}).get(key)
        require(isinstance(s, dict) and s.get("definition_sha256") == definition_sha(definition), "State definition SHA missing/different: " + key)
        require(s.get("status") == "completed" and type(s.get("returncode")) is int and s.get("returncode") == 0 and s.get("missing_outputs") == []
                and not any("stop_requested" in name for name in s) and s.get("argv") == definition["argv"], "Incomplete/failed/stopped/mismatched actual argv: " + key)
        result[key] = s
    return result


def metadata_attestations(manifest, digest):
    result = []
    for job in manifest["jobs"]:
        group = manifest["baseline_groups"][job["baseline_group"]]
        pairs = [(None, job["output"])] + [(r, r["output"]) for r in manifest["records"] if r["inference_job_id"] == job["id"] and r["output"] != job["output"]]
        for logical, output in pairs:
            ap, raw = small(attestation_path(output)); a = json.loads(raw); expected = logical or job
            require(a.get("manifest_sha256") == digest and a.get("job_id") == job["id"] and a.get("baseline_group") == job["baseline_group"]
                    and a.get("verified_frozen_inputs") is True and a.get("checkpoint_sha256") == expected.get("checkpoint_sha256")
                    and a.get("weights_identity") == expected.get("weights_identity") and a.get("actual_argv") == job["argv"], "Attestation source/scorer/checkpoint/model-state identity differs")
            require(a.get("inference_alias") is (logical is not None) and a.get("source_model_equivalence") == expected.get("source_model_equivalence"), "Canonical/alias model proof differs")
            if logical:
                require(a.get("logical_record_id") == logical["id"] and a.get("canonical_output") == str(Path(job["output"]).resolve())
                        and a.get("inference_subprocess_wall_seconds") == 0, "Alias must bind canonical inference with zero extra forward")
            require(set(a["output_sha256"]) == output_set(output, group["tasks"]) and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in a["output_sha256"].values()), "Exact complete task output set/SHA required")
            require(all(a["preflight_environment"].get(k) == group["inference_policy"][k] for k in ("torch_version", "cuda_version", "gpu_name")), "Attested original worker environment differs")
            output_stats = {}
            for path in a["output_sha256"]:
                require(Path(path).is_file(), "Output missing after completed scheduler")
                output_stats[path] = signature(path)
            result.append({"job_id": job["id"], "logical_record_id": logical["id"] if logical else None,
                           "attestation": ap, "output_sha256": a["output_sha256"], "output_five_stat": output_stats})
    return result


def inspect_state(plan_proof, actual_queue, state_path, output):
    plan = load_plan(plan_proof); queue_proof, raw = small(actual_queue); queue = json.loads(raw)
    state_proof, state_raw = small(state_path); state = json.loads(state_raw)
    selected = state_records(plan["frozen_definitions"], queue, state)
    manifest = read_proof(plan["manifest"]); attestations = metadata_attestations(manifest, MANIFEST_SHA)
    require(len(attestations) == 73, "67 canonical plus six logical alias attestations required")
    snapshot = exclusive_bytes(Path(output).with_name(Path(output).stem + "_state_snapshot.json"), state_raw)
    value = {"schema_version": 1, "status": "complete_primary_state_and_attestation_metadata_only", "ready": True, "plan": plan_proof,
             "counts": COUNTS, "actual_queue": queue_proof, "actual_state_observation": state_proof, "state_snapshot": snapshot,
             "selected_completed_records": selected, "attestations": attestations, "source_sha256": plan["source_sha256"],
             "no_result_or_prediction_JSON_parsed": True, "result_or_prediction_bytes_hashed": False,
             "scope_note": "Metadata source closure only. Original check still required under ROOT authority for full output byte SHA integrity."}
    return exclusive(output, value)


def authorize(plan_proof, state_proof, authority_proof):
    plan = load_plan(plan_proof); state = read_proof(state_proof); a = read_proof(authority_proof)
    expected = {"authorized_by": "root", "allow_score_analysis": True, "plan": plan_proof, "manifest": plan["manifest"],
                "manifest_sha256": MANIFEST_SHA, "frozen_queue": plan["frozen_queue"], "state_proof": state_proof,
                "source_sha256": plan["source_sha256"], "expected_counts": COUNTS, "expected_inference_jobs": 67,
                "expected_logical_endpoints": 72, "journal_path": str(JOURNAL), "allow_partial": False, "analysis_output": plan["analysis_output"]}
    require(all(a.get(k) == v for k, v in expected.items()), "Exact explicit ROOT true authority required")
    require(state.get("ready") is True and state.get("status") == "complete_primary_state_and_attestation_metadata_only" and state["plan"] == plan_proof and state["counts"] == COUNTS and state["source_sha256"] == plan["source_sha256"], "Exact complete state/source proof required")
    return plan, state


def refresh_state(plan, proof):
    queue = read_proof(proof["actual_queue"])
    _, raw = small(proof["actual_state_observation"]["path"])
    selected = state_records(plan["frozen_definitions"], queue, json.loads(raw))
    require(selected == proof["selected_completed_records"], "Selected completed state changed after ROOT reviewed it")
    snapshot_jobs = read_proof(proof["state_snapshot"])["jobs"]
    require(all(snapshot_jobs.get(key) == value for key, value in selected.items()), "Immutable state snapshot does not contain same selected records")
    current = metadata_attestations(read_proof(plan["manifest"]), MANIFEST_SHA)
    require(current == proof["attestations"], "Attestation/output source metadata changed after proof")
    return {"current_state_path": proof["actual_state_observation"]["path"], "current_state_sha256": sha_bytes(raw),
            "unrelated_live_scheduler_state_changes_allowed": True, "all_67_selected_records_unchanged": True}


def check_result(value):
    require(value.get("ready") is True and value.get("all_output_integrity_verified") is True
            and value.get("manifest_sha256") == MANIFEST_SHA and value.get("expected_inference_jobs") == 67
            and value.get("logical_endpoints") == 72 and value.get("scores_or_predictions_parsed") is False
            and value.get("model_inference_started") is False, "Original full check not successful")


def create_journal(path, bindings):
    return exclusive(path, {"schema_version": 1, "status": "semantic_access_enabled", "semantic_access_enabled_at_utc": now(),
                            "before_original_analyze_invocation": True, "authorization_attempt_not_analysis_success": True, **bindings})


def execute(plan_proof, state_proof, authority_proof, attempt):
    plan, state = authorize(plan_proof, state_proof, authority_proof)
    require(not JOURNAL.exists() and not Path(plan["analysis_output"]).exists(), "Journal/output already exists; never reset or overwrite")
    observation = refresh_state(plan, state)
    base_argv = ["/usr/bin/python", str(GUARD), "--manifest", str(MANIFEST), "--manifest-sha256", MANIFEST_SHA]
    with (attempt / "check_stdout.json").open("xb") as stdout, (attempt / "check_stderr.log").open("xb") as stderr:
        checked = subprocess.run(base_argv + ["--expected-inference-jobs", "67"], cwd=ROOT, stdout=stdout, stderr=stderr)
    require(checked.returncode == 0, "Original check subprocess failed")
    check_proof, raw = small(attempt / "check_stdout.json"); check_result(json.loads(raw))
    plan = load_plan(plan_proof); observation = refresh_state(plan, state)
    read_proof(authority_proof)
    bindings = {"plan": plan_proof, "manifest": plan["manifest"], "frozen_queue": plan["frozen_queue"], "state_proof": state_proof,
                "actual_queue": state["actual_queue"], "live_state_confirmation": observation, "authority": authority_proof,
                "source_sha256": plan["source_sha256"], "original_successful_check": check_proof,
                "counts": COUNTS, "analysis_output": plan["analysis_output"]}
    journal_proof = create_journal(JOURNAL, bindings)
    argv = base_argv + ["--analyze", "--authorization", authority_proof["path"], "--output", plan["analysis_output"]]
    with (attempt / "analyze_stdout.json").open("xb") as stdout, (attempt / "analyze_stderr.log").open("xb") as stderr:
        analyzed = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr)
    require(analyzed.returncode == 0, "Original analyze failed after semantic access enabled; journal preserved")
    metadata_proof, raw = small(Path(plan["analysis_output"]) / "analysis_manifest.json"); meta = json.loads(raw)
    require(meta.get("complete") is True and meta.get("manifest_sha256") == MANIFEST_SHA and meta.get("logical_endpoints") == 72,
            "Original analysis did not complete full declared scope")
    return exclusive(attempt / "COMPLETE.json", {"complete": True, "journal": journal_proof, "plan": plan_proof, "authority": authority_proof,
                     "state_proof": state_proof, "original_check": check_proof, "original_analysis_metadata": metadata_proof,
                     "original_helpers_or_statistics_modified": False, "new_model_inference": False})


def run(plan_proof, state_proof, authority_proof, attempt):
    plan, _ = authorize(plan_proof, state_proof, authority_proof)
    require(not JOURNAL.exists() and not Path(plan["analysis_output"]).exists(), "Existing first-read/output; cannot reset")
    attempt = Path(attempt).resolve(); attempt.mkdir(parents=True, exist_ok=False)
    launch = {"plan": plan_proof, "state_proof": state_proof, "authority": authority_proof, "attempt": str(attempt), "parent_pid": os.getpid()}
    launch_proof = exclusive(attempt / "launch.json", launch)
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    env["PRIMARY_BEHAVIOR_ANALYSIS_PARENT_PID"] = str(os.getpid())
    argv = [sys.executable, str(Path(__file__).resolve()), "_execute", "--launch", launch_proof["path"], "--launch-sha256", launch_proof["sha256"]]
    before = resource.getrusage(resource.RUSAGE_CHILDREN); start = time.perf_counter()
    with (attempt / "worker_stdout.log").open("xb") as stdout, (attempt / "worker_stderr.log").open("xb") as stderr:
        child = subprocess.run(argv, env=env, cwd=ROOT, stdout=stdout, stderr=stderr)
    wall = time.perf_counter() - start; after = resource.getrusage(resource.RUSAGE_CHILDREN)
    timing = exclusive(attempt / "whole_process_timing.json", {"schema_version": 1, "returncode": child.returncode,
                       "whole_child_process_wall_seconds": wall, "whole_child_user_cpu_seconds": after.ru_utime-before.ru_utime,
                       "whole_child_sys_cpu_seconds": after.ru_stime-before.ru_stime, "launch": launch_proof,
                       "scope": "Complete child + its waited original check/analyze descendants, including full closure, original check, exclusive semantic journal and original analyze",
                       "subtimers_are_nested_do_not_add_again": True, "parent_prelaunch_authority_check_not_in_child_timer": True,
                       "journal_exists_after_attempt": JOURNAL.exists(), "failed_attempt_not_reset": child.returncode != 0})
    require(child.returncode == 0, "Authorized analysis failed; attempt/timing/journal preserved: " + timing["path"])
    return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--output", type=Path, required=True); prep.add_argument("--analysis-output", type=Path, required=True)
    for action in ("inspect-state", "run"):
        p = sub.add_parser(action); p.add_argument("--plan", type=Path, required=True); p.add_argument("--plan-sha256", required=True)
        if action == "inspect-state":
            p.add_argument("--queue", type=Path, required=True); p.add_argument("--state", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
        else:
            for name in ("state-proof", "authority"):
                p.add_argument("--" + name, type=Path, required=True); p.add_argument("--" + name + "-sha256", required=True)
            p.add_argument("--attempt-dir", type=Path, required=True)
    child = sub.add_parser("_execute"); child.add_argument("--launch", type=Path, required=True); child.add_argument("--launch-sha256", required=True)
    a = parser.parse_args()
    if a.action == "prepare": result = prepare(a.output, a.analysis_output)
    elif a.action == "inspect-state": result = inspect_state({"path": str(a.plan.resolve()), "sha256": a.plan_sha256}, a.queue, a.state, a.output)
    elif a.action == "run":
        result = run({"path": str(a.plan.resolve()), "sha256": a.plan_sha256}, {"path": str(a.state_proof.resolve()), "sha256": a.state_proof_sha256},
                     {"path": str(a.authority.resolve()), "sha256": a.authority_sha256}, a.attempt_dir)
    else:
        launch = read_proof({"path": str(a.launch.resolve()), "sha256": a.launch_sha256})
        require(str(os.getppid()) == os.environ.get("PRIMARY_BEHAVIOR_ANALYSIS_PARENT_PID") == str(launch["parent_pid"]), "Use timed run parent; direct _execute prohibited")
        require(Path(a.launch).resolve().parent == Path(launch["attempt"]) and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Timed CPU attempt/env mismatch")
        result = execute(launch["plan"], launch["state_proof"], launch["authority"], Path(launch["attempt"]))
    print(json.dumps(result))


if __name__ == "__main__": main()
