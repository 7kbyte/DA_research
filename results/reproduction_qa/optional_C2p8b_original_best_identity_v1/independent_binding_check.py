"""Independent small-evidence binding audit; zero checkpoint payload reads."""
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
START = time.perf_counter()
BEFORE = resource.getrusage(resource.RUSAGE_SELF)
READ_BYTES = 0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source(path, expected=None):
    global READ_BYTES
    path = Path(path).resolve()
    require(path.stat().st_size < 16 << 20, "Small evidence only")
    raw = path.read_bytes()
    READ_BYTES += len(raw)
    digest = hashlib.sha256(raw).hexdigest()
    require(expected is None or expected == digest, "Small source SHA mismatch")
    return {"path": str(path), "sha256": digest, "bytes": len(raw)}, raw


def read(path, expected=None):
    proof, raw = source(path, expected)
    return proof, json.loads(raw)


def stat(path):
    value = Path(path).stat()
    return {key: getattr(value, "st_" + key) for key in ("dev", "ino", "size", "mtime_ns", "ctime_ns")}


proof_source, proof = read(HERE / "worker/proof.json", "3d8ab51efa4fb3fa427bb0b0241141d9acda40e4d5b9a851689b5e737636cd5e")
timing_source, timing = read(HERE / "whole_process_timing.json", "1b69ca0a7242a8b904d49410c6d204dcb99ac2e552e5325d18eda52f2c8082f2")
launch_source, launch = read(HERE / "launch.json")
require(timing["launch"] == {key: launch_source[key] for key in ("path", "sha256")}, "Timing/launch binding differs")
producer_source, _ = source(launch["producer"]["path"], launch["producer"]["sha256"])
require(proof["producer"] == producer_source, "Actual producer differs")
require(proof["status"] == "passed_original_best_complete_model_state_and_fresh_file_SHA" and timing["returncode"] == 0, "Producer not complete")
require(launch["explicit_env"]["CUDA_VISIBLE_DEVICES"] == "" and all(launch["explicit_env"][k] == "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Explicit CPU environment differs")
require(proof["CUDA_initialized"] is False and proof["CPU_threads"] == 2, "CPU-only constraint differs")
for item in proof["source_proofs"]:
    source(item["path"], item["sha256"])
by_path = {item["path"]: item for item in proof["source_proofs"]}
template_source, template = read(ROOT / "docs/optional_C2p8b_B4_seed11_QA_ROOT_FALSE_TEMPLATE.json", "05fe83d2a327c065190883774365641fac432a0546e1b13946a479ada0fc0881")
expected_template_source, _ = source(ROOT / "docs/optional_C2p8b_B4_seed11_QA_expected_identity_TEMPLATE.json", "824f5a991ac496d864ade865b649f73f3a0cbd4ef0573d980da81d98a4f3e40e")
require(template["root_authorized"] is False, "Original QA authority changed")
rid = proof["original_run_id"]
expected_best = {"step": 612, "target_prediction_tokens": 20014848, "replay_prediction_tokens": 1000742}
require(proof["full_training_counters"] == {"step": 765, "target_prediction_tokens": 25000000, "replay_prediction_tokens": 1250000}, "Full training exposure differs")
require(proof["actual_best_counters"] == proof["expected_best_counters"] == expected_best, "Actual best exposure differs")
registry_path = ROOT / "configs/extended_v2_pythia_2p8b_m8_h25m_formal_active_registry.json"
registry_source, registry = read(registry_path, by_path[str(registry_path)]["sha256"])
matches = [item for item in registry if item["id"] == rid]
require(len(matches) == 1, "Registry fixed run missing/ambiguous")
registered = matches[0]
config_source, config = read(registered["config_path"], registered["config_sha256"])
summary_source, summary = read(registered["summary_path"], by_path[registered["summary_path"]]["sha256"])
require(summary["config"] == config and summary["status"] == "completed", "Source summary/config binding differs")
require({k: summary["counters"][k] for k in proof["full_training_counters"]} == proof["full_training_counters"], "Source final counters differ")
opaque = hashlib.sha256(json.dumps(summary["best_target_validation"], sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
require(proof["selection"]["opaque_sha256"] == opaque and proof["selection"]["NLL_value_emitted"] is False, "Opaque selection binding differs")
job_record = proof["original_training_job"]["completed_record"]
_, state = read(ROOT / "logs/supplemental_job_state.json")
require(state["jobs"][rid] == job_record and job_record["status"] == "completed" and job_record["returncode"] == 0 and not job_record["missing_outputs"] and "stop_requested_at" not in job_record, "Live completed source job differs")
definition = proof["original_training_job"]["definition"]
require(hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest() == job_record["definition_sha256"], "Original definition binding differs")
snapshot_source, snapshot = read(proof["historical_file_SHA_evidence"]["snapshot"]["path"], proof["historical_file_SHA_evidence"]["snapshot"]["sha256"])
historical_source, historical = read(proof["historical_file_SHA_evidence"]["receipt"]["path"], proof["historical_file_SHA_evidence"]["receipt"]["sha256"])
require(snapshot["entries"][proof["source_checkpoint"]] == {key: historical_source[key] for key in ("path", "sha256")}, "Snapshot/read receipt binding differs")
require(historical["method"] == "sha256_entire_open_fd_to_eof_v1" and historical["complete_to_eof"] is True, "Historical full read not established")
current = stat(proof["source_checkpoint"])
require(proof["five_stat_before"] == proof["five_stat_after"] == current, "Original checkpoint current five-stat differs")
require(all(historical[key] == current for key in ("stat_before", "stat_after", "fd_before", "fd_after")), "Historical source identity differs")
operations = proof["read_operations"]
require(len(operations) == 2 and all(op["completed"] is True and op["fd_before"] == op["fd_after"] == current for op in operations), "Fresh stable-fd operations incomplete")
require(operations[1]["complete_to_eof"] is True and operations[1]["serialized_bytes_read"] == current["size"], "Fresh sequential SHA not full-file")
require(proof["fresh_full_checkpoint_SHA_bytes"] == current["size"] == historical["read_bytes"], "Whole-file byte counts differ")
require(proof["source_checkpoint_sha256"] == historical["sha256"] == "4d813191a3d598a66142106bd00533bf3e7eb3e828abd3abb82578dbdf1bb49d", "Fresh and prewarm serialized SHA differ")
require(proof["serialized_checkpoint_read_bytes_total"] == sum(op["serialized_bytes_read"] for op in operations) >= 22200000000, "Actual byte total differs")
fingerprint = proof["model_state_fingerprint"]
require(fingerprint == {"sha256_model_state": "1c142978e5969417f8f1e54226aa7c960fbff231270cb1de8328d05c0a7eb1c6", "state_elements": 2775208960}, "Whole state fingerprint receipt differs")
require(proof["model_tensor_count"] == 388 and proof["model_state_tensor_bytes_hashed"] == 2775208960 * 4, "Complete FP32 inventory differs")
require(proof["whole_model_state_fingerprinted"] and proof["original_identity_only_not_a_retraining_comparison"], "Proof scope differs")
require(proof["model_fingerprint_and_file_SHA_are_distinct_identities"] is True, "Distinct hash semantics missing")
require(proof["NLL_values_emitted"] is False and proof["QA_launch_authority_granted"] is False and proof["training_relocation_archive_or_score_analysis_started"] is False, "Scope violation")
after = resource.getrusage(resource.RUSAGE_SELF)
result = {"schema_version": 1, "status": "passed_independent_small_evidence_binding_check", "created_at_utc": datetime.now(timezone.utc).isoformat(),
          "scope": "Independent source/stat/receipt binding check; full fingerprint computation remains the separately SHA-bound producer, not a second model payload pass",
          "proof": proof_source, "whole_process_timing": timing_source, "producer": producer_source, "checker": source(__file__)[0],
          "historical_full_file_read_receipt": historical_source, "cache_snapshot": snapshot_source,
          "source_registry": registry_source, "source_config": config_source, "source_summary": summary_source,
          "immutable_ROOT_FALSE_template": template_source, "immutable_expected_template": expected_template_source,
          "model_state_fingerprint": fingerprint, "source_checkpoint_sha256": proof["source_checkpoint_sha256"],
          "serialized_SHA_and_model_state_fingerprint_are_distinct": True, "source_current_five_stat": current,
          "actual_best_counters": expected_best, "tensor_count": 388, "extra_checkpoint_payload_bytes_read": 0,
          "serialized_returned_bytes_total": proof["serialized_checkpoint_read_bytes_total"],
          "nominal_two_file_size_bytes": 2 * current["size"], "torch_load_nonsequential_ZIP_bytes_skipped_relative_to_file_size": current["size"] - operations[0]["serialized_bytes_read"],
          "nominal_estimate_not_actual_lower_bound": "Launch's two-file-size nominal estimate is 34,240 bytes above actual serialized returns: torch.load seeks over some non-tensor ZIP bytes; fresh sequential SHA covers every file byte. Actual bytes are reported, not inflated.",
          "wall_seconds": time.perf_counter() - START, "user_cpu_seconds": after.ru_utime - BEFORE.ru_utime, "sys_cpu_seconds": after.ru_stime - BEFORE.ru_stime,
          "small_evidence_bytes_returned": READ_BYTES, "timing_is_separate_postproof_CPU_operation": True,
          "QA_training_or_relocation_or_new_archive_authorized": False, "NLL_values_emitted": False,
          "remaining_pending": ["persistent_C2p8b_package_and_base_closure", "actual_relocated_runtime_plan_and_complete_expected", "later_explicit_ROOT_TRUE_QA_authority_and_all_time/resource_gates"]}
with (HERE / "independent_binding_check.json").open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
print(json.dumps({"status": result["status"], "proof": str(HERE / "independent_binding_check.json"), "extra_checkpoint_payload_bytes_read": 0}))
