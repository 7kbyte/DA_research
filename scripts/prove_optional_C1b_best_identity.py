"""CPU-only identity proof of the predeclared C1B original best checkpoint.

Thin fixed-identity adaptation of prove_optional_C2p8b_best_identity.py;
the existing C2.8B implementation and evidence are not modified.

No training, archive creation, relocation, model forward, or score analysis.
Selection is bound opaquely; its validation NLL is never emitted.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "extended_v2_pythia_1b_m32_formal_B4_s11_w0p05"
TEMPLATE_SHA = "e84360ca3d1e7bd0dcb722bb90c0971340f11c2cf3845d6e3af95693c19708a9"
TEMPLATE = ROOT / "docs/optional_C1b_B4_seed11_original_best_identity_CPU_INPUT.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signature(info):
    return {k: getattr(info, "st_" + k) for k in ("dev", "ino", "size", "mtime_ns", "ctime_ns")}


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def small_proof(path, expected=None):
    path = Path(path).resolve(); before = signature(path.stat())
    require(before["size"] < 16 << 20, "Small-source byte bound exceeded")
    raw = path.read_bytes(); after = signature(path.stat())
    require(before == after, "Small source changed during read")
    digest = hashlib.sha256(raw).hexdigest()
    require(expected is None or digest == expected, "Bound source SHA differs: " + str(path))
    return {"path": str(path), "sha256": digest, "bytes": len(raw)}, raw


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def counters(value):
    return {k: value[k] for k in ("step", "target_prediction_tokens", "replay_prediction_tokens")}


def timer():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return time.perf_counter(), r.ru_utime, r.ru_stime


def elapsed(start):
    end = timer()
    return dict(wall_seconds=end[0]-start[0], user_cpu_seconds=end[1]-start[1], sys_cpu_seconds=end[2]-start[2])


@contextmanager
def tracked_read(receipt, stream, rawfile, operation):
    start = timer()
    item = dict(operation=operation, completed=False, serialized_bytes_read=0,
                fd_before=signature(os.fstat(rawfile.fileno())))
    receipt["read_operations"].append(item)
    try:
        yield item
        item["completed"] = True
    finally:
        item.update(serialized_bytes_read=stream.bytes_read, **elapsed(start),
                    fd_after=signature(os.fstat(rawfile.fileno())))


def prove(snapshot, snapshot_sha, receipt):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Explicit empty CVD required")
    require(all(os.environ.get(k) == "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Two CPU threads required")
    template_proof, raw = small_proof(TEMPLATE, TEMPLATE_SHA); template = json.loads(raw)
    require(template["original_run_id"] == RUN_ID and template["root_authorized"] is False, "Predetermined identity/template changed")
    sources = [template_proof]
    for item in template["source_proofs"]:
        p, _ = small_proof(item["path"], item["sha256"]); sources.append(p)
    by_source = {p["path"]: p for p in sources}
    verifier_path = ROOT / "scripts/verify_reproduction_qa_postrun.py"
    qa = load_module(verifier_path, "optional_original_best_qa")
    state_proof, state_raw = small_proof(ROOT / "logs/supplemental_job_state.json")
    state = json.loads(state_raw); record = state["jobs"][RUN_ID]
    queue_proof, raw = small_proof(ROOT / "configs/supplemental_v1_queue.json")
    definitions = [j for j in json.loads(raw) if j["id"] == RUN_ID]
    require(len(definitions) == 1, "Exactly one original job required"); job = definitions[0]
    definition_sha = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    require(record["status"] == "completed" and record["returncode"] == 0 and not record["missing_outputs"] and "stop_requested_at" not in record, "Original complete rc0/no-stop gate failed")
    require(record["definition_sha256"] == definition_sha and record["argv"] == job["argv"], "Actual original definition/argv differs")
    cfg = json.loads(Path(template["source_config"]["path"]).read_text())
    draft = json.loads(Path(template["final_expectation"]["draft_record"]["path"]).read_text())
    summary_path = next(p["path"] for p in sources if p["path"].endswith(RUN_ID + "_summary.json"))
    summary = json.loads(Path(summary_path).read_text())
    require(summary["config"] == cfg and summary["status"] == "completed", "Original summary/config not complete and equal")
    require(counters(summary["counters"]) == template["full_counters"] == dict(step=3058, target_prediction_tokens=100000000, replay_prediction_tokens=5000000), "Original full exposure differs")
    require(draft["definition_sha256"] == definition_sha and draft["completed_scheduler_record"] == record, "Original completed record differs from verified final draft")
    qa.provenance_matches(summary["provenance"], template["required_original_execution_provenance"])
    qa.strict_policy(template["required_original_execution_provenance"], cfg)
    expected_best = template["best_expectation"]["summary_only_checkpoint_counters"]
    require(expected_best == dict(step=612, target_prediction_tokens=20014848, replay_prediction_tokens=1000742), "Predetermined best exposure differs")
    best = Path(template["best_expectation"]["source_checkpoint"])
    require(not best.is_symlink() and best.resolve() == best and best.name == "checkpoint_best.pt", "Original best path differs")
    require(summary["best_target_validation"]["step"] == expected_best["step"] and summary["best_target_validation"]["checkpoint_path"] == str(best), "Original best identity differs")
    snapshot_proof, raw = small_proof(snapshot, snapshot_sha); snapshot_obj = json.loads(raw)
    cache_source = ROOT / "scripts/stat_bound_digest_cache.py"
    reader_source = by_source.get(str(cache_source))
    # Cache reader is separately frozen by the aux protocol, not part of QA sources.
    if reader_source is None:
        protocol = json.loads((ROOT / "docs/supplemental_aux_analysis_protocol_v2.json").read_text())
        reader_source = protocol["checkpoint_cache_reader_source"]
    p, _ = small_proof(cache_source, reader_source["sha256"]); sources.append(p)
    cache_module = load_module(cache_source, "optional_original_best_cache")
    cache = cache_module.StatBoundHashes(snapshot=snapshot, snapshot_sha256=snapshot_sha)
    previous = cache._receipt(str(best)); require(previous is not None, "Frozen historical full-read receipt missing")
    historical_proof = snapshot_obj["entries"][str(best)]
    p, _ = small_proof(historical_proof["path"], historical_proof["sha256"]); sources.append(p)
    require(previous["source_authority"]["kind"] == "completed_training_checkpoint" and previous["source_authority"]["run_id"] == RUN_ID and previous["source_authority"]["endpoint"] == "best" and previous["source_authority"]["returncode"] == 0, "Historical read source identity differs")
    before = signature(best.stat())
    require(previous["stat_before"] == before, "Original best changed since historical full read")
    receipt.update(source_checkpoint=str(best), five_stat_before=before, source_proofs=sources,
                   original_training_job={"state_observation": state_proof, "queue": queue_proof, "definition": job, "definition_sha256": definition_sha, "completed_record": record},
                   full_training_counters=template["full_counters"], expected_best_counters=expected_best,
                   historical_file_SHA_evidence={"snapshot": snapshot_proof, "receipt": historical_proof, "sha256": previous["sha256"], "read_bytes": previous["read_bytes"], "current_checkpoint_byte_reads": 0})
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    require(not torch.cuda.is_initialized(), "CUDA must remain uninitialized")
    codec_path = ROOT / "src/replay_study/weight_archive.py"
    codec = load_module(codec_path, "optional_original_best_codec")
    with best.open("rb") as rawfile:
        opened = signature(os.fstat(rawfile.fileno())); require(opened == before, "Path/fd differs on open")
        stream = qa.CountingReader(rawfile)
        with tracked_read(receipt, stream, rawfile, "CPU_load_metadata_and_full_state_fingerprint"):
            saved = torch.load(stream, map_location="cpu", weights_only=False)
            require(saved["config"] == cfg and counters(saved["counters"]) == expected_best, "Best payload config/actual exposure differs")
            require(saved["checkpoint_kind"] == "model_only_best_target_validation" and "optimizer_state" not in saved, "Original best payload kind differs")
            qa.provenance_matches(saved["provenance"], template["required_original_execution_provenance"])
            selection = saved["selection"]
            require(all(selection[k] == v for k, v in (("step", 612), ("target_prediction_tokens", 20014848), ("checkpoint_path", str(best)))), "Original best selection identity differs")
            opaque_selection = hashlib.sha256(encoded(selection)).hexdigest()
            require(opaque_selection == hashlib.sha256(encoded(summary["best_target_validation"])).hexdigest(), "Opaque best selection differs from source summary")
            model_state = saved["model_state"]
            require(all(t.dtype == torch.float32 for t in model_state.values()), "Original state dtype differs")
            tensor_count = len(model_state); fingerprint = codec.model_fingerprint(model_state)
            require(fingerprint["state_elements"] == 1011781632, "Complete model element inventory differs")
            fd_after = signature(os.fstat(rawfile.fileno())); require(fd_after == before == signature(best.stat()), "Original best changed during full model fingerprint")
            receipt.update(model_state_fingerprint=fingerprint, model_tensor_count=tensor_count,
                           model_state_tensor_bytes_hashed=sum(t.numel() * t.element_size() for t in model_state.values()),
                           actual_best_counters=counters(saved["counters"]),
                           checkpoint_kind=saved["checkpoint_kind"], required_execution_provenance=template["required_original_execution_provenance"],
                           selection={"opaque_sha256": opaque_selection, "encoding": "UTF8 json sort_keys separators comma-colon ensure_ascii=False allow_nan=False", "step": 612, "target_prediction_tokens": 20014848, "checkpoint_path": str(best), "NLL_value_emitted": False},
                           fingerprint_source={**by_source[str(codec_path)], "function": "model_fingerprint", "argument": "entire model_state", "algorithm": "sha256_sorted_key_dtype_shape_contiguous_uint8_v1"})
        del model_state, saved; gc.collect()
        rawfile.seek(0); stream = qa.CountingReader(rawfile)
        with tracked_read(receipt, stream, rawfile, "fresh_sequential_entire_checkpoint_SHA256") as item:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(8 << 20), b""): digest.update(block)
            fresh_sha = digest.hexdigest(); fd_after = signature(os.fstat(rawfile.fileno()))
            require(fd_after == before == signature(best.stat()) and stream.bytes_read == before["size"], "Fresh whole-file read incomplete or source changed")
            item["complete_to_eof"] = True
    require(fresh_sha == previous["sha256"], "Fresh serialized file SHA differs from historical prewarm receipt")
    for item in sources: small_proof(item["path"], item["sha256"])
    small_proof(snapshot_proof["path"], snapshot_proof["sha256"])
    state_now = json.loads((ROOT / "logs/supplemental_job_state.json").read_text())["jobs"][RUN_ID]
    require(state_now == record, "Original completed job record changed")
    require(signature(best.stat()) == before and best.resolve() == best, "Source identity changed before proof completion")
    require(not torch.cuda.is_initialized(), "Unexpected CUDA initialization")
    receipt.update(status="passed_original_best_complete_model_state_and_fresh_file_SHA", source_checkpoint_sha256=fresh_sha,
                   fresh_file_SHA_equals_historical_complete_read=True, model_fingerprint_and_file_SHA_are_distinct_identities=True,
                   five_stat_after=signature(best.stat()), path_and_open_fd_stable=True, whole_model_state_fingerprinted=True,
                   original_identity_only_not_a_retraining_comparison=True,
                   CPU_threads=2, CUDA_initialized=False, CPU_loader_torch_version=torch.__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-snapshot", type=Path, required=True); parser.add_argument("--cache-snapshot-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    start = timer(); args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema_version=1, status="failed_original_best_identity_not_verified", created_at_utc=datetime.now(timezone.utc).isoformat(),
                   scope="ROOT authorized original B4s11 best CPU identity only", original_run_id=RUN_ID, read_operations=[],
                   training_relocation_archive_or_score_analysis_started=False, QA_launch_authority_granted=False, NLL_values_emitted=False,
                   producer=small_proof(__file__)[0], parent_timing_scope="Whole producer command after stdlib/argv startup; user/sys across all CPU threads, operations nested", planning_runtime_replaced_by_measurement=False)
    try:
        prove(args.cache_snapshot, args.cache_snapshot_sha256, receipt)
    except Exception as error:
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        receipt["parent_timing"] = elapsed(start)
        receipt["serialized_checkpoint_read_bytes_total"] = sum(x["serialized_bytes_read"] for x in receipt["read_operations"])
        receipt["fresh_full_checkpoint_SHA_bytes"] = sum(x["serialized_bytes_read"] for x in receipt["read_operations"] if x["operation"] == "fresh_sequential_entire_checkpoint_SHA256")
        receipt["read_bytes_are_serialized_returned_bytes_not_physical_disk_bytes"] = True
        receipt["subtimers_nested_in_parent_do_not_add_again"] = True
        with (args.output_dir / "proof.json").open("x") as handle:
            json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps({"status": receipt["status"], "proof": str(args.output_dir / "proof.json"), "NLL_values_emitted": False}))
    raise SystemExit(0 if receipt["status"].startswith("passed_") else 1)


if __name__ == "__main__":
    main()
