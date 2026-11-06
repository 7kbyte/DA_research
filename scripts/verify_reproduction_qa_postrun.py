"""Independent CPU acceptance of archive-QA final AND best checkpoints.

Preparation reads accepted metadata only. Verification never starts training,
inference, or CUDA. It invokes the explicitly SHA-bound archived full-state
fingerprint with a counting file object, without patching that implementation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime, timezone
from fractions import Fraction


COUNTERS = ("step", "target_prediction_tokens", "replay_prediction_tokens")
POLICY_KEYS = (
    "torch_version", "cuda_version", "gpu", "attention_backend_policy",
    "attention_implementation", "precision", "deterministic_algorithms",
    "deterministic_warn_only", "cublas_workspace_config", "allow_tf32",
    "allow_tf32_matmul", "allow_tf32_cudnn", "cudnn_deterministic",
    "cudnn_benchmark", "sdpa_backends_enabled", "model_weight_dtype", "teacher_weight_dtype",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def five_stat(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def proof(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def bound(reference):
    path = Path(reference["path"])
    require(sha(path) == reference["sha256"], "Bound source SHA changed: " + str(path))
    return path


def exact_path(path, expected, base=None):
    p, wanted = Path(path), Path(expected)
    require(p.is_absolute() and p == wanted, "Unexpected/nonabsolute path: " + str(path))
    require(not p.is_symlink(), "Final path itself is a symlink: " + str(p))
    if base is not None:
        require(p.resolve().is_relative_to(Path(base).resolve()), "Resolved path escapes QA namespace: " + str(p))
    return p


def subset_counters(payload):
    return {key: payload[key] for key in COUNTERS}


def provenance_matches(actual, expected):
    require(all(k in expected for k in POLICY_KEYS), "Expected strict execution evidence missing")
    require(all(k in actual and actual[k] == v for k, v in expected.items()), "Actual numerical provenance differs/missing")


def strict_policy(expected, config):
    provenance_matches(expected, expected)
    require(expected["deterministic_algorithms"] is True and expected["deterministic_warn_only"] is False,
            "Strict deterministic execution evidence required")
    require(expected["model_weight_dtype"] == expected["teacher_weight_dtype"] == "torch.float32",
            "FP32 student/teacher weight evidence required")
    require(expected["cublas_workspace_config"] == ":4096:8" and expected["cudnn_benchmark"] is False,
            "Frozen cuBLAS/cuDNN policy differs")
    require(expected["cudnn_deterministic"] == config["deterministic_algorithms"] is True,
            "Strict cuDNN/config policy differs")
    for key, cfg_key in (("precision", "precision"), ("attention_backend_policy", "attention_backend"),
                         ("attention_implementation", "attention_implementation"),
                         ("allow_tf32", "allow_tf32"), ("allow_tf32_matmul", "allow_tf32"),
                         ("allow_tf32_cudnn", "allow_tf32")):
        require(expected[key] == config[cfg_key], "Config/actual numerical policy differs: " + key)
    require(set(expected["sdpa_backends_enabled"]) == {"flash", "math", "memory_efficient", "cudnn"},
            "Four SDPA backends must be explicit")


def prepare(plan_path, plan_sha256, final_manifest, best_manifest, best_sha256,
            best_metadata_root, best_acceptance, algorithm, algorithm_sha256):
    require(sha(plan_path) == plan_sha256, "Original QA plan SHA differs")
    plan = read(plan_path)
    rows = read(bound(plan["registry"]))
    jobs = read(bound(plan["queue"]))
    originals = plan["future_full_run_comparison"]["expected_runs"]
    ids = {r["reproduction_run_id"] for r in originals}
    require(len(ids) == len(originals) == len(rows) == len(jobs) > 0, "Plan/registry/queue count differs")
    require(ids == {r["id"] for r in rows} == {j["id"] for j in jobs}, "Plan/registry/queue IDs differ")
    require(jobs == plan["queue"]["jobs"], "Plan exact jobs differ")
    require(sha(final_manifest) == plan["future_full_run_comparison"]["source_weight_manifest_sha256"], "Final manifest SHA differs")
    require(sha(best_manifest) == best_sha256, "Accepted best manifest SHA differs")
    fm, bm = read(final_manifest), read(best_manifest)
    require(bm["status"] == "all_unique_weights_roundtrip_verified", "Best archive not accepted")
    acceptance = read(best_acceptance)
    require(acceptance["manifest_sha256"] == best_sha256, "Best acceptance binds another manifest")
    require(acceptance["receipt"]["status"] == "passed" and acceptance["receipt"]["all_original_weight_bytes_compared"], "Best durable acceptance incomplete")
    bound({"path": acceptance["receipt_path"], "sha256": acceptance["receipt_sha256"]})
    require(read(acceptance["receipt_path"])["manifest_sha256"] == best_sha256, "Durable receipt binds another manifest")
    workspace = Path(jobs[0]["argv"][1]).resolve().parents[1]
    require(Path(algorithm).resolve().is_relative_to(workspace), "Fingerprint implementation is outside archived workspace")
    require(sha(algorithm) == algorithm_sha256, "Archived fingerprint source SHA differs")
    all_sources = [proof(p) for p in (plan_path, final_manifest, best_manifest, best_acceptance,
                                    acceptance["receipt_path"], algorithm)]
    all_sources += [plan["registry"], {k: v for k, v in plan["queue"].items() if k in {"path", "sha256"}}]
    identity_audit = read(bound(plan["scientific_identity_audit"]))
    all_sources.append(plan["scientific_identity_audit"])
    relocation = read(bound(identity_audit["relocation_manifest"]))
    all_sources.append(identity_audit["relocation_manifest"])
    scratch = Path(relocation["new_scratch"])
    require(Path(relocation["new_workspace"]).resolve() == workspace, "Relocation workspace differs")
    require(scratch.is_absolute() and scratch.resolve() == scratch, "Independent QA scratch must be canonical")
    for source in identity_audit["scientific_core_files"]:
        bound({"path": source["restored_path"], "sha256": source["sha256"]})
        all_sources.append({"path": source["restored_path"], "sha256": source["sha256"]})
    final_by_id = {r["condition_run_id"]: r for r in fm["logical_runs"]}
    best_by_id = {r["run_id"]: r for r in bm["logical_runs"] if r["kind"] == "best"}
    registry = {r["id"]: r for r in rows}
    expected_runs = []
    import torch
    for old in originals:
        run_id, origin = old["reproduction_run_id"], old["reproduction_of"]
        row, final, best = registry[run_id], final_by_id[origin], best_by_id[origin]
        config_path = bound(old["relocated_config"])
        require(sha(config_path) == row["config_sha256"], "Registry config SHA differs")
        config, source_config = read(config_path), read(bound(old["source_config"]))
        require(config["run_label"] == run_id and run_id != origin and run_id.startswith(plan["namespace"] + "__"), "QA identity is not a new namespace")
        require(config["resume_from"] is None, "QA must be a fresh full run")
        require(final["original_checkpoint_metadata"]["config"] == source_config, "Final source config differs")
        require(final["model_fingerprint"] == old["expected_final_model_state_fingerprint"], "Final fingerprint disagrees with original plan")
        metadata_path = Path(best_metadata_root) / best["metadata_member"]
        require(metadata_path.resolve().is_relative_to(Path(best_metadata_root).resolve()), "Best metadata escapes accepted root")
        require(sha(metadata_path) == best["metadata_sha256"] and metadata_path.stat().st_size < 1 << 24, "Best metadata identity differs or is not metadata-sized")
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
        require("model_state" not in metadata and "optimizer_state" not in metadata, "Best expected input is not metadata-only")
        require(metadata["config"] == source_config and best["source_config_sha256"] == old["source_config"]["sha256"], "Best source config differs")
        full = subset_counters(final["original_checkpoint_metadata"]["counters"])
        require(full == subset_counters(final["summary_counters"]), "Original final counters disagree")
        require(full["target_prediction_tokens"] == config["target_prediction_tokens"], "Original target horizon differs")
        replay = int(Fraction(str(config["replay_ratio"])) * config["target_prediction_tokens"])
        require(full["replay_prediction_tokens"] == replay, "Original final replay exposure differs")
        bc = subset_counters(metadata["counters"])
        exposure = best["actual_exposure"]
        require(bc == {"step": exposure["checkpoint_step"], "target_prediction_tokens": exposure["target_prediction_tokens"], "replay_prediction_tokens": exposure["replay_prediction_tokens"]}, "Best counter/manifest exposure differs")
        require(exposure["full_training_target_prediction_tokens"] == full["target_prediction_tokens"] and exposure["full_training_replay_prediction_tokens"] == full["replay_prediction_tokens"], "Best origin was not full training")
        require(metadata["checkpoint_kind"] == "model_only_best_target_validation", "Wrong original best kind")
        selection = metadata["selection"]
        require(math.isfinite(selection["nll"]) and selection["step"] == bc["step"] and selection["target_prediction_tokens"] == bc["target_prediction_tokens"], "Best selection differs from exposure")
        require(selection["checkpoint_path"] == best["source_checkpoint_path"], "Best source selection path differs")
        provenance_matches(metadata["provenance"], old["required_original_execution_provenance"])
        strict_policy(old["required_original_execution_provenance"], config)
        output_dir = Path(config["output_dir"])
        require(output_dir == Path(row["output_dir"]) and output_dir.name == run_id, "QA output/run ID differs")
        require(output_dir.resolve().is_relative_to(scratch / "runs"), "Output escapes independent QA scratch")
        final_path = output_dir / "checkpoints/latest.pt"
        best_path = output_dir / "checkpoints/checkpoint_best.pt"
        require(str(final_path) == old["planned_final_checkpoint"], "Final QA path differs from plan")
        mapped_selection = {**selection, "checkpoint_path": str(best_path)}
        expected_runs.append({
            "run_id": run_id, "original_run_id": origin, "group": old["group"],
            "config": proof(config_path), "output_dir": str(output_dir), "summary_copy": row["summary_path"],
            "internal_summary": str(output_dir / "summary.json"), "full_counters": full,
            "required_provenance": old["required_original_execution_provenance"],
            "endpoints": [
                {"kind": "final", "checkpoint": str(final_path), "counters": full,
                 "expected_fingerprint": old["expected_final_model_state_fingerprint"],
                 "checkpoint_kind_present": "checkpoint_kind" in final["original_checkpoint_metadata"],
                 "checkpoint_kind": final["original_checkpoint_metadata"].get("checkpoint_kind"),
                 "original_checkpoint": final["source_checkpoint_path"], "original_checkpoint_sha256": final["source_checkpoint_sha256"]},
                {"kind": "best", "checkpoint": str(best_path), "counters": bc,
                 "expected_fingerprint": best["model_fingerprint"], "checkpoint_kind_present": True,
                 "checkpoint_kind": metadata["checkpoint_kind"], "selection": mapped_selection,
                 "original_selection": selection, "original_checkpoint": best["source_checkpoint_path"],
                 "original_checkpoint_sha256": best["source_checkpoint_sha256"],
                 "accepted_metadata": proof(metadata_path)},
            ],
        })
        all_sources += [proof(metadata_path), proof(config_path), old["source_config"]]
    unique_sources = {str(Path(s["path"]).resolve()): s for s in all_sources}
    result = {
        "schema_version": 1, "status": "prepared_expected_endpoints_only_not_run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "namespace": plan["namespace"],
        "plan": proof(plan_path), "registry": plan["registry"],
        "queue": {k: v for k, v in plan["queue"].items() if k in {"path", "sha256"}},
        "scheduler_state": plan["planned_state"], "planned_gpus": plan["planned_gpus"],
        "archived_workspace": str(workspace), "independent_scratch": str(scratch), "source_proofs": list(unique_sources.values()),
        "archived_fingerprint_algorithm": {**proof(algorithm), "function": "model_fingerprint"},
        "runs": expected_runs, "expected_endpoint_count": len(expected_runs) * 2,
        "new_checkpoint_reads_or_retraining_performed": False, "root_launch_authority_granted": False,
        "planning_CPU_postrun_allowance_seconds": 300, "planning_allowance_is_measured": False,
        "full_weight_archive_rehash_performed": False,
    }
    require(all(not Path(e["checkpoint"]).exists() for r in expected_runs for e in r["endpoints"]), "New QA checkpoints already exist; cannot claim preparation only")
    return result


class CountingReader:
    """Count bytes delivered by explicit serialized reads, including rereads.

    This is not a physical-disk/page-cache counter. Passing it as torch.load's
    documented file-like input leaves the archived algorithm unmodified.
    """
    def __init__(self, raw):
        self.raw = raw
        self.bytes_read = 0

    def read(self, n=-1):
        value = self.raw.read(n); self.bytes_read += len(value); return value

    def readinto(self, buffer):
        n = self.raw.readinto(buffer); self.bytes_read += n or 0; return n

    def readline(self, n=-1):
        value = self.raw.readline(n); self.bytes_read += len(value); return value

    def seek(self, *args):
        return self.raw.seek(*args)

    def tell(self):
        return self.raw.tell()


def load_algorithm(reference):
    path = bound(reference)
    spec = importlib.util.spec_from_file_location("_archived_qa_fullstate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(sha(path) == reference["sha256"], "Archived source changed during import")
    return module.model_fingerprint


def process_gone(pid, group=False):
    require(isinstance(pid, int) and pid > 1, "Missing/invalid recorded process identity")
    try:
        (os.killpg if group else os.kill)(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def measure_read(path, operation, receipt, callback):
    item = {"operation": operation, "checkpoint": str(path), "serialized_bytes_read": 0,
            "wall_seconds": 0., "process_cpu_seconds": 0., "completed": False}
    receipt["checkpoint_read_operations"].append(item)
    wall, cpu = time.monotonic(), time.process_time()
    with path.open("rb") as raw:
        stream = CountingReader(raw)
        try:
            result = callback(stream)
            item["completed"] = True
            return result
        finally:
            item.update(serialized_bytes_read=stream.bytes_read, wall_seconds=time.monotonic() - wall,
                        process_cpu_seconds=time.process_time() - cpu)


def hash_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        h.update(block)
    return h.hexdigest()


def validate_sources(expected):
    for source in expected["source_proofs"]:
        bound(source)
    plan = read(bound(expected["plan"]))
    require(plan["namespace"] == expected["namespace"], "Namespace differs from original plan")
    require(plan["planned_state"] == expected["scheduler_state"], "Scheduler state differs from original plan")
    require(len(expected["runs"]) == len(plan["future_full_run_comparison"]["expected_runs"]), "Expected run grid differs")
    return plan


def verify(expected_path, expected_sha256, receipt):
    require(sha(expected_path) == expected_sha256, "Expected endpoint JSON SHA differs")
    expected = read(expected_path)
    plan = validate_sources(expected)
    jobs = read(bound(expected["queue"]))
    registry = read(bound(expected["registry"]))
    state_path = Path(expected["scheduler_state"])
    state_before = five_stat(state_path)
    state = read(state_path)
    state_sha = sha(state_path)
    ids = {r["run_id"] for r in expected["runs"]}
    require(set(state["jobs"]) == ids and not state["active_jobs"], "All QA jobs must finish; no partial grid")
    require(process_gone(state["scheduler_pid"]), "QA scheduler still exists")
    require(state["deadline_utc"] == plan["qa_deadline_utc"], "QA scheduler deadline differs")
    require(len({state["jobs"][i]["gpu"] for i in ids}) == len(ids), "QA jobs did not occupy separate frozen GPUs")
    require({str(state["jobs"][i]["gpu"]) for i in ids} == {str(i) for i in expected["planned_gpus"]}, "QA GPU identity differs")
    by_job, by_registry = {j["id"]: j for j in jobs}, {r["id"]: r for r in registry}
    inputs, endpoints, inodes = [], [], set()
    # Complete ALL structural gates before reading ANY new checkpoint storage.
    for run in expected["runs"]:
        run_id = run["run_id"]; record = state["jobs"][run_id]; job = by_job[run_id]
        require(record["status"] == "completed" and record["returncode"] == 0 and not record["missing_outputs"], "QA job incomplete/failed: " + run_id)
        require("stop_requested_at" not in record, "QA job received a stop request")
        require(record["definition_sha256"] == hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(), "QA job definition differs")
        require(record["argv"] == job["argv"] and process_gone(record["pid"], group=True), "QA child argv differs or process group still exists")
        config_path = bound(run["config"]); cfg = read(config_path)
        require(run["config"]["sha256"] == by_registry[run_id]["config_sha256"], "Registry/config differs")
        require(cfg["run_label"] == run_id and run_id.startswith(expected["namespace"] + "__") and cfg["resume_from"] is None, "New QA identity/resumption differs")
        output = exact_path(cfg["output_dir"], run["output_dir"])
        require(output.name == run_id and output == Path(by_registry[run_id]["output_dir"]), "QA output namespace differs")
        require(output.resolve().is_relative_to(Path(expected["independent_scratch"]) / "runs"), "QA scratch parent was redirected")
        cp_root = output / "checkpoints"
        summary_path = exact_path(run["summary_copy"], by_registry[run_id]["summary_path"], expected["archived_workspace"])
        internal = exact_path(run["internal_summary"], output / "summary.json", output)
        require(set(job["outputs"]) == {str(summary_path), str(internal)}, "Queue expected outputs differ")
        for p in (config_path, summary_path, internal):
            inputs.append({"path": str(p), "sha256": sha(p), "five_stat": five_stat(p)})
        summary = read(summary_path)
        require(summary == read(internal) and summary["config"] == cfg, "Internal/copied summary or config differs")
        require(summary["status"] == "completed" and subset_counters(summary["counters"]) == run["full_counters"], "Summary incomplete or actual final exposure differs")
        provenance_matches(summary["provenance"], run["required_provenance"])
        strict_policy(run["required_provenance"], cfg)
        best = next(e for e in run["endpoints"] if e["kind"] == "best")
        require(summary["best_target_validation"] == best["selection"], "Actual best selection differs from accepted original")
        final = next(e for e in run["endpoints"] if e["kind"] == "final")
        require(summary["checkpoint_path"] == final["checkpoint"], "Summary final checkpoint path differs")
        require({e["kind"] for e in run["endpoints"]} == {"final", "best"} and len(run["endpoints"]) == 2, "Both endpoint kinds required")
        for endpoint in run["endpoints"]:
            name = "latest.pt" if endpoint["kind"] == "final" else "checkpoint_best.pt"
            path = exact_path(endpoint["checkpoint"], cp_root / name, output)
            before = five_stat(path)
            require(before[:2] not in [list(x) for x in inodes], "Checkpoints share an inode; no endpoint may reuse another file")
            inodes.add(tuple(before[:2]))
            original = Path(endpoint["original_checkpoint"])
            require(path.resolve() != original.resolve() and not (original.exists() and os.path.samefile(path, original)), "QA checkpoint reuses original file")
            ended = datetime.fromisoformat(record["ended_at_utc"].replace("Z", "+00:00")).timestamp()
            require(record["started_at_epoch"] <= before[3] / 1e9 <= ended, "Checkpoint mtime lies outside this actual QA run")
            endpoints.append((run, endpoint, path, cfg, summary, before))
    require(len(endpoints) == expected["expected_endpoint_count"] == len(ids) * 2, "Incomplete final+best endpoint grid")
    receipt["structural_gate_complete_before_checkpoint_reads"] = True
    receipt["scheduler_state"] = {"path": str(state_path), "sha256": state_sha, "five_stat": state_before}
    receipt["expected_endpoint_count"] = len(endpoints)
    algorithm = load_algorithm(expected["archived_fingerprint_algorithm"])
    receipt["archived_fingerprint_algorithm"] = expected["archived_fingerprint_algorithm"]
    import torch
    torch.set_num_threads(2)
    for run, endpoint, path, cfg, summary, before in endpoints:
        saved = measure_read(path, "metadata_full_torch_load_cpu", receipt,
                             lambda f: torch.load(f, map_location="cpu", weights_only=False))
        require(saved["config"] == cfg, "Checkpoint config differs from exact QA config")
        require(subset_counters(saved["counters"]) == endpoint["counters"], "Checkpoint actual endpoint exposure differs")
        provenance_matches(saved["provenance"], run["required_provenance"])
        require(("checkpoint_kind" in saved) == endpoint["checkpoint_kind_present"] and saved.get("checkpoint_kind") == endpoint["checkpoint_kind"], "Checkpoint kind differs")
        require(all(str(t.dtype) == run["required_provenance"]["model_weight_dtype"] for t in saved["model_state"].values()), "Actual model tensor dtype differs")
        if endpoint["kind"] == "best":
            require(saved["selection"] == endpoint["selection"] and "optimizer_state" not in saved, "Best selection/model-only payload differs")
        else:
            require("optimizer_state" in saved and saved["best_target_validation"] == summary["best_target_validation"], "Final full checkpoint/best reference differs")
        del saved
        gc.collect()
        fingerprint = measure_read(path, "archived_full_model_fingerprint_cpu", receipt, algorithm)
        fresh_sha = measure_read(path, "fresh_entire_checkpoint_sha256", receipt, hash_stream)
        after = five_stat(path)
        require(before == after, "Checkpoint changed during metadata/fingerprint/SHA reads")
        require(fresh_sha != endpoint["original_checkpoint_sha256"], "Serialized original checkpoint copied as a new QA endpoint")
        require(fingerprint == endpoint["expected_fingerprint"], "Full model_state bytes differ: " + run["run_id"] + " " + endpoint["kind"])
        receipt["endpoint_proofs"].append({
            "run_id": run["run_id"], "original_run_id": run["original_run_id"], "kind": endpoint["kind"],
            "checkpoint": str(path), "checkpoint_sha256": fresh_sha, "five_stat_before": before, "five_stat_after": after,
            "actual_exposure": endpoint["counters"], "actual_selection": endpoint.get("selection"),
            "checkpoint_kind": endpoint["checkpoint_kind"], "model_state_fingerprint": fingerprint,
            "bitwise_full_model_state_equal": True,
        })
    for item in inputs:
        require(five_stat(item["path"]) == item["five_stat"] and sha(item["path"]) == item["sha256"], "Summary/config changed during verification")
    for run, endpoint, path, cfg, summary, before in endpoints:
        require(five_stat(path) == before, "Earlier checkpoint changed before whole-phase acceptance")
    require(five_stat(state_path) == state_before and sha(state_path) == state_sha, "Scheduler state changed during verification")
    require(sha(expected_path) == expected_sha256, "Expected endpoint JSON changed during verification")
    validate_sources(expected)
    receipt["new_run_inputs"] = inputs
    receipt["status"] = "passed_all_final_and_best_full_model_state_bytes"
    receipt["all_expected_endpoint_weights_bitwise_equal"] = True


def verify_to_receipt(expected_path, expected_sha256, output):
    # Reserve the result exclusively before doing work, so a rerun cannot erase
    # either a passed or a failed first attempt. Failure is explicitly persisted.
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        wall, cpu = time.monotonic(), time.process_time()
        receipt = {
            "schema_version": 1, "status": "failed_complete_reproduction_not_verified",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "expected": {"path": str(Path(expected_path).resolve()), "sha256": expected_sha256},
            "verifier": proof(__file__), "checkpoint_read_operations": [], "endpoint_proofs": [],
            "GPU_used": False, "new_scientific_replicates": 0, "cost_category": "engineering_QA_diagnostic_CPU_postrun_acceptance",
            "training_or_validation_trajectories_compared": False, "optimizer_or_RNG_or_stream_states_compared": False,
            "serialized_original_checkpoint_equality_required": False, "planning_300s_is_not_a_measurement": True,
            "serialized_read_bytes_meaning": "actual bytes returned by file read/readinto/readline, with repeat reads counted; not physical disk bytes or CUDA work",
            "operation_timers_nested_in_parent_wall_and_CPU": True,
        }
        try:
            require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Explicit CUDA_VISIBLE_DEVICES='' is required for CPU verification")
            verify(expected_path, expected_sha256, receipt)
        except Exception as error:
            receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        finally:
            receipt["wall_seconds"] = time.monotonic() - wall
            receipt["process_cpu_seconds"] = time.process_time() - cpu
            receipt["serialized_checkpoint_read_bytes_total"] = sum(x["serialized_bytes_read"] for x in receipt["checkpoint_read_operations"])
            receipt["fresh_full_checkpoint_SHA_bytes"] = sum(x["serialized_bytes_read"] for x in receipt["checkpoint_read_operations"] if x["operation"] == "fresh_entire_checkpoint_sha256")
            receipt["checkpoint_read_operations_count"] = len(receipt["checkpoint_read_operations"])
            json.dump(receipt, handle, ensure_ascii=False, indent=2); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare-expected")
    for flag in ("plan", "final-manifest", "best-manifest", "best-metadata-root", "best-acceptance", "algorithm", "output"):
        prep.add_argument("--" + flag, type=Path, required=True)
    for flag in ("plan-sha256", "best-sha256", "algorithm-sha256"):
        prep.add_argument("--" + flag, required=True)
    accept = sub.add_parser("verify")
    accept.add_argument("--expected", type=Path, required=True)
    accept.add_argument("--expected-sha256", required=True)
    accept.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare-expected":
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Explicit CUDA_VISIBLE_DEVICES='' is required")
        require(not args.output.exists(), "Expected metadata output is immutable")
        result = prepare(args.plan, args.plan_sha256, args.final_manifest, args.best_manifest,
                         args.best_sha256, args.best_metadata_root, args.best_acceptance,
                         args.algorithm, args.algorithm_sha256)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(result, f, ensure_ascii=False, indent=2); f.write("\n")
        print(json.dumps({"status": result["status"], "expected_endpoint_count": result["expected_endpoint_count"], "output_sha256": sha(args.output)}))
    else:
        result = verify_to_receipt(args.expected, args.expected_sha256, args.output)
        print(json.dumps({"status": result["status"], "endpoint_proofs_completed": len(result["endpoint_proofs"]), "output": str(args.output), "wall_seconds": result["wall_seconds"]}))
        if not result["status"].startswith("passed_"):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
