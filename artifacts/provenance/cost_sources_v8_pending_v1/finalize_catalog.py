#!/usr/bin/env python3
"""Small-source-only QA and append-only, ROOT-confirmed completed-cost supplement."""
import datetime
import hashlib
import json
from pathlib import Path
import resource
import time

from prepare_catalog import OUT, ROOT, EXPECTED_SPEC, SPEC, pointer, relative


def write(name, value):
    path = OUT / name
    if path.exists():
        raise ValueError(f"refuse replacing existing final preparation evidence: {name}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return {"path": relative(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def small_source(path):
    path = ROOT / path
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("only bounded small metadata may be read")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
        if getattr(before, key) != getattr(after, key):
            raise ValueError("small metadata changed during read")
    sha = hashlib.sha256(raw).hexdigest()
    retained = OUT / "source_bytes" / f"{sha}.json"
    if retained.exists() and retained.read_bytes() != raw:
        raise ValueError("retained bytes conflict")
    if not retained.exists():
        retained.write_bytes(raw)
    return json.loads(raw), {"path": relative(path), "sha256": sha, "bytes": len(raw),
                            "retained_bytes": relative(retained),
                            "stat": {k: getattr(after, k) for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}}


def main():
    start = time.monotonic()
    cpu = resource.getrusage(resource.RUSAGE_SELF)
    if hashlib.sha256(SPEC.read_bytes()).hexdigest() != EXPECTED_SPEC:
        raise ValueError("v7 source specification changed")
    original = json.loads((OUT / "pending_sources.json").read_bytes())
    QA = []
    for row in original["candidates"]:
        obj, proof = small_source(row["source"]["path"])
        if proof["sha256"] != row["source"]["sha256"]:
            raise ValueError("candidate source changed")
        if pointer(obj, row["wall_pointer"]) != row["wall_seconds"]:
            raise ValueError("candidate wall pointer drift")
        if {p: pointer(obj, p) for p in row["cpu_component_pointers"]} != row["cpu_component_values"]:
            raise ValueError("candidate CPU pointer drift")
        QA.append({"id": row["id"], "source_sha256": proof["sha256"], "status": "passed"})

    additions = []
    for id_, path, scope in [
        ("C410_best_batch", "artifacts/provenance/additional_endpoint_actual_v4/C410_best_batch/whole_child_timing.json",
         "ROOT confirmed complete after initial catalog freeze. Actual exporter spawn-to-exit: startup/resume verification, 20 new weights, three proven-alias metadata operations and assembly. Original 105.168669581 sample encode is a separate prior operation; sample resume checks do not re-charge that encode. Assembly 109.004355 seconds is nested and not additive."),
        ("C410_best_publication_metadata_review", "artifacts/provenance/C410_best_publication_metadata_review_actual_v1/whole_child_timing.json",
         "ROOT confirmed complete after initial catalog freeze. Independent whole metadata reviewer spawn-to-exit; internal reviewer regions are nested. No large payload rehash, no publication included.")]:
        obj, proof = small_source(path)
        if type(obj.get("returncode")) is not int or obj["returncode"] != 0:
            raise ValueError("ROOT-confirmed completed supplement requires strict integer rc0")
        if "stop_requested" in obj or "stop_requested_at" in obj:
            raise ValueError("stopped operation is not a completed supplement")
        cpus = ["/user_cpu_seconds", "/sys_cpu_seconds"]
        additions.append({"id": id_, "source": proof,
            "wall_pointer": "/wall_seconds", "wall_seconds": obj["wall_seconds"],
            "cpu_component_pointers": cpus, "cpu_component_values": {p: pointer(obj, p) for p in cpus},
            "recorded_CPU_seconds": sum(pointer(obj, p) for p in cpus),
            "CPU_scope": "whole_child_process_tree", "returncode_pointer": "/returncode", "returncode": 0,
            "classification": "pending_measured_independent_parent",
            "timing_semantics": "independent_measured_operation", "scope": scope,
            "nested_subtimers_added": False, "registered_in_active_spec": False})
    supplement = write("supplement_01_completed_C410_best.json", {
        "schema_version": 1, "status": "pending_source_supplement_only",
        "basis": "ROOT explicitly confirmed both completed parent timers in the current task; source receipts independently checked here. This is not an execution or publication authority.",
        "initial_catalog_sha256": hashlib.sha256((OUT / "pending_sources.json").read_bytes()).hexdigest(),
        "candidates": additions,
        "C410_best_publisher_registered": False, "new_project_snapshot_registered": False,
        "initial_freeze_limitations": "The initial catalog's no-Cbest-batch statement describes its freeze boundary; only these two subsequently ROOT-confirmed completed parents are now appended."})

    registered_examples = []
    spec = json.loads(SPEC.read_bytes())
    for operation in spec["additional_operations"]:
        if operation["id"] == "extra_export_A_remaining_final_single_sample_whole_child" or operation["id"].startswith("native_Cfinal_"):
            obj, proof = small_source(operation["source"]["path"])
            if proof["sha256"] != operation["source"]["sha256"]:
                raise ValueError("previously registered context changed")
            registered_examples.append({"id": operation["id"], "registered_source": proof,
                "original_source": obj.get("original_source"), "original_wall_pointer": obj.get("original_wall_pointer"),
                "cost_added_again": False})
    context = write("already_registered_v7_context.json", {"status": "existing_context_not_new_cost", "operations": registered_examples,
        "note": "Native-C publication reviewer preparation/test parents here were already registered by v7. Current Cmodel-NLL reviewer synthetic tests are distinct actual calls and use different source paths."})

    prep_rows = []
    for leaf in ["preparation_whole_child_timing.json", "preparation_attempt2_whole_child_timing.json"]:
        obj, proof = small_source(relative(OUT / leaf))
        prep_rows.append({"source": proof, "wall_pointer": "/whole_child_wall_seconds",
            "wall_seconds": obj["whole_child_wall_seconds"],
            "cpu_component_pointers": ["/child_user_seconds", "/child_system_seconds"],
            "cpu_component_values": {p: pointer(obj, p) for p in ["/child_user_seconds", "/child_system_seconds"]},
            "returncode": obj["returncode"], "nested_regions_excluded": True,
            "note": "This source-list preparation call is a newly measured parent candidate, separate from the 38 historical candidates. Failed attempt source/log bytes retained; manual preceding inventory and authoring whole unknown."})
    prep = write("this_preparation_cost_supplement.json", {"status": "measured_preparation_calls_for_future_registration_only", "parents": prep_rows,
        "failure": "First preparation attempt failed only because an excluded A acceptance copy used whole_child_cost rather than whole_process_cost; corrected source and original source/log/receipt retained.",
        "unknown_manual_engineering_parent_seconds": None})
    totals = {"initial_historical_candidates": len(original["candidates"]), "subsequently_confirmed_completed_candidates": len(additions),
              "historical_candidate_total": len(original["candidates"]) + len(additions),
              "historical_independent_parents": 31, "historical_partial_regions": 7,
              "this_preparation_additional_measured_parent_calls": len(prep_rows)}
    identity_unique = len({(r["source"]["path"], r["wall_pointer"]) for r in original["candidates"] + additions}) == 38
    if not identity_unique:
        raise ValueError("historical candidate identities are not unique")
    qa = write("pointer_and_source_QA.json", {"status": "passed_small_source_only_QA", "checks": QA,
        "candidate_identity_unique": identity_unique,
        "v7_specification_SHA_unchanged": True, "collector_executed": False, "large_payload_bytes_read": 0,
        "active_configs_modified": False, "score_semantics_read": False, "counts": totals})
    end = resource.getrusage(resource.RUSAGE_SELF)
    final_region = write("finalization_region_receipt.json", {"status": "completed_small_metadata_finalization_region",
        "region_wall_seconds": time.monotonic() - start,
        "region_self_user_seconds": end.ru_utime - cpu.ru_utime,
        "region_self_system_seconds": end.ru_stime - cpu.ru_stime,
        "scope": "This script main-entry through fresh source/pointer QA and append-only supplement writes; excludes startup/imports/final receipt/COMPLETE serialization. Manual parent unknown.",
        "manual_engineering_parent_seconds": None, "not_an_additional_whole_parent": True})
    outputs = []
    for path in sorted(OUT.glob("*.json")):
        outputs.append({"path": relative(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    final = write("COMPLETE.json", {"schema_version": 1, "status": "completed_pending_cost_source_preparation_only",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "counts": totals,
        "historical_source_catalog": {"path": relative(OUT / "pending_sources.json"), "sha256": hashlib.sha256((OUT / "pending_sources.json").read_bytes()).hexdigest()},
        "completed_after_initial_freeze_supplement": supplement,
        "existing_registered_context": context, "source_pointer_QA": qa, "this_preparation_parent_candidates": prep,
        "finalization_region": final_region, "outputs": outputs,
        "collector_run": False, "ROOT_collection_authority": False, "score_or_model_payload_reads": False,
        "Cbest_publisher_and_new_snapshot_not_pre_registered": True, "manual_engineering_whole_unknown": True})
    print(json.dumps({"completion": final, "counts": totals}, ensure_ascii=False))


if __name__ == "__main__":
    main()
