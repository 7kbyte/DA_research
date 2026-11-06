#!/usr/bin/env python3
"""Freeze an explicit small-receipt candidate list; never run a cost collector."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SPEC = ROOT / "configs/cost_inventory_final_sources_v7.json"
EXPECTED_SPEC = "0cab816da531e857ded88a004624d9a946a1bd3dae75f295ed68e1e4f40b1f0e"


def pointer(obj, path):
    for part in path.split("/")[1:]:
        key = part.replace("~1", "/").replace("~0", "~")
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj


def relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


cache = {}


def read_source(path):
    path = (ROOT / path).resolve()
    if str(path) in cache:
        return cache[str(path)]
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > 16 * 1024 * 1024:
            raise ValueError(f"not an allowed small regular metadata file: {path}")
        data = stream.read()
        after = os.fstat(stream.fileno())
    now = path.stat()
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, k) != getattr(after, k) or getattr(after, k) != getattr(now, k) for k in keys):
        raise ValueError(f"metadata source changed during read: {path}")
    digest = hashlib.sha256(data).hexdigest()
    proof = {"path": relative(path), "sha256": digest, "bytes": len(data),
             "stat": {k: getattr(now, k) for k in keys},
             "retained_bytes": f"source_bytes/{digest}.json"}
    target = OUT / proof["retained_bytes"]
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError("content-addressed retained-byte collision")
    else:
        target.write_bytes(data)
    result = (json.loads(data), proof)
    cache[str(path)] = result
    return result


def desc(id_, path, wall, cpu, scope, *, region=False, cpu_scope="whole_child_process_tree", note=""):
    return dict(id=id_, path=path, wall=wall, cpu=cpu, scope=scope,
                region=region, cpu_scope=cpu_scope, note=note)


def main():
    start = time.monotonic()
    self_before = resource.getrusage(resource.RUSAGE_SELF)
    (OUT / "source_bytes").mkdir(exist_ok=True)
    spec, spec_proof = read_source(SPEC)
    if spec_proof["sha256"] != EXPECTED_SPEC:
        raise ValueError("existing v7 specification changed")
    registered = {}
    for operation in spec["additional_operations"]:
        source = operation["source"]
        source_obj, proof = read_source(source["path"])
        if proof["sha256"] != source["sha256"]:
            raise ValueError(f"registered v7 receipt SHA mismatch: {operation['id']}")
        original = source_obj.get("original_source")
        raw = relative(original["path"]) if original else proof["path"]
        wall = source_obj.get("original_wall_pointer", operation.get("wall_pointer"))
        sha = original["sha256"] if original else proof["sha256"]
        registered.setdefault((raw, wall), []).append({"id": operation["id"],
            "original_source_sha256": sha, "registered_source": proof})

    D = []
    def add(*args, **kwargs):
        D.append(desc(*args, **kwargs))
    def short(id_, path, scope, **kwargs):
        add(id_, path, "/wall_seconds", ["/user_cpu_seconds", "/sys_cpu_seconds"], scope, **kwargs)

    add("cost_v6_successful_collection", "artifacts/provenance/cost_sources_v6/trial_whole_child_timing.json",
        "/whole_child_wall_seconds", ["/whole_child_user_cpu_seconds", "/whole_child_sys_cpu_seconds"],
        "Successful v6 collector spawn-to-exit; not previously charged to its own ledger.")
    add("cost_v7_successful_collection", "artifacts/provenance/cost_sources_v7/collection_whole_child_timing.json",
        "/whole_child_wall_seconds", ["/whole_child_user_seconds", "/whole_child_sys_seconds"],
        "Successful v7 collector plus IO-audit child; its internal collector timer is nested.")
    short("A_remaining93_batch", "artifacts/provenance/additional_endpoint_actual_v4/A_remaining_final_batch/whole_child_timing.json",
          "Actual A batch: startup/resume verification, 92 new encodes and whole TAR assembly. The original first-sample encode was a separate earlier v7-registered operation; reused sample metadata does not re-charge it.")
    add("A_remaining93_metadata_review", "artifacts/provenance/A_remaining_final_publication_metadata_review_actual_v1/whole_child_timing.json",
        "/whole_parent_wall_seconds", ["/parent_self_user_seconds", "/parent_self_system_seconds", "/children_user_seconds", "/children_system_seconds"],
        "Independent full outer Popen metadata review parent; child/reviewer-region timers nested.", cpu_scope="parent_self_plus_children")
    short("A_remaining93_publication", "artifacts/provenance/A_remaining_final_publication_actual_v1/whole_child_timing.json",
          "Actual publisher spawn-to-exit; source/full-prefix reads, copy and durable commit included; publisher internal timer nested.")
    add("A_remaining93_independent_metadata_region", "docs/A_remaining_final_durable_publication_independent_review_v1.json",
        "/timing/wall_seconds", ["/timing/self_user_seconds", "/timing/self_sys_seconds"],
        "Small-metadata/stat independent post-publication audit region only; manual review/authoring whole unknown.", region=True, cpu_scope="self_region")
    short("C410_best_first_sample", "artifacts/provenance/additional_endpoint_actual_v4/C410_best_sample/whole_child_timing.json",
          "Completed first sample spawn-to-exit only; encode/decode/uint8 verification endpoint timer nested. No future batch/publication registered here.")
    add("C410_best_sample_independent_region", "artifacts/provenance/C410_best_sample_independent_review_v1/REVIEW.json",
        "/audit_region_wall_seconds", ["/audit_region_user_seconds", "/audit_region_system_seconds"],
        "Completed independent sample small-metadata/stat audit region; embedded sample and collector timer copies excluded.", region=True, cpu_scope="self_region")
    add("B_native_preparation_region", "artifacts/provenance/B_native_restore_preparation_v1/preparation_receipt.json",
        "/timing/wall_seconds", ["/timing/self_user_seconds", "/timing/self_sys_seconds"],
        "Measured small-source preparation region only; manual authoring whole unknown. The recorded read counter covers the metadata/tool loop, not every helper hash/import read.", region=True, cpu_scope="self_region")
    add("B_native_schema_guard_region", "artifacts/provenance/B_native_restore_preparation_v1/schema_guard_checks.json",
        "/wall_seconds", ["/process_cpu_seconds"],
        "Seven pure schema-guard cases; only local region measured, not a whole engineering task.", region=True, cpu_scope="process_clock_region")
    add("B_native_shared_lock_regression", "artifacts/provenance/B_native_restore_preparation_v1/lock_regression_tests/whole_child_timing.json",
        "/whole_wall_seconds", ["/children_user_seconds", "/children_sys_seconds"],
        "Independent actual two-test child; known CPU covers children only, wrapper self CPU unrecorded.")
    for id_, leaf in [("B_native_independent_identity_probe", "whole_bound_identity_probe_timing.json"),
                      ("B_native_independent_lock_probe", "whole_lock_revision_probe_timing.json")]:
        add(id_, "artifacts/provenance/B_native_restore_independent_review_v1/" + leaf,
            "/wall_seconds", ["/user_cpu_seconds", "/system_cpu_seconds"],
            "Independent actual preparation probe child; manual review/authoring whole unknown.")
    short("B_native_actual_restore", "artifacts/provenance/B_native_restore_actual_parent_v1/whole_child_timing.json",
          "Actual fixed B first-sample durable native restore: complete outer Python spawn-to-exit including imports, admission, extraction, decode, independent full tensor fingerprint, receipt and process-group cleanup. All internal wrapper/body/stage timers are nested.")
    add("B_native_independent_actual_metadata_review", "artifacts/provenance/B_native_restore_independent_acceptance_v1/whole_child_timing.json",
        "/whole_child_wall_seconds", ["/child_user_seconds", "/child_system_seconds"],
        "Separate automated small-metadata/stat post-restoration child; not nested inside the prior ROOT restore. Manual reading/doc writing whole unknown.")
    for id_, leaf, note in [
        ("Cbest_preparation_collection_failure", "preparation_tests", "Actual failed pytest collection child, rc2; cost retained despite failure."),
        ("Cbest_preparation_importlib171_tests", "preparation_tests_importlib", "Actual 171-test rerun child: 113 new +39 native-v2 +19 frozen-export tests. This is a distinct rerun, not a copy of an earlier registered test invocation.")]:
        add(id_, f"artifacts/provenance/C_best_publication_runbook_v1/{leaf}/whole_child_timing.json",
            "/whole_wall_seconds", ["/parent_user_seconds", "/parent_sys_seconds", "/children_user_seconds", "/children_sys_seconds"],
            note, cpu_scope="parent_self_plus_children")
    add("Cbest_explicit_permission_tests", "artifacts/provenance/C_best_publication_runbook_v1/permission_gate_tests/whole_child_timing.json",
        "/whole_wall_seconds", ["/children_user_seconds", "/children_sys_seconds"],
        "Actual five-case explicit-authority guard child; wrapper self CPU unrecorded.")
    add("Cbest_preparation_region", "artifacts/provenance/C_best_publication_runbook_v1/preparation_receipt.json",
        "/timer/wall_seconds", ["/timer/self_user_seconds", "/timer/self_sys_seconds"],
        "Small-source/scope/preparation receipt region only; manual authoring whole unknown.", region=True, cpu_scope="self_region")
    for id_, leaf in [("Cbest_independent_original_probe", "whole_probe_timing.json"),
                      ("Cbest_independent_explicit_permission_probe", "whole_explicit_authority_v2_probe_timing.json")]:
        add(id_, "artifacts/provenance/C_best_publication_independent_review_v1/" + leaf,
            "/wall_seconds", ["/user_cpu_seconds", "/system_cpu_seconds"],
            "Independent actual small-metadata/synthetic probe child; manual review whole unknown.")
    for n in (1, 2):
        add(f"behavioral_presentation_synthetic_smoke_v{n}", f"artifacts/provenance/behavioral_presentation_preparation_v{n}/whole_smoke_timing.json",
            "/wall_seconds", ["/user_cpu_seconds", "/system_cpu_seconds"],
            "Actual whole synthetic five-phase presentation smoke child; fixture values are not study measurements. Per-phase presentation/plot timers nested; no real behavior scores were read by this catalog.")
    add("delivery_guide_v5_manifest_union_region", "results/delivery_capacity_guide_v5/metadata_calculation.json",
        "/timing/wall_seconds", ["/timing/self_user_seconds", "/timing/self_sys_seconds"],
        "Actual seven-small-manifest reparse/union/stat calculation region; manual engineering whole unknown. The copy in guide source receipt is excluded.", region=True, cpu_scope="self_region")
    add("delivery_guide_v5_navigation_region", "docs/交付文件与恢复容量指南_来源核对_v5.json",
        "/measured_regions/guide_and_navigation_region/wall_seconds",
        ["/measured_regions/guide_and_navigation_region/self_user_seconds", "/measured_regions/guide_and_navigation_region/self_sys_seconds"],
        "Actual separate small metadata/navigation writing region only; manual engineering whole unknown.", region=True, cpu_scope="self_region")
    for id_, leaf, note in [
        ("tests_attempt1", "tests_attempt1/whole_child_timing.json", "Failed first synthetic development test child, rc1."),
        ("tests_attempt2", "tests_attempt2/whole_child_timing.json", "Second independent synthetic development test child."),
        ("tests_attempt3", "tests_attempt3/whole_child_timing.json", "Third independent synthetic development test child."),
        ("tests_final", "tests_final/tests_whole_child_timing.json", "Independent final targeted synthetic test child."),
        ("tests_stable_source", "tests_stable_source/whole_child_timing.json", "Independent stable-source targeted 36-test child."),
        ("toy_cli", "toy_cli_whole_child_timing.json", "Whole synthetic CLI smoke child includes both nested CLI/plot/bootstrap executions."),
        ("toy_cli_stable_source", "toy_cli_stable_source/whole_child_timing.json", "Independent stable-source synthetic CLI smoke child includes both nested CLI/plot/bootstrap executions.")]:
        add("Cmodel_review_" + id_, "artifacts/provenance/C_model_nll_review_v2_preparation/" + leaf,
            "/whole_child_wall_seconds", ["/child_user_seconds", "/child_system_seconds"], note,
            note="Separate actual child invocations. Not every development iteration's source bytes were snapshotted; the final stable source is frozen. Synthetic fixture timers must never be charged as actual operations.")
    for id_, leaf, note in [
        ("regression_failed", "regression_whole_child_timing.json", "Actual failed synthetic regression child (four failures), rc1."),
        ("regression_fixed41", "regression_attempt2_whole_child_timing.json", "Actual corrected 41-test synthetic regression child."),
        ("schema_audit_syntax_failure", "actual_schema_audit_whole_child_timing.json", "Actual schema-audit child failed with syntax error, rc1."),
        ("schema_audit_actual_metadata", "actual_schema_audit_attempt2_whole_child_timing.json", "Actual corrected small C410 metadata/schema audit child only; no score payload hash/semantics, no model payloads.")]:
        add("Cmodel_review_" + id_, "artifacts/provenance/C_model_nll_review_v2_schema_repair_v1/" + leaf,
            "/whole_child_wall_seconds", ["/child_user_seconds", "/child_system_seconds"], note)

    rows = []
    for entry in D:
        obj, proof = read_source(entry["path"])
        components = {p: pointer(obj, p) for p in entry["cpu"]}
        wall = pointer(obj, entry["wall"])
        for value in [wall, *components.values()]:
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid measured timing in {entry['id']}")
        rc = obj.get("returncode")
        if rc is not None and type(rc) is not int:
            raise ValueError("returncode must be strict int")
        prior = registered.get((proof["path"], entry["wall"]), [])
        if prior and any(x["original_source_sha256"] != proof["sha256"] for x in prior):
            raise ValueError("same registered original source pointer changed SHA")
        rows.append({"id": entry["id"], "source": proof,
            "wall_pointer": entry["wall"], "wall_seconds": wall,
            "cpu_component_pointers": entry["cpu"], "cpu_component_values": components,
            "recorded_CPU_seconds": sum(components.values()), "CPU_scope": entry["cpu_scope"],
            "returncode_pointer": "/returncode" if "returncode" in obj else None,
            "returncode": rc, "original_status": obj.get("status"),
            "classification": "already_registered_v7_not_new" if prior else (
                "pending_measured_region_parent_unknown" if entry["region"] else "pending_measured_independent_parent"),
            "timing_semantics": "partial_region_parent_unknown_not_additive" if entry["region"] else "independent_measured_operation",
            "proposed_primary_parent": not entry["region"] and not prior,
            "previous_v7_matches": prior, "scope": entry["scope"], "source_version_limitations": entry["note"],
            "nested_subtimers_added": False, "registered_in_active_spec": bool(prior)})
    identities = [(r["source"]["path"], r["source"]["sha256"], r["wall_pointer"]) for r in rows]
    if len(set(identities)) != len(identities) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate candidate operation")

    excluded_specs = [
        ("/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/A_remaining_final/completion.json", "/wall_seconds", "A_remaining93_batch", "Assembly-only timer is inside the actual A batch parent."),
        ("docs/A_remaining_final_weights_durable_publication.json", "/wall_seconds", "A_remaining93_publication", "Internal publisher timer is nested; no second publication occurred."),
        ("artifacts/provenance/A_remaining_final_publication_actual_v1/ROOT_acceptance.json", "/whole_process_cost", "A_remaining93_publication", "Acceptance contains copies/references to the publication timing, not a new measured publication."),
        ("artifacts/provenance/C410_best_sample_independent_review_v1/REVIEW.json", "/sample_cost", "C410_best_first_sample", "Copied sample whole timer, not independent work; collector copies in the same receipt are also context only."),
        ("artifacts/provenance/B_native_restore_actual_v1/whole_child_timing.json", "/outer_wrapper_parent_wall_seconds", "B_native_actual_restore", "Inner main-entry wrapper, process-group child and all extraction/decode/fingerprint stages are nested under the actual outer spawn-to-exit parent."),
        ("artifacts/provenance/B_native_restore_actual_parent_v1/small_QA_evidence/actual_restore_receipt.json", "/runner_body_wall_seconds", "B_native_actual_restore", "Inner body and extraction/decode/full-fingerprint timers are nested."),
        ("artifacts/provenance/B_native_restore_actual_parent_v1/ROOT_acceptance.json", "/whole_process_cost", "B_native_actual_restore", "Copied original timing in ROOT acceptance, not a new restore call."),
        ("artifacts/provenance/B_native_restore_preparation_v1/preparation_receipt_v2.json", "/timer_boundary", None, "Revision receipt references existing original preparation and independent lock test; does not record a new whole engineering parent."),
        ("artifacts/provenance/C_best_publication_runbook_v1/preparation_receipt_v2.json", "/validation", None, "Revision receipt copies/references original preparation and guard-test timers, not another test invocation."),
        ("artifacts/provenance/C_model_nll_review_v2_preparation/toy_cli_smoke.json", "/nested_cli_timers", "Cmodel_review_toy_cli", "Both actual nested CLI timers belong to the single toy CLI smoke whole parent."),
        ("artifacts/provenance/C_model_nll_review_v2_preparation/toy_cli_stable_source/toy_cli_smoke.json", "/nested_cli_timers", "Cmodel_review_toy_cli_stable_source", "Both actual nested CLI timers belong to the stable toy CLI smoke whole parent."),
        ("docs/交付文件与恢复容量指南_来源核对_v5.json", "/measured_regions/seven_manifest_reparse_and_union", "delivery_guide_v5_manifest_union_region", "Same measured region copied from metadata_calculation.json; register only its canonical source."),
    ]
    exclusions = []
    for path, ptr, parent, reason in excluded_specs:
        obj, proof = read_source(path)
        pointer(obj, ptr)
        exclusions.append({"source": proof, "pointer": ptr, "parent_candidate_id": parent,
                           "classification": "nested_or_copied_context_excluded", "reason": reason})

    limitations = [
        "This is a source candidate catalog only; no v8 collector run, no normalized active v8 specification, and no change to v6/v7 accounting.",
        "Wall times of overlapping operations are not calendar elapsed time; do not sum concurrent worker times and call the result study elapsed duration.",
        "Failure attempts are real measured CPU operations and are retained with their original nonzero returncodes, not relabeled successful.",
        "A first sample is already charged in v7; the batch parent includes only later actual work plus sample resume verification, not a repeated initial encode.",
        "Two behavioral-presentation smoke whole times total 42.40141221508384 seconds; per-phase synthetic outputs are nested, and fixture values are not actual measured scientific scores.",
        "Cmodel toy input analyze/check timing JSONs contain deliberately fake 1.0-second fixture values and must never be treated as actual cost receipts; only measured outer smoke parents listed here are candidates.",
        "The standalone first 89-test Cbest run lacks an independent parent JSON timer; its process whole is unknown, not inferred from pytest internal duration or wall-clock timestamps.",
        "Manual literature/review/coding/receipt writing full task durations are unknown; local measured preparation regions do not repair this missing parent evidence.",
        "RUSAGE_CHILDREN CPU values cover child process trees only unless explicit parent-self components are listed; unknown wrapper self CPU must not be manufactured.",
        "Cbest batch/publication and any new project snapshot are explicitly outside this preparation; no incomplete or future timer is pre-registered.",
    ]
    payload = {"schema_version": 1, "status": "pending_cost_v8_source_candidates_only",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "collector_executed": False, "ROOT_collection_authority": False,
        "active_cost_configs_modified": False, "GPU_used": False,
        "score_semantics_read": False, "large_payload_bytes_read": 0,
        "existing_v7_specification": spec_proof,
        "candidate_count": len(rows), "classification_counts": {k: sum(r["classification"] == k for r in rows) for k in sorted({r["classification"] for r in rows})},
        "candidates": rows, "nested_or_copied_context_exclusions": exclusions,
        "unknown_manual_and_scope_limitations": limitations,
        "future_registration_requires_ROOT_authorization": True}
    target = OUT / "pending_sources.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    head = ["# v8 成本待登记来源（仅准备）", "", "未运行 collector，未修改 v6/v7，也没有登记未来 Cbest 整批、发布或新快照。所有引用均为现存小型收据；来源原字节按 SHA 保存在 source_bytes。", "", "| ID | 原始 wall（秒） | 状态 / 范围 |", "|---|---:|---|"]
    for row in rows:
        label = "局部区域；人工整体未知" if "region" in row["classification"] else ("失败调用" if row["returncode"] not in (None, 0) else "独立已测父调用")
        head.append(f"| {row['id']} | {row['wall_seconds']:.9f} | {label} |")
    head.extend(["", "实际登记时：优先使用每行原始 wall_pointer 与 cpu_component_pointers；同一父项的 encode/decode/assembly/发布内部计时及复制进 acceptance 的计时均不重复相加。局部区域单列，不能冒充完整人工工作成本。", "", "A93 首例已在 v7 登记；新批量父项包括后续 92 份及打包。B native 真正恢复的外部 36.184441 秒父项包含内层 wrapper/body/提取/解码/独立指纹。C model 的两次 toy whole 各含两条嵌套 CLI；toy 输入里伪造的 1 秒 timing 不是成本证据。", "", "两次行为演示 synthetic smoke 父调用合计 42.401412 秒，与已登记的 17:30 答辩渲染不是同一次调用。失败测试和语法失败保留原 rc；它们的真实成本不得丢弃。", "", "完整每项来源 SHA、精确 JSON 指针、CPU 分量与父子解释见 pending_sources.json。所有人工整体时长未知，不由 UTC 差、pytest 内部 duration 或局部计时补造。"])
    (OUT / "待登记说明.md").write_text("\n".join(head) + "\n")
    script = Path(__file__).read_bytes()
    source_manifest = {"schema_version": 1, "status": "frozen_small_source_copies",
        "preparation_source": {"path": relative(__file__), "sha256": hashlib.sha256(script).hexdigest(), "bytes": len(script)},
        "sources": [v[1] for _, v in sorted(cache.items())],
        "total_original_small_source_bytes_read": sum(v[1]["bytes"] for v in cache.values()),
        "candidate_catalog": {"path": relative(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()},
        "large_payload_bytes_read": 0}
    (OUT / "source_manifest.json").write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n")
    end = resource.getrusage(resource.RUSAGE_SELF)
    receipt = {"status": "completed_candidate_catalog_preparation_only", "collector_executed": False,
        "region_wall_seconds": time.monotonic() - start,
        "region_self_user_seconds": end.ru_utime - self_before.ru_utime,
        "region_self_system_seconds": end.ru_stime - self_before.ru_stime,
        "scope": "Script main-entry through source snapshot/catalog/README/source_manifest writes; excludes interpreter startup/imports/final receipt serialization and all preceding manual reading/authoring.",
        "manual_engineering_parent_wall_seconds": None,
        "nested_in_external_preparation_whole_child": True,
        "source_manifest_sha256": hashlib.sha256((OUT / "source_manifest.json").read_bytes()).hexdigest(),
        "candidate_count": len(rows), "region_timer_must_not_be_added_to_external_parent": True}
    (OUT / "preparation_region_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(rows), "classifications": payload["classification_counts"], "candidate_catalog_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "small_source_bytes": source_manifest["total_original_small_source_bytes_read"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
