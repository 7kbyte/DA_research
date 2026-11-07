#!/usr/bin/env python3
"""Metadata-only progress and command references; never calls check/analyze."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PHASES = [
    ("extended_pythia_1b_100m_nll", "C1B_nll", "e828aa3b0d31c9548b48cca9a3abf597bef708b5da96ee96598a0b9683f18aa4"),
    ("extended_pythia_2p8b_25m_nll", "C2p8B_nll", "71b80a6b563a712ccceb416001022205ebfcdcc273a857c67320588c32d27412"),
    ("extended_pythia_410m_100m_behavior", "C410_behavior", "537e591bd43085b99d2bfcaed17a74ee963180913ac5b746400e4580e731e061"),
    ("extended_pythia_1b_100m_behavior", "C1B_behavior", "0d6fe37a1c376cb56be8b46f6d88b06cf4e7ef8116345ae584c7bc3788eace8c"),
    ("extended_pythia_2p8b_25m_behavior", "C2p8B_behavior", "e49b2ccc6d5963c6596c360e9fe916513d16d6e1d6836ee6ee3217372c652829"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    sources = {}

    def load(path, expected=None):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("Metadata source SHA changed: " + str(path))
        proof = {"path": str(path), "sha256": digest, "bytes": len(raw)}
        if str(path) in sources and sources[str(path)] != proof:
            raise ValueError("Source changed within this observation")
        sources[str(path)] = proof
        return json.loads(raw), {"path": str(path), "sha256": digest}

    successor = ROOT / "artifacts/provenance/final_evaluation_handoff_actual_v1/successor"
    queue, queue_proof = load(successor / "queue.json", "5468a5641cfcc097f2c84c2224df78bdf94fea92a42411b8ef077b951772b8f7")
    state, state_proof = load(successor / "state.json")
    by_id = {j["id"]: j for j in queue}
    if len(by_id) != len(queue):
        raise ValueError("Duplicate successor ID")
    rows, phases, commands = [], [], []
    cli = ROOT / "scripts/analyze_supplemental_phase_v3.py"
    guard = ROOT / "src/replay_study/supplemental_analysis_guard_v3.py"
    for path, expected in [(cli, "bd0463e34fe9373ffcb37587f6838f5f5bb1a852b6eafc023bd67923ec97ed0c"),
                           (guard, "6190ded0f8c21c035a1829f0d4f451d3842af9f5f03942dcc1fceb94bdcbc4cb")]:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Accepted v3 source changed")
        sources[str(path)] = {"path": str(path), "sha256": expected, "bytes": len(raw)}
    for contract, repair_tag, expected_plan in PHASES:
        plan, plan_proof = load(ROOT / "artifacts/analysis_prepared_v2" / (contract + ".json"), expected_plan)
        repair, repair_proof = load(ROOT / "docs" / ("supplemental_aux_analysis_operational_repair_v3_" + repair_tag + ".json"))
        if (repair["retained_plan"] != plan_proof or repair["retained_manifest"] != plan["manifest"]
                or repair["retained_protocol"] != plan["protocol"] or repair["score_access_authorized"] is not False):
            raise ValueError("Accepted repair no longer binds retained plan")
        for field in ("protocol", "manifest", "ready"):
            load(plan[field]["path"], plan[field]["sha256"])
        suffix = "_root_authority_TEMPLATE.json" if repair_tag == "C410_behavior" else "_root_authority_TEMPLATE_v3.json"
        template, template_proof = load(ROOT / "artifacts/analysis_prepared_v2" / (contract + suffix))
        if template["allow_score_analysis"] is not False or template["plan"] != plan_proof:
            raise ValueError("Expected retained FALSE template")
        local = []
        for item in plan["closure_jobs"]:
            job_id, definition = item["job_id"], item["definition"]
            actual = state["jobs"].get(job_id, {})
            digest = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
            stopped = sorted(k for k in actual if k.startswith("stop"))
            started = actual.get("status") in ("running", "completed")
            row = {"contract_id": contract, "job_id": job_id, "job_kind": item["kind"],
                "status": actual.get("status", "missing"),
                "definition_equal_successor_queue": by_id.get(job_id) == definition,
                "definition_SHA_equal_state": actual.get("definition_sha256") == digest,
                "actual_argv_matches_if_started": actual.get("argv") == definition["argv"] if started else "not_started",
                "strict_integer_rc0": actual.get("status") == "completed" and type(actual.get("returncode")) is int and actual["returncode"] == 0,
                "stop_fields": ";".join(stopped), "missing_outputs_recorded": bool(actual.get("missing_outputs")),
                "expected_definition_sha256": digest}
            local.append(row)
            rows.append(row)
        statuses = collections.Counter(r["status"] for r in local)
        bad = [r["job_id"] for r in local if not r["definition_equal_successor_queue"] or not r["definition_SHA_equal_state"]
               or r["actual_argv_matches_if_started"] is False or r["stop_fields"] or r["missing_outputs_recorded"]
               or r["status"] not in ("pending", "running", "completed")
               or (r["status"] == "completed" and not r["strict_integer_rc0"])]
        complete = not bad and all(r["strict_integer_rc0"] for r in local)
        journal = Path(plan["first_read_receipt_path"])
        analysis_out = ROOT / "results/supplemental_analysis_v3" / contract
        phases.append({"contract_id": contract, "closure_jobs": len(local), "completed_strict0": sum(r["strict_integer_rc0"] for r in local),
            "running": statuses["running"], "pending": statuses["pending"], "failed_or_other": sum(v for k, v in statuses.items() if k not in ("completed", "running", "pending")),
            "all_definitions_and_started_argv_match": all(r["definition_equal_successor_queue"] and r["definition_SHA_equal_state"] and r["actual_argv_matches_if_started"] is not False for r in local),
            "no_any_job_stop_field": all(not r["stop_fields"] for r in local), "metadata_full_closure": complete,
            "first_read_journal_exists": journal.exists(), "analysis_output_exists": analysis_out.exists(), "bad_job_ids": ";".join(bad)})
        future_op = ROOT / "artifacts/provenance/C_pending_phase_analysis_execution_v1" / contract
        future_state = future_op / "actual_successor_state_immutable.json"
        authority = future_op / "ROOT_AUTHORITY_v3.json"
        common = ["--plan", plan_proof["path"], "--plan-sha256", plan_proof["sha256"],
                  "--operational-repair", repair_proof["path"], "--operational-repair-sha256", repair_proof["sha256"],
                  "--state", str(future_state)]
        commands.append({"contract_id": contract, "prepared_only": True, "execution_authorized": False,
            "plan": plan_proof, "operational_repair": repair_proof, "FALSE_template": template_proof,
            "expected_counts": plan["counts"], "first_read_receipt_path_immutable": str(journal),
            "ROOT_future_operation_directory_not_created": str(future_op),
            "ROOT_must_copy_actual_full_successor_state_only_after_phase_full_closure": {
                "source": state_proof["path"], "destination": str(future_state), "sha256": "<ROOT computes from immutable copied original bytes>",
                "same_path_and_SHA_required_from_check_through_analyze": True},
            "check_argv_REFERENCE_ONLY": ["/usr/bin/python", str(cli), "check", *common, "--output", str(future_op / "check_readiness.json")],
            "analyze_argv_REFERENCE_ONLY": ["/usr/bin/python", str(cli), "analyze", *common, "--authority", str(authority),
                "--authority-sha256", "<SHA256_OF_ROOT_NEW_TRUE_AUTHORITY>", "--output", str(analysis_out)],
            "future_ROOT_authority_path_not_created": str(authority),
            "authority_core_fields_from_retained_FALSE_template": template,
            "ROOT_required_additions_after_real_check": {"operational_repair": repair_proof,
                "actual_successor_immutable_state": {"path": str(future_state), "sha256": "<ROOT fixed actual-state SHA>",
                    "pending_actual_completed_phase_snapshot": False, "same_exact_source_required_for_check_and_analyze": True},
                "whole_phase_check": {"path": str(future_op / "check_readiness.json"), "sha256": "<ROOT hashes successful original check output>"},
                "allow_score_analysis": "ROOT must explicitly authorize true after check; this reference does not authorize it"},
            "new_output_path": str(analysis_out), "whole_timing": "Use separate external Popen spawn-to-exit wall/user/sys plus stdout/stderr for original check and analyze; no nested additions."})

    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for filename, values in [("progress_by_phase.csv", phases), ("closure_job_metadata.csv", rows)]:
        with (out / filename).open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    command_doc = {"prepared_only": True, "score_access_authorized": False, "state_observation_is_not_an_analysis_state_binding": True, "phases": commands}
    (out / "future_commands_PREPARED_ONLY.json").write_text(json.dumps(command_doc, ensure_ascii=False, indent=2) + "\n")
    lines = ["# C五组：实际successor进度与ROOT运行清单", "", f"只读观察state.updated_at_utc：{state.get('updated_at_utc')}。这不是原check，不核评分/attestation载荷，也不是授权或冻结分析state。", "",
        "| 阶段 | closure | strict0已完成 | running | pending | 定义/已启动argv | 无stop字段 |", "|---|---:|---:|---:|---:|---|---|"]
    for p in phases:
        lines.append(f"| {p['contract_id']} | {p['closure_jobs']} | {p['completed_strict0']} | {p['running']} | {p['pending']} | {p['all_definitions_and_started_argv_match']} | {p['no_any_job_stop_field']} |")
    lines += ["", "全部closure job ID及逐项元数据在closure_job_metadata.csv；精确原CLI、plan/repair SHA、authority字段与未来输出路径在future_commands_PREPARED_ONLY.json。未改变既有计划、原protocol、cache或固定first-read路径。", "",
        "1. 某一组全部closure completed、严格整数rc0、无任何stop字段、definition/实际argv一致且无missing输出后，ROOT在该组全新的operation目录独占保存实际successor完整state原字节副本，计算并固定其SHA。其他phase可以继续运行；不得使用旧master或本次观察摘要替代原state。",
        "2. 保持该副本path/SHA不变，ROOT用对应check_argv运行原v3 check，并完整记录外部父wall/user/sys、rc、stdout/stderr。check将执行原完整source/output/attestation和CP缓存current-stat/允许fresh fallback门禁；本清单没有预先通过这些门禁。检查ready=true，scheduler_state及其SHA精确等于绑定的副本，并绑定check输出SHA。",
        "3. ROOT新建独立授权文件：从保留的FALSE模板逐字段继承plan/protocol/READY/manifest/expected_counts/checkpoint_verification，authorized_by必须小写root、allow_partial=false、冻结选择字段不变；显式加入本组operational_repair、成功check证明及actual_successor_immutable_state path/SHA。只有ROOT明确许可后才设置allow_score_analysis=true并计算新文件SHA。C410行为的旧FALSE没有repair/state字段，须在新的ROOT文件中增加，原模板不可覆盖。",
        "4. 使用相同state副本、真实新authority SHA及全新analysis output，运行对应原analyze_argv；保持5000次bootstrap、seed20261106及原端点/统计循环。原analyze会再次完整inspect，并在语义解析前独占创建固定first-read journal。不得预写、覆盖或重置journal，失败必须保留并停止，不能换输出目录绕过首次读取。",
        "5. check与analyze分别保留完整父计时；内部guard/verification计时是嵌套项，不能重复加总。只有原analyze实际rc0且本组COMPLETE及其来源闭合后，才进入独立数值复核/报告。", "",
        "本次只创建进度表和命令参考；没有读取任何评分/CP/base/归档载荷，没有运行原check/analyze、创建TRUE或首次读取journal，也未启动GPU、修改queue/core/plan。当前state的SHA仅绑定本次观察，未来必须由ROOT重新保存实际完整state原字节。"]
    (out / "ROOT_RUNBOOK.md").write_text("\n".join(lines) + "\n")
    receipt = {"status": "metadata_only_non_authorizing_runbook_prepared", "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "actual_successor_state_observation": state_proof, "state_updated_at_utc": state.get("updated_at_utc"),
        "successor_queue": queue_proof, "phases": phases, "closure_rows": len(rows),
        "source_proofs": list(sources.values()), "check_analyze_or_first_journal_executed": False,
        "score_CP_base_archive_payloads_read": False,
        "output_proofs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in sorted(out.iterdir())]}
    (out / "metadata_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"state_updated_at_utc": state.get("updated_at_utc"), "phases": phases, "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
