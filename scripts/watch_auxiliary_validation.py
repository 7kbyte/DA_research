#!/usr/bin/env python3
"""CPU-only complete-cohort B/C validation exporter; no heldout score reader."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from replay_study.analysis import _check_summary, read_json, sha256
from replay_study.aux_validation import (cohort_gate, collect_extended_validation, proof,
    validate_bank_activation, verify_proof, write_extended_analysis, write_json)

PHASE_B = "B_conditional_bank_variance"
DEFAULT_PROTOCOL = ROOT / "docs/auxiliary_validation_analysis_protocol_v2.json"


def prepare(path):
    if path.exists():
        raise FileExistsError("never replace a registered analysis protocol")
    freeze_path = ROOT / "results/extended_formal_freeze.json"
    freeze = read_json(freeze_path)
    if freeze.get("status") != "passed" or freeze.get("scientific_core_changed") is not False:
        raise ValueError("explicit frozen extended authority required")
    index_path = ROOT / "configs/supplemental_v1_registry_index.json"
    index = read_json(index_path)
    master = read_json(index["master_queue"])
    master_by_id = {j["id"]:j for j in master}
    source = index["phases"][PHASE_B]
    original, runtime, sources = validate_bank_activation(source["analysis_registry"], source["execution_registry"], source["activation_path"])
    if len(original) != 60:
        raise ValueError("B must retain all 60 registered conditional draws")
    for row in runtime:
        argv=master_by_id[row["id"]]["argv"]
        for flag,wanted in (("--config",row["config_path"]),("--summary-copy",row["summary_path"])):
            if argv.count(flag)!=1 or Path(argv[argv.index(flag)+1]).resolve()!=Path(wanted).resolve():
                raise ValueError("actual master launch does not use the activated config/summary identity")
    b = dict(phase=PHASE_B, role="supplement_B_conditional_bank_variance", expected_runs=60,
             source_registry=proof(source["analysis_registry"]), runtime_registry=proof(source["execution_registry"]),
             activation=proof(source["activation_path"]), source_binding=sources,
             frozen_job_definitions=[master_by_id[r["id"]] for r in runtime],
             baseline=proof(ROOT / "results/pilots/lr_deterministic_v3/initial_cudnn_matched.json"),
             output=str(ROOT / "results/analysis/supplement_B_validation_complete_v1"))
    phases = []
    for phase in freeze["phases"]:
        for key in ("active_registry", "snapshot", "protocol", "queue"):
            verify_proof(dict(path=phase[key], sha256=phase[key + "_sha256"]))
        protocol = read_json(phase["protocol"])
        rows = read_json(phase["active_registry"])
        if len(rows) != phase["runs"] or phase["runs"] not in {24,15}:
            raise ValueError("frozen C phase run count mismatch")
        baseline = dict(path=protocol["initial_validation_path"], sha256=protocol["initial_validation_sha256"])
        verify_proof(baseline)
        has_d = any(r["group"] == "D" for r in rows)
        phases.append(dict(phase=phase["phase"], model=phase["model"], expected_runs=phase["runs"],
            target_prediction_tokens=phase["target_prediction_tokens"], role="supplement_C_extended",
            registry=proof(phase["active_registry"]), snapshot=proof(phase["snapshot"]),
            protocol=proof(phase["protocol"]), queue=proof(phase["queue"]), baseline=baseline,
            frozen_job_definitions=read_json(phase["queue"]),
            completion=str(ROOT / "results/extended_phase_equivalence" / phase["phase"] / "completion.json") if has_d else None,
            audit_mode="independent_training_phase_completion" if has_d else "complete_15_run_source_state_audit",
            output=str(ROOT / "results/analysis" / (phase["phase"] + "_validation_v1"))))
    code = [ROOT / "src/replay_study" / name for name in ("analysis.py", "bank_variance.py", "aux_validation.py", "execution.py")]
    code += [Path(__file__), ROOT / "scripts/audit_extended_runs.py"]
    state=read_json(ROOT/"logs/supplemental_job_state.json")
    cohort_gate(runtime,state,b["frozen_job_definitions"],master)
    for phase in phases:
        cohort_gate(read_json(phase["registry"]["path"]),state,phase["frozen_job_definitions"],master)
    result = dict(schema_version=1, registered_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status="prepared_validation_only", original_freeze=proof(freeze_path),
        B=b, C=phases, code=[proof(p) for p in code],
        state=str(ROOT / "logs/supplemental_job_state.json"), master_queue=str(Path(index["master_queue"]).resolve()),
        cohort_gate="all registered physical jobs completed; exact frozen master/definition SHA; rc0 and all outputs exist; no partial scientific summary",
        B_replicate_unit="selection_seed; generation pool11 and training shuffle11 fixed; not 10 training seeds",
        C_replicate_unit="training_seed; three seeds within each model-specific phase",
        C_scale_comparison="within-model only; no causal scale comparison across model/data/horizon",
        checkpoint_rule="final fixed frozen horizon; best first target-validation minimum, actual exposure explicit",
        costs="full-training parent costs once per actual run; teacher timers nested, never added twice",
        uncertainty="sample SD ddof1 across declared replicate unit, not SEM or document CI",
        heldout_scores_access_permitted=False, behavior_scores_access_permitted=False, gpu_work=False)
    write_json(path, result)
    return result


def verify_plan(plan):
    if plan.get("heldout_scores_access_permitted") is not False or plan.get("behavior_scores_access_permitted") is not False:
        raise ValueError("validation-only access boundary must remain explicit")
    for item in [plan["original_freeze"], *plan["code"], *plan["B"]["source_binding"], plan["B"]["baseline"]]:
        verify_proof(item)
    for phase in plan["C"]:
        for key in ("registry", "snapshot", "protocol", "queue", "baseline"):
            verify_proof(phase[key])


def validate_completed_output(path, protocol_sha):
    manifest = read_json(path)
    if manifest.get("status") != "complete" or manifest.get("analysis_protocol_sha256") != protocol_sha:
        raise ValueError("previous output does not belong to this complete frozen analysis")
    for item in manifest["outputs"]:
        verify_proof(item)
    return manifest


def validate_completion(phase, rows):
    path = Path(phase["completion"])
    completion = read_json(path)
    if (completion.get("status") != "passed" or completion.get("phase") != phase["phase"] or
            completion.get("model") != phase["model"] or completion.get("completed_runs") != len(rows) or
            completion.get("expected_runs") != len(rows)):
        raise ValueError("independent phase completion identity mismatch")
    for key, wanted in (("source_registry", phase["registry"]), ("input_snapshot", phase["snapshot"])):
        if completion[key]["sha256"] != wanted["sha256"] or Path(completion[key]["path"]).resolve() != Path(wanted["path"]).resolve():
            raise ValueError("phase audit is not bound to the frozen registry/snapshot")
    records = [completion["source_registry"], completion["input_snapshot"], completion["audit"],
               *completion["proofs"].values(), *completion["source_inputs"]]
    for record in records:
        verify_proof(record)
    audit = read_json(completion["audit"]["path"])
    if audit.get("status") != "passed" or audit.get("source_violations") or audit.get("completed_runs") != len(rows):
        raise ValueError("phase source/state audit has not passed")
    expected = {r["id"]: sha256(r["summary_path"]) for r in rows}
    got = {r["id"]: r["summary_sha256"] for r in audit["phase_audit"]["runs"]}
    if expected != got:
        raise ValueError("completed summaries differ from independently audited summaries")
    return [proof(path), *records], {"status": audit["numerical_null_control_status"],
        "diagnostics": audit["numerical_null_control_diagnostics"],
        "all_pairs_bitwise_equal": completion["all_pairs_bitwise_equal"],
        "completion": proof(path), "interpretation": "identical-bank numerical control; never deduplication effect"}


def run_extended(phase, state, protocol_sha, plots=True):
    rows = read_json(phase["registry"]["path"])
    gate = cohort_gate(rows, state, phase["frozen_job_definitions"], read_json(ROOT/"configs/supplemental_v1_queue.json"))
    if not gate["ready"]:
        raise ValueError("complete cohort required before reading validation summaries")
    sources = [phase[key] for key in ("registry", "snapshot", "protocol", "queue", "baseline")]
    null = None
    if phase["completion"]:
        more, null = validate_completion(phase, rows)
        sources.extend(more)
    else:
        from audit_extended_runs import audit_phase
        audit = audit_phase(phase["snapshot"]["path"], state, allow_incomplete=False, compare_weights=False)
        if audit["status"] != "passed" or audit["completed_runs"] != len(rows):
            raise ValueError(f"complete 15-run extended audit failed: {audit['violations']}")
        audit_path = Path(phase["output"]) / "source_state_audit.json"
        write_json(audit_path, audit)
        sources.append(proof(audit_path))
    snapshot = read_json(phase["snapshot"]["path"])
    summaries = {}
    for row in rows:
        cfgpath = Path(row["config_path"])
        expected = snapshot["configs"][str(cfgpath.resolve().relative_to(ROOT.resolve()))]["sha256"]
        if sha256(cfgpath) != expected or row["config_sha256"] != expected:
            raise ValueError("extended frozen config SHA mismatch")
        summaries[row["id"]] = read_json(row["summary_path"])
        sources.extend([proof(cfgpath), proof(row["summary_path"])])
    protocol = read_json(phase["protocol"]["path"])
    for name, path in protocol["validation_paths"].items():
        verify_proof(dict(path=path, sha256=protocol["validation_sha256"][name]))
        sources.append(proof(path))
    tables, execution = collect_extended_validation(rows, protocol, read_json(phase["baseline"]["path"]), summaries,
                                                   phase=phase["phase"], model=phase["model"])
    result = write_extended_analysis(phase["output"], tables, execution, phase=phase["phase"], model=phase["model"],
        target=phase["target_prediction_tokens"], sources=sources, numerical_null=null, plots=plots)
    result["analysis_protocol_sha256"] = protocol_sha
    write_json(Path(phase["output"]) / "analysis_manifest.json", result)
    return result


def plot_bank_print(output):
    """Seven-inch print variants, leaving the existing analyzer unchanged."""
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "pdf.fonttype":42, "svg.fonttype":"none", "savefig.dpi":400})
    output = Path(output)
    rows = list(csv.DictReader((output / "validation_endpoints.csv").open()))
    colors = {"B4":"#0072B2", "M":"#D55E00", "C":"#CC79A7"}
    fig, axes = plt.subplots(1,2,figsize=(7,3.25))
    for ax, metric, label in zip(axes,("f_old_c4","f_old_wikitext"),("C4","WikiText-103")):
        for group in colors:
            for size, marker in ((20,"o"),(196,"s")):
                r = [r for r in rows if r["group"] == group and int(r["bank_chunks"]) == size]
                x, y = np.array([float(a["g_new"]) for a in r]), np.array([float(a[metric]) for a in r])
                ax.scatter(x,y,s=12,alpha=.3,color=colors[group],marker=marker)
                ax.errorbar(x.mean(),y.mean(),xerr=x.std(ddof=1),yerr=y.std(ddof=1),color=colors[group],marker=marker,
                            ms=4,capsize=2,linestyle="none",label=f"{group}, K={size}")
        ax.set(xlabel="Target validation gain (nats/token)",ylabel=f"{label} validation forgetting")
    axes[0].legend(fontsize=6,ncol=2)
    fig.suptitle("Selection draws (n=10); fixed pool and training seed11; mean ± draw SD",fontsize=9)
    fig.tight_layout()
    for ext in ("png","pdf","svg"):
        fig.savefig(output / ("conditional_validation_frontier_7in."+ext))
    plt.close(fig)


def bank_discussion(output):
    import csv
    output = Path(output)
    stats = list(csv.DictReader((output / "endpoint_statistics.csv").open()))
    paired = list(csv.DictReader((output / "paired_validation_statistics.csv").open()))
    lines = ["# 固定生成池与训练顺序下的选库条件变异", "",
        "60 项全部完成后才作此验证分析；重复单位是 selection_seed（1001–1010），生成池与训练 shuffle 都固定为11。它不是十个独立训练 seed，也不估计跨生成池的不确定性。",
        "各方法、K 内报告10个 draw 的均值和样本 SD（ddof=1）；同 draw 标签的 M−B4/C−B4 先配对相减，不声称不同算法的随机数严格对应。固定10M目标曝光、b=.05、共享冻结 LR/λ。只读验证，不读补充测试或行为。", "",
        "|方法|K|指标|均值±draw SD|n|", "|---|---:|---|---:|---:|"]
    for r in stats:
        if r["metric"] in {"g_new","f_old_c4","f_old_wikitext"}:
            lines.append(f"|{r['group']}|{r['bank_chunks']}|{r['metric']}|{float(r['mean']):+.6f} ± {float(r['std_ddof1']):.6f}|{r['n']}|")
    lines += ["", "同 draw 配对统计：", "", "|对照|K|指标|均值±draw SD|", "|---|---:|---|---:|"]
    for r in paired:
        if r["metric"] in {"g_new","f_old_c4","f_old_wikitext"}:
            lines.append(f"|{r['group']}|{r['bank_chunks']}|{r['metric']}|{float(r['mean']):+.6f} ± {float(r['std_ddof1']):.6f}|")
    lines += ["", "组成 NLL 与结果的波动不直接证明梯度约束、保留机制或普遍优势。teacher 子计时和聚类子计时属于嵌套项，不重复加至父成本；共享 generation/feature 准备费也不乘以60。"]
    (output / "条件选库验证讨论.md").write_text("\n".join(lines)+"\n")


def run_bank(spec, state, protocol_sha, plots=True):
    from replay_study.bank_variance import analyze_bank_variance
    source, runtime, sources = validate_bank_activation(spec["source_registry"]["path"], spec["runtime_registry"]["path"], spec["activation"]["path"])
    if len(runtime) != spec["expected_runs"] or not cohort_gate(runtime, state, spec["frozen_job_definitions"], read_json(ROOT/"configs/supplemental_v1_queue.json"))["ready"]:
        raise ValueError("all 60 conditional draw runs must complete first")
    for row in runtime:
        summary = read_json(row["summary_path"])
        _check_summary(summary, {**row, "seed":row["training_seed"]})
        if summary.get("status") != "completed":
            raise ValueError("scheduler and saved summary completion disagree")
        sources.append(proof(row["summary_path"]))
    result = analyze_bank_variance(spec["source_registry"]["path"], spec["baseline"]["path"], spec["output"], allow_partial=False, plots=False)
    if not result["complete_trained_results"] or result["completed_draw_runs"] != 60:
        raise ValueError("full bank analyzer did not obtain the complete registered matrix")
    bank_discussion(spec["output"])
    if plots:
        plot_bank_print(spec["output"])
    result.update(status="complete", analysis_protocol_sha256=protocol_sha, runtime_registry=spec["runtime_registry"],
                  activation=spec["activation"], sources=[*sources,spec["baseline"]],
                  completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), gpu_work_started=False)
    result["outputs"] = [proof(p) for p in sorted(Path(spec["output"]).iterdir()) if p.is_file() and p.name != "analysis_manifest.json"]
    write_json(Path(spec["output"])/"analysis_manifest.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--protocol",type=Path,default=DEFAULT_PROTOCOL)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--watch",action="store_true")
    parser.add_argument("--poll-seconds",type=float,default=45)
    parser.add_argument("--progress",type=Path,default=ROOT/"logs/auxiliary_validation_analysis_watch.json")
    parser.add_argument("--no-plots",action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.prepare:
        result=prepare(args.protocol)
        print(json.dumps(dict(status=result["status"],protocol=proof(args.protocol))))
        return
    if not 1 <= args.poll_seconds <= 60:
        parser.error("poll-seconds must be between 1 and 60")
    plan = read_json(args.protocol)
    verify_plan(plan)
    protocol_sha=sha256(args.protocol)
    specs=[plan["B"],*plan["C"]]
    outcomes={}
    while True:
        state=read_json(plan["state"])
        master=read_json(plan["master_queue"])
        for spec in specs:
            name=spec["phase"]
            if outcomes.get(name,{}).get("status") in {"complete","failed_requires_review"}:
                continue
            path=Path(spec["output"])/"analysis_manifest.json"
            try:
                if path.exists():
                    result=validate_completed_output(path,protocol_sha)
                    outcomes[name]=dict(status="complete",manifest=proof(path))
                    continue
                registry=spec["runtime_registry"] if name==PHASE_B else spec["registry"]
                rows=read_json(registry["path"])
                gate=cohort_gate(rows,state,spec["frozen_job_definitions"],master)
                outcomes[name]=dict(status="waiting_complete_cohort",gate=gate)
                if gate["failed_jobs"]:
                    raise ValueError(f"cohort scheduler failures: {gate['failed_jobs']}")
                if not gate["ready"]:
                    continue
                if spec.get("completion") and not Path(spec["completion"]).exists():
                    outcomes[name]["status"]="waiting_independent_phase_audit"
                    continue
                print(json.dumps(dict(at=dt.datetime.now(dt.timezone.utc).isoformat(),phase=name,event="complete_gate_passed_start_validation_analysis")),flush=True)
                verify_plan(plan)
                if name==PHASE_B:
                    run_bank(spec,state,protocol_sha,plots=not args.no_plots)
                else:
                    run_extended(spec,state,protocol_sha,plots=not args.no_plots)
                outcomes[name]=dict(status="complete",manifest=proof(path))
                print(json.dumps(dict(phase=name,event="complete",manifest=proof(path))),flush=True)
            except Exception as error:
                outcomes[name]=dict(status="failed_requires_review",error=repr(error))
                print(json.dumps(dict(phase=name,event="failed_requires_review",error=repr(error))),flush=True)
        record=dict(updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),pid=os.getpid(),protocol=proof(args.protocol),
                    phases=outcomes,test_or_behavior_scores_read=False,gpu_work_started=False)
        write_json(args.progress,record)
        terminal=all(v.get("status") in {"complete","failed_requires_review"} for v in outcomes.values())
        if not args.watch or terminal:
            if any(v.get("status")=="failed_requires_review" for v in outcomes.values()):
                raise SystemExit(1)
            return
        time.sleep(args.poll_seconds)


if __name__=="__main__":
    main()
