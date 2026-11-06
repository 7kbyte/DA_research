#!/usr/bin/env python3
"""Readable primary-only endpoint tables and conditional document intervals."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path("results/analysis/primary_nll_v2")


def readcsv(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def writecsv(path,rows):
    with path.open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def interval_row(value):
    return dict(estimate=value["estimate"],ci_low=value["ci_low"],ci_high=value["ci_high"],
                interval_includes_zero=value["ci_low"]<=0<=value["ci_high"])


def main():
    scope=json.loads((ROOT/"scope_and_costs.json").read_text())
    if scope.get("status")!="complete_primary_only" or scope.get("physical_jobs")!=67 or scope.get("bootstrap_replicates")!=5000:
        raise ValueError("complete primary-only, already-unsealed analysis required")
    sources={};tables={};allrows={}
    for endpoint in ("final","best"):
        folder=ROOT/endpoint
        for name in ("analysis_manifest.json","endpoints.csv","endpoint_statistics.csv","paired_effects.csv","paired_statistics.csv",
                     "document_bootstrap.json","endpoint_document_bootstrap.json"):
            p=folder/name;sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        rows=readcsv(folder/"endpoints.csv");lookup={r["run_id"]:r for r in rows};allrows[endpoint]=lookup
        endpoint_intervals=[]
        for result in json.loads((folder/"endpoint_document_bootstrap.json").read_text()):
            source=lookup[result["run_id"]]
            for weighting in ("token_weighted","equal_document"):
                endpoint_intervals.append(dict(run_id=source["run_id"],actual_run_id=source["actual_run_id"],group=source["group"],training_seed=source["training_seed"],
                    ratio=source["ratio"],weight=source["weight"],endpoint=endpoint,target_prediction_tokens=source["target_prediction_tokens"],
                    replay_prediction_tokens=source["replay_prediction_tokens"],full_training_target_prediction_tokens=source["full_training_target_prediction_tokens"],
                    full_training_replay_prediction_tokens=source["full_training_replay_prediction_tokens"],
                    metric=result["metric"],weighting=weighting,**interval_row(result[weighting]),documents=result["documents"],
                    prediction_tokens=result["prediction_tokens"],bootstrap_replicates=result["bootstrap_replicates"],
                    scope=result["scope"],confidence=result["confidence"]))
        writecsv(folder/"endpoint_bootstrap_intervals_readable.csv",endpoint_intervals)
        rawpairs=readcsv(folder/"paired_effects.csv");paired=[]
        for result in json.loads((folder/"document_bootstrap.json").read_text()):
            # Bootstrap JSON's raw target NLL A-B is reversed to the preregistered G_new difference here.
            metric=("g_new" if result["corpus"]=="target" else "f_old_"+result["corpus"]) if result["field"]=="sum_nll" else "kl_"+result["corpus"]
            matches=[r for r in rawpairs if r["comparison"]==result["comparison"] and int(r["training_seed"])==result["training_seed"] and
                     float(r["ratio"])==result["ratio"] and float(r["weight"])==result["weight"] and
                     r["run_a"]==result["run_a"] and r["run_b"]==result["run_b"]]
            if len(matches)!=1:raise ValueError("bootstrap scientific identity is missing, duplicated or foreign")
            setting=matches[0]
            expected=(-1 if metric=="g_new" else 1)*result["token_weighted"]["estimate"]
            if abs(float(setting[metric])-expected)>1e-10:raise ValueError("bootstrap point estimate differs from the same identified endpoint")
            for weighting in ("token_weighted","equal_document"):
                value=dict(result[weighting])
                if metric=="g_new":value=dict(estimate=-value["estimate"],ci_low=-value["ci_high"],ci_high=-value["ci_low"])
                paired.append(dict(comparison=result["comparison"],training_seed=result["training_seed"],ratio=setting["ratio"],
                    weight=setting["weight"],endpoint=endpoint,metric=metric,weighting=weighting,**interval_row(value),
                    run_a=result["run_a"],run_b=result["run_b"],actual_run_id_a=result["actual_run_id_a"],actual_run_id_b=result["actual_run_id_b"],
                    equal_checkpoint_exposure=setting["equal_checkpoint_exposure"],target_tokens_a=setting["target_tokens_a"],
                    target_tokens_b=setting["target_tokens_b"],replay_tokens_a=result["replay_tokens_a"],replay_tokens_b=result["replay_tokens_b"],
                    documents=result["documents"],bootstrap_replicates=result["bootstrap_replicates"],
                    sign="G_new A-B, or F_old/KL A-B",scope=result["scope"]))
        writecsv(folder/"paired_bootstrap_intervals_readable.csv",paired)
        stats=readcsv(folder/"endpoint_statistics.csv");pstats=readcsv(folder/"paired_statistics.csv")
        lines=[f"## {endpoint}","", "以下为均值±样本SD(ddof=1,n=3)，不是bootstrap CI。G_new越大越好，F_old越小越好。",""]
        for weighting,suffix in (("预测token加权",""),("等文档平均","_equal_document")):
            lines.extend([f"### {weighting}","","| 方法 | G_new | F_old C4 | F_old Wiki |","|---|---:|---:|---:|"])
            def fmt(item):return f"{float(item['mean']):+.5f}±{float(item['std_ddof1']):.5f}"
            for group in ("B0","B1","B2","B3","B4","D","M","C"):
                items=[next(r for r in stats if r["group"]==group and (group=="B0" or float(r["ratio"])==.05) and r["metric"]==metric+suffix)
                       for metric in ("g_new","f_old_c4","f_old_wikitext")]
                lines.append("| "+group+" | "+" | ".join(fmt(item) for item in items)+" |")
            lines.extend(["","| 同seed差 A−B | ΔG_new | ΔF_old C4 | ΔF_old Wiki |","|---|---:|---:|---:|"])
            for comparison in ("M-B4","C-B4","B2-B1","B4-B3","B3-B1","B4-B2"):
                items=[next(r for r in pstats if r["comparison"]==comparison and float(r["ratio"])==.05 and r["metric"]==metric+suffix)
                       for metric in ("g_new","f_old_c4","f_old_wikitext")]
                lines.append("| "+comparison+" | "+" | ".join(fmt(item) for item in items)+" |")
            lines.append("")
        (folder/"主结果与六组配对表.md").write_text("\n".join(lines)+"\n")
    descriptive=[]
    for run,final in allrows["final"].items():
        best=allrows["best"][run]
        descriptive.append(dict(run_id=run,group=final["group"],training_seed=final["training_seed"],ratio=final["ratio"],
            final_target_prediction_tokens=final["target_prediction_tokens"],best_target_prediction_tokens=best["target_prediction_tokens"],
            **{metric+"_best_minus_final":float(best[metric])-float(final[metric]) for metric in ("g_new","f_old_c4","f_old_wikitext")},
            independent_replicates=False,extra_training_cost_for_best_endpoint=0))
    writecsv(ROOT/"best_vs_final_descriptive.csv",descriptive)
    (ROOT/"readable_summary_provenance.json").write_text(json.dumps(dict(
        scope="Complete primary NLL only; behavior/other phase test scores remain sealed",source_sha256=sources,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        best_vs_final_descriptive_counts=dict(g_new_higher=sum(r["g_new_best_minus_final"]>0 for r in descriptive),
          c4_loss_lower=sum(r["f_old_c4_best_minus_final"]<0 for r in descriptive),wiki_loss_lower=sum(r["f_old_wikitext_best_minus_final"]<0 for r in descriptive)),
        counts_are_correlated_descriptive_conditions_not_new_training_replicates=True),ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":main()
