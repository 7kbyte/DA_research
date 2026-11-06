#!/usr/bin/env python3
"""Independent read-only recomputation of complete A validation selection."""
import csv
import datetime as dt
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from replay_study.analysis import (_check_summary,endpoint_metrics,sha256,
    training_validation_execution,verify_recorded_execution,write_csv,PAIRS)

SELECTION=ROOT/"results/supplement_A_validation/validation_lambda_selection.json"
EXPECTED_SELECTION_SHA="9548085fc18278f69fad584b67485bceb5193a18b10c78f516f6b09b4fa4e477"
OUTPUT=ROOT/"results/supplement_A_validation_review"


def read(path):return json.loads(Path(path).read_text())
def proof(path):return dict(path=str(Path(path).resolve()),sha256=sha256(path))
def check_close(a,b):
    if not math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError(f"independent numeric mismatch: {a} versus {b}")


def plot_scores(candidates,selections):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"svg.fonttype":"none","savefig.dpi":400})
    fig,axes=plt.subplots(1,2,figsize=(7,3.3))
    for ax,methods,title in zip(axes,(("B1","B3"),("B2","B4","M","C")),("CE replay","Teacher-KL replay")):
        for method in methods:
            rows=sorted([r for r in candidates if r["group"]==method],key=lambda r:r["weight"])
            line=ax.errorbar([r["weight"] for r in rows],[r["mean"] for r in rows],
                yerr=[r["std_ddof1"] for r in rows],marker="o",ms=3,lw=1,capsize=2,label=method)
            chosen=selections[method]
            ax.scatter(chosen["weight"],chosen["mean"],marker="*",s=90,color=line[0].get_color(),zorder=5)
        ax.set_xscale("log");ax.axhline(0,color=".6",lw=.6)
        ax.axvline(.05,color=".5",lw=.7,ls="--")
        ax.set(xlabel="Replay weight lambda (log scale)",ylabel="Validation score S (lower is better)",title=title)
        ax.legend(fontsize=7,frameon=False)
    fig.suptitle("Supplement A | final 10M target tokens | mean ± seed SD (n=3)",fontsize=9)
    fig.tight_layout()
    for ext in ("png","pdf","svg"):fig.savefig(OUTPUT/("all42_lambda_validation_scores_7in."+ext))
    plt.close(fig)


def main():
    if sha256(SELECTION)!=EXPECTED_SELECTION_SHA:raise ValueError("frozen A selection SHA changed")
    selected=read(SELECTION)
    assert selected["complete"] and selected["completed_conditions"]==selected["expected_conditions"]==126
    assert not selected["test_scores_accessed"] and not selected["selection_uses_test"]
    for path,digest in selected["sources_sha256"].items():
        if sha256(path)!=digest:raise ValueError(f"A source changed: {path}")
    registry_path=ROOT/"configs/supplemental_v1_A_registry.json"
    registry=read(registry_path)
    protocol_path=ROOT/"docs/formal_deterministic_protocol.json"
    protocol=read(protocol_path)
    baseline=read(protocol["initial_validation_path"])
    endpoint_path=SELECTION.parent/"validation_endpoints.csv"
    endpoints=list(csv.DictReader(endpoint_path.open()))
    by_id={r["condition_id"]:r for r in endpoints}
    assert len(endpoints)==len(by_id)==126 and set(by_id)=={r["id"] for r in registry}
    calculations=[];groups={};selected_points={};summary_proofs=[]
    for row in registry:
        actual=row["alias_of"] if row["execution_kind"]=="alias" else row["id"]
        summary=read(row["summary_path"])
        summary_proofs.append(proof(row["summary_path"]))
        _check_summary(summary,{**row,"id":actual})
        assert summary["status"]=="completed" and summary["config"]["model_revision"]==baseline["revision"]
        assert summary["counters"]["target_prediction_tokens"]==10_000_000
        assert summary["counters"]["replay_prediction_tokens"]==500_000
        assert summary["config"]["validation_paths"]==protocol["validation_paths"]
        verify_recorded_execution(baseline["execution_provenance"],training_validation_execution(summary))
        point=summary["validation_trajectory"][-1]
        assert point["target_prediction_tokens"]==10_000_000 and point["replay_prediction_tokens"]==500_000
        metrics=endpoint_metrics(point["metrics"],baseline["corpora"])
        stored=by_id[row["id"]]
        assert stored["phase"]=="supplement_A_lambda_sweep" and stored["split"]=="validation" and stored["checkpoint_kind"]=="final"
        assert stored["actual_run_id"]==actual and int(stored["seed"])==row["seed"]
        assert float(stored["weight"])==row["weight"] and float(stored["ratio"])==row["ratio"]==.05
        for metric in ("g_new","f_old_c4","f_old_wikitext","nll_target","nll_c4","nll_wikitext","kl_target","kl_c4","kl_wikitext"):
            check_close(metrics[metric],stored[metric])
        score=-metrics["g_new"]+.5*(metrics["f_old_c4"]+metrics["f_old_wikitext"])
        entry=dict(condition_id=row["id"],actual_run_id=actual,group=row["group"],seed=row["seed"],weight=row["weight"],
            score=score,execution_kind=row["execution_kind"],target_prediction_tokens=10_000_000,replay_prediction_tokens=500_000,**metrics)
        calculations.append(entry)
        groups.setdefault((row["group"],float(row["weight"])),[]).append(entry)
    candidates=[]
    expected_candidates={(r["group"],float(r["weight"])):r for r in selected["all_candidate_scores"]}
    assert len(groups)==len(expected_candidates)==42
    for (group,weight),rows in sorted(groups.items()):
        assert len(rows)==3 and {r["seed"] for r in rows}=={11,22,33}
        values=[r["score"] for r in rows];mean=statistics.fmean(values);sd=statistics.stdev(values)
        candidate=expected_candidates[group,weight]
        check_close(mean,candidate["mean"]);check_close(sd,candidate["std_ddof1"])
        assert set(candidate["condition_ids"])=={r["condition_id"] for r in rows}
        assert set(candidate["actual_run_ids"])=={r["actual_run_id"] for r in rows}
        recorded={r["seed"]:r for r in candidate["per_seed_scores"]}
        for r in rows:
            check_close(r["score"],recorded[r["seed"]]["score"])
            assert r["actual_run_id"]==recorded[r["seed"]]["actual_run_id"]
        candidates.append(dict(group=group,weight=weight,mean=mean,std_ddof1=sd,n=3))
    selection_by_group={r["group"]:r for r in selected["selections"]}
    selected_summary=[]
    for group in sorted(selection_by_group):
        best=min((r for r in candidates if r["group"]==group),key=lambda r:(r["mean"],r["weight"]))
        assert best["weight"]==selection_by_group[group]["weight"]
        check_close(best["mean"],selection_by_group[group]["mean"])
        current=groups[group,best["weight"]];common=groups[group,.05]
        row=dict(group=group,selected_lambda=best["weight"],relative_per_token_weight_lambda_over_b=best["weight"]/.05,
                 selected_mean_S=best["mean"],selected_SD_S=best["std_ddof1"],common_lambda=.05,
                 common_mean_S=statistics.fmean(r["score"] for r in common),n_training_seeds=3)
        for metric in ("g_new","f_old_c4","f_old_wikitext"):
            row[metric+"_selected_mean"]=statistics.fmean(r[metric] for r in current)
            row[metric+"_selected_SD"]=statistics.stdev(r[metric] for r in current)
        selected_summary.append(row)
        for r in current:selected_points[group,r["seed"]]=r
    policy_pairs=[]
    for a,b in PAIRS:
        for seed in (11,22,33):
            left,right=selected_points[a,seed],selected_points[b,seed]
            policy_pairs.append(dict(comparison=a+"-"+b,seed=seed,run_a=left["actual_run_id"],run_b=right["actual_run_id"],
                lambda_a=left["weight"],lambda_b=right["weight"],same_lambda=left["weight"]==right["weight"],
                contrast="validation-selected method policies; distinct from fixed-lambda source/objective contrast",
                **{m:left[m]-right[m] for m in ("g_new","f_old_c4","f_old_wikitext","score")}))
    pilot_path=Path(protocol["shared_lambda_selection_record"])
    assert sha256(pilot_path)==protocol["shared_lambda_selection_record_sha256"]
    pilot=read(pilot_path)
    assert len(pilot["runs"])==16 and protocol["seed"]==101 and protocol["target_prediction_tokens"]==2_000_000
    pilot_groups={};pilot_legacy_policy_gaps=[]
    for run in pilot["runs"]:
        summary=read(run["summary_path"])
        assert summary["config"]["seed"]==101 and run["counters"]["target_prediction_tokens"]==2_000_000
        assert run["counters"]["replay_prediction_tokens"]==100_000
        pilot_execution=training_validation_execution(summary)
        # These historical pilots predate recording the fourth SDPA switch.
        # Do not mistake the adapter's absent-field fallback for an observation.
        recorded_backends=summary["provenance"]["sdpa_backends_enabled"]
        if "cudnn" not in recorded_backends:
            known_fields=set(baseline["execution_provenance"])-{"cudnn_sdp_enabled","model_weight_dtype"}
            for field in known_fields:
                assert pilot_execution[field]==baseline["execution_provenance"][field],field
            pilot_legacy_policy_gaps.append(dict(run_id=run["id"],missing_observation="cudnn_sdp_enabled",
                status="historical provenance incomplete; not asserted as independently execution-matched"))
        else:
            verify_recorded_execution(baseline["execution_provenance"],pilot_execution)
        metrics=endpoint_metrics(run["validation"],baseline["corpora"])
        values=run["delta_nll"]
        for name in ("target","c4","wikitext"):
            check_close(values[name],run["validation"][name]["nll"]-pilot["initial_validation"][name]["nll"])
            check_close(pilot["initial_validation"][name]["nll"],baseline["corpora"][name]["nll"])
        score=-metrics["g_new"]+.5*(metrics["f_old_c4"]+metrics["f_old_wikitext"])
        check_close(score,run["score"])
        pilot_groups.setdefault(run["group"],[]).append(dict(weight=run["weight"],score=score))
    for aggregate in pilot["aggregates"]:
        check_close(aggregate["mean_score"],statistics.fmean(aggregate["groups"].values()))
        assert set(aggregate["groups"])=={"B1","B2","B3","B4"}
    assert min(pilot["aggregates"],key=lambda r:(r["mean_score"],r["weight"]))["weight"]==.05
    result=dict(status="passed",audited_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        full_conditions=126,candidates=42,new_training=108,main_aliases=18,source_files_sha_verified=len(selected["sources_sha256"]),
        selected_lambda_by_method={r["group"]:r["selected_lambda"] for r in selected_summary},
        primary_shared=dict(seed=101,target_prediction_tokens=2_000_000,replay_prediction_tokens=100_000,lambda_grid=protocol["lambda_grid"],
            aggregate="unweighted mean of four B1-B4 scores at seed101",selected_lambda=.05,selected_mean_S=pilot["selected_mean_score"],
            per_group_pilot_minimizers={g:min(rows,key=lambda r:(r["score"],r["weight"])) for g,rows in pilot_groups.items()},
            legacy_execution_metadata_gaps=pilot_legacy_policy_gaps,
            initial_backend_amendment=protocol["initial_execution_amendment"]),
        supplement_A=dict(seeds=[11,22,33],target_prediction_tokens=10_000_000,replay_prediction_tokens=500_000,
            aggregate="per-method unweighted mean of three seed scores",endpoint="final only; no best or heldout selection"),
        score="-G_target + .5*(F_C4+F_Wiki); lower is better",selected_strategy_pairs_not_fixed_lambda=True,
        source_selection=proof(SELECTION),primary_protocol=proof(protocol_path),primary_pilot=proof(pilot_path),
        source_endpoint_csv=proof(endpoint_path),summary_proofs=summary_proofs,
        producer_source=proof(__file__),
        test_or_behavior_scores_read=False,scientific_parameters_modified=False,supplement_C_frozen_lambda_unchanged=.05)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write_csv(OUTPUT/"all126_recomputed_validation.csv",calculations)
    write_csv(OUTPUT/"all42_recomputed_scores.csv",candidates)
    write_csv(OUTPUT/"selected_vs_common_lambda_validation.csv",selected_summary)
    write_csv(OUTPUT/"selected_policy_paired_validation.csv",policy_pairs)
    plot_scores(candidates,selection_by_group)
    result["outputs"]=[proof(p) for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name!="independent_audit.json"]
    (OUTPUT/"independent_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"summary_proofs","outputs"}},ensure_ascii=False))


if __name__=="__main__":main()
