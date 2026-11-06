#!/usr/bin/env python3
"""B review v2: identical arithmetic, row-shared K scales and explicit draw labels."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("g_new", "f_old_c4", "f_old_wikitext")
SIX = METRICS + tuple(m + "_equal_document" for m in METRICS)
LABELS = ("G_new (higher is better)", "F_C4 (lower is better)", "F_Wiki (lower is better)")
COLORS = {"B4": "#4477AA", "M": "#EE7733", "C": "#228833"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, values):
    with Path(path).open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader(); w.writerows(values)


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")


def summary(values):
    return {"mean": statistics.mean(values), "std_ddof1": statistics.stdev(values), "n": len(values)}


def direction(metric, value):
    return (1 if value > 0 else -1 if value < 0 else 0) * (1 if metric.startswith("g_new") else -1)


def document_means(corpus):
    totals = defaultdict(lambda: [0.0, 0])
    for chunk in corpus["chunks"]:
        d = totals[str(chunk["doc_id"])]; d[0] += chunk["sum_nll"]; d[1] += chunk["valid_prediction_tokens"]
    return {d: (s / n, n) for d, (s, n) in totals.items()}


def raw_metrics(current, initial):
    result = {}
    for corpus, metric in zip(("target", "c4", "wikitext"), METRICS):
        a, b = current[corpus], initial[corpus]
        assert a["valid_prediction_tokens"] == b["valid_prediction_tokens"]
        da, db = document_means(a), document_means(b)
        assert {d: n for d, (_, n) in da.items()} == {d: n for d, (_, n) in db.items()}
        identities = lambda c: [(x["chunk_index"], str(x["doc_id"]), x["valid_prediction_tokens"]) for x in c["chunks"]]
        assert identities(a) == identities(b)
        sign = -1 if corpus == "target" else 1
        result[metric] = sign * (a["sum_nll"] / a["valid_prediction_tokens"] - b["sum_nll"] / b["valid_prediction_tokens"])
        result[metric + "_equal_document"] = sign * statistics.mean(da[d][0] - db[d][0] for d in da)
    return result


def bootstrap_metric(row, metric, weighting, reverse_target):
    part = dict(row[weighting])
    if reverse_target and metric.startswith("g_new"):
        part = {"estimate": -part["estimate"], "ci_low": -part["ci_high"], "ci_high": -part["ci_low"]}
    assert all(math.isfinite(v) for v in part.values()) and part["ci_low"] <= part["ci_high"]
    return part


def validate_inventory(endpoints):
    expected = {(k, g, d) for k in (20, 196) for g in COLORS for d in range(1001, 1011)}
    actual = [(int(r["bank_chunks"]), r["group"], int(r["selection_seed"])) for r in endpoints]
    if len(actual) != 60 or len(set(actual)) != 60 or set(actual) != expected:
        raise ValueError("Complete unique 3 methods x 2 K x 10 draws required")
    for r in endpoints:
        if (r["training_seed"] != "11" or r["generation_seed"] != "11" or r["checkpoint"] != "final"
                or int(r["target_prediction_tokens"]) != 10_000_000 or int(r["replay_prediction_tokens"]) != 500_000
                or float(r["weight"]) != .05 or float(r["ratio"]) != .05):
            raise ValueError("B frozen seed/exposure/condition changed")


def plot(endpoints, pairs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    for paired in (False, True):
        fig, axes = plt.subplots(3, 2, figsize=(7, 6.5), sharex=True, sharey="row")
        for col, k in enumerate((20, 196)):
            labels = ("M-B4", "C-B4") if paired else tuple(COLORS)
            for mi, metric in enumerate(METRICS):
                ax = axes[mi, col]
                for x, label in enumerate(labels):
                    subset = sorted([r for r in (pairs if paired else endpoints) if int(r["bank_chunks"]) == k and r["comparison" if paired else "group"] == label], key=lambda r: int(r["selection_seed"]))
                    ys = [float(r[metric]) for r in subset]
                    color = COLORS[label[0] if paired else label]
                    xs = [x + (i - 4.5) * .025 for i in range(10)]
                    ax.scatter(xs, ys, s=22, alpha=.78, color=color, edgecolors="white", linewidths=.3, zorder=3)
                    st = summary(ys)
                    ax.errorbar(x + .20, st["mean"], yerr=st["std_ddof1"], fmt="D", color=color, capsize=3, markersize=4, zorder=4)
                    if paired:
                        for i, (xp, yp) in enumerate(zip(xs, ys)):
                            ax.annotate(str(1001+i)[-2:], (xp, yp), xytext=(2, 1), textcoords="offset points", fontsize=5.3, color=color)
                ax.axhline(0, color=".5", ls="--", lw=.7)
                ax.grid(axis="y", alpha=.2)
                ax.set_xticks(range(len(labels)), labels)
                if col == 0: ax.set_ylabel(LABELS[mi] + "\n(nats/token)")
                if mi == 0: ax.set_title(f"K = {k} chunks")
                if mi == 2: ax.set_xlabel("Method difference (left minus B4)" if paired else "Method")
        fig.suptitle("B test: conditional paired draw differences" if paired else "B test: all 60 conditional bank draws", y=.993, fontsize=11)
        caption = ("Labels 01-10 = selection seeds 1001-1010; same labels do not establish strict CRN.\n"
                   "Diamonds/bars: mean +/- sample SD (n=10), conditional on pool=11 and train=11.\n"
                   "Final 10M target + 0.5M replay; b=lambda=0.05; each metric row shares its K scales.") if paired else (
                   "Dots: selection draws 1001-1010; diamonds/bars: mean +/- sample SD (n=10).\n"
                   "Pool=11, train=11; final 10M target + 0.5M replay; b=lambda=0.05.")
        fig.text(.5, .014, caption, ha="center", fontsize=7.6)
        fig.tight_layout(rect=(0, .10 if paired else .075, 1, .97))
        name = "paired_draws_token_7in" if paired else "six_conditions_token_7in"
        for ext in ("png", "pdf", "svg"): fig.savefig(out/(name+"."+ext), dpi=220)
        plt.close(fig)


def run(complete_path, expected_complete, output):
    start, cpu = time.perf_counter(), time.process_time()
    complete_path, output = Path(complete_path), Path(output)
    if sha(complete_path) != expected_complete: raise ValueError("COMPLETE SHA differs from root authority")
    complete = read(complete_path)
    assert complete["complete"] and complete["endpoints"] == [{"endpoint": "final", "logical_conditions": 60}]
    for key in ("authority", "plan", "readiness", "first_read_receipt"):
        p = complete[key]; assert sha(p["path"]) == p["sha256"]
    authority, plan, readiness = (read(complete[k]["path"]) for k in ("authority", "plan", "readiness"))
    assert authority["allow_score_analysis"] is True and authority["plan"] == complete["plan"]
    assert readiness["ready"] and readiness["closure_job_count"] == 61 and not readiness["pending_job_ids"]
    assert plan["contract_id"] == "supplement_B_nll"
    manifest = read(plan["manifest"]["path"]); assert sha(plan["manifest"]["path"]) == plan["manifest"]["sha256"]
    base = complete_path.parent/"final"; analysis = read(base/"analysis_manifest.json")
    assert analysis["complete"] and analysis["completed_conditions"] == analysis["expected_conditions"] == 60
    assert analysis["replicate_unit"] == "selection_seed" and analysis["bootstrap_replicates"] == 5000
    verified_sources, source_bytes = {}, 0
    for path, expected in analysis["sources_sha256"].items():
        if "/checkpoints/" in path: raise ValueError("Review must not hash large training checkpoints")
        assert sha(path) == expected
        verified_sources[path] = expected; source_bytes += Path(path).stat().st_size
    endpoints, pairs = rows(base/"endpoints.csv"), rows(base/"paired_effects.csv")
    validate_inventory(endpoints)
    assert len(pairs) == 40
    by_id, records = {r["run_id"]: r for r in endpoints}, {r["run_id"]: r for r in manifest["records"]}
    initial = read(next(iter(manifest["baseline_groups"].values()))["initial_output"])["corpora"]
    errors, check_count = [], 0
    def same(a, b):
        nonlocal check_count
        errors.append(abs(float(a)-float(b))); check_count += 1
        if errors[-1] > 1e-10: raise ValueError(f"Numerical inconsistency: {a} != {b}")
    for r in endpoints:
        value = read(records[r["run_id"]]["output"])["corpora"]
        for m, number in raw_metrics(value, initial).items(): same(number, r[m])
    endpoint_stats, paired_stats = [], []
    for k in (20, 196):
        for g in COLORS:
            for m in SIX:
                subset = [r for r in endpoints if int(r["bank_chunks"]) == k and r["group"] == g]
                endpoint_stats.append({"bank_chunks": k, "group": g, "metric": m, **summary([float(r[m]) for r in subset])})
    directions = []
    for r in pairs:
        a, b = by_id[r["run_a"]], by_id[r["run_b"]]
        assert a["selection_seed"] == b["selection_seed"] == r["selection_seed"] and a["bank_chunks"] == b["bank_chunks"] == r["bank_chunks"]
        for m in SIX:
            same(float(a[m])-float(b[m]), r[m])
            directions.append({"bank_chunks": int(r["bank_chunks"]), "comparison": r["comparison"], "selection_seed": int(r["selection_seed"]), "metric": m,
                               "difference": float(r[m]), "favorable_direction": direction(m, float(r[m])), "run_a": r["run_a"], "run_b": r["run_b"]})
    direction_counts = []
    for k in (20, 196):
        for comparison in ("M-B4", "C-B4"):
            for m in SIX:
                ds = [r for r in directions if r["bank_chunks"] == k and r["comparison"] == comparison and r["metric"] == m]
                paired_stats.append({"bank_chunks": k, "comparison": comparison, "metric": m, **summary([r["difference"] for r in ds])})
                direction_counts.append({"bank_chunks": k, "comparison": comparison, "metric": m,
                    "favorable": sum(r["favorable_direction"] > 0 for r in ds), "unfavorable": sum(r["favorable_direction"] < 0 for r in ds), "ties": sum(r["favorable_direction"] == 0 for r in ds)})
    for oldfile, newrows, key in (("endpoint_statistics", endpoint_stats, "group"), ("paired_statistics", paired_stats, "comparison")):
        lookup = {(int(r["bank_chunks"]), r[key], r["metric"]): r for r in rows(base/(oldfile+".csv"))}
        for r in newrows:
            old = lookup[r["bank_chunks"], r[key], r["metric"]]
            same(r["mean"], old["mean"]); same(r["std_ddof1"], old["std_ddof1"])
    rankings = []
    for k in (20, 196):
        for m in SIX:
            ss = [r for r in endpoint_stats if r["bank_chunks"] == k and r["metric"] == m]
            ordered = sorted(ss, key=lambda r: r["mean"], reverse=m.startswith("g_new"))
            rankings.append({"bank_chunks": k, "metric": m, "better_to_worse_mean": ">".join(r["group"] for r in ordered)})
    pair_ci, endpoint_ci = [], []
    pair_lookup = {(int(r["bank_chunks"]), r["comparison"], int(r["selection_seed"])): r for r in pairs}
    for raw in read(base/"document_bootstrap.json"):
        if raw["field"] != "sum_nll": continue
        m = METRICS[("target", "c4", "wikitext").index(raw["corpus"])]
        old = pair_lookup[raw["bank_chunks"], raw["comparison"], raw["selection_seed"]]
        for weighting, suffix in (("token_weighted", ""), ("equal_document", "_equal_document")):
            part = bootstrap_metric(raw, m, weighting, True); metric = m+suffix; same(part["estimate"], old[metric])
            pair_ci.append({"bank_chunks": raw["bank_chunks"], "comparison": raw["comparison"], "selection_seed": raw["selection_seed"], "metric": metric,
                **part, "documents": raw["documents"], "prediction_tokens": raw["prediction_tokens"], "run_a": raw["run_a"], "run_b": raw["run_b"],
                "bootstrap_replicates": raw["bootstrap_replicates"], "bootstrap_seed": raw["bootstrap_seed"], "confidence": raw["confidence"], "conditional_on_both_checkpoints": True})
    for raw in read(base/"endpoint_document_bootstrap.json"):
        if raw["field"] != "sum_nll": continue
        r = by_id[raw["run_id"]]
        for weighting, suffix in (("token_weighted", ""), ("equal_document", "_equal_document")):
            part = bootstrap_metric(raw, raw["metric"], weighting, False); metric = raw["metric"]+suffix; same(part["estimate"], r[metric])
            endpoint_ci.append({"run_id": r["run_id"], "group": r["group"], "bank_chunks": raw["bank_chunks"], "selection_seed": raw["selection_seed"], "metric": metric, **part,
                "documents": raw["documents"], "prediction_tokens": raw["prediction_tokens"], "bootstrap_replicates": raw["bootstrap_replicates"], "bootstrap_seed": raw["bootstrap_seed"], "confidence": raw["confidence"]})
    ci_counts = []
    for r in direction_counts:
        subset = [s for s in pair_ci if all(s[k] == r[k] for k in ("bank_chunks", "comparison", "metric"))]
        fav = lambda s: s["ci_low"] > 0 if r["metric"].startswith("g_new") else s["ci_high"] < 0
        unfav = lambda s: s["ci_high"] < 0 if r["metric"].startswith("g_new") else s["ci_low"] > 0
        ci_counts.append({**{k:r[k] for k in ("bank_chunks", "comparison", "metric")}, "conditional_CI_entirely_favorable": sum(fav(s) for s in subset), "conditional_CI_entirely_unfavorable": sum(unfav(s) for s in subset), "conditional_CI_includes_zero": sum(s["ci_low"] <= 0 <= s["ci_high"] for s in subset), "n_individual_draw_CIs": len(subset)})
    validation = ROOT/"results/analysis/supplement_B_validation_complete_v1"
    valstats = {(int(r["bank_chunks"]),r["group"],r["metric"]):r for r in rows(validation/"endpoint_statistics.csv")}
    val_comparison = [{"bank_chunks":r["bank_chunks"], "group":r["group"], "metric":r["metric"], "validation_mean":float(valstats[r["bank_chunks"],r["group"],r["metric"]]["mean"]), "validation_std_ddof1":float(valstats[r["bank_chunks"],r["group"],r["metric"]]["std_ddof1"]), "test_mean":r["mean"], "test_std_ddof1":r["std_ddof1"]} for r in endpoint_stats if r["metric"] in METRICS]
    runtime_path = ROOT/"results/evaluation_timing_budget_v5_sources/B_runtime_completion.json"
    runtime = read(runtime_path); assert runtime["status"] == "passed" and runtime["completed_jobs"] == 60
    assert {j["id"] for j in runtime["jobs"]} == {j["id"] for j in manifest["jobs"]}
    traincost = rows(base/"actual_training_costs.csv"); infercost = rows(base/"physical_inference_costs.csv")
    composition = rows(validation/"bank_composition.csv")
    generation = read(ROOT/"results/generation_summary.json")["pools"]["11"]
    cost = {"new_test_forward_jobs":60,"new_initial_forward_jobs":0,"baseline_reused_once_not_recharged":True,
        "whole_slot_seconds_sum":runtime["actual_whole_slot_seconds_sum"],"mixed_master_calendar_window_seconds":runtime["mixed_master_first_start_to_last_exit_seconds"],
        "calendar_window_exclusive":False,"scorer_subprocess_wall_seconds_nested":sum(float(r["incremental_inference_subprocess_wall_seconds"]) for r in infercost),
        "input_verification_wall_seconds_nested":sum(j["input_verification_wall_seconds_nested"] for j in runtime["jobs"]),
        "training_parent_components_seconds":{k:sum(float(r[k]) for r in traincost) for k in ("setup_seconds","training_seconds","validation_seconds","checkpoint_seconds")},
        "training_committed_parent_sum_seconds":sum(float(r["committed_parent_sum_seconds"]) for r in traincost),
        "bank_selection_wall_seconds_sum":sum(float(r["selection_wall_seconds"]) for r in composition),
        "clustering_seconds_nested":sum(float(r["clustering_nested_seconds"]) for r in composition),
        "shared_existing_pool11_generation_seconds_not_new_B_inference":generation["generation"]["generation_wall_seconds"],
        "shared_existing_pool11_joint_stats_seconds_not_recharged_each_method":generation["teacher_statistics"]["wall_seconds"],
        "original_authorized_CPU_analysis_process_wall_seconds":None,"original_authorized_CPU_analysis_process_CPU_seconds":None,
        "original_analysis_parent_timer_missing":True,"analysis_calendar_timestamps_are_not_parent_timers":True,
        "whole_phase_guard_wall_seconds_local_only":readiness["score_blind_verification_wall_seconds"],
        "nested_scorer_verification_teacher_clustering_not_added_to_parents":True,"all_wall_seconds_are_not_pure_GPU_seconds":True}
    output.mkdir(parents=True, exist_ok=False)
    tables = {"all_60_endpoints":endpoints,"all_40_pairs":pairs,"six_metric_statistics":endpoint_stats,"paired_six_metric_statistics":paired_stats,"paired_draw_directions":directions,
        "direction_counts":direction_counts,"six_metric_rankings":rankings,"conditional_paired_document_CIs":pair_ci,"conditional_endpoint_document_CIs":endpoint_ci,"conditional_CI_counts":ci_counts,"validation_vs_test":val_comparison}
    for name, table in tables.items(): write_csv(output/(name+".csv"),table)
    write_json(output/"cost_accounting.json",cost)
    plot(endpoints, pairs, output)
    inputs = {str(p.resolve()):sha(p) for p in base.iterdir() if p.is_file()}
    for p in (complete_path,Path(plan["manifest"]["path"]),runtime_path,ROOT/"results/generation_summary.json",ROOT/"results/bank_diagnostics/conditional_nll_sampling_reference.json",validation/"analysis_manifest.json",validation/"bank_composition.csv",validation/"composition_statistics.csv",validation/"endpoint_statistics.csv"):
        inputs[str(p.resolve())]=sha(p)
    for path, digest in verified_sources.items(): assert sha(path)==digest
    receipt = {"schema_version":1,"review_complete":True,"completed_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "review_source":{"path":str(Path(__file__).resolve()),"sha256":sha(__file__)},"authorized_COMPLETE":{"path":str(complete_path.resolve()),"sha256":expected_complete},
        "authority":complete["authority"],"first_read_receipt":complete["first_read_receipt"],"input_sha256":inputs,"verified_collector_source_sha256":verified_sources,
        "collector_source_bytes_verified_each_pass":source_bytes,"collector_source_verification_passes":2,
        "independent_scalar_checks":check_count,"maximum_absolute_numerical_error":max(errors),"all_60_draws_preserved":True,"replicate_unit":"selection_seed",
        "six_token_equal_document_mean_rankings_match":all(next(r["better_to_worse_mean"] for r in rankings if r["bank_chunks"]==k and r["metric"]==m)==next(r["better_to_worse_mean"] for r in rankings if r["bank_chunks"]==k and r["metric"]==m+"_equal_document") for k in (20,196) for m in METRICS),
        "new_bootstrap_runs":0,"new_model_forward_calls":0,"selection_changed":False,"original_analysis_or_first_read_modified":False,"C_or_behavior_scores_accessed":False,
        "outputs_sha256":{str(p.resolve()):sha(p) for p in output.iterdir() if p.is_file()},
        "review_wall_seconds":time.perf_counter()-start,"review_process_CPU_seconds":time.process_time()-cpu,
        "review_timer_scope":"This independent authorized source/numerical review and plots through final source recheck; excludes interactive writing and receipt serialization; not original collector timing or inference slot"}
    write_json(output/"review_receipt.json",receipt)
    return {"output":str(output),"receipt_sha256":sha(output/"review_receipt.json"),"scalar_checks":check_count,"max_error":max(errors),"cost":cost,"review_wall_seconds":receipt["review_wall_seconds"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete",type=Path,required=True); parser.add_argument("--complete-sha256",required=True); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); print(json.dumps(run(args.complete,args.complete_sha256,args.output),ensure_ascii=False,indent=2))
