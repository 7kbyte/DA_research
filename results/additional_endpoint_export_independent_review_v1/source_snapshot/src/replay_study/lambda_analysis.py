"""Validation-only supplement A; aliases retain their actual training identity.

This module never rewrites phase labels, opens test scores, or launches jobs.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from .analysis import (CORPORA, METRICS, LABELS, _check_summary, _protocol_status,
                       aggregate_endpoints, endpoint_metrics, mean_std,
                       paired_seed_differences, read_json, sha256,
                       training_costs, training_validation_execution,
                       verify_recorded_execution, write_csv)


PHASE = "supplement_A_lambda_sweep"
OPERATIONAL = {"output_dir", "device", "resume_from", "max_steps", "deadline_utc",
               "deadline_margin_seconds", "save_every_steps", "keep_step_checkpoints"}
SCORE = "S = delta_target + 0.5 * (delta_C4 + delta_Wiki); lower is better"


def _write(path, payload):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _science(config):
    return {k: v for k, v in config.items() if k not in OPERATIONAL}


def _strict_execution(execution):
    if execution.get("deterministic_algorithms") is not True or execution.get("deterministic_warn_only") is not False:
        raise ValueError("supplement A requires strict deterministic execution proof")


def check_registry(registry_path, preregistered_registry_path, preregistration_path,
                   parent_registry_path, protocol_status_path):
    """Bind the complete supplement grid to its preregistration and active parent."""
    prereg = read_json(preregistration_path)
    if sha256(preregistered_registry_path) != prereg["registry_sha256"]:
        raise ValueError("preregistered supplement registry SHA changed")
    original = read_json(preregistered_registry_path)
    rows = read_json(registry_path)
    _protocol_status(protocol_status_path, parent_registry_path, "active")
    parents = read_json(parent_registry_path)
    if not parents or any(r.get("phase") != "formal" or r.get("active") is not True or
                          r.get("status") in {"provisional", "invalid", "superseded"} for r in parents):
        raise ValueError("supplement parent must be the explicitly active formal registry")
    parent_by_id = {r["id"]: r for r in parents}
    if len(parent_by_id) != len(parents):
        raise ValueError("duplicate formal parent IDs")
    expected = {(g, int(s), float(w)) for g in prereg["groups"] for s in prereg["seeds"] for w in prereg["lambda_grid"]}
    if len(rows) != len(expected) or len(original) != len(expected):
        raise ValueError("complete preregistered supplement grid is required")
    original_by_id = {r["id"]: r for r in original}
    if len(original_by_id) != len(original) or {r["id"] for r in rows} != set(original_by_id):
        raise ValueError("supplement condition IDs differ from preregistration")
    scientific_cells = [(r["group"], int(r["seed"]), float(r["weight"])) for r in rows]
    if set(scientific_cells) != expected or len(set(scientific_cells)) != len(rows):
        raise ValueError("duplicate or missing supplement scientific condition")
    changed_row_fields = {"active", "launch_status", "config_path", "config_sha256",
                          "preregistered_config_path", "preregistered_config_sha256", "operational_override"}
    sources = {str(Path(p)): sha256(p) for p in (registry_path, preregistered_registry_path,
               preregistration_path, parent_registry_path, protocol_status_path)}
    checked = []
    actual_ids = set()
    for row in rows:
        fixed = original_by_id[row["id"]]
        if row.get("phase") != PHASE or fixed.get("phase") != PHASE:
            raise ValueError("explicit supplement_A_lambda_sweep phase required")
        if row.get("status") in {"provisional", "invalid", "superseded"}:
            raise ValueError("invalid or provisional supplement condition")
        if ({k: v for k, v in row.items() if k not in changed_row_fields} !=
                {k: v for k, v in fixed.items() if k not in changed_row_fields}):
            raise ValueError("runtime registry changed a preregistered scientific field")
        cfg, fixed_cfg, source_cfg = [read_json(p) for p in
            (row["config_path"], fixed["config_path"], fixed["source_config_path"])]
        for path, expected_sha in ((row["config_path"], row["config_sha256"]),
                                   (fixed["config_path"], fixed["config_sha256"]),
                                   (fixed["source_config_path"], fixed["source_config_sha256"])):
            if sha256(path) != expected_sha:
                raise ValueError("supplement or source config SHA changed")
            sources[str(Path(path))] = expected_sha
        if _science(cfg) != _science(fixed_cfg):
            raise ValueError("runtime config changed scientific fields")
        if (row.get("preregistered_config_sha256", fixed["config_sha256"]) != fixed["config_sha256"] or
                Path(row.get("preregistered_config_path", fixed["config_path"])).resolve() != Path(fixed["config_path"]).resolve()):
            raise ValueError("runtime config provenance names a different preregistered source")
        if any(k not in OPERATIONAL or cfg.get(k) != v for k, v in row.get("operational_override", {}).items()):
            raise ValueError("declared operational override differs from actual config")
        if row["source_formal_run_id"] not in parent_by_id:
            raise ValueError("source formal run is outside the active parent registry")
        parent = parent_by_id[row["source_formal_run_id"]]
        if (parent["group"], parent["seed"], parent["ratio"], parent["weight"]) != (row["group"], row["seed"], row["ratio"], .05):
            raise ValueError("source formal scientific identity differs")
        if sha256(parent["config_path"]) != fixed["source_config_sha256"]:
            raise ValueError("active formal source config is not the frozen supplement source")
        if (cfg.get("target_prediction_tokens") != prereg["target_prediction_tokens_per_run"] or
                cfg.get("replay_ratio") != prereg["replay_ratio"] or cfg.get("learning_rate") != prereg["learning_rate"] or
                cfg.get("replay_weight") != row["weight"] or cfg.get("seed") != row["seed"] or
                cfg.get("deterministic_algorithms") is not True or not cfg.get("model_revision")):
            raise ValueError("supplement scientific budget or deterministic config differs")
        if row["execution_kind"] == "alias":
            actual_id = row["alias_of"]
            if (actual_id != parent["id"] or row["weight"] != .05 or
                    row.get("counted_as_new_training") is not False or fixed_cfg != source_cfg or
                    Path(row["summary_path"]).resolve() != Path(parent["summary_path"]).resolve()):
                raise ValueError("alias must reuse exactly its formal run and summary")
        elif row["execution_kind"] == "train":
            actual_id = row["id"]
            if row.get("alias_of") is not None or row.get("counted_as_new_training") is not True or row["weight"] == .05:
                raise ValueError("new training and alias identity disagree")
            allowed = {"replay_weight", "output_dir", "run_label"}
            if {k: v for k, v in fixed_cfg.items() if k not in allowed} != {k: v for k, v in source_cfg.items() if k not in allowed}:
                raise ValueError("preregistered child changes more than lambda and run location")
        else:
            raise ValueError("execution_kind must be train or alias")
        if cfg.get("run_label") != actual_id or cfg.get("output_dir") != row["output_dir"]:
            raise ValueError("actual run identity differs from config")
        if actual_id in actual_ids:
            raise ValueError("same actual training run counted as multiple supplement cells")
        actual_ids.add(actual_id)
        checked.append({**row, "actual_run_id": actual_id, "config": cfg, "source_parent": parent})
    return checked, prereg, sources


def select_lambdas(endpoints, registry, preregistration):
    """Never select from an incomplete method grid, even under --allow-partial."""
    expected_seeds = set(map(int, preregistration["seeds"]))
    weights = sorted(map(float, preregistration["lambda_grid"]))
    scores, selections, statuses = [], [], []
    for group in preregistration["groups"]:
        candidates = []
        for weight in weights:
            rows = [r for r in endpoints if r["group"] == group and float(r["weight"]) == weight]
            if len({r["seed"] for r in rows}) != len(rows):
                raise ValueError("duplicate seed cannot increase selection replication")
            if not {r["seed"] for r in rows}.issubset(expected_seeds):
                raise ValueError("unexpected seed in lambda scoring")
            values = [-r["g_new"] + .5 * (r["f_old_c4"] + r["f_old_wikitext"]) for r in rows]
            stats = mean_std(values)
            complete = {r["seed"] for r in rows} == expected_seeds
            candidate = {"group": group, "weight": weight, **stats, "complete": complete,
                         "expected_n": len(expected_seeds), "score_definition": SCORE,
                         "source": "validation_final_fixed_exposure_only",
                         "condition_ids": [r["condition_id"] for r in sorted(rows, key=lambda r: r["seed"])],
                         "actual_run_ids": [r["actual_run_id"] for r in sorted(rows, key=lambda r: r["seed"])]}
            candidate["per_seed_scores"] = [{"seed": r["seed"], "condition_id": r["condition_id"],
                "actual_run_id": r["actual_run_id"],
                "score": -r["g_new"] + .5 * (r["f_old_c4"] + r["f_old_wikitext"])}
                for r in sorted(rows, key=lambda r: r["seed"])]
            scores.append(candidate)
            candidates.append(candidate)
        ready = all(r["complete"] for r in candidates)
        statuses.append({"group": group, "selection_ready": ready, "expected_weights": len(weights),
                         "complete_weights": sum(r["complete"] for r in candidates)})
        if ready:
            chosen = min(candidates, key=lambda r: (r["mean"], r["weight"]))
            selections.append({**chosen, "selection_frozen": True, "shared_across_seeds": True,
                               "tie_policy": "strict equal numeric score chooses smaller lambda",
                               "selected_by_best_checkpoint": False, "test_scores_used": False})
    return scores, selections, statuses


def candidate_frontier(endpoints, registry):
    aggregates = aggregate_endpoints(endpoints, registry)
    grouped = defaultdict(dict)
    for row in aggregates:
        grouped[(row["group"], row["weight"])][row["metric"]] = row
    candidates = []
    for (group, weight), metrics in sorted(grouped.items()):
        complete = all(metrics[m]["complete"] for m in ("g_new", "f_old_c4", "f_old_wikitext"))
        candidates.append({"group": group, "weight": weight, "complete": complete,
                           "n": metrics["g_new"]["n"],
                           **{f"{m}_{s}": metrics[m][s] for m in ("g_new", "f_old_c4", "f_old_wikitext")
                              for s in ("mean", "std_ddof1")}})
    ready = [r for r in candidates if r["complete"]]
    def dominates(a, b, olds):
        av = [a["g_new_mean"]] + [-a[f"f_old_{c}_mean"] for c in olds]
        bv = [b["g_new_mean"]] + [-b[f"f_old_{c}_mean"] for c in olds]
        return all(x >= y for x, y in zip(av, bv)) and any(x > y for x, y in zip(av, bv))
    for row in candidates:
        for name, olds in (("c4", ("c4",)), ("wikitext", ("wikitext",)), ("joint", ("c4", "wikitext"))):
            row[f"pareto_{name}"] = (not any(dominates(other, row, olds) for other in ready)) if row["complete"] else None
        row["frontier_is_partial"] = len(ready) != len(candidates)
        row["frontier_scope"] = "all preregistered methods/lambdas with complete three-seed final validation"
    return aggregates, candidates


def collect_lambda_analysis(registry_path, baseline_path, *, preregistered_registry_path,
                            preregistration_path, parent_registry_path, protocol_status_path):
    rows, prereg, sources = check_registry(registry_path, preregistered_registry_path,
                                          preregistration_path, parent_registry_path, protocol_status_path)
    baseline = read_json(baseline_path)
    sources[str(Path(baseline_path))] = sha256(baseline_path)
    project = Path(__file__).resolve().parents[2]
    for path in (Path(__file__), Path(__file__).with_name("analysis.py"),
                 Path(__file__).with_name("execution.py"), project / "scripts/analyze_lambda_sweep.py"):
        sources[str(path)] = sha256(path)
    if baseline.get("checkpoint") is not None or not baseline.get("execution_provenance"):
        raise ValueError("unadapted matched execution baseline is required")
    _strict_execution(baseline["execution_provenance"])
    endpoint_metrics(baseline["corpora"], baseline["corpora"])
    if any(abs(float(baseline["corpora"][name].get("kl", 0))) > 1e-8 for name in CORPORA):
        raise ValueError("initial teacher self-KL must be zero")
    endpoints, availability, costs, identities = [], [], [], []
    for row in rows:
        cfg = row["config"]
        if baseline.get("revision") != cfg["model_revision"]:
            raise ValueError("supplement baseline revision mismatch")
        actual_id = row["actual_run_id"]
        identity = {"condition_id": row["id"], "actual_run_id": actual_id, "run_id": actual_id,
                    "phase": PHASE, "execution_kind": row["execution_kind"], "alias_of": row["alias_of"],
                    "group": row["group"], "seed": row["seed"], "ratio": row["ratio"], "weight": row["weight"]}
        identities.append({**identity, "summary_path": row["summary_path"],
                           "future_final_evaluation_relative_path": f"{actual_id}/final.json",
                           "initial_evaluation_is_shared": True,
                           "counted_as_new_training": row["counted_as_new_training"]})
        status = "missing_summary"
        path = Path(row["summary_path"])
        if path.exists():
            summary = read_json(path)
            sources[str(path)] = sha256(path)
            actual_entry = {**row, "id": actual_id}
            _check_summary(summary, actual_entry)
            execution = training_validation_execution(summary)
            _strict_execution(execution)
            verify_recorded_execution(baseline["execution_provenance"], execution)
            status = "incomplete_training"
            counters = summary["counters"]
            if counters["target_prediction_tokens"] > cfg["target_prediction_tokens"]:
                raise ValueError("supplement exposure exceeds its frozen target budget")
            costs.append({**identity, **training_costs(summary),
                          "incremental_training_cost_factor": 1 if row["execution_kind"] == "train" else 0,
                          "cost_scope": "alias is historical main cost, incremental supplement cost zero" if row["execution_kind"] == "alias" else "new supplement training"})
            if summary["status"] == "completed":
                parent_row = row["source_parent"]
                parent_path = Path(parent_row["summary_path"])
                if not parent_path.exists():
                    raise ValueError("completed supplement lacks its completed active formal source")
                parent_summary = read_json(parent_path)
                _check_summary(parent_summary, parent_row)
                if parent_summary.get("status") != "completed":
                    raise ValueError("completed supplement source formal run is incomplete")
                sources[str(parent_path)] = sha256(parent_path)
                parent_provenance = parent_summary.get("provenance", {})
                parent_execution = training_validation_execution(parent_summary)
                _strict_execution(parent_execution)
                verify_recorded_execution(baseline["execution_provenance"], parent_execution)
                for field in ("target", "replay", "source_sha256"):
                    if field not in parent_provenance or summary.get("provenance", {}).get(field) != parent_provenance[field]:
                        raise ValueError("supplement data/code provenance differs from its frozen formal source")
                replay = int(cfg["target_prediction_tokens"] * Fraction(str(cfg["replay_ratio"])))
                points = summary.get("validation_trajectory", [])
                if (counters["target_prediction_tokens"] != cfg["target_prediction_tokens"] or
                        counters["replay_prediction_tokens"] != replay or not points or
                        points[-1]["target_prediction_tokens"] != cfg["target_prediction_tokens"] or
                        points[-1]["replay_prediction_tokens"] != replay or points[-1]["step"] != counters["step"]):
                    raise ValueError("completed supplement lacks exact final fixed-exposure validation")
                endpoints.append({**identity, "split": "validation", "checkpoint_kind": "final",
                                  "target_prediction_tokens": counters["target_prediction_tokens"],
                                  "replay_prediction_tokens": replay, "checkpoint_step": counters["step"],
                                  "checkpoint_path": summary.get("checkpoint_path"),
                                  **endpoint_metrics(points[-1]["metrics"], baseline["corpora"])})
                status = "ready"
        availability.append({**identity, "status": status})
    scores, selections, selection_status = select_lambdas(endpoints, rows, prereg)
    statistics, frontier = candidate_frontier(endpoints, rows)
    paired, paired_statistics = paired_seed_differences(endpoints, rows)
    manifest = {"generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "phase": PHASE,
                "complete": len(endpoints) == len(rows), "completed_conditions": len(endpoints),
                "expected_conditions": len(rows), "new_training_conditions": sum(r["execution_kind"] == "train" for r in rows),
                "main_alias_conditions": sum(r["execution_kind"] == "alias" for r in rows),
                "replicate_unit": "training seed, three per method/lambda; main aliases do not add replicates",
                "endpoint_policy": "final exact target exposure; never best checkpoint",
                "score_definition": SCORE, "tie_policy": "strict equal numeric score chooses smaller lambda",
                "test_scores_accessed": False, "training_launched": False, "sources_sha256": sources}
    return {"manifest": manifest, "validation_endpoints": endpoints, "endpoint_statistics": statistics,
            "paired_same_lambda": paired, "paired_same_lambda_statistics": paired_statistics,
            "all_candidates_frontier": frontier, "candidate_scores": scores,
            "lambda_selections": selections, "selection_status": selection_status,
            "availability": availability, "run_identities": identities, "observed_costs": costs}


def write_lambda_analysis(bundle, output, *, allow_partial=False, plots=True):
    if not bundle["manifest"]["complete"] and not allow_partial:
        raise ValueError("supplement lambda grid incomplete; --allow-partial required, no incomplete-method lambda selected")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for key, value in bundle.items():
        if isinstance(value, list):
            write_csv(output / f"{key}.csv", value)
    _write(output / "analysis_manifest.json", bundle["manifest"])
    _write(output / "validation_lambda_selection.json", {
        **bundle["manifest"], "all_candidate_scores": bundle["candidate_scores"],
        "method_status": bundle["selection_status"], "selections": bundle["lambda_selections"],
        "selection_uses_test": False, "shared_initial_test_baseline_relative_path": "initial_pythia-410m.json",
        "future_evaluation_identity_policy": "actual_run_id/final.json; reuse main alias evaluations, never condition_id paths"})
    if plots:
        plot_lambda_analysis(bundle, output)
    else:
        # A later partial refresh must not leave older counts beside new tables.
        for extension in ("png", "pdf", "svg"):
            (output / f"lambda_gain_forgetting.{extension}").unlink(missing_ok=True)
    (output / "统计与选参说明.md").write_text(
        "补充A保持supplement_A_lambda_sweep身份，不混入主36。所有端点均为最终固定10M目标预测token的验证结果；"
        "均值与样本SD(ddof=1)以三个训练seed为单位。18个main alias复用原训练与评估，不新增重复或训练成本。"
        "每方法必须全部预设λ×三个seed完整才选一个共享λ：先逐seed计算S=Δtarget+.5(ΔC4+ΔWiki)，再对seed无权平均取最小；"
        "严格相同数值的分数取更小λ，不用近似平局容差。缺失候选不能忽略，partial不会替缺失方法挑λ。"
        "前沿来自完整三seed候选的均值，只作描述、不额外选择λ；partial前沿不能当全部网格前沿。"
        "全部候选与配对同λ差均保留，best和test均不参与选择。未来测试以actual_run_id/final.json复用原始评估，"
        "共享初始baseline；本入口未开放test分数读取，也不启动训练。\n")
    return bundle["manifest"]


def plot_lambda_analysis(bundle, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "savefig.dpi": 320,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {"B1": "#009E73", "B2": "#56B4E9", "B3": "#E69F00", "B4": "#0072B2", "M": "#D55E00", "C": "#CC79A7"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ready = [r for r in bundle["all_candidates_frontier"] if r["complete"]]
    for ax, old, title in zip(axes, ("c4", "wikitext"), ("C4", "WikiText-103")):
        for group in colors:
            rows = sorted([r for r in ready if r["group"] == group], key=lambda r: r["weight"])
            if not rows:
                continue
            ax.plot([r["g_new_mean"] for r in rows], [r[f"f_old_{old}_mean"] for r in rows],
                    color=colors[group], linewidth=.7, alpha=.5)
            for i, row in enumerate(rows):
                ax.errorbar(row["g_new_mean"], row[f"f_old_{old}_mean"], xerr=row["g_new_std_ddof1"],
                            yerr=row[f"f_old_{old}_std_ddof1"], marker="o", color=colors[group],
                            capsize=2, linestyle="none", label=LABELS[group] if i == 0 else None)
                ax.annotate(f"{row['weight']:g}", (row["g_new_mean"], row[f"f_old_{old}_mean"]),
                            xytext=(3, 3), textcoords="offset points", fontsize=6, color=colors[group])
        if not ready:
            ax.text(.5, .5, "Prepared analysis only\nNo complete three-seed candidate", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel("New-domain gain (nats/token)")
        ax.set_ylabel("Old-domain forgetting (nats/token)")
        ax.grid(alpha=.18)
    if ready:
        axes[0].legend(fontsize=6, frameon=False)
    status = "COMPLETE" if bundle["manifest"]["complete"] else "PARTIAL"
    fig.suptitle(f"SUPPLEMENT A {status}: final validation; labels = lambda; mean +/- seed SD", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"lambda_gain_forgetting.{ext}", bbox_inches="tight")
    plt.close(fig)
