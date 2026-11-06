"""Validation-only adapters for explicitly separate supplemental cohorts.

No primary-registry rewriting, evaluation-result reader, or inference is used.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

from .analysis import (CORPORA, METRICS, PAIRS, _check_summary, aggregate_endpoints,
                       endpoint_metrics, paired_seed_differences, read_json, sha256,
                       selected_checkpoint_exposure, training_costs,
                       training_validation_execution, trajectory_statistics,
                       verify_recorded_execution, write_csv)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def proof(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def verify_proof(record):
    if sha256(record["path"]) != record["sha256"]:
        raise ValueError(f"Frozen input changed: {record['path']}")


def cohort_gate(rows, state, definitions=None, master=None):
    """Read scheduler identities only, never partial endpoint scores."""
    ids = [r["id"] for r in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("nonempty unique cohort IDs required")
    statuses = {i: state.get("jobs", {}).get(i, {}).get("status", "missing") for i in ids}
    if definitions is not None:
        frozen = {j["id"]:j for j in definitions}
        actual = {j["id"]:j for j in master}
        if len(frozen) != len(definitions) or set(frozen) != set(ids):
            raise ValueError("frozen phase jobs must cover exactly the registered cohort")
        for run_id in ids:
            job = state.get("jobs", {}).get(run_id,{})
            digest = hashlib.sha256(json.dumps(frozen[run_id],sort_keys=True).encode()).hexdigest()
            if actual.get(run_id) != frozen[run_id] or job.get("definition_sha256") != digest:
                raise ValueError("master/scheduler definition differs from the frozen phase job")
            if job.get("status") == "completed":
                if job.get("returncode") != 0 or job.get("missing_outputs"):
                    raise ValueError("completed status lacks successful exit and complete outputs")
                for path in frozen[run_id].get("outputs",[]):
                    if not Path(path).is_file():
                        raise ValueError("registered completed output is missing")
    failed = {i: s for i, s in statuses.items() if s in {"failed", "blocked"}}
    return {"ready": all(s == "completed" for s in statuses.values()),
            "expected_runs": len(ids), "completed_runs": sum(s == "completed" for s in statuses.values()),
            "status_counts": dict(Counter(statuses.values())), "failed_jobs": failed}


def validate_bank_activation(source_path, runtime_path, activation_path):
    """Only the explicit operational deadline override may differ from preparation."""
    source, runtime, activation = map(read_json, (source_path, runtime_path, activation_path))
    if (activation.get("phase") != "conditional_bank_variance" or
            activation.get("source_artifacts_modified") is not False):
        raise ValueError("explicit unmodified conditional activation required")
    for key, path in (("source_registry", source_path), ("execution_registry", runtime_path)):
        if Path(activation[key]).resolve() != Path(path).resolve() or sha256(path) != activation[key + "_sha256"]:
            raise ValueError("activation registry binding mismatch")
    verify_proof({"path": activation["preparation_manifest"], "sha256": activation["preparation_manifest_sha256"]})
    if Path(activation["analysis_registry"]).resolve() != Path(source_path).resolve():
        raise ValueError("analysis must retain the prepared registry identity")
    old, new = ({r["id"]: r for r in rows} for rows in (source, runtime))
    if (len(old) != len(source) or len(new) != len(runtime) or set(old) != set(new) or
            set(activation["config_overrides"]) != set(old)):
        raise ValueError("activation must bind exactly the complete prepared cohort")
    sources = [proof(p) for p in (source_path, runtime_path, activation_path, activation["preparation_manifest"])]
    mutable = {"active", "status", "config_path", "prepared_config_path", "prepared_config_sha256", "config_sha256"}
    for run_id, original in old.items():
        actual = new[run_id]
        if (original.get("phase") != "conditional_bank_variance" or "seed" in original or
                original.get("training_seed") != 11 or original.get("generation_seed") != 11 or
                actual.get("active") is not True or actual.get("status") != "activated"):
            raise ValueError("selection draw must remain conditional on train/generation seed11")
        if {k: v for k, v in original.items() if k not in mutable} != {k: v for k, v in actual.items() if k not in mutable}:
            raise ValueError("runtime registry changed a scientific field")
        override = activation["config_overrides"][run_id]
        if set(override["operational_override"]) != {"deadline_utc"}:
            raise ValueError("only the explicitly recorded deadline override is permitted")
        for prefix, path in (("prepared", original["config_path"]), ("runtime", actual["config_path"])):
            if Path(override[prefix + "_config_path"]).resolve() != Path(path).resolve():
                raise ValueError("activation config path mismatch")
            verify_proof({"path": path, "sha256": override[prefix + "_config_sha256"]})
            sources.append(proof(path))
        if (Path(actual["prepared_config_path"]).resolve() != Path(original["config_path"]).resolve() or
                actual["prepared_config_sha256"] != override["prepared_config_sha256"] or
                actual["config_sha256"] != override["runtime_config_sha256"]):
            raise ValueError("runtime row config hashes do not match activation")
        expected = {**read_json(original["config_path"]), **override["operational_override"]}
        if read_json(actual["config_path"]) != expected:
            raise ValueError("runtime config changed beyond the recorded operational override")
    return source, runtime, sources


def collect_extended_validation(rows, protocol, baseline, summaries, *, phase, model):
    """Pure complete-cohort computation; callers bind frozen files and audit receipts."""
    # Protocol.groups is the lambda-calibration grid, not the formal registry.
    formal_groups = {"pythia-410m": ("B0","B1","B2","B3","B4","D","M","C"),
                     "pythia-1b": ("B0","B1","B2","B3","B4","D","M","C"),
                     "pythia-2.8b": ("B0","B1","B2","B4","M")}
    if model not in formal_groups:
        raise ValueError("unregistered extended model")
    groups = formal_groups[model]
    matrix = {(g, s) for g in groups for s in (11, 22, 33)}
    if (not phase.startswith("extended_") or protocol["model"] != model or
            protocol["data_version"] != "extended_v1" or
            len(rows) != len(matrix) or {(r["group"], r["seed"]) for r in rows} != matrix or
            len({r["id"] for r in rows}) != len(rows) or
            any(r.get("phase") != "formal" or r.get("active") is not True or r.get("model") != model or
                not r["id"].startswith(phase + "_") for r in rows)):
        raise ValueError("explicit complete model-specific extended cohort required")
    if set(summaries) != {r["id"] for r in rows}:
        raise ValueError("incomplete cohort: no partial final statistics")
    if (baseline.get("revision") != protocol["model_revision"] or baseline.get("checkpoint") is not None or
            not baseline.get("execution_provenance")):
        raise ValueError("matched initial revision and execution provenance required")
    endpoints, trajectories, costs, executions = [], [], [], {}
    target = protocol["formal_target_prediction_tokens"]
    for row in rows:
        summary = summaries[row["id"]]
        _check_summary(summary, row)
        cfg, counters = summary["config"], summary["counters"]
        if (summary.get("status") != "completed" or cfg["target_prediction_tokens"] != target or
                counters["target_prediction_tokens"] != target or
                counters["replay_prediction_tokens"] != int(target * Fraction(str(row["ratio"]))) or
                cfg["model_revision"] != protocol["model_revision"] or
                cfg["model_name_or_path"] != baseline["model"] or
                cfg["validation_paths"] != protocol["validation_paths"] or
                cfg["learning_rate"] != protocol["shared_learning_rate"] or
                (row["group"] != "B0" and cfg["replay_weight"] != protocol["shared_lambda"])):
            raise ValueError("completed extended budget/model/data/hyperparameter binding mismatch")
        execution = training_validation_execution(summary)
        verify_recorded_execution(baseline["execution_provenance"], execution)
        executions[row["id"]] = execution
        identity = dict(run_id=row["id"], actual_run_id=row["id"], phase=phase, role="supplement_C_extended",
                        model=model, data_family="extended_v1", split="validation", group=row["group"],
                        seed=row["seed"], ratio=row["ratio"], weight=row["weight"], learning_rate=row["learning_rate"])
        points = summary.get("validation_trajectory", [])
        if not points or points[-1]["target_prediction_tokens"] != target:
            raise ValueError("full-exposure final validation missing")
        exposures = [p["target_prediction_tokens"] for p in points]
        if exposures != sorted(set(exposures)) or exposures[0] <= 0:
            raise ValueError("validation exposures must be strictly increasing")
        for point in points:
            if (set(point["metrics"]) != set(CORPORA) or point["replay_prediction_tokens"] !=
                    int(point["target_prediction_tokens"] * Fraction(str(row["ratio"])))):
                raise ValueError("trajectory corpus or replay exposure mismatch")
            trajectories.append({**identity, "step": point["step"],
                                 "target_prediction_tokens": point["target_prediction_tokens"],
                                 "replay_prediction_tokens": point["replay_prediction_tokens"],
                                 **endpoint_metrics(point["metrics"], baseline["corpora"])})
        # Enforce the frozen target-validation-only, first-minimum selection rule.
        best = min(points, key=lambda p: p["metrics"]["target"]["nll"])
        selected = selected_checkpoint_exposure(summary, "best")
        if selected["checkpoint_step"] != best["step"]:
            raise ValueError("best endpoint is not the first minimum of target validation")
        for kind in ("final", "best"):
            exposure = selected_checkpoint_exposure(summary, kind)
            point = next(p for p in points if p["step"] == exposure["checkpoint_step"])
            endpoints.append({**identity, "checkpoint": kind, **exposure,
                              **endpoint_metrics(point["metrics"], baseline["corpora"])})
        costs.append({**identity, **training_costs(summary),
                      "full_training_target_prediction_tokens": target,
                      "full_training_replay_prediction_tokens": counters["replay_prediction_tokens"]})
    tables = {"validation_endpoints": endpoints, "validation_trajectories": trajectories,
              "full_training_costs_once_per_run": costs,
              "validation_trajectory_statistics": trajectory_statistics(trajectories, rows)}
    for kind in ("final", "best"):
        selected = [r for r in endpoints if r["checkpoint"] == kind]
        tables[kind + "_endpoint_statistics"] = aggregate_endpoints(selected, rows)
        tables[kind + "_paired_differences"], tables[kind + "_paired_statistics"] = paired_seed_differences(selected, rows)
    availability = [{"comparison": a + "-" + b, "registered": a in groups and b in groups,
                     "reason": "registered within-model comparison" if a in groups and b in groups else
                     "one or both groups absent from this frozen model-specific registry"} for a, b in PAIRS]
    tables["comparison_availability"] = availability
    return tables, executions


def plot_extended(tables, output, *, model, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 400})
    colors = dict(zip(("B0", "B1", "B2", "B3", "B4", "D", "M", "C"), plt.get_cmap("tab10").colors))
    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(Path(output) / (name + "." + ext))  # exact 7-inch canvas; no tight bbox expansion
        plt.close(fig)
    for kind in ("final", "best"):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
        rows = [r for r in tables["validation_endpoints"] if r["checkpoint"] == kind]
        for ax, metric, label in zip(axes, ("f_old_c4", "f_old_wikitext"), ("C4", "WikiText-103")):
            for group in sorted({r["group"] for r in rows}):
                selected = [r for r in rows if r["group"] == group]
                x, y = np.array([r["g_new"] for r in selected]), np.array([r[metric] for r in selected])
                ax.scatter(x, y, s=13, alpha=.35, color=colors[group])
                ax.errorbar(x.mean(), y.mean(), xerr=x.std(ddof=1), yerr=y.std(ddof=1),
                            marker="o", ms=4, capsize=2, color=colors[group], label=group, linestyle="none")
            ax.axhline(0, color=".6", lw=.5); ax.axvline(0, color=".6", lw=.5)
            ax.set(xlabel="Target validation gain (nats/token)", ylabel=f"{label} validation forgetting")
        axes[0].legend(ncol=4, fontsize=6, loc="best")
        fig.suptitle(f"{model} | {kind} | mean ± seed SD (n=3)")
        save(fig, kind + "_validation_frontier_7in")
        fig, axes = plt.subplots(1, 3, figsize=(7, 3.0))
        stats = tables[kind + "_paired_statistics"]
        comparisons = sorted({r["comparison"] for r in stats})
        for ax, metric, label in zip(axes, ("g_new", "f_old_c4", "f_old_wikitext"), ("Target gain", "C4 forgetting", "Wiki forgetting")):
            lookup = {r["comparison"]:r for r in stats if r["metric"] == metric}
            for index, comparison in enumerate(comparisons):
                r = lookup[comparison]
                ax.errorbar(r["mean"], index, xerr=r["std_ddof1"], fmt="o", ms=4, capsize=2)
            ax.axvline(0, color=".5", lw=.7)
            ax.set(yticks=range(len(comparisons)), yticklabels=comparisons, title=label, xlabel="Paired A−B (nats/token)")
        fig.suptitle(f"{model} | {kind} | same-seed mean ± SD (n=3)")
        save(fig, kind + "_validation_paired_7in")
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))
    stats = tables["validation_trajectory_statistics"]
    for ax, metric, label in zip(axes, ("g_new", "f_old_c4", "f_old_wikitext"), ("Target gain", "C4 forgetting", "Wiki forgetting")):
        for group in sorted({r["group"] for r in stats}):
            current = sorted([r for r in stats if r["group"] == group and r["metric"] == metric], key=lambda r:r["target_prediction_tokens"])
            x = np.array([r["target_prediction_tokens"] / 1e6 for r in current]); y = np.array([r["mean"] for r in current])
            sd = np.array([r["std_ddof1"] for r in current])
            ax.plot(x, y, lw=1, color=colors[group], label=group)
            ax.fill_between(x, y-sd, y+sd, color=colors[group], alpha=.12)
        ax.set(title=label, xlabel="Target exposure (M tokens)", ylabel="nats/token")
    axes[0].legend(ncol=4, fontsize=5)
    fig.suptitle(f"{model} | extended validation trajectories | horizon {target / 1e6:g}M")
    save(fig, "validation_trajectories_7in")


def write_extended_analysis(output, tables, executions, *, phase, model, sources, target, numerical_null=None, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(output / (name + ".csv"), rows)
    write_json(output / "execution_provenance.json", executions)
    lines = [f"# {model} 扩展实验验证结果", "", f"独立 phase：`{phase}`；数据为 extended_v1；最终目标曝光 {target:,}。",
             "只报告该模型内冻结条件的验证结果。扩展训练含主训练前缀，扩展评估与主评估共享 hash partition 且并非独立，不能称新的独立测试；不同模型、horizon、数据之间不作因果尺度比较。",
             "", "G 为初始数学 NLL 减适应后 NLL，F 为适应后旧域 NLL 减初始 NLL。正 G 表示适应，正 F 表示遗忘。误差为三个训练 seed 的样本 SD（ddof=1）。",
             "best 仅按目标验证集选择，和 final 分开；best 成对比较若曝光不同不能解释为等预算。训练成本按每个完整运行计一次，teacher 子计时嵌套于父项，不重加。",
             "逐文档指标仅在原验证记录包含对齐 chunk 贡献时可得；空值不补造文档置信区间。未读取本补充测试 NLL 或任何行为分数。", ""]
    for kind in ("final", "best"):
        lines += [f"## {kind}", "", "|方法|数学 G（均值±SD）|C4 F|Wiki F|目标曝光范围|", "|---|---:|---:|---:|---:|"]
        stats = {(r["group"], r["metric"]): r for r in tables[kind + "_endpoint_statistics"]}
        def fmt(group, metric):
            item = stats[group, metric]
            return f"{item['mean']:+.6f} ± {item['std_ddof1']:.6f}"
        for group in sorted({r["group"] for r in tables["validation_endpoints"]}):
            exposure = [r["target_prediction_tokens"] for r in tables["validation_endpoints"] if r["group"] == group and r["checkpoint"] == kind]
            lines.append(f"|{group}|{fmt(group,'g_new')}|{fmt(group,'f_old_c4')}|{fmt(group,'f_old_wikitext')}|{min(exposure):,}–{max(exposure):,}|")
        lines += ["", "全部预定且本模型存在的成对差异见 paired CSV，差值为 A−B，先逐 seed 相减再汇总。缺少的方法组合在 comparison_availability.csv 显式说明。", ""]
    if numerical_null:
        lines += ["同库 D/B4 审计为数值一致性诊断，不能称去重效果。实际证据：", "```json", json.dumps(numerical_null, ensure_ascii=False, indent=2), "```"]
    (output / "验证结果说明.md").write_text("\n".join(lines) + "\n")
    if plots:
        plot_extended(tables, output, model=model, target=target)
    manifest = {"schema_version": 1, "status": "complete", "phase": phase, "role": "supplement_C_extended",
                "model": model, "data_family": "extended_v1", "split": "validation", "replicate_unit": "training_seed",
                "training_seeds": [11,22,33], "target_prediction_tokens_final": target,
                "logical_endpoint_count": len(tables["validation_endpoints"]), "sources": sources,
                "outputs": [proof(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "analysis_manifest.json"],
                "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "test_or_behavior_scores_read": False, "gpu_work_started": False}
    write_json(output / "analysis_manifest.json", manifest)
    return manifest
