"""Render authorized A-only tables from completed immutable analysis CSVs.

This display-only adapter neither selects a lambda nor resamples any examples.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


METHODS = ("B1", "B2", "B3", "B4", "M", "C")
PAIRS = ("M-B4", "C-B4", "B2-B1", "B4-B3", "B3-B1", "B4-B2")
METRICS = ("g_new", "f_old_c4", "f_old_wikitext")
LABELS = ("G_new", "F_C4", "F_Wiki")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(headers, rows):
    return "\n".join(["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"] + ["|" + "|".join(map(str, row)) + "|" for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-complete", type=Path, required=True)
    parser.add_argument("--watch-complete-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.watch_complete) != args.watch_complete_sha256:
        raise ValueError("Watcher completion SHA mismatch")
    watch = json.loads(args.watch_complete.read_text())
    if watch.get("status") != "completed" or watch.get("scope") != "ONLY_A_NLL_full126_and_validation_selected18" or watch.get("returncode") != 0:
        raise ValueError("Full A-only authorized completion is required")
    complete_ref = watch["analysis_complete"]
    if sha(complete_ref["path"]) != complete_ref["sha256"]:
        raise ValueError("Analysis completion changed")
    complete = json.loads(Path(complete_ref["path"]).read_text())
    for key in ("plan", "authority", "readiness", "first_read_receipt"):
        ref = complete[key]
        if sha(ref["path"]) != ref["sha256"]:
            raise ValueError(f"{key} changed")
    ready = json.loads(Path(complete["readiness"]["path"]).read_text())
    if not complete["complete"] or not ready["ready"] or ready["closure_job_count"] != 145:
        raise ValueError("Full phase gate is not complete")
    root = Path(complete_ref["path"]).parent
    sources = {str(args.watch_complete.resolve()): sha(args.watch_complete)}

    def load(relative):
        path = root / relative
        sources[str(path.resolve())] = sha(path)
        with path.open() as handle:
            return list(csv.DictReader(handle))

    endpoints = load("final/endpoints.csv")
    curves = load("final/endpoint_statistics.csv")
    paired = load("final/paired_statistics.csv")
    paired_individual = load("final/paired_effects.csv")
    selected = load("validation_selected_policy/selected_endpoints.csv")
    selected_stats = load("validation_selected_policy/selected_policy_statistics.csv")
    selected_pairs = load("validation_selected_policy/selected_policy_pairs.csv")
    bootstrap = load("validation_selected_policy/selected_document_bootstrap.csv")
    if len(endpoints) != 126 or len(selected) != 18 or len(bootstrap) != 108:
        raise ValueError("Unexpected completed output cardinality")
    if {x["group"] for x in selected} != set(METHODS):
        raise ValueError("Selected methods changed")
    for x in endpoints:
        if (x["role"], x["model_tag"], x["data_family"], x["checkpoint"]) != ("supplement_A", "pythia-410m", "main_v3", "final"):
            raise ValueError("Mixed phase/model/data/endpoint")
        if (int(x["target_prediction_tokens"]), int(x["replay_prediction_tokens"])) != (10000000, 500000):
            raise ValueError("Unexpected actual checkpoint exposure")

    curve_index = {(x["group"], float(x["weight"]), x["metric"]): x for x in curves}
    same_index = {(x["comparison"], x["metric"]): x for x in paired if float(x["weight"]) == .05}
    tuned_index = {(x["comparison"], x["metric"]): x for x in selected_stats}
    weights = sorted({float(x["weight"]) for x in endpoints})
    if len(weights) != 7 or len(curve_index) != 630:
        raise ValueError("Incomplete 7 lambda x 6 method grid")

    # Verify displayed paired moments against the already frozen individual draws.
    for summary, individuals in ((same_index, [x for x in paired_individual if float(x["weight"]) == .05]), (tuned_index, selected_pairs)):
        for pair in PAIRS:
            points = [x for x in individuals if x["comparison"] == pair]
            if {int(x["training_seed"]) for x in points} != {11, 22, 33}:
                raise ValueError("Incomplete same-seed comparison")
            for metric in METRICS:
                numbers = [float(x[metric]) for x in points]
                row = summary[pair, metric]
                if not math.isclose(statistics.mean(numbers), float(row["mean"]), abs_tol=1e-12):
                    raise ValueError("Paired mean mismatch")
                if not math.isclose(statistics.stdev(numbers), float(row["std_ddof1"]), abs_tol=1e-12):
                    raise ValueError("Paired SD mismatch")

    intervals = []
    for x in bootstrap:
        if x["field"] != "sum_nll":
            continue
        row = dict(x)
        target = x["corpus"] == "target"
        row["display_metric"] = {"target": "G_new", "c4": "F_C4", "wikitext": "F_Wiki"}[x["corpus"]]
        for weighting in ("token_weighted", "equal_document"):
            estimate, low, high = (float(x[weighting + suffix]) for suffix in ("_estimate", "_ci_low", "_ci_high"))
            if target:
                estimate, low, high = -estimate, -high, -low
            row[weighting + "_display_estimate"] = estimate
            row[weighting + "_display_ci_low"] = low
            row[weighting + "_display_ci_high"] = high
            # Same pair/doc and sign must reproduce the existing point estimate.
            metric = METRICS[LABELS.index(row["display_metric"])]
            if weighting == "equal_document":
                metric += "_equal_document"
            matching = [p for p in selected_pairs if p["comparison"] == x["comparison"] and p["training_seed"] == x["training_seed"]]
            if len(matching) != 1 or not math.isclose(estimate, float(matching[0][metric]), abs_tol=1e-10):
                raise ValueError("Conditional interval sign/point estimate mismatch")
        intervals.append(row)
    if len(intervals) != 54:
        raise ValueError("Incomplete NLL conditional intervals")

    out = args.output_root.resolve()
    if out.exists():
        raise FileExistsError("Appendix output is immutable; choose a new path")
    out.mkdir(parents=True)

    def write_csv(name, rows):
        with (out / name).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("selected_nll_token_equal_document_conditional_intervals.csv", intervals)
    write_csv("common_lambda_0p05_paired_statistics.csv", [x for x in paired if float(x["weight"]) == .05])
    write_csv("common_lambda_0p05_same_seed_pairs.csv", [x for x in paired_individual if float(x["weight"]) == .05])
    write_csv("all_42_candidates_statistics.csv", curves)
    for index, weighting in enumerate(("token", "equal_document")):
        rows = []
        for method in METHODS:
            points = [x for x in selected if x["group"] == method]
            row = {"method": method, "lambda": points[0]["weight"], "weighting": weighting}
            for metric in METRICS:
                key = metric if index == 0 else metric + "_equal_document"
                values = [float(x[key]) for x in points]
                row[metric + "_mean"] = statistics.mean(values)
                row[metric + "_sample_sd"] = statistics.stdev(values)
            rows.append(row)
        write_csv("selected_policy_" + weighting + "_seed_statistics.csv", rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(7, 6.6), sharey=True)
    for axis, metric, label in zip(axes, METRICS, LABELS):
        for policy, stats, individual, shift, color in (
            ("Common lambda=0.05", same_index, [x for x in paired_individual if float(x["weight"]) == .05], -.13, "#0072B2"),
            ("Each method's validation-selected lambda", tuned_index, selected_pairs, .13, "#D55E00"),
        ):
            for y, pair in enumerate(PAIRS):
                row = stats[pair, metric]
                axis.errorbar(float(row["mean"]), y + shift, xerr=float(row["std_ddof1"]), fmt="_", markersize=8, capsize=2, color=color, label=policy if y == 0 else None)
                points = sorted([x for x in individual if x["comparison"] == pair], key=lambda x: int(x["training_seed"]))
                for j, (point, marker) in enumerate(zip(points, ("o", "s", "^"))):
                    axis.scatter(float(point[metric]), y + shift + (j - 1) * .045, marker=marker, s=12, alpha=.8, color=color, zorder=3)
        axis.axvline(0, color="0.45", linewidth=.7)
        axis.set_yticks(range(6), PAIRS)
        axis.invert_yaxis()
        axis.set_xlabel("Paired " + label + " (left minus right)", fontsize=9)
        axis.tick_params(labelsize=8)
        axis.grid(axis="x", alpha=.2)
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, 1.4), fontsize=8, ncol=1, frameon=False)
    fig.suptitle("Supplement A: all 3 paired seeds; mean +/- sample SD\nSeed markers: 11 circle, 22 square, 33 triangle", fontsize=10, y=1)
    fig.tight_layout(rect=(0, 0, 1, .96))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / ("common_vs_validation_selected_paired." + extension), dpi=200)
    plt.close(fig)

    def fmt(row):
        return f"{float(row['mean']):.5f} ± {float(row['std_ddof1']):.5f}"

    lines = ["# 补充A专用附表：完整λ曲线、共有λ对照与条件区间", "", "本附表仅展示已获授权且完整通过145项依赖闭包的A-NLL。原始first-read与统计产物保持原字节；18个主实验复用结果此前已见。全部选择沿用冻结validation，未按test重选λ，也不改变B/C条件。行为与B/C评分继续封存。", "", "每个final端点为实际10,000,000 target与500,000 replay prediction tokens，b=.05，LR=3e-5，训练seed11/22/33。以下均为三seed均值±样本SD；G_new越大越好，F越小越好。", "", "## 全42候选：不按test筛选λ", ""]
    for weight in weights:
        lines += [f"### λ={weight:g}", "", table(["方法", "G_new", "F_C4", "F_Wiki"], [[method] + [fmt(curve_index[method, weight, metric]) for metric in METRICS] for method in METHODS]), ""]
    lines += ["## 共有λ=.05：预定六组同seed对照", "", "这些是同一共有λ下的配对，保留与原主实验相同的解释；不能与各自选择λ的策略对照混写。", "", table(["左−右", "ΔG_new", "ΔF_C4", "ΔF_Wiki"], [[pair] + [fmt(same_index[pair, metric]) for metric in METRICS] for pair in PAIRS]), "", "## 各自验证选λ后的政策对照", "", "固定选择为B1=.05、B2=1、B3=.1、B4=5、M=5、C=1。所选18点全部从已评126端点复用，新增forward为0。两边method与λ可能共同变化，应称“各自验证选λ后的策略比较”，不能解释为固定λ的objective效应。", "", table(["左−右", "ΔG_new", "ΔF_C4", "ΔF_Wiki"], [[pair] + [fmt(tuned_index[pair, metric]) for metric in METRICS] for pair in PAIRS]), "", f"![全部三个配对seed与均值±样本SD]({out / 'common_vs_validation_selected_paired.png'})", "", "图7×6.6in，CSV与PNG/PDF/SVG配套。共同λ与所选策略都显示全部seed，不将条件bootstrap区间混作训练seed SD。", "", "## Token加权与等文档加权", "", "主NLL为所有有效prediction token的sum_nll/valid_tokens；等文档结果先对每文档计算token平均，再对文档等权。两者是不同estimand，不能把长文档权重变化称为新训练重复。target、C4、WikiText分别为466/595/60文档、523264/523264/270319有效prediction tokens。", ""]
    for weighting in ("token", "equal_document"):
        rows = list(csv.DictReader((out / ("selected_policy_" + weighting + "_seed_statistics.csv")).open()))
        lines += [("### Token加权所选策略" if weighting == "token" else "### 等文档加权所选策略"), "", table(["方法", "λ", "G_new", "F_C4", "F_Wiki"], [[x["method"], x["lambda"]] + [f"{float(x[m + '_mean']):.5f} ± {float(x[m + '_sample_sd']):.5f}" for m in METRICS] for x in rows]), ""]
    lines += ["## 所选策略的逐seed条件95%区间", "", "以下全部54个NLL比较区间直接转写现有5000次paired-document bootstrap（seed20261106）；无新抽样。target NLL左−右在转换到ΔG_new时反号并交换区间端点。C4/WikiText保留左−右符号。原始KL及NLL共108行区间仍在原CSV。", "", "区间条件于已训练checkpoint与这份评价文档集，依赖文档独立/可交换重采样假设；同源或主题相关性可能使总体覆盖含义不成立。它不是训练随机性、重复选参流程或独立目标总体的置信区间，亦未作多重比较校正。三seed样本SD与个体值独立保留，不能把文档当训练replicates。", ""]
    for metric in LABELS:
        rows = []
        for pair in PAIRS:
            points = sorted([x for x in intervals if x["comparison"] == pair and x["display_metric"] == metric], key=lambda x: int(x["training_seed"]))
            for point in points:
                cells = [point["comparison"], point["training_seed"]]
                for weighting in ("token_weighted", "equal_document"):
                    cells.append(f"{point[weighting + '_display_estimate']:.5f} [{point[weighting + '_display_ci_low']:.5f}, {point[weighting + '_display_ci_high']:.5f}]")
                rows.append(cells)
        lines += ["### Δ" + metric, "", table(["左−右", "seed", "Token估计 [95%区间]", "等文档估计 [95%区间]"], rows), ""]
    lines += ["## 可供正文使用的描述边界", "", "完整曲线显示CE回放的B1/B3在大λ下目标增益转负且通用域NLL增量扩大；KL回放的B2/B4/M/C曲线形状不同。此为已冻结网格的描述，不据此重设λ或新增训练。验证所选B4的G_new为0.07574±0.00102，F_C4为0.02435±0.00081、F_Wiki为0.05271±0.00141。M−B4所选策略的ΔG_new为−0.00166±0.00348；C−B4为−0.00241±0.00269，并未显示三seed下稳定全面优于随机库。C4与WikiText的取舍分别列出，不能宣称行为能力或总体retention。", "", "表格中的策略差与固定λ差需分别使用。例如B4−B3共有λ=.05的ΔG_new仅0.00121±0.01333；各自验证选择5/.1后的策略差为0.05301±0.00140，两者回答不同问题。全部同λ与全部seed均保留。", "", "本附表只是显示层；完整训练、评分占槽、CPU核验和共享准备费用见《补充A_NLL完整结果.md》及原成本CSV。scorer wall已嵌在scheduler whole-slot，teacher子计时已嵌在父计时，不能相加。", ""]
    report = out / "补充A_共有lambda与条件区间附表.md"
    report.write_text("\n".join(lines))
    outputs = {str(p): sha(p) for p in out.iterdir() if p.is_file()}
    proof = {"scope": "ONLY_authorized_A_NLL_display_adapter", "selection_recomputed": False, "bootstrap_resampled": False, "GPU_forward_calls": 0, "source_code": {str(Path(__file__).resolve()): sha(__file__)}, "sources_sha256": sources, "output_sha256": outputs, "all42_candidates": True, "selected_endpoints": 18, "NLL_conditional_interval_rows": 54, "prior_main18_were_previously_seen": True, "other_scores_remain_sealed": True}
    (out / "appendix_provenance.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(report), "report_sha256": sha(report), "provenance_sha256": sha(out / "appendix_provenance.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
