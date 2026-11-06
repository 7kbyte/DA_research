"""Presentation only: plot existing frozen A-gradient CSVs, no statistics/log read."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "results/A_gradient_clipping_posttest_v1/analysis"
COMPLETE_SHA = "d424e859a4857c6f2050b6036a3c34c6540f68568e3dcd69398249a4a42a9c5b"
CSV_SHAS = {
    "per_run_windows.csv": "b43e39a977b2f6d0978c92012c207e0455080c31439b89fb5fca5096df0ec6d6",
    "method_lambda_windows.csv": "3e195cfa3662a86c3a1427ab2433fdec672b4874a8f6ca5a49dc920cf4521740",
}
METHODS = ("B1", "B2", "B3", "B4", "M", "C")
LAMBDAS = (.01, .05, .1, .5, 1., 5., 10.)


def require(condition, message):
    if not condition: raise ValueError(message)


def proof(path):
    path = Path(path).resolve(); digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""): digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def file_stats(tree):
    """Metadata only: do not reopen archived logs/source bytes."""
    result = {}
    for path in sorted(tree.rglob("*")):
        if path.is_file():
            info = path.stat()
            result[str(path)] = {key: getattr(info, "st_" + key) for key in ("dev", "ino", "size", "mtime_ns", "ctime_ns")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(k) == "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Explicit empty CVD/two CPU threads required")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    source_stat_before = file_stats(ORIGINAL / "source_bytes")
    complete_proof = proof(ORIGINAL / "COMPLETE.json"); require(complete_proof["sha256"] == COMPLETE_SHA, "Original COMPLETE changed")
    complete = json.loads((ORIGINAL / "COMPLETE.json").read_text())
    original_proofs = []
    for item in complete["output_proofs_before_complete"]:
        path = Path(item["path"])
        require(path.parent == ORIGINAL, "Original proof outside direct analysis outputs")
        actual = proof(path); require(actual["sha256"] == item["sha256"], "Original output changed")
        original_proofs.append(actual)
    require(complete["logical_conditions"] == 126 and complete["train_events"] == 38556 and complete["total_steps_strictly_above_threshold"] == 38556
            and complete["all_run_window_fraction_range"] == [1., 1.] and complete["window_step_counts"] == {"all": [306], "early": [61], "late": [62]}, "Frozen shared information differs")
    csv_proofs = [proof(ORIGINAL / name) for name in CSV_SHAS]
    require(all(item["sha256"] == CSV_SHAS[Path(item["path"]).name] for item in csv_proofs), "Frozen numeric CSV changed")
    individuals = list(csv.DictReader((ORIGINAL / "per_run_windows.csv").open()))
    groups = list(csv.DictReader((ORIGINAL / "method_lambda_windows.csv").open()))
    require(len(individuals) == 378 and len(groups) == 126, "Existing CSV scope differs")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(METHODS, ("#377eb8", "#ff7f00", "#4daf4a", "#e41a1c", "#984ea3", "#8c564b")))
    fig, axes = plt.subplots(3, 1, figsize=(7, 6.6), sharex=True, sharey=True)
    fig.suptitle("A gradient logs: post-test descriptive diagnostic", fontsize=10, y=.982)
    shared = ("ALL 126 runs / 38,556 logged updates / configured threshold = 1\n"
              "Strict norm > threshold: 100% in ALL / EARLY / LATE for every run\n"
              "Updates per run: ALL 306 | EARLY (target ≤ 2M) 61 | LATE (target > 8M) 62")
    fig.text(.5, .922, shared, ha="center", va="center", fontsize=8,
             bbox={"boxstyle": "round,pad=.45", "facecolor": "#f2f4f6", "edgecolor": "#999999", "linewidth": .6})
    labels = {"all": "All: 306 updates / run", "early": "Early: target ≤ 2M (61 updates / run)", "late": "Late: target > 8M (62 updates / run)"}
    for ax, window in zip(axes, ("all", "early", "late")):
        for method in METHODS:
            selected = sorted([r for r in groups if r["method"] == method and r["window"] == window], key=lambda r: float(r["lambda"]))
            require(len(selected) == 7 and tuple(float(r["lambda"]) for r in selected) == LAMBDAS, "All lambda cells required")
            require(all(.85 < float(r["mean_mean"]) - float(r["mean_sample_sd"]) and float(r["mean_mean"]) + float(r["mean_sample_sd"]) < 500 for r in selected), "Shared limits would clip precomputed SD bars")
            ax.errorbar([float(r["lambda"]) for r in selected], [float(r["mean_mean"]) for r in selected],
                        yerr=[float(r["mean_sample_sd"]) for r in selected], color=colors[method], marker="o", mfc="white", ms=3,
                        lw=1., elinewidth=.7, capsize=2, label=method, zorder=2)
            for seed, marker in ((11, "."), (22, "x"), (33, "+")):
                points = sorted([r for r in individuals if r["method"] == method and r["window"] == window and int(r["pipeline_seed"]) == seed], key=lambda r: float(r["lambda"]))
                require(len(points) == 7, "All seed points required")
                require(all(.85 < float(r["mean"]) < 500 for r in points), "Shared limits would hide seed points")
                ax.scatter([float(r["lambda"]) for r in points], [float(r["mean"]) for r in points],
                           color=colors[method], marker=marker, s=13, alpha=.75, linewidths=.7, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(.85, 500)
        ax.set_xticks(LAMBDAS, labels=(".01", ".05", ".1", ".5", "1", "5", "10"))
        ax.axhline(1., color=".4", ls="--", lw=.6)
        ax.text(.018, .91, labels[window], transform=ax.transAxes, fontsize=8, va="top")
        ax.set_ylabel("Mean preclip L2 norm\n(shared log scale)", fontsize=8)
        ax.tick_params(labelsize=8); ax.grid(alpha=.2, linewidth=.5)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(.53, .857))
    axes[-1].set_xlabel("Replay loss weight λ (log scale)", fontsize=8)
    fig.text(.5, .037, "All 3 seeds (11: dot; 22: ×; 33: +) and mean ± sample SD; λ/windows share random sources.\nTotal gradient norm does not determine Adam update magnitude/direction or a causal mechanism.", ha="center", fontsize=7)
    fig.subplots_adjust(left=.112, right=.982, top=.802, bottom=.135, hspace=.20)
    for suffix in ("png", "pdf", "svg"): fig.savefig(output / ("gradient_diagnostic_v2." + suffix), dpi=220)
    plt.close(fig)
    require(file_stats(ORIGINAL / "source_bytes") == source_stat_before, "Original archived source/log metadata changed")
    require(proof(ORIGINAL / "COMPLETE.json") == complete_proof and all(proof(item["path"]) == item for item in original_proofs), "Original analysis output bytes changed")
    source = proof(__file__)
    with (output / "producer.py").open("xb") as handle: handle.write(Path(__file__).read_bytes())
    result = {"schema_version": 1, "status": "complete_presentation_only_v2_no_reanalysis", "producer": source,
              "original_COMPLETE": complete_proof, "unchanged_original_direct_output_proofs": original_proofs,
              "numeric_CSV_inputs": csv_proofs, "all_original_numeric_CSV_SHA_unchanged": True,
              "original_source_bytes_tree_stat_unchanged": True, "original_source_bytes_files": len(source_stat_before),
              "raw_training_logs_read": False, "statistics_recomputed": False, "source_only_reads_are_original_numeric_CSV_and_direct_output_integrity_checks": True,
              "layout": {"inches": [7, 6.6], "panels": "Shared 100-percent information box plus 3 full-width norm panels", "shared_norm_y_range": [.85, 500], "all_lambda_method_seed_points_retained": True},
              "scope": "All 126 original A runs; only plot existing per-run mean and precomputed 3seed mean/sample SD",
              "figures": [proof(output / ("gradient_diagnostic_v2." + suffix)) for suffix in ("png", "pdf", "svg")]}
    with (output / "presentation_receipt.json").open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(output)}))


if __name__ == "__main__": main()
