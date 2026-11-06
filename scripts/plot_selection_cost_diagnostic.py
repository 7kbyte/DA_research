#!/usr/bin/env python3
"""Plot recorded preparation measurements only; never launch model inference."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    project=Path(__file__).resolve().parents[1]
    summary_path=project/"results/selection_cost_diagnostic_summary.json"
    summary=json.loads(summary_path.read_text())
    measurement_path=project/summary["measurement"]
    if hashlib.sha256(measurement_path.read_bytes()).hexdigest()!=summary["measurement_sha256"]:
        raise ValueError("diagnostic measurement SHA differs from summary")
    raw=json.loads(measurement_path.read_text())
    if not summary["complete"] or not summary["all_recomputed_tensors_exact_equal_to_frozen_stats"]:
        raise ValueError("complete exactly matched diagnostic required")
    modes=("nll_only","feature_only","joint_frozen_pipeline")
    labels=("M: NLL only","C: head-free features","Shared: joint")
    colors=("#D55E00","#CC79A7","#0072B2")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,
                         "svg.fonttype":"none","savefig.dpi":320,"axes.spines.top":False,
                         "axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.6))
    for i,mode in enumerate(modes):
        trials=raw["modes"][mode]["trials"]
        values=np.asarray([t["wall_seconds"] for t in trials])
        median=float(np.median(values))
        if not np.isclose(median,summary["modes"][mode]["median_wall_seconds"],rtol=0,atol=1e-12):
            raise ValueError("summary/trial median mismatch")
        axes[0].bar(i,median,color=colors[i],alpha=.8,width=.65)
        axes[0].scatter(i+np.linspace(-.09,.09,len(values)),values,s=15,color="black",zorder=4)
        axes[0].text(i,median+.17,f"{median:.2f}",ha="center",va="bottom",fontsize=8)
        allocated=summary["modes"][mode]["peak_gpu_allocated_bytes_including_model"]/1024**3
        reserved=summary["modes"][mode]["peak_gpu_reserved_bytes_including_model"]/1024**3
        axes[1].bar(i-.17,allocated,color=colors[i],width=.32)
        axes[1].bar(i+.17,reserved,color=colors[i],alpha=.32,width=.32,hatch="//")
    for ax in axes:
        ax.set_xticks(range(3),labels,fontsize=8);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    axes[0].set_ylabel("Whole-pool wall time (seconds)")
    axes[0].set_title("Median and 3 measurement repeats")
    axes[0].set_ylim(0,10.2)
    axes[1].set_ylabel("Peak GPU memory including model (GiB)")
    axes[1].set_title("Allocated (solid) / reserved (hatched)")
    fig.suptitle("Preparation diagnostic: Pythia-410M, H100, pool seed 11, batch 16",fontsize=10)
    fig.text(.5,.015,"Statistics pass only; excludes generation, loading, warmup and CPU selection. Repeats are not training seeds.",
             ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.055,1,.92))
    output=project/"results/method_illustrations";output.mkdir(parents=True,exist_ok=True)
    for ext in ("png","pdf","svg"):fig.savefig(output/f"selection_statistics_cost.{ext}",bbox_inches="tight")
    plt.close(fig)
    provenance={"summary":str(summary_path.relative_to(project)),
                "summary_sha256":hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                "measurement":summary["measurement"],"measurement_sha256":summary["measurement_sha256"],
                "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "model_or_test_inference_performed_by_plotter":False,
                "repetition_unit":"measurement repeat on one fixed pool, not training seed",
                "not_additive":"Do not add shared joint to per-method statistics modes."}
    (output/"selection_statistics_cost.json").write_text(json.dumps(provenance,indent=2)+"\n")


if __name__=="__main__":main()
