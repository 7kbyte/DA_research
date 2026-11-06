#!/usr/bin/env python3
"""Seven-inch primary NLL figures from complete, already-unsealed analysis only."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))


def table(path):
    with Path(path).open() as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        for key,value in row.items():
            if value=="":row[key]=None
            else:
                try:row[key]=float(value)
                except (ValueError,TypeError):pass
    return rows


def figures(root):
    scope=json.loads((root/"scope_and_costs.json").read_text())
    if scope.get("status")!="complete_primary_only" or scope.get("physical_jobs")!=67 or scope.get("bootstrap_replicates")!=5000:
        raise ValueError("complete, explicitly unsealed primary-only analysis required")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from replay_study.nll_v2_analysis import plot_frontier
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.labelsize":8,
                         "xtick.labelsize":7,"ytick.labelsize":7,"pdf.fonttype":42,"ps.fonttype":42})
    comparisons=("M-B4","C-B4","B2-B1","B4-B3","B3-B1","B4-B2")
    for endpoint in ("final","best"):
        out=root/endpoint
        meta=json.loads((out/"analysis_manifest.json").read_text())
        identity=meta["identity"]
        if (meta.get("complete"),meta.get("expected_conditions"),identity.get("role"),identity.get("data_family"),identity.get("checkpoint"))!=(
                True,36,"primary","main_v3",endpoint):raise ValueError("foreign or incomplete analysis stratum")
        stats=table(out/"endpoint_statistics.csv");pairs=table(out/"paired_statistics.csv")
        rawpairs=table(out/"paired_effects.csv")
        plot_frontier(dict(identity=identity,complete=True,replicate_unit="training_seed",
                           endpoints=table(out/"endpoints.csv"),all_candidates_frontier=table(out/"all_candidates_frontier.csv")),
                      out,figure_width=7)
        # Six original comparisons at b=lambda=.05; points are paired trained seeds.
        fig,axes=plt.subplots(1,3,figsize=(7,3.25),sharey=True,constrained_layout=True)
        for ax,metric,label in zip(axes,("g_new","f_old_c4","f_old_wikitext"),
                                  ("Delta adaptation gain","Delta retention loss: C4","Delta retention loss: Wiki")):
            for i,comparison in enumerate(comparisons):
                item=next(r for r in pairs if r["comparison"]==comparison and r["metric"]==metric and r["ratio"]==.05 and r["weight"]==.05)
                values=[r[metric] for r in rawpairs if r["comparison"]==comparison and r["ratio"]==.05 and r["weight"]==.05]
                if item["n"]!=3 or len(values)!=3:raise ValueError("three paired seeds required for every original contrast")
                ax.scatter(values,[i]*len(values),s=13,color="#007d9c",alpha=.55,zorder=3)
                ax.errorbar(item["mean"],i,xerr=item["std_ddof1"],fmt="s",color="black",capsize=3,ms=3,zorder=5)
            ax.axvline(0,color="grey",linewidth=.7,linestyle=":")
            ax.set_xlabel(label+"\n(nats / prediction token)")
            ax.set_yticks(range(len(comparisons)),comparisons)
            ax.grid(axis="x",alpha=.2);ax.spines[["top","right"]].set_visible(False)
        axes[0].invert_yaxis()
        fig.suptitle(f"Primary paired contrasts | {endpoint} | b = lambda = 0.05\nPoints: 3 seeds; squares: paired mean +/- sample SD",fontsize=9)
        for ext in ("png","pdf","svg"):fig.savefig(out/("paired_contrasts_7in."+ext),dpi=400)
        plt.close(fig)
        # Original b sensitivity: lambda=.05, all budgets retained, no test selection.
        fig,axes=plt.subplots(1,3,figsize=(7,2.75),constrained_layout=True)
        for ax,metric,label in zip(axes,("g_new","f_old_c4","f_old_wikitext"),("Adaptation gain","Retention loss: C4","Retention loss: Wiki")):
            for group,color in (("B4","#007d9c"),("M","#b35a00")):
                rows=sorted((r for r in stats if r["group"]==group and r["metric"]==metric and r["weight"]==.05),key=lambda r:r["ratio"])
                if [r["ratio"] for r in rows]!=[.01,.05,.1] or any(r["n"]!=3 for r in rows):raise ValueError("complete three-budget sensitivity required")
                ax.errorbar([r["ratio"] for r in rows],[r["mean"] for r in rows],yerr=[r["std_ddof1"] for r in rows],
                            fmt="o-",color=color,label=group,capsize=3,ms=3,linewidth=.9)
            ax.set_xticks([.01,.05,.1]);ax.set_xlabel("Replay exposure ratio b");ax.set_ylabel(label+"\n(nats / prediction token)")
            ax.grid(alpha=.2);ax.spines[["top","right"]].set_visible(False)
        axes[-1].legend(frameon=False,fontsize=8)
        title="Final: fixed 10M target exposure" if endpoint=="final" else "Validation-selected best: actual exposure may differ"
        fig.suptitle(title+" | lambda = 0.05 | mean +/- sample SD (n=3)",fontsize=9)
        for ext in ("png","pdf","svg"):fig.savefig(out/("budget_sensitivity_7in."+ext),dpi=400)
        plt.close(fig)
    (root/"打印图说明.md").write_text("所有PDF/SVG画布宽7英寸，PNG为400dpi。paired_contrasts为六组原定同seed配对差，均A−B，G_new正值更有利、F_old负值更有利；误差条是三seed差值的样本SD，不是文档bootstrap区间。budget_sensitivity保留全部b=.01/.05/.1及固定λ=.05，不按test选择预算。final固定10M目标曝光；best为验证选出、实际曝光可能不同，不能当等预算。D与B4的严格null/推理别名仍见完整CSV及模型等价证明，不能把D当新的独立训练重复。文档bootstrap两种权重和条件性CI另见JSON，不以这些SD图替代。\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--analysis-root",type=Path,default=Path("results/analysis/primary_nll_v2"));a=p.parse_args();figures(a.analysis_root)
