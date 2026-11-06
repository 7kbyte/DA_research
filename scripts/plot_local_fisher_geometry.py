#!/usr/bin/env python3
"""Draw an analytic two-context illustration, never a measured model result."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    output=Path(__file__).resolve().parents[1]/"results/method_illustrations"
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,
                         "svg.fonttype":"none","savefig.dpi":320})
    grid=np.linspace(-2,2,301);x,y=np.meshgrid(grid,grid)
    weights=(.9,.5,.1);levels=(.025,.05,.1,.2)
    fig,axes=plt.subplots(1,3,figsize=(9.4,3.4),sharex=True,sharey=True)
    records=[]
    for ax,w in zip(axes,weights):
        # At theta0=0, q(y=1|c)=1/2 for both contexts; a_c1=e1, a_c2=e2.
        fisher=np.diag([w/4,(1-w)/4])
        local=(w*x*x+(1-w)*y*y)/8
        contours=ax.contour(x,y,local,levels=levels,colors="#0072B2",linewidths=1.1)
        ax.clabel(contours,fmt="%.3g",fontsize=7)
        ax.axhline(0,color=".8",linewidth=.6);ax.axvline(0,color=".8",linewidth=.6)
        ax.annotate("",xy=(1,0),xytext=(0,0),arrowprops={"arrowstyle":"->","color":"#D55E00","lw":1.8})
        ax.set_aspect("equal");ax.set_xticks([-2,-1,0,1,2]);ax.set_yticks([-2,-1,0,1,2])
        ax.set_title(rf"$\mu(c_1)={w:.1f},\ \mu(c_2)={1-w:.1f}$"+"\n"+rf"$F_\mu=\mathrm{{diag}}({w/4:.3f},{(1-w)/4:.3f})$",fontsize=9)
        ax.set_xlabel(r"Parameter displacement $\delta_1$")
        ax.text(.5,-.3,rf"All token NLLs = $\log 2$; $R_{{quad}}(1,0)={w/8:.4f}$",
                transform=ax.transAxes,ha="center",fontsize=8)
        records.append({"context_1_weight":w,"context_2_weight":1-w,
                        "fisher_diagonal":np.diag(fisher).tolist(),"all_hard_label_nll":float(np.log(2)),
                        "quadratic_penalty_at_displacement_1_0":w/8})
    axes[0].set_ylabel(r"Parameter displacement $\delta_2$")
    fig.suptitle("Analytic illustration only: equal NLL does not imply equal KL constraint directions",fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.92))
    for ext in ("png","pdf","svg"):fig.savefig(output/f"local_fisher_geometry.{ext}",bbox_inches="tight")
    plt.close(fig)
    manifest={"scope":"analytic binary-logistic local quadratic KL; not fitted to Pythia or experimental results",
              "teacher_initial_probabilities":[.5,.5],"context_vectors":[[1,0],[0,1]],
              "initial_parameters":[0,0],"contour_levels":list(levels),"records":records,
              "model_or_test_inference_performed":False,
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/"local_fisher_geometry.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":main()
