"""Conditional selection-draw experiment; deliberately separate from primary seeds."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .analysis import (_check_summary, _protocol_status, endpoint_metrics, mean_std,
                       read_json, sha256, training_costs, write_csv,
                       verify_recorded_execution, training_validation_execution)
from .evaluate import load_tensor_data, prediction_mask
from .selection import chunk_hashes, save_bank


METHODS={"B4":"random","M":"nll_stratified","C":"cluster"}
PHASE="conditional_bank_variance"
SCOPE="selection-draw variability conditional on generation pool 11 and training shuffle 11; not independent training-seed uncertainty"


def _write(path: str | Path, value: Any) -> None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def _scientific_registry_sha(rows: Sequence[Mapping[str,Any]]) -> str:
    normalized=[{k:v for k,v in row.items() if k not in {"active","status"}} for row in sorted(rows,key=lambda r:r["id"])]
    return hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def completed_parent(registry_path: str | Path, protocol_status_path: str | Path) -> tuple[list,dict,dict]:
    """Reject provisional/partial parents before reading or selecting any pool."""
    _protocol_status(protocol_status_path,registry_path,"active")
    rows=read_json(registry_path)
    if not rows or any(r.get("active") is not True or r.get("phase")!="formal" or
                       r.get("status") in {"provisional","invalid","superseded"} for r in rows):
        raise ValueError("parent must be an explicitly active formal registry")
    chosen=[];proof={}
    for row in rows:
        if not Path(row["summary_path"]).exists():
            raise ValueError(f"parent formal registry is incomplete: {row['id']}")
        summary=read_json(row["summary_path"]);cfg=read_json(row["config_path"])
        _check_summary(summary,row)
        if (summary.get("status")!="completed" or summary["counters"]["target_prediction_tokens"]!=cfg["target_prediction_tokens"] or
                summary["counters"]["replay_prediction_tokens"]!=int(cfg["target_prediction_tokens"]*Fraction(str(cfg["replay_ratio"])))):
            raise ValueError(f"parent formal registry is incomplete or budget-invalid: {row['id']}")
        provenance=summary.get("provenance",{})
        if (cfg.get("deterministic_algorithms") is not True or provenance.get("deterministic_algorithms") is not True or
                provenance.get("deterministic_warn_only") is not False):
            raise ValueError("parent requires saved strict deterministic execution proof")
        proof[row["id"]]={"config_sha256":sha256(row["config_path"]),"summary_sha256":sha256(row["summary_path"])}
        if row["group"]=="B4" and int(row["seed"])==11 and math.isclose(float(row["ratio"]),.05):
            chosen.append((row,cfg,summary))
    if len(chosen)!=1:
        raise ValueError("exactly one completed B4 training-seed11 b=.05 parent is required")
    row,cfg,summary=chosen[0]
    if cfg.get("replay_objective")!="kl" or int(cfg["seed"])!=11:
        raise ValueError("B4 parent must use KL and fixed training seed11")
    if cfg["target_prediction_tokens"]!=10_000_000:
        raise ValueError("conditional study requires the full 10M-target-token parent protocol")
    return rows,{"row":row,"config":cfg,"summary":summary},proof


def _nll_composition(nll: np.ndarray, strata: np.ndarray, indices: Sequence[int]) -> dict[str,Any]:
    indices=np.asarray(indices,dtype=np.int64);values=nll[indices]
    counts=np.bincount(strata[indices],minlength=5)
    return {"teacher_nll_mean":float(values.mean()),"teacher_nll_std_ddof1":float(values.std(ddof=1)) if len(values)>1 else None,
            "teacher_nll_min":float(values.min()),"teacher_nll_max":float(values.max()),
            "teacher_nll_quantiles":np.quantile(values,[0,.25,.5,.75,1]).tolist(),
            "pool_nll_stratum_counts":counts.tolist(),"pool_nll_stratum_fractions":(counts/len(indices)).tolist(),
            "strata_policy":"five equal-frequency full-pool bins; stable original-index ties"}


def make_bank_variance(parent_registry: str | Path, protocol_status_path: str | Path, output: str | Path,
                       *, run_root: str | Path, result_root: str | Path,
                       bank_sizes: Sequence[int]=(20,196), selection_seeds: Sequence[int]=tuple(range(1001,1011))) -> dict[str,Any]:
    started=time.perf_counter()
    parent_rows,parent,proof=completed_parent(parent_registry,protocol_status_path)
    cfg,parent_row,parent_summary=parent["config"],parent["row"],parent["summary"]
    output=Path(output).absolute()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new bank-variance output namespace; no existing bank is replaced")
    sizes=tuple(int(k) for k in bank_sizes);draws=tuple(int(s) for s in selection_seeds)
    if not sizes or min(sizes)<1 or len(set(sizes))!=len(sizes) or not draws or len(set(draws))!=len(draws):
        raise ValueError("positive unique bank sizes and unique selection seeds required")
    source_bank=read_json(Path(cfg["replay_path"]).with_suffix(".json"))
    pool_path=Path(source_bank["source_pool"]);statistics_path=Path(source_bank["teacher_statistics_artifact"])
    pool_metadata=read_json(pool_path.with_suffix(".json"))
    if (pool_metadata.get("config",{}).get("seed",pool_metadata.get("seed"))!=11 or
            pool_metadata.get("revision")!=cfg["model_revision"] or
            pool_metadata.get("split")!="generated_training_candidates" or
            pool_metadata.get("experiment_role",pool_metadata.get("config",{}).get("role"))!="formal"):
        raise ValueError("fixed pool must be training-only generation seed11 from the same pinned teacher")
    pool_sha=sha256(pool_path)
    if source_bank.get("source_pool_sha256")!=pool_sha or pool_metadata.get("artifact_sha256",pool_sha)!=pool_sha:
        raise ValueError("parent/pool content hash mismatch")
    statistics_sha=sha256(statistics_path)
    if source_bank.get("teacher_statistics_sha256")!=statistics_sha:
        raise ValueError("teacher statistics content differs from the parent bank's frozen statistics")
    stats=torch.load(statistics_path,map_location="cpu",weights_only=False)
    if stats.get("source_pool_sha256")!=pool_sha:
        raise ValueError("teacher statistics do not match the fixed generation pool")
    if stats.get("revision")!=cfg["model_revision"]:
        raise ValueError("teacher-statistics revision differs from the pinned parent teacher")
    pool=load_tensor_data(pool_path);hashes=chunk_hashes(pool)
    if len(set(hashes))!=len(hashes):
        raise ValueError("registered fixed pool is expected exact-unique; random K would otherwise differ in unique capacity")
    ids=pool["input_ids"];mask=pool.get("attention_mask",torch.ones_like(ids,dtype=torch.bool))
    counts=prediction_mask(mask).sum(1)
    if not torch.equal(counts,counts[0].expand_as(counts)) or int(counts[0])<1 or max(sizes)>len(ids):
        raise ValueError("insufficient equal-budget source chunks")
    nll=np.asarray(stats["teacher_nll"],dtype=np.float64)
    features=np.asarray(stats.get("features",[]),dtype=np.float32)
    if (nll.ndim!=1 or len(nll)!=len(ids) or not np.isfinite(nll).all() or
            features.ndim!=2 or len(features)!=len(ids) or not np.isfinite(features).all()):
        raise ValueError("finite aligned teacher NLL and features are required")
    strata=np.empty(len(nll),dtype=np.int64)
    for layer,indices in enumerate(np.array_split(np.argsort(nll,kind="stable"),5)):
        strata[indices]=layer
    output.mkdir(parents=True,exist_ok=True);registry=[];jobs=[];records=[]
    suffix=lambda value:str(value).replace(".","p").replace("-","m")
    for size in sizes:
        for draw in draws:
            for group,method in METHODS.items():
                run_id=f"bankvar_{group}_tr11_gen11_draw{draw}_k{size}_b0p05_w{suffix(cfg['replay_weight'])}"
                bank_path=output/"banks"/f"{run_id}.pt"
                bank=save_bank(pool_path,bank_path,method=method,seed=draw,bank_size=size,statistics=statistics_path)
                if bank["bank_unique_chunks"]!=size:
                    raise AssertionError("selected unique bank capacity differs from registered K")
                composition=_nll_composition(nll,strata,bank["selected_indices"])
                bank.update(phase=PHASE,experiment_role=PHASE,training_seed=11,generation_seed=11,selection_seed=draw,
                            parent_formal_run_id=parent_row["id"],parent_registry_sha256=sha256(parent_registry),
                            conditional_scope=SCOPE,nll_composition=composition)
                _write(bank_path.with_suffix(".json"),bank)
                child=copy.deepcopy(cfg)
                child.update(seed=11,replay_path=str(bank_path),output_dir=str(Path(run_root)/run_id),
                             run_label=run_id,resume_from=None,max_steps=None)
                config_path=output/"configs"/f"{run_id}.json";_write(config_path,child)
                row={"id":run_id,"phase":PHASE,"active":False,"status":"prepared_only","group":group,
                     "training_seed":11,"generation_seed":11,"selection_seed":draw,"bank_chunks":size,
                     "unique_bank_prediction_tokens":size*int(counts[0]),"ratio":.05,"weight":cfg["replay_weight"],
                     "learning_rate":cfg["learning_rate"],"config_path":str(config_path),"bank_path":str(bank_path),
                     "summary_path":str(Path(result_root)/f"{run_id}_summary.json"),"output_dir":child["output_dir"],
                     "parent_formal_run_id":parent_row["id"],"parent_registry":str(parent_registry),
                     "parent_registry_sha256":sha256(parent_registry),"conditional_scope":SCOPE}
                registry.append(row)
                jobs.append({"id":run_id,"argv":[sys.executable,str(Path(__file__).resolve().parents[2]/"scripts/make_experiments.py"),
                                                   "run","--config",str(config_path),"--summary-copy",row["summary_path"]],
                             "env":{"CUBLAS_WORKSPACE_CONFIG":":4096:8"},
                             "outputs":[str(Path(child["output_dir"])/"summary.json"),row["summary_path"]],
                             "expected_seconds":parent_summary["counters"].get("total_accounted_wall_seconds",600)})
                records.append({"id":run_id,"bank_sha256":bank["artifact_sha256"],"bank_manifest_sha256":sha256(bank_path.with_suffix(".json")),
                                "config_sha256":sha256(config_path),"selection_wall_seconds":bank["wall_seconds"],
                                "clustering_nested_seconds":bank.get("clustering_wall_seconds",0)})
    registry_path=output/"registry.json";_write(registry_path,registry)
    _write(output/"queue_template.json",{"launch_authorized":False,"prepared_only":True,
                                         "activation_rule":"parent formal audit must pass; root explicitly authorizes supplementary training",
                                         "jobs":jobs})
    manifest={"prepared_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"phase":PHASE,"prepared_only":True,
              "training_started":False,"model_or_test_inference_performed":False,"conditional_scope":SCOPE,
              "parent_registry":str(parent_registry),"parent_registry_sha256":sha256(parent_registry),
              "parent_registry_snapshot":parent_rows,"parent_proof":proof,"parent_b4_config_snapshot":cfg,
              "fixed_pool":str(pool_path),"fixed_pool_sha256":pool_sha,"statistics":str(statistics_path),
              "statistics_sha256":statistics_sha,"bank_sizes_chunks":list(sizes),"selection_seeds":list(draws),
              "training_seed":11,"generation_seed":11,"registry_sha256":sha256(registry_path),
              "cpu_torch_threads":torch.get_num_threads(),
              "scientific_registry_sha256":_scientific_registry_sha(registry),
              "banks":records,"preparation_wall_seconds":time.perf_counter()-started,
              "cost_policy":"reuse fixed generation/statistics once; new selection measured per bank; clustering nested in selection"}
    _write(output/"preparation_manifest.json",manifest)
    return manifest


def _conditional_statistics(rows: Sequence[Mapping[str,Any]], metrics: Sequence[str], expected_n: int) -> list[dict]:
    groups=defaultdict(list)
    for row in rows:
        groups[(row["group"],row["bank_chunks"])].append(row)
    out=[]
    for (group,size),current in sorted(groups.items()):
        if len({r["selection_seed"] for r in current})!=len(current):
            raise ValueError("duplicate selection draw within method/K")
        for metric in metrics:
            values=[r[metric] for r in current if r.get(metric) is not None]
            stats=mean_std(values)
            out.append({"group":group,"bank_chunks":size,"metric":metric,**stats,
                        "expected_selection_draws":expected_n,"complete":stats["n"]==expected_n,"replicate_unit":"selection_seed",
                        "n_training_seeds":1,"n_generation_pools":1,"scope":SCOPE})
    return out


def _conditional_pairs(rows: Sequence[Mapping[str,Any]],metrics: Sequence[str]) -> list:
    lookup={(r["group"],r["bank_chunks"],r["selection_seed"]):r for r in rows};paired=[]
    if len(lookup)!=len(rows):raise ValueError("duplicate conditional draw identity")
    for group in ("M","C"):
        for (g,size,draw),a in sorted(lookup.items()):
            b=lookup.get(("B4",size,draw))
            if g!=group or b is None:continue
            row={"group":f"{group}-B4","comparison":f"{group}-B4","bank_chunks":size,"selection_seed":draw,
                 "training_seed":11,"generation_seed":11,"scope":SCOPE,"sign":"A minus B"}
            for metric in metrics:
                row[metric]=a[metric]-b[metric] if a.get(metric) is not None and b.get(metric) is not None else None
            paired.append(row)
    return paired


def analyze_bank_variance(registry_path: str | Path, baseline_path: str | Path, output: str | Path,
                          *, allow_partial: bool=False,plots: bool=True) -> dict[str,Any]:
    registry=read_json(registry_path)
    if not registry or any(r.get("phase")!=PHASE for r in registry):
        raise ValueError("conditional bank-variance registry required; never mix primary/pilot runs")
    if any(r.get("training_seed")!=11 or r.get("generation_seed")!=11 or "seed" in r for r in registry):
        raise ValueError("draw seeds must not be presented as independent training seeds")
    if len({r["id"] for r in registry})!=len(registry):raise ValueError("duplicate conditional run IDs")
    preparation=read_json(Path(registry_path).parent/"preparation_manifest.json")
    if _scientific_registry_sha(registry)!=preparation["scientific_registry_sha256"]:
        raise ValueError("conditional registry scientific contents changed since preparation")
    frozen={r["id"]:r for r in preparation["banks"]}
    expected_matrix={(g,k,s) for g in METHODS for k in preparation["bank_sizes_chunks"] for s in preparation["selection_seeds"]}
    if {(r["group"],r["bank_chunks"],r["selection_seed"]) for r in registry}!=expected_matrix:
        raise ValueError("conditional registry does not contain the complete registered selection matrix")
    expected=len({r["selection_seed"] for r in registry})
    baseline_payload=read_json(baseline_path);baseline=baseline_payload["corpora"]
    compositions=[];endpoints=[];availability=[];costs=[]
    for row in registry:
        metadata=read_json(Path(row["bank_path"]).with_suffix(".json"))
        proof=frozen[row["id"]]
        if (sha256(Path(row["bank_path"]).with_suffix(".json"))!=proof["bank_manifest_sha256"] or
                sha256(row["config_path"])!=proof["config_sha256"] or metadata["artifact_sha256"]!=proof["bank_sha256"]):
            raise ValueError("prepared bank manifest or training config changed")
        if sha256(row["bank_path"])!=metadata["artifact_sha256"]:raise ValueError("prepared bank content changed")
        if any(metadata.get(k)!=row[k] for k in ("training_seed","generation_seed","selection_seed")):
            raise ValueError("bank/registry conditional seeds differ")
        composition=metadata["nll_composition"]
        record={k:row[k] for k in ("id","group","training_seed","generation_seed","selection_seed","bank_chunks")}
        record.update({k:composition[k] for k in ("teacher_nll_mean","teacher_nll_std_ddof1","teacher_nll_min","teacher_nll_max")})
        for i,value in enumerate(composition["pool_nll_stratum_fractions"]):record[f"stratum_{i}_fraction"]=value
        record.update(selection_wall_seconds=metadata["wall_seconds"],clustering_nested_seconds=metadata.get("clustering_wall_seconds",0))
        compositions.append(record)
        summary_path=Path(row["summary_path"]);status="missing_summary"
        if summary_path.exists():
            summary=read_json(summary_path);cfg=read_json(row["config_path"])
            primary_identity={"id":row["id"],"seed":11,"ratio":row["ratio"],"weight":row["weight"],
                              "learning_rate":row["learning_rate"],"config_path":row["config_path"]}
            _check_summary(summary,primary_identity)
            costs.append({**{k:row[k] for k in ("id","group","training_seed","generation_seed","selection_seed","bank_chunks")},
                          "run_status":summary["status"],**training_costs(summary)})
            if (summary["status"]=="completed" and summary["counters"]["target_prediction_tokens"]==cfg["target_prediction_tokens"] and
                    summary["counters"]["replay_prediction_tokens"]==int(cfg["target_prediction_tokens"]*Fraction(str(cfg["replay_ratio"])))):
                points=summary.get("validation_trajectory",[])
                if not points or points[-1]["target_prediction_tokens"]!=cfg["target_prediction_tokens"]:
                    raise ValueError("conditional endpoint lacks fixed-exposure final validation")
                if baseline_payload.get("revision")!=cfg["model_revision"]:raise ValueError("conditional baseline revision mismatch")
                if not baseline_payload.get("execution_provenance"):raise ValueError("matched execution provenance required")
                verify_recorded_execution(baseline_payload["execution_provenance"],training_validation_execution(summary))
                endpoints.append({**{k:row[k] for k in ("id","group","training_seed","generation_seed","selection_seed","bank_chunks")},
                                  **endpoint_metrics(points[-1]["metrics"],baseline),"scope":SCOPE})
                status="ready"
            else:status="training_not_complete"
        availability.append({"id":row["id"],"selection_seed":row["selection_seed"],"status":status})
    complete=len(endpoints)==len(registry)
    if not complete and not allow_partial:
        raise ValueError("conditional trained results incomplete; --allow-partial required for prepared-bank diagnostics")
    composition_metrics=("teacher_nll_mean","teacher_nll_std_ddof1","teacher_nll_min","teacher_nll_max",
                         "stratum_0_fraction","stratum_1_fraction","stratum_2_fraction","stratum_3_fraction","stratum_4_fraction",
                         "selection_wall_seconds")
    metrics=("g_new","f_old_c4","f_old_wikitext","kl_target","kl_c4","kl_wikitext")
    cp=_conditional_pairs(compositions,composition_metrics);ep=_conditional_pairs(endpoints,metrics)
    tables={"bank_composition":compositions,"composition_statistics":_conditional_statistics(compositions,composition_metrics,expected),
            "paired_composition":cp,"paired_composition_statistics":_conditional_statistics(cp,composition_metrics,expected),
            "validation_endpoints":endpoints,"endpoint_statistics":_conditional_statistics(endpoints,metrics,expected),
            "paired_validation":ep,"paired_validation_statistics":_conditional_statistics(ep,metrics,expected),
            "availability":availability,"observed_run_costs":costs}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():write_csv(output/f"{name}.csv",rows)
    manifest={"phase":PHASE,"complete_trained_results":complete,"completed_draw_runs":len(endpoints),"expected_draw_runs":len(registry),
              "replicate_unit":"selection_seed","training_seeds":[11],"generation_seeds":[11],"scope":SCOPE,
              "registry_sha256":sha256(registry_path),"baseline_sha256":sha256(baseline_path),"test_scores_accessed":False}
    _write(output/"analysis_manifest.json",manifest)
    if plots:plot_bank_variance(compositions,endpoints,output,manifest)
    return manifest


def plot_bank_variance(compositions, endpoints, output, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none","savefig.dpi":320})
    colors={"B4":"#0072B2","M":"#D55E00","C":"#CC79A7"};output=Path(output)
    sizes=sorted({r["bank_chunks"] for r in compositions})
    fig,axes=plt.subplots(1,len(sizes),figsize=(4*len(sizes),3.6),squeeze=False)
    for ax,size in zip(axes[0],sizes):
        for i,group in enumerate(METHODS):
            rows=sorted([r for r in compositions if r["group"]==group and r["bank_chunks"]==size],key=lambda r:r["selection_seed"])
            values=np.asarray([r["teacher_nll_mean"] for r in rows])
            if not len(values):continue
            ax.scatter(i+np.linspace(-.12,.12,len(values)),values,color=colors[group],s=16,alpha=.45)
            ax.errorbar(i,values.mean(),yerr=values.std(ddof=1) if len(values)>1 else None,
                        marker="o",color=colors[group],capsize=4,linestyle="none")
        ax.set_title(f"K = {size} chunks");ax.set_xticks(range(3),["B4: random","M: NLL strata","C: cluster"])
        ax.set_ylabel("Bank mean frozen-teacher NLL (nats/token)");ax.grid(axis="y",alpha=.18)
    fig.suptitle("Conditional selection draws; one fixed generation pool (seed 11)",fontsize=10)
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ("png","pdf","svg"):fig.savefig(output/f"bank_nll_composition.{ext}",bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.8))
    for ax,corpus,label in zip(axes,("c4","wikitext"),("C4","WikiText-103")):
        for size in sizes:
            for group in METHODS:
                rows=[r for r in endpoints if r["group"]==group and r["bank_chunks"]==size]
                if not rows:continue
                x=np.asarray([r["g_new"] for r in rows]);y=np.asarray([r[f"f_old_{corpus}"] for r in rows])
                marker="o" if size==sizes[0] else "s"
                ax.scatter(x,y,color=colors[group],marker=marker,alpha=.3,s=15)
                ax.errorbar(x.mean(),y.mean(),xerr=x.std(ddof=1) if len(x)>1 else None,
                            yerr=y.std(ddof=1) if len(y)>1 else None,marker=marker,color=colors[group],
                            capsize=3,linestyle="none",label=f"{group}, K={size}, n_draws={len(rows)}")
        if not endpoints:ax.text(.5,.5,"Prepared banks only\nNo trained conditional endpoints",ha="center",va="center",transform=ax.transAxes,color=".4")
        ax.set_title(label);ax.set_xlabel("New-domain gain (nats/token)");ax.set_ylabel("Old-domain forgetting (nats/token)")
        ax.grid(alpha=.18)
    if endpoints:axes[0].legend(fontsize=6,frameon=False)
    status="COMPLETE" if manifest["complete_trained_results"] else "PARTIAL"
    fig.suptitle(f"CONDITIONAL {status}: selection draws; training seed = generation seed = 11",fontsize=10)
    fig.tight_layout(rect=(0,0,1,.93))
    for ext in ("png","pdf","svg"):fig.savefig(output/f"conditional_gain_forgetting.{ext}",bbox_inches="tight")
    plt.close(fig)
    (output/"图示与统计说明.md").write_text("本补充的重复单位为 selection_seed，生成池和训练shuffle都固定为11。点为选库draw，误差棒为draw间样本SD（ddof=1），不是十个独立训练seed。教师NLL组成图只描述库，不代表梯度或retention收益。成对差为相同draw标签下M−B4或C−B4，不声称不同算法共享完全相同的随机数。未完成模型明确标partial；只读取验证端点，未读取模型测试或行为分数。\n")
