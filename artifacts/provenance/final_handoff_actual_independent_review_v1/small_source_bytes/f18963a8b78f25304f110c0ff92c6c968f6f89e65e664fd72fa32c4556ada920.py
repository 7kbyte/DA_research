#!/usr/bin/env python3
"""Freeze initial-only timing for planned models from completed LR execution facts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from evaluation_plan import ROOT, OPERATIONAL, Hashes, absolute, command, encoded, identity, outputs, read, training_execution
from make_behavioral_queue import task_inputs


def build(output_root, manifest_path):
    h=Hashes(); scope_path=ROOT/"configs/behavioral_scope_execution_freeze_v2.json";scope=read(scope_path)
    if scope["state"]!="root_frozen_before_first_real_score":raise ValueError("Root-frozen execution/scope required")
    assets_path=ROOT/"artifacts/assets/extended_model_provenance.json";provenance=read(assets_path)
    source_files=["scripts/evaluation_plan.py","scripts/run_frozen_evaluation.py","src/replay_study/evaluate.py","src/replay_study/execution.py","scripts/evaluate_tasks.py","src/replay_study/tasks.py","scripts/make_behavioral_queue.py"]
    scorer={str(ROOT/p):h.digest(ROOT/p) for p in source_files}
    data_root=ROOT/"artifacts/behavioral";task_sources,tasks=task_inputs(data_root)
    task_sources={str(absolute(p)):v for p,v in task_sources.items()}
    for task in tasks.values():task["choices"]=sum(len(json.loads(line)["choices"]) for line in Path(task["artifact"]).open() if line.strip())
    common={**scorer,str(scope_path):h.digest(scope_path),str(assets_path):h.digest(assets_path),str(Path(__file__).resolve()):h.digest(__file__)}
    groups,jobs,evidence={},[],[];output_root,manifest_path=absolute(output_root),absolute(manifest_path)
    for model,namespace,micro in [("pythia-1b","pythia_1b",32),("pythia-2.8b","pythia_2p8b",8)]:
        selection_path=ROOT/f"results/extended/{model}/m{micro}/learning_rate_selection.json";selection=read(selection_path)
        registry_path=ROOT/f"configs/extended_v2_{namespace}_m{micro}_lr_registry.json"
        h.proof({"path":registry_path,"sha256":selection["source_registry_sha256"]})
        protocol_path,protocol_sha=h.proof({"path":selection["source_protocol_path"],"sha256":selection["source_protocol_sha256"]});protocol=read(protocol_path)
        rows=read(registry_path)
        if len(rows)!=3 or any(r["phase"]!="lr" or r["model"]!=model for r in rows):raise ValueError("The completed model-specific LR pilot registry is required")
        observed=[];source_inputs={str(registry_path):selection["source_registry_sha256"],protocol_path:protocol_sha,str(selection_path):h.digest(selection_path)}
        for row in rows:
            cfg,summary=read(row["config_path"]),read(row["summary_path"])
            if summary["status"]!="completed" or summary["counters"]["target_prediction_tokens"]!=2_000_000 or any(summary["config"].get(k)!=v for k,v in cfg.items() if k not in OPERATIONAL):raise ValueError("Complete and scientifically matched 2M LR pilot required")
            h.proof({"path":row["config_path"],"sha256":row["config_sha256"]})
            source_inputs[row["config_path"]]=row["config_sha256"];source_inputs[row["summary_path"]]=h.digest(row["summary_path"])
            policy={"model":str(absolute(cfg["model_name_or_path"])),"revision":cfg["model_revision"],"batch_size":scope["behavior_batch_size_by_model"][model],"max_length":scope["max_length"],"device":"cuda:0",**training_execution(cfg,summary)}
            observed.append(policy)
        if any(p!=observed[0] for p in observed):raise ValueError("LR pilots disagree on actual model/execution facts")
        policy=observed[0]; pinned=next(p for p in provenance["models"] if p["repo"].split("/")[-1]==model)
        if policy["model"]!=str(absolute(pinned["path"])) or policy["revision"]!=pinned["revision"] or protocol["model_revision"]!=pinned["revision"]:raise ValueError("Pinned model revision/path mismatch")
        h.proof({"path":pinned["weight_file"],"sha256":pinned["sha256"]})
        model_dir=Path(policy["model"]);weights=sorted([*model_dir.glob("*.safetensors"),*model_dir.glob("pytorch_model*.bin")])
        if not weights:raise ValueError("Local pinned base model required")
        files=sorted(set(weights+list(model_dir.glob("*.json"))+list(model_dir.glob("*.model"))+list(model_dir.glob("*.txt"))))
        model_inputs={str(p):h.digest(p) for p in files}
        key=identity({"mode":"behavior","policy":policy,"model_assets":model_inputs,"evaluation_inputs":task_sources,"scorer":scorer})
        jid=f"behavior_{model}_{key[:20]}_initial";output=str(output_root/model/"baselines"/f"{key}.json")
        group={"id":key,"model_tag":model,"inference_policy":policy,"base_model_assets_sha256":model_inputs,"evaluation_input_sha256":task_sources,"test_paths":{},"tasks":tasks,"initial_job_id":jid,"initial_output":output,"planned_model":True}
        groups[key]=group
        jobs.append({"id":jid,"kind":"initial","phase":"planned_model_initial_timing","model_tag":model,"baseline_group":key,"output":output,"record_ids":[],"input_sha256":{**source_inputs,**model_inputs,**task_sources},"argv":command(policy,"behavior",data_root,{})+["--output",output]})
        evidence.append({"model":model,"source_kind":"completed_LR_pilot_execution_only","source_registry":str(registry_path),"source_registry_sha256":selection["source_registry_sha256"],"actual_execution":observed[0],"formal_training_completed_claim":False,"baseline_identity_matches_standard_v2_builder":True})
    manifest={"schema_version":2,"mode":"behavior","phase":"planned_model_initial_timing","role":"planned_model_initial_timing","prepared_only":True,"launch_authorized":False,"formal_training_completed_claim":False,"if_no_formal_adaptation_results":"initial-only diagnostic; do not present as completed model-comparison experiment","output_root":str(output_root),"common_input_sha256":common,"scorer_input_sha256":scorer,"baseline_groups":groups,"jobs":jobs,"records":[],"source_execution_evidence":evidence,"analysis_policy":{"bootstrap_replicates":5000,"bootstrap_seed":20261106},"workload":{"initial_baselines":2,"adapted_inference_jobs":0,"total_inference_jobs":2,"future_formal_reuse":"Pass this original manifest in prior_plans; identical model/data/scorer/execution identity reuses baseline"},"scorer_role_note":"Frozen scorer evaluation_role=formal denotes full task split, not formal adapted training completion","model_inference_performed_by_generator":False}
    digest=hashlib.sha256(encoded(manifest)).hexdigest()
    queue=[{"id":j["id"],"argv":[sys.executable,str(ROOT/"scripts/run_frozen_evaluation.py"),"--manifest",str(manifest_path),"--manifest-sha256",digest,"--job-id",j["id"],"--device","cuda:0"],"env":{"CUBLAS_WORKSPACE_CONFIG":":4096:8"},"expected_seconds":600,"outputs":outputs(j["output"],"behavior")} for j in jobs]
    return queue,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("output-root","queue","manifest"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    if a.queue.exists() or a.manifest.exists() or a.queue.resolve()==a.manifest.resolve():raise FileExistsError("New distinct output paths required")
    q,m=build(a.output_root,a.manifest)
    from schedule_jobs import validate_queue
    validate_queue(q)
    for path,value in ((a.manifest,m),(a.queue,q)):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encoded(value))
    print("Prepared two planned-model initial-only jobs; inference not started")


if __name__=="__main__":main()
