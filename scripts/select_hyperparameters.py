#!/usr/bin/env python3
"""Select one shared replay weight using the complete preregistered pilot grid."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path


def select_shared_lambda(registry: list[dict], initial: dict, protocol: dict,
                         summaries: dict[str,dict], scheduler_state: dict | None = None) -> dict:
    """Validate complete matched budgets, then minimize mean S; exact ties use λ."""
    groups, weights = protocol["groups"], protocol["lambda_grid"]
    expected={(group,float(weight)) for group in groups for weight in weights}
    actual={(record["group"],float(record["weight"])) for record in registry}
    if len(registry)!=len(expected) or actual!=expected:
        raise ValueError("Calibration registry must contain each preregistered group/lambda exactly once")
    if len({r["id"] for r in registry})!=len(registry):
        raise ValueError("Duplicate calibration run IDs")
    baseline=initial.get("corpora",initial)
    scores=[]
    stream=None
    for record in registry:
        if scheduler_state is not None and scheduler_state["jobs"][record["id"]]["status"]!="completed":
            raise ValueError(f"Scheduler has not completed {record['id']}")
        summary=summaries[record["id"]]
        config,counters=summary["config"],summary["counters"]
        if summary["status"]!="completed" or counters["target_prediction_tokens"]!=protocol["target_prediction_tokens"]:
            raise ValueError(f"Incomplete target exposure: {record['id']}")
        if counters["replay_prediction_tokens"]!=protocol["replay_prediction_tokens"]:
            raise ValueError(f"Replay budget mismatch: {record['id']}")
        for field,expected_value in [("seed",protocol["seed"]),("replay_ratio",protocol["replay_ratio"]),
                    ("learning_rate",protocol["shared_learning_rate"]),("replay_weight",record["weight"]),
                    ("microbatch_size",protocol["microbatch_size"]),
                    ("gradient_accumulation_steps",protocol["gradient_accumulation_steps"])]:
            if config[field]!=expected_value:
                raise ValueError(f"Unmatched {field}: {record['id']}")
        if stream is None:stream=summary["target_stream"]
        elif summary["target_stream"]!=stream:
            raise ValueError("Target streams differ across calibration runs")
        trajectory=summary.get("validation_trajectory",[])
        if not trajectory or trajectory[-1]["target_prediction_tokens"]!=protocol["target_prediction_tokens"]:
            raise ValueError(f"Missing fixed-exposure final validation: {record['id']}")
        metrics=trajectory[-1]["metrics"]
        for name in ["target","c4","wikitext"]:
            if not metrics[name]["complete_corpus"] or not math.isfinite(metrics[name]["nll"]):
                raise ValueError(f"Invalid validation corpus {name}: {record['id']}")
            if metrics[name]["valid_prediction_tokens"]!=baseline[name]["valid_prediction_tokens"]:
                raise ValueError(f"Validation token count changed for {name}: {record['id']}")
            if "kl" not in metrics[name]:
                raise ValueError(f"Missing teacher KL on {name}: {record['id']}")
        deltas={name:metrics[name]["nll"]-baseline[name]["nll"] for name in ["target","c4","wikitext"]}
        score=deltas["target"]+.5*(deltas["c4"]+deltas["wikitext"])
        scores.append({"id":record["id"],"group":record["group"],"weight":record["weight"],
            "score":score,"delta_nll":deltas,"validation":metrics,
            "counters":counters,"summary_path":record.get("summary_path"),
            "external_evaluation_path":record.get("external_evaluation_path")})
    aggregates=[{"weight":weight,"mean_score":math.fsum(row["score"] for row in scores if row["weight"]==weight)/len(groups),
                 "groups":{row["group"]:row["score"] for row in scores if row["weight"]==weight}} for weight in weights]
    selected=min(aggregates,key=lambda row:(row["mean_score"],row["weight"]))
    return {"status":"complete","selected_shared_lambda":selected["weight"],
            "shared_learning_rate":protocol["shared_learning_rate"],"selected_mean_score":selected["mean_score"],
            "selection_rule":protocol["lambda_selection_rule"],"score_formula":protocol["lambda_group_score"],
            "aggregate_rule":protocol["lambda_aggregate"],"initial_validation":baseline,
            "target_stream":stream,"aggregates":aggregates,"runs":scores,
            "timing_coverage":{"training_teacher_forward_seconds":{
                "available_run_ids":[row["id"] for row in scores if "training_teacher_forward_seconds" in row["counters"]],
                "missing_run_ids":[row["id"] for row in scores if "training_teacher_forward_seconds" not in row["counters"]],
                "nested_in":"training_seconds","add_to_parent":False}},
            "selected_at_utc":dt.datetime.now(dt.timezone.utc).isoformat()}


def select_learning_rate(records, initial, protocol, evaluations, state=None):
    expected=set(protocol["learning_rate_grid_completed"])
    if len(records)!=len(expected) or {r["learning_rate"] for r in records}!=expected:
        raise ValueError("LR registry does not match the frozen three-point grid")
    baseline=initial["corpora"]
    scores=[]
    stream=None
    for record in records:
        evaluation=evaluations[f"lr_{record['label']}"]
        if state and state["jobs"][evaluation["job_id"]]["status"]!="completed":
            raise ValueError("LR reevaluation is not complete")
        summary=json.loads((Path(record["output_dir"])/"summary.json").read_text())
        if summary["status"]!="completed" or summary["counters"]["target_prediction_tokens"]!=protocol["target_prediction_tokens"]:
            raise ValueError("LR training budget is incomplete")
        if stream is None:stream=summary["target_stream"]
        elif stream!=summary["target_stream"]:raise ValueError("LR target streams differ")
        measured=json.loads(Path(evaluation["evaluation_path"]).read_text())["corpora"]
        for name in ["target","c4","wikitext"]:
            metrics=measured[name]
            if (not metrics["complete_corpus"] or not math.isfinite(metrics["nll"]) or "kl" not in metrics
                    or metrics["valid_prediction_tokens"]!=baseline[name]["valid_prediction_tokens"]):
                raise ValueError(f"Invalid LR validation: {name}")
        scores.append({**record,"validation_path":evaluation["evaluation_path"],
            "selection_score_target_nll":measured["target"]["nll"],
            "validation":{name:{k:v for k,v in metrics.items() if k!="chunks"} for name,metrics in measured.items()},
            "target_gain":baseline["target"]["nll"]-measured["target"]["nll"],
            "c4_forgetting":measured["c4"]["nll"]-baseline["c4"]["nll"],
            "wikitext_forgetting":measured["wikitext"]["nll"]-baseline["wikitext"]["nll"]})
    selected=min(scores,key=lambda row:(row["selection_score_target_nll"],row["learning_rate"]))
    return {"status":"complete","selected_learning_rate":selected["learning_rate"],
        "selection_rule":protocol["learning_rate_selection_rule"],"pilots":scores,
        "initial_validation":{name:{k:v for k,v in metrics.items() if k!="chunks"} for name,metrics in baseline.items()},
        "target_stream":stream,"selected_at_utc":dt.datetime.now(dt.timezone.utc).isoformat()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry",type=Path,default=Path("configs/lambda_registry.json"))
    parser.add_argument("--initial-validation",type=Path,default=Path("results/pilots/lr_validation_initial.json"))
    parser.add_argument("--protocol",type=Path,default=Path("docs/hyperparameter_selection_protocol.json"))
    parser.add_argument("--state",type=Path)
    parser.add_argument("--output",type=Path,default=Path("results/pilot_lambda_selection.json"))
    parser.add_argument("--result-role",default="calibration")
    parser.add_argument("--phase",choices=["lr","lambda"],default="lambda")
    parser.add_argument("--evaluation-registry",type=Path)
    args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text())
    if (protocol.get("initial_validation_sha256") and
            hashlib.sha256(args.initial_validation.read_bytes()).hexdigest()!=protocol["initial_validation_sha256"]):
        raise ValueError("Initial validation changed after protocol freeze")
    for name,path in protocol["validation_paths"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=protocol["validation_sha256"][name]:
            raise ValueError(f"Frozen validation data changed: {name}")
    registry=json.loads(args.registry.read_text())
    initial=json.loads(args.initial_validation.read_text())
    state=json.loads(args.state.read_text()) if args.state else None
    evaluations=({row["id"]:row for row in json.loads(args.evaluation_registry.read_text())}
                 if args.evaluation_registry else None)
    if args.phase=="lr":
        if evaluations is None:raise ValueError("LR selection requires an evaluation registry")
        if "initial" not in evaluations:
            raise ValueError("LR reevaluation requires exactly one initial baseline record")
        if Path(evaluations["initial"]["evaluation_path"]).resolve()!=args.initial_validation.resolve():
            raise ValueError("Initial baseline path differs from evaluation registry")
        if state and state["jobs"][evaluations["initial"]["job_id"]]["status"]!="completed":
            raise ValueError("Initial baseline evaluation is incomplete")
        result=select_learning_rate(registry,initial,protocol,evaluations,state)
    else:
        summaries={record["id"]:json.loads(Path(record["summary_path"]).read_text()) for record in registry}
        if evaluations:
            if set(evaluations)!=set(summaries):raise ValueError("External evaluation grid differs from training grid")
            for record in registry:
                evaluation=evaluations[record["id"]]
                measured=json.loads(Path(evaluation["evaluation_path"]).read_text())
                if measured["checkpoint"]!=evaluation["checkpoint_path"]:
                    raise ValueError("External evaluation used an unexpected checkpoint")
                summary=copy.deepcopy(summaries[record["id"]])
                summary["validation_trajectory"]=[{"step":summary["counters"]["step"],
                    "target_prediction_tokens":summary["counters"]["target_prediction_tokens"],
                    "metrics":measured["corpora"]}]
                summaries[record["id"]]=summary
                record["external_evaluation_path"]=evaluation["evaluation_path"]
            if state:
                state={"jobs":{name:state["jobs"][evaluation["job_id"]] for name,evaluation in evaluations.items()}}
        result=select_shared_lambda(registry,initial,protocol,summaries,state)
    result["data_version"]=protocol.get("data_version")
    result["result_role"]=args.result_role
    result["protocol_sha256"]=hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    result["registry_sha256"]=hashlib.sha256(args.registry.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(args.output.suffix+".tmp")
    temporary.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    os.replace(temporary,args.output)
    print(json.dumps({key:value for key,value in result.items()
                      if key in {"selected_shared_lambda","selected_mean_score","selected_learning_rate"}}))


if __name__=="__main__":main()
