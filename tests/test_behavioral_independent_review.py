"""Temporary synthetic files only; never open prepared real behavior results."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from replay_study import behavioral_analysis as original
from replay_study import selected_policy_analysis as selected

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("independent_behavior_reviewer", ROOT/"scripts/review_behavioral_results.py")
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(reviewer.encoded(value));return path


def proof(path):
    return {"path":str(Path(path).resolve()),"sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def table(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def fixture(tmp_path, supplemental=False, family="C"):
    groups=selected.GROUPS if family=="A" else ("B4","M")
    kinds=("final",) if supplemental else ("final","best")
    tasks={}
    prepared={}
    for task in ("piqa","hellaswag"):
        rows=[{"task":task,"item_id":str(i),"fingerprint":task+str(i),"gold":(0,0,1,1,0,1)[i],"choices":["a","bbbb"]} for i in range(6)]
        path=tmp_path/(task+".jsonl");path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        tasks[task]={"artifact":str(path),"sha256":proof(path)["sha256"]}
        prepared[task]=rows
    records=[];jobs=[{"id":"initial","kind":"initial","baseline_group":"base","output":str(tmp_path/"initial.json"),"argv":["synthetic","initial"]}]
    for gi,group in enumerate(groups):
        for si,seed in enumerate((11,22,33)):
            for ki,kind in enumerate(kinds):
                run=f"{group}_{seed}";key=run+"_"+kind
                record=dict(id=key,run_id=run,actual_run_id=run,phase="formal",phase_name="synthetic",role="supplemental" if supplemental else "primary",
                    data_family="synthetic",model_tag="tiny",kind=kind,group=group,training_seed=seed,selection_seed=None,
                    ratio=.05,weight=(gi+1)*.1 if family=="A" else .5,learning_rate=1e-5,checkpoint=str(tmp_path/(key+".pt")),
                    checkpoint_sha256=hashlib.sha256(key.encode()).hexdigest(),weights_identity="checkpoint:"+hashlib.sha256(key.encode()).hexdigest(),
                    baseline_group="base",initial_job_id="initial",inference_job_id=key,output=str(tmp_path/(key+".json")),inference_alias_of=None,
                    step=10 if kind=="final" else 4,target_prediction_tokens_at_checkpoint=100 if kind=="final" else 40,
                    replay_prediction_tokens_at_checkpoint=5 if kind=="final" else 2)
                records.append(record)
                jobs.append({"id":key,"kind":"adapted","baseline_group":"base","output":record["output"],"argv":["synthetic",key]})
    source=tmp_path/"opaque_source.py";source.write_text("# synthetic source, never execute\n")
    manifest={"schema_version":2,"mode":"behavior","common_input_sha256":{str(source):proof(source)["sha256"]},
        "analysis_policy":{"bootstrap_replicates":5000,"bootstrap_seed":20261106},"records":records,"jobs":jobs,
        "baseline_groups":{"base":{"tasks":tasks,"initial_output":str(tmp_path/"initial.json"),"initial_job_id":"initial"}},
        "workload":{"total_inference_jobs":len(jobs)}}
    mp=proof(write(tmp_path/"manifest.json",manifest))
    all_predictions={};attestations=[]
    for job in jobs:
        record=next((r for r in records if r["inference_job_id"]==job["id"]),None)
        offset=0 if record is None else groups.index(record["group"])+(0,0,1)[record["training_seed"]//11-1]+(record["kind"]=="best")
        metrics={};outputs={}
        for task,rows in prepared.items():
            current=[]
            for i,row in enumerate(rows):
                scores=([-1.,-2.],[-3.,-2.],[-1.,-8.])[(i+offset)%3]
                a=int(np.argmax(scores));b=int(np.argmax(np.array(scores)/np.array([1,4])))
                current.append({**row,"choice_logprobs":scores,"choice_character_lengths":[1,4],"prediction_acc":a,"prediction_acc_norm":b,
                    "correct_acc":int(a==row["gold"]),"correct_acc_norm":int(b==row["gold"])})
            path=Path(job["output"]).with_name(Path(job["output"]).stem+"_"+task+".jsonl")
            path.write_text("".join(json.dumps(r)+"\n" for r in reversed(current))) # order must not matter
            outputs[str(path)]=proof(path)["sha256"]
            metrics[task]={"complete_split":True,"items":len(rows),"data_sha256":tasks[task]["sha256"],
                "predictions_path":str(path),"predictions_sha256":outputs[str(path)],
                **{m:sum(r["correct_"+m] for r in current)/len(rows) for m in ("acc_norm","acc")}}
            all_predictions[job["id"],task]=current
        write(job["output"],{"complete":True,"tasks":metrics});outputs[job["output"]]=proof(job["output"])["sha256"]
        ap=write(Path(job["output"]).with_name(Path(job["output"]).stem+"_execution.json"),
            {"manifest_sha256":mp["sha256"],"job_id":job["id"],"baseline_group":"base","verified_frozen_inputs":True,
             "checkpoint_sha256":record["checkpoint_sha256"] if record else None,"weights_identity":record["weights_identity"] if record else None,
             "output_sha256":outputs})
        attestations.append({"job_id":job["id"],"attestation":proof(ap),"output_sha256":outputs})
    bundle=tmp_path/"analysis"/("behavior" if supplemental else "")
    endpoints=[]
    for record in records:
        for task in tasks:
            for metric in ("acc_norm","acc"):
                seed=reviewer.stream(f"{record['id']}|{task}|{metric}")
                stats=original.paired_binary_summary(all_predictions["initial",task],all_predictions[record["id"],task],metric,5000,seed)
                row={k:record.get(k) for k in (*reviewer.STRATA,"run_id","actual_run_id","group","ratio","weight","learning_rate","training_seed","step","checkpoint_sha256","weights_identity","inference_job_id","target_prediction_tokens_at_checkpoint","replay_prediction_tokens_at_checkpoint")}
                row.update(checkpoint_kind=record["kind"],seed=record["training_seed"],task=task,metric=metric,primary_metric=metric=="acc_norm",phase="formal",
                    items=stats["items"],**stats["counts"],initial_accuracy=stats["initial_accuracy"],accuracy=stats["accuracy"]["estimate"],
                    accuracy_ci_low=stats["accuracy"]["ci_low"],accuracy_ci_high=stats["accuracy"]["ci_high"],accuracy_delta=stats["accuracy_delta"],
                    delta_ci_low=stats["delta_ci_low"],delta_ci_high=stats["delta_ci_high"],conditional_forgetting_rate=stats["conditional_forgetting"]["estimate"],
                    conditional_acquisition_rate=stats["conditional_acquisition"]["estimate"],bootstrap_seed=seed)
                endpoints.append(row)
    context={k:records[0][k] for k in reviewer.STRATA}
    pairs,pair_aggregates=original.paired_seed_summary(endpoints)
    aggregates=original.grouped_seed_summary(endpoints)
    for name,rows in (("endpoints",endpoints),("seed_aggregates",[{**context,**r} for r in aggregates]),
        ("paired_seed_differences",[{**context,**r} for r in pairs]),("paired_seed_aggregates",[{**context,**r} for r in pair_aggregates])):
        table(bundle/(name+".csv"),rows)
    counts={"logical_endpoints":len(records),"physical_jobs":len(jobs),"queue_jobs":len(jobs)}
    plan={"manifest":mp,"counts":counts,"source_sha256":manifest["common_input_sha256"],"frozen_definitions":jobs}
    if supplemental:
        phase={"name":"synthetic","rows":len(records)}
        protocol=proof(write(tmp_path/"protocol.json",{"analysis_source_sha256":manifest["common_input_sha256"]}))
        plan.update(contract={"mode":"behavior","family":family,"phase":phase,"logical_endpoints":len(records)},protocol=protocol,
            closure_jobs=jobs,first_read_receipt_path=str(tmp_path/"first.json"))
        if family=="A":
            sp=proof(write(tmp_path/"selection.json",{"synthetic_complete_validation_only":True}))
            rp=proof(write(tmp_path/"registry.json",{"synthetic_complete_validation_only":True}))
            plan["selection"]={"selection":sp,"source_registry":rp,"selected_ids":[r["run_id"] for r in records],
                "selected_identity":{r["run_id"]:{k:r[k] for k in ("group","weight","ratio","training_seed","actual_run_id")} for r in records}}
    pp=proof(write(tmp_path/"plan.json",plan))
    authority={"authorized_by":"root","allow_score_analysis":True,"allow_partial":False,"manifest":mp,"plan":pp,
        "expected_logical_endpoints":len(records),"expected_inference_jobs":len(jobs),"expected_counts":counts}
    ap=proof(write(tmp_path/"ROOT_AUTHORITY.json",authority))
    meta={"complete":True,"manifest_sha256":mp["sha256"],"logical_endpoints":len(records),"primary_metric":"acc_norm","auxiliary_metric":"acc"}
    if supplemental:
        readiness=proof(write(tmp_path/"analysis"/"whole_phase_readiness.json",{"ready":True,"pending_job_ids":[],
            "all_source_baseline_alias_output_integrity_verified":True,"plan":pp,"closure_job_count":len(jobs),"counts":counts,"attestations":attestations}))
        jp=proof(write(tmp_path/"first.json",{"before_first_score_parse":True,"complete_phase_source_and_output_SHA_verified":True,
            "readiness":readiness,"plan":pp,"authority":ap}))
        meta.update(first_read_receipt=jp,whole_phase_readiness=readiness)
        complete={"complete":True,"plan":pp,"authority":ap,"readiness":readiness,"first_read_receipt":jp,"parsed_phase":phase,
            "endpoints":[{"endpoint":"final_behavior","logical_conditions":len(records)}]}
        if family=="A":
            aux=selected.behavior_selected({"endpoints":endpoints},plan["selection"],lambda r:all_predictions[r["run_id"]+"_final",r["task"]])
            for name,rows in aux.items():
                if name!="metadata":table(tmp_path/"analysis"/"validation_selected_policy"/(name+".csv"),rows)
            write(tmp_path/"analysis"/"validation_selected_policy/analysis_manifest.json",aux["metadata"])
    else:
        state=proof(write(tmp_path/"stateproof.json",{"ready":True,"plan":pp,"attestations":attestations,
            "selected_completed_records":{j["id"]:{"status":"completed","returncode":0,"missing_outputs":[],"argv":j["argv"],
                "definition_sha256":reviewer.sha_bytes(json.dumps(j,sort_keys=True).encode())} for j in jobs}}))
        cp=proof(write(tmp_path/"check.json",{"ready":True,"all_output_integrity_verified":True,"logical_endpoints":len(records),
            "expected_inference_jobs":len(jobs),"manifest_sha256":mp["sha256"]}))
        jp=proof(write(tmp_path/"first.json",{"status":"semantic_access_enabled","before_original_analyze_invocation":True,
            "manifest":mp,"state_proof":state,"plan":pp,"authority":ap}))
        complete={"complete":True,"plan":pp,"authority":ap,"state_proof":state,"journal":jp,"original_check":cp,
            "original_analysis_metadata":proof(write(bundle/"analysis_manifest.json",meta))}
    write(bundle/"analysis_manifest.json",meta)
    return proof(write(tmp_path/"analysis_COMPLETE.json" if not supplemental else tmp_path/"analysis"/"COMPLETE.json",complete)),mp,all_predictions


@pytest.mark.parametrize("supplemental,family",[(False,"C"),(True,"C"),(True,"A")])
def test_full_tiny_phase_matches_frozen_statistics_and_both_score_formats(tmp_path,supplemental,family):
    complete,manifest,_=fixture(tmp_path,supplemental,family)
    out=tmp_path/"review"
    result=reviewer.review(complete,manifest,out)
    assert result["complete"] and result["maximum_absolute_numeric_error"] < 1e-14
    assert result["numeric_and_identity_checks"]>1000
    assert result["checkpoint_payload_bytes_read"]==result["model_forward_calls"]==0
    assert all(not p.endswith(".pt") for p in result["source_files"])
    if family=="A":
        assert result["validation_selected_review"]["pairs"]==72
        rows=list(csv.DictReader((out/"selected_policy_score_resolved_statistics.csv").open()))
        assert len(rows)==24 and {r["score"] for r in rows}=={"acc","acc_norm"}
        # Frozen source grouped the two scores correctly, but then erased their
        # labels. The new view must preserve two genuinely different means.
        original_rows=list(csv.DictReader((tmp_path/"analysis/validation_selected_policy/selected_policy_statistics.csv").open()))
        collisions=[r for r in original_rows if r["comparison"]=="B4-B3" and r["task"]=="piqa"]
        assert len(collisions)==2 and {r["metric"] for r in collisions}=={"accuracy_left_minus_right"}
        resolved=[r for r in rows if r["comparison"]=="B4-B3" and r["task"]=="piqa"]
        assert len({r["mean"] for r in resolved})==2


@pytest.mark.parametrize("change",["missing_complete","false_complete","missing_firstread","false_authority","partial_authority",
    "changed_authority","changed_source","changed_manifest","missing_endpoint","changed_output","changed_table","failed_job","bool_rc0","stopped_job","changed_definition"])
def test_fail_closed_without_ever_opening_checkpoint(tmp_path,change):
    complete,manifest,_=fixture(tmp_path)
    c=json.loads(Path(complete["path"]).read_text())
    if change=="missing_complete":Path(complete["path"]).unlink()
    elif change=="false_complete":c["complete"]=False;complete=proof(write(complete["path"],c))
    elif change=="missing_firstread":Path(c["journal"]["path"]).unlink()
    elif change in ("false_authority","partial_authority","changed_authority"):
        a=json.loads(Path(c["authority"]["path"]).read_text())
        if change=="false_authority":a["allow_score_analysis"]=False
        elif change=="partial_authority":a["allow_partial"]=True
        else:a["plan"]=manifest
        c["authority"]=proof(write(c["authority"]["path"],a));complete=proof(write(complete["path"],c))
    elif change=="changed_source":(tmp_path/"opaque_source.py").write_text("# changed\n")
    elif change=="changed_manifest":Path(manifest["path"]).write_text("{}")
    elif change=="changed_output":(tmp_path/"B4_11_final.json").write_text("{}")
    elif change in ("missing_endpoint","changed_table"):
        p=tmp_path/"analysis/endpoints.csv";rows=list(csv.DictReader(p.open()))
        if change=="missing_endpoint":rows.pop()
        else:rows[0]["accuracy"]=".123"
        table(p,rows)
    else:
        state=json.loads(Path(c["state_proof"]["path"]).read_text());row=state["selected_completed_records"]["B4_11_final"]
        if change=="failed_job":row["returncode"]=1
        elif change=="bool_rc0":row["returncode"]=False
        elif change=="stopped_job":row["stop_requested_at"]="synthetic"
        else:row["definition_sha256"]="wrong"
        c["state_proof"]=proof(write(c["state_proof"]["path"],state))
        journal=json.loads(Path(c["journal"]["path"]).read_text());journal["state_proof"]=c["state_proof"]
        c["journal"]=proof(write(c["journal"]["path"],journal));complete=proof(write(complete["path"],c))
    with pytest.raises((ValueError,FileNotFoundError,KeyError)):
        reviewer.review(complete,manifest,tmp_path/"review")
    assert not (tmp_path/"review/COMPLETE.json").exists()


def test_hand_computed_transition_sd_ties_and_independent_bootstrap_moments():
    a={i:{"fingerprint":str(i),"gold":0,"acc":v} for i,v in enumerate([1,1,0,0])}
    b={i:{"fingerprint":str(i),"gold":0,"acc":v} for i,v in enumerate([1,0,1,0])}
    stats=reviewer.binary(a,b,"acc",17)
    assert [stats[k] for k in ("CC","CW","WC","WW")]==[1]*4
    assert stats["accuracy"]==stats["initial_accuracy"]==.5
    assert stats["accuracy_delta"]==0 and stats["conditional_forgetting_rate"]==stats["conditional_acquisition_rate"]==.5
    assert reviewer.mean_sd([.01,.02,.03])["std_ddof1"]==pytest.approx(.01)
    # IID paired-question bootstrap enumeration, independent of multinomial RNG:
    # four possible deltas {-1,0,0,1}; enumerate all 4^4 ordered resamples.
    import itertools
    possible=np.array([sum(x)/4 for x in itertools.product((0,-1,1,0),repeat=4)])
    assert possible.mean()==0 and possible.var()==.125
    assert np.quantile(possible,[.025,.975]).tolist()==[-.75,.75]
    assert stats["delta_ci_low"]==-.75 and stats["delta_ci_high"]==.75
    assert reviewer.wilson(0,0)==(None,None,None)
    assert max(enumerate([-1.,-1.]),key=lambda x:x[1])[0]==0


def test_reviewer_refuses_checkpoint_and_repeated_namespace(tmp_path):
    cp=tmp_path/"checkpoint.pt";cp.write_bytes(b"never read")
    with pytest.raises(ValueError,match="refuses payload"):
        reviewer.Reader().bytes(cp)
    complete,manifest,_=fixture(tmp_path)
    reviewer.review(complete,manifest,tmp_path/"review")
    with pytest.raises(ValueError,match="immutable"):
        reviewer.review(complete,manifest,tmp_path/"review")


@pytest.mark.parametrize("change",["duplicate_item","fingerprint","character_length","tie_prediction","missing_task"])
def test_independent_question_identity_and_argmax_fail_closed(tmp_path,change):
    path=tmp_path/"prepared.jsonl"
    source={"task":"piqa","item_id":"0","fingerprint":"f","gold":1,"choices":["a","bbbb"]}
    path.write_text(json.dumps(source)+"\n")
    pred=tmp_path/"prediction.jsonl"
    row={**source,"choice_logprobs":[-1.,-4.],"choice_character_lengths":[1,4],
         "prediction_acc":0,"correct_acc":0,"prediction_acc_norm":0,"correct_acc_norm":0}
    if change=="fingerprint":row["fingerprint"]="wrong"
    elif change=="character_length":row["choice_character_lengths"]=[1,2]
    elif change=="tie_prediction":row["prediction_acc_norm"]=1;row["correct_acc_norm"]=1
    elif change=="missing_task":row["task"]="other"
    pred.write_text(json.dumps(row)+"\n"+(json.dumps(row)+"\n" if change=="duplicate_item" else ""))
    info={"artifact":str(path),"sha256":proof(path)["sha256"]}
    metrics={"predictions_path":str(pred),"predictions_sha256":proof(pred)["sha256"],"complete_split":True,"items":1,"data_sha256":info["sha256"]}
    with pytest.raises(ValueError):reviewer.questions(reviewer.Reader(),"piqa",info,metrics,{str(pred):proof(pred)["sha256"]})


def test_CLI_complete_parent_timer_is_separate_from_review_receipt(tmp_path):
    complete,manifest,_=fixture(tmp_path,True,"A")
    output=tmp_path/"cli_review"
    done=subprocess.run([sys.executable,str(ROOT/"scripts/review_behavioral_results.py"),"--complete",complete["path"],"--complete-sha256",complete["sha256"],
        "--manifest",manifest["path"],"--manifest-sha256",manifest["sha256"],"--output",str(output)],capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    timer=json.loads(output.with_name(output.name+"_whole_child_timing.json").read_text())
    assert timer["wall_seconds"]>0 and timer["returncode"]==0 and timer["nested_subtimers_not_additive"]
    assert json.loads((output/"COMPLETE.json").read_text())["complete"]
