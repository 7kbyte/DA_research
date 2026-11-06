"""Additional tiny complete phases: v3 repair, D alias, imported baseline/reuse.

Only temporary synthetic JSON/JSONL. The original 26 tests and reviewer source
remain unchanged; no prepared real result/score is opened.
"""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("behavior_review_old_fixture",ROOT/"tests/test_behavioral_independent_review.py")
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
reviewer=f.reviewer


def read(path):return json.loads(Path(path).read_text())
def attpath(output):
    p=Path(output);return p.with_name(p.stem+"_execution.json")


def copy_result(source,destination):
    value=read(source);dest=Path(destination)
    for task,metrics in value["tasks"].items():
        pred=dest.with_name(dest.stem+"_"+task+".jsonl")
        pred.parent.mkdir(parents=True,exist_ok=True)
        pred.write_bytes(Path(metrics["predictions_path"]).read_bytes())
        metrics.update(predictions_path=str(pred),predictions_sha256=f.proof(pred)["sha256"])
    f.write(dest,value)


def result_hashes(output):
    value=read(output)
    return {str(p):f.proof(p)["sha256"] for p in [Path(output),*[Path(m["predictions_path"]) for m in value["tasks"].values()]]}


def expected_tables(root,manifest):
    """Reference fixture values from the frozen functions, never the reviewer."""
    rows=[];before={task:f.original.jsonl(metrics["predictions_path"]) for task,metrics in read(manifest["baseline_groups"]["base"]["initial_output"])["tasks"].items()}
    for record in manifest["records"]:
        for task,metrics in read(record["output"])["tasks"].items():
            for metric in ("acc_norm","acc"):
                seed=reviewer.stream(f"{record['id']}|{task}|{metric}")
                stats=f.original.paired_binary_summary(before[task],f.original.jsonl(metrics["predictions_path"]),metric,5000,seed)
                row={k:record.get(k) for k in (*reviewer.STRATA,"run_id","actual_run_id","group","ratio","weight","learning_rate","training_seed","step","checkpoint_sha256","weights_identity","inference_job_id","target_prediction_tokens_at_checkpoint","replay_prediction_tokens_at_checkpoint")}
                row.update(checkpoint_kind=record["kind"],seed=record["training_seed"],task=task,metric=metric,primary_metric=metric=="acc_norm",phase="formal",
                    items=stats["items"],**stats["counts"],initial_accuracy=stats["initial_accuracy"],accuracy=stats["accuracy"]["estimate"],
                    accuracy_ci_low=stats["accuracy"]["ci_low"],accuracy_ci_high=stats["accuracy"]["ci_high"],accuracy_delta=stats["accuracy_delta"],
                    delta_ci_low=stats["delta_ci_low"],delta_ci_high=stats["delta_ci_high"],conditional_forgetting_rate=stats["conditional_forgetting"]["estimate"],
                    conditional_acquisition_rate=stats["conditional_acquisition"]["estimate"],bootstrap_seed=seed)
                rows.append(row)
    pairs,aggregate=f.original.paired_seed_summary(rows)
    context={k:rows[0][k] for k in reviewer.STRATA}
    for name,values in (("endpoints",rows),("seed_aggregates",f.original.grouped_seed_summary(rows)),
        ("paired_seed_differences",pairs),("paired_seed_aggregates",aggregate)):
        f.table(root/"behavior"/(name+".csv"),[{**context,**r} for r in values])


def sources_fixture(tmp_path, branch, change=None):
    cp,_,_=f.fixture(tmp_path,True,"A" if branch=="import_reuse" else "C")
    complete=read(cp["path"]);plan=read(complete["plan"]["path"]);manifest=read(plan["manifest"]["path"])
    metadata_path=tmp_path/"analysis/behavior/analysis_manifest.json"
    metadata=read(metadata_path)
    prior=None;logical_aliases=[]
    if branch=="D_alias":
        by={(r["group"],r["training_seed"]):r for r in manifest["records"]}
        comparisons=[]
        for seed in (11,22,33):
            left,right=by["B4",seed],by["M",seed]
            old_right=copy.deepcopy(right)
            fingerprint={"sha256_model_state":f.reviewer.sha_bytes(f"SYNTHETIC_STATE_{seed}".encode()),"state_elements":2}
            for field in ("id","run_id","actual_run_id","checkpoint","output"):
                right[field]=right[field].replace("M_","D_")
            right.update(group="D",inference_job_id=left["inference_job_id"],inference_alias_of=left["id"])
            for record in (left,right):record["weights_identity"]="model_state:"+fingerprint["sha256_model_state"]
            copy_result(left["output"],right["output"])
            comparisons.append({"seed":seed,"endpoint":"final","kind":"identical_bank_null_control","bitwise_model_state_equal":True,
                "left":{"checkpoint":left["checkpoint"],"checkpoint_sha256":left["checkpoint_sha256"],"model_state_fingerprint":fingerprint},
                "right":{"checkpoint":right["checkpoint"],"checkpoint_sha256":right["checkpoint_sha256"],"model_state_fingerprint":fingerprint}})
            logical_aliases.append((right,left))
            assert not Path(left["checkpoint"]).exists() and not Path(right["checkpoint"]).exists()
        algorithm=tmp_path/"synthetic_fingerprint.py";algorithm.write_text("# opaque synthetic algorithm source; never execute\n")
        eq=f.proof(f.write(tmp_path/"equivalence.json",{"schema_version":1,"status":"passed",
            "algorithm":"sha256_sorted_key_dtype_shape_contiguous_uint8_v1","algorithm_source":f.proof(algorithm),"comparisons":comparisons}))
        for i,(alias,canonical) in enumerate(logical_aliases):
            comparison=comparisons[i]
            evidence={"proof":eq["path"],"proof_sha256":eq["sha256"],"comparison_index":i,"fingerprint":comparison["left"]["model_state_fingerprint"],
                "left_checkpoint_sha256":comparison["left"]["checkpoint_sha256"],"right_checkpoint_sha256":comparison["right"]["checkpoint_sha256"]}
            alias["source_model_equivalence"]=canonical["source_model_equivalence"]=evidence
        manifest["jobs"]=[j for j in manifest["jobs"] if not j["id"].startswith("M_")]
    elif branch=="import_reuse":
        # Three B1 endpoints reuse a prior manifest; initial is imported too.
        prior=copy.deepcopy(manifest)
        prior["records"]=[copy.deepcopy(r) for r in manifest["records"] if r["group"]=="B1"]
        prior["jobs"]=[copy.deepcopy(j) for j in manifest["jobs"] if j["kind"]=="initial" or j["id"].startswith("B1_")]
        for job in prior["jobs"]:
            destination=tmp_path/"prior_outputs"/Path(job["output"]).name
            copy_result(job["output"],destination);job["output"]=str(destination)
        prior_jobs={j["id"]:j for j in prior["jobs"]}
        for job in prior["jobs"]:
            old_id=job["id"];job["id"]="prior_"+old_id
            rr=[r for r in prior["records"] if r["inference_job_id"]==old_id]
            job["record_ids"]=[r["id"] for r in rr]
            if rr:
                for key in ("checkpoint","checkpoint_sha256","weights_identity"):job[key]=rr[0][key]
        for record in prior["records"]:
            job=prior_jobs[record["inference_job_id"]]
            record.update(output=job["output"],inference_job_id=job["id"],initial_job_id="prior_initial")
        prior["baseline_groups"]["base"]["initial_job_id"]="prior_initial"
        prior["baseline_groups"]["base"]["initial_output"]=prior_jobs["initial"]["output"]
        prior["workload"].update(total_inference_jobs=4,total_queue_jobs=4,logical_checkpoint_endpoints=3,reuse_materialization_jobs=0)
        prior_proof=f.proof(f.write(tmp_path/"prior_manifest.json",prior))
        group=manifest["baseline_groups"]["base"]
        group.update(initial_job_id="prior_initial",initial_output=prior_jobs["initial"]["output"],initial_origin={"manifest":prior_proof["path"],"manifest_sha256":prior_proof["sha256"],"job_id":"prior_initial","output":prior_jobs["initial"]["output"]})
        for record in manifest["records"]:record["initial_job_id"]="prior_initial"
        manifest["jobs"]=[j for j in manifest["jobs"] if j["kind"]!="initial"]
        for job in manifest["jobs"]:
            if job["id"].startswith("B1_"):
                record=next(r for r in manifest["records"] if r["inference_job_id"]==job["id"])
                job.update(kind="reuse",output=prior_jobs[job["id"]]["output"],
                    inference_origin={"manifest":prior_proof["path"],"manifest_sha256":prior_proof["sha256"],"job_id":prior_jobs[job["id"]]["id"],"output":prior_jobs[job["id"]]["output"]})
                copy_result(job["output"],record["output"])
                logical_aliases.append((record,job))
    elif branch!="repair":raise AssertionError(branch)

    # Frozen metadata shapes: canonical/reuse jobs identify all logical records.
    for job in manifest["jobs"]:
        rr=[r for r in manifest["records"] if r["inference_job_id"]==job["id"]]
        job["record_ids"]=[r["id"] for r in rr]
        if rr:
            for k in ("checkpoint","checkpoint_sha256","weights_identity","source_model_equivalence"):
                job[k]=rr[0].get(k)
    manifest["workload"].update(total_inference_jobs=sum(j["kind"]!="reuse" for j in manifest["jobs"]),
        total_queue_jobs=len(manifest["jobs"]),logical_checkpoint_endpoints=len(manifest["records"]),
        reuse_materialization_jobs=sum(j["kind"]=="reuse" for j in manifest["jobs"]))
    mp=f.proof(f.write(plan["manifest"]["path"],manifest));plan["manifest"]=mp
    closure=[{"job_id":j["id"],"kind":j["kind"],"manifest":mp,"definition":{"id":j["id"],"argv":["synthetic-wrapper",j["id"]]}} for j in manifest["jobs"]]
    if prior:
        closure += [{"job_id":j["id"],"kind":j["kind"],"manifest":prior_proof,"definition":{"id":j["id"],"argv":["synthetic-prior-wrapper",j["id"]]}} for j in prior["jobs"]]
    counts={"logical_endpoints":len(manifest["records"]),"queue_jobs":len(manifest["jobs"]),
        "new_forward_jobs":manifest["workload"]["total_inference_jobs"],"reuse_jobs":manifest["workload"]["reuse_materialization_jobs"],
        "logical_aliases":len(logical_aliases),"imported_baselines":int(prior is not None),"closure_queue_jobs":len(closure)}
    plan.update(counts=counts,closure_jobs=closure,closure_job_count=len(closure))
    pp=f.proof(f.write(complete["plan"]["path"],plan))
    active=tmp_path/"active_guard.py";active.write_text("# opaque v3 source; never execute\n")
    retained=tmp_path/"retained_guard.py";retained.write_text("# opaque retained source; never execute\n")
    repair={"schema_version":3,"operational_revision":3,"kind":"supplemental_analysis_guard_operational_bugfix",
        "retained_plan":pp,"retained_manifest":mp,"retained_protocol":plan["protocol"],"scientific_or_statistics_changes":False,
        "score_access_authorized":False,"first_read_rule":"exact_retained_v2_plan_manifest_namespace",
        "active_source_sha256":{str(active):f.proof(active)["sha256"]},"retained_source_sha256":{str(retained):f.proof(retained)["sha256"]}}
    if change=="repair_wrong_manifest":repair["retained_manifest"]={"path":mp["path"],"sha256":"f"*64}
    if change=="repair_wrong_plan":repair["retained_plan"]={"path":pp["path"],"sha256":"f"*64}
    repair_proof=f.proof(f.write(tmp_path/"repair_v3.json",repair))
    authority=read(complete["authority"]["path"])
    authority.update(plan=pp,manifest=mp,expected_counts=counts,operational_repair=repair_proof)
    if change=="authority_old_repair":authority["operational_repair"]={"path":repair_proof["path"],"sha256":"e"*64}
    ap=f.proof(f.write(complete["authority"]["path"],authority))
    attestations=[]
    def attest(job,record,digest,alias=False):
        output=record["output"] if record else job["output"]
        value={"manifest_sha256":digest,"job_id":job["id"],"baseline_group":"base","verified_frozen_inputs":True,
            "checkpoint_sha256":record["checkpoint_sha256"] if record else job.get("checkpoint_sha256"),
            "weights_identity":record["weights_identity"] if record else job.get("weights_identity"),
            "source_model_equivalence":(record or job).get("source_model_equivalence"),"actual_argv":job["argv"],
            "inference_alias":alias,"output_sha256":result_hashes(output),"inference_subprocess_wall_seconds":0 if alias else .1}
        if alias:value.update(logical_record_id=record["id"],canonical_output=str(Path(job["output"]).resolve()),canonical_inference_subprocess_wall_seconds=.1)
        if job["kind"]=="reuse":value.update(inference_origin=job["inference_origin"],inference_subprocess_wall_seconds=0)
        if change=="alias_nonzero" and alias:value["inference_subprocess_wall_seconds"]=.01
        if change=="alias_wrong_manifest" and alias:value["manifest_sha256"]="f"*64
        if change=="prior_wrong_manifest" and digest!=mp["sha256"]:value["manifest_sha256"]="f"*64
        att=f.proof(f.write(attpath(output),value))
        attestations.append({"manifest_sha256":digest,"job_id":job["id"],"logical_record_id":record["id"] if alias else None,
            "attestation":att,"output_sha256":value["output_sha256"]})
    if prior:
        prior_records={r["inference_job_id"]:r for r in prior["records"]}
        for job in prior["jobs"]:
            attest(job,None,prior_proof["sha256"])
    for job in manifest["jobs"]:
        rr=[r for r in manifest["records"] if r["inference_job_id"]==job["id"]]
        if job["kind"]!="reuse":attest(job,None,mp["sha256"])
        for record in rr:
            if record["output"]!=job["output"]:attest(job,record,mp["sha256"],True)
    readiness={"schema_version":3,"operational_revision":3,"operational_repair":repair_proof,"plan":pp,
        "ready":True,"pending_job_ids":[],"all_source_baseline_alias_output_integrity_verified":True,
        "closure_job_count":len(closure),"counts":counts,"attestations":attestations,"score_files_hashed":True,"score_json_parsed":False}
    rp=f.proof(f.write(complete["readiness"]["path"],readiness))
    journal=read(complete["first_read_receipt"]["path"]);journal.update(plan=pp,authority=ap,readiness=rp)
    jp=f.proof(f.write(complete["first_read_receipt"]["path"],journal))
    metadata.update(manifest_sha256=mp["sha256"],first_read_receipt=jp,whole_phase_readiness=rp)
    f.write(metadata_path,metadata)
    expected_tables(tmp_path/"analysis",manifest)
    complete.update(plan=pp,authority=ap,readiness=rp,first_read_receipt=jp,operational_repair=repair_proof)
    cp=f.proof(f.write(cp["path"],complete))
    return cp,mp,counts


@pytest.mark.parametrize("branch",["repair","D_alias","import_reuse"])
def test_real_source_shapes_pass_full_independent_review(tmp_path,branch):
    complete,manifest,counts=sources_fixture(tmp_path,branch)
    result=reviewer.review(complete,manifest,tmp_path/"review")
    assert result["complete"] and result["checkpoint_payload_bytes_read"]==0
    assert result["maximum_absolute_numeric_error"]<1e-14
    if branch=="D_alias":
        assert counts["logical_aliases"]==3 and result["logical_endpoints"]==6 and result["manifest_jobs"]==4
        rows=list(csv.DictReader((tmp_path/"review/paired_seed_differences.csv").open()))
        assert len(rows)==12 and {r["left_group"] for r in rows}=={"D"}
        assert all(float(r["accuracy_left_minus_right"])==0 for r in rows)
    if branch=="import_reuse":
        assert counts["reuse_jobs"]==3 and counts["closure_queue_jobs"]==22 and counts["imported_baselines"]==1
        assert result["validation_selected_review"]["pairs"]==72
        assert (tmp_path/"review/selected_policy_score_resolved_statistics.csv").exists()


@pytest.mark.parametrize("branch,change",[("repair","repair_wrong_manifest"),("repair","repair_wrong_plan"),
    ("repair","authority_old_repair"),("D_alias","alias_nonzero"),("D_alias","alias_wrong_manifest"),
    ("import_reuse","alias_nonzero"),("import_reuse","alias_wrong_manifest"),("import_reuse","prior_wrong_manifest")])
def test_source_mismatches_refuse_before_semantic_prediction_parse(tmp_path,branch,change,monkeypatch):
    complete,manifest,_=sources_fixture(tmp_path,branch,change)
    def forbidden(*_args,**_kwargs):raise AssertionError("No question JSONL should be semantically read before source gate passes")
    monkeypatch.setattr(reviewer.Reader,"rows",forbidden)
    with pytest.raises(ValueError):reviewer.review(complete,manifest,tmp_path/"review")
    assert not (tmp_path/"review/COMPLETE.json").exists()
