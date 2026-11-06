"""Hand-computed CPU invariants; fixtures never read real heldout results."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from replay_study.analysis import sha256, training_validation_execution
from replay_study.aux_validation import (cohort_gate, collect_extended_validation, proof,
    validate_bank_activation, write_extended_analysis, write_json)


def fixture(model="pythia-410m"):
    phase="extended_synthetic_"+model.replace("-","_")
    groups=("B0","B1","B2","B4","M") if model=="pythia-2.8b" else ("B0","B1","B2","B3","B4","D","M","C")
    paths={c:c+"_validation.pt" for c in ("target","c4","wikitext")}
    protocol=dict(model=model,model_revision="pinned",data_version="extended_v1",groups=["B1","B2","B3","B4"],
                  formal_target_prediction_tokens=1000,validation_paths=paths,shared_learning_rate=.001,shared_lambda=.05)
    provenance=dict(torch_version="synthetic",cuda_version="synthetic",device="cuda:0",precision="bf16",gpu="synthetic GPU",
        attention_implementation="sdpa",attention_backend_policy="auto",sdpa_backends_enabled=dict(flash=True,memory_efficient=True,math=True,cudnn=True),
        deterministic_algorithms=True,deterministic_warn_only=False,allow_tf32_matmul=True,allow_tf32_cudnn=True,
        cudnn_deterministic=True,cudnn_benchmark=False,cublas_workspace_config=":4096:8",model_weight_dtype="torch.float32")
    def corpora(value):
        return {c:dict(nll=value,kl=.2,complete_corpus=True,precision="bf16",model_weight_dtype="torch.float32",prediction_tokens=100) for c in paths}
    rows=[];summaries={}
    for group_index,group in enumerate(groups):
        for seed in (11,22,33):
            run_id=f"{phase}_{group}_s{seed}"
            ratio=0. if group=="B0" else .05
            weight=0. if group=="B0" else .05
            row=dict(id=run_id,model=model,phase="formal",active=True,group=group,seed=seed,ratio=ratio,weight=weight,learning_rate=.001)
            config=dict(seed=seed,replay_ratio=ratio,replay_weight=weight,learning_rate=.001,run_label=run_id,
                model_revision="pinned",model_name_or_path="synthetic-model-"+model,validation_paths=copy.deepcopy(paths),target_prediction_tokens=1000,
                precision="bf16",device="cuda:0",attention_implementation="sdpa")
            # Same-seed heterogeneity cancels in paired effects, exactly 0.01 per group index.
            value=3 + seed/100 + group_index*.01
            points=[dict(step=1,target_prediction_tokens=100,replay_prediction_tokens=int(100*ratio),metrics=corpora(value-1)),
                    dict(step=10,target_prediction_tokens=1000,replay_prediction_tokens=int(1000*ratio),metrics=corpora(value))]
            counters=dict(step=10,target_prediction_tokens=1000,replay_prediction_tokens=int(1000*ratio),
                setup_seconds=1,training_seconds=10,validation_seconds=2,checkpoint_seconds=1,total_accounted_wall_seconds=14,
                training_teacher_forward_seconds=3,validation_teacher_forward_seconds=1)
            summaries[run_id]=dict(status="completed",config=config,counters=counters,provenance=copy.deepcopy(provenance),
                validation_trajectory=points,best_target_validation=dict(step=1,target_prediction_tokens=100,nll=value-1))
            rows.append(row)
    baseline=dict(model="synthetic-model-"+model,revision="pinned",checkpoint=None,corpora=corpora(5),
                  execution_provenance=training_validation_execution(next(iter(summaries.values()))))
    return rows,protocol,baseline,summaries,dict(phase=phase,model=model)


def test_complete_fixed_model_validation_and_best_actual_exposure():
    rows,p,b,s,kw=fixture()
    tables,execution=collect_extended_validation(rows,p,b,s,**kw)
    assert len(tables["validation_endpoints"])==48 and len(execution)==24
    best=[r for r in tables["validation_endpoints"] if r["checkpoint"]=="best"]
    assert {r["target_prediction_tokens"] for r in best}=={100}
    assert {r["full_training_target_prediction_tokens"] for r in best}=={1000}
    assert all(r["split"]=="validation" and r["role"]=="supplement_C_extended" for r in best)
    # Costs are once per trained run, and nested teacher time is not re-added.
    assert len(tables["full_training_costs_once_per_run"])==24
    assert sum(r["parent_component_sum_seconds"] for r in tables["full_training_costs_once_per_run"])==24*14
    assert all(not r["nested_timers_additive"] for r in tables["full_training_costs_once_per_run"])
    gain=next(r for r in tables["final_endpoint_statistics"] if r["group"]=="B0" and r["metric"]=="g_new")
    assert gain["mean"]==pytest.approx(1.78) and gain["std_ddof1"]==pytest.approx(.11) and gain["n"]==3
    pair=next(r for r in tables["final_paired_statistics"] if r["comparison"]=="B3-B1" and r["metric"]=="g_new")
    assert pair["mean"]==pytest.approx(-.02) and pair["std_ddof1"]==pytest.approx(0,abs=1e-14)
    assert {r["comparison"] for r in tables["final_paired_statistics"]}=={"M-B4","C-B4","B2-B1","B4-B3","B3-B1","B4-B2"}


def test_missing_cohort_rejected_and_absent_2p8_groups_explicit():
    rows,p,b,s,kw=fixture("pythia-2.8b")
    with pytest.raises(ValueError,match="incomplete"):
        collect_extended_validation(rows,p,b,{k:v for k,v in s.items() if k!=rows[0]["id"]},**kw)
    tables,_=collect_extended_validation(rows,p,b,s,**kw)
    absent={r["comparison"] for r in tables["comparison_availability"] if not r["registered"]}
    assert absent=={"C-B4","B4-B3","B3-B1"}
    assert len(tables["validation_endpoints"])==30


@pytest.mark.parametrize("change",["model","revision","data","policy","budget","best","seed"])
def test_refuse_scientific_identity_or_execution_changes(change):
    rows,p,b,s,kw=fixture()
    row=rows[0]; current=s[row["id"]]
    if change=="model": rows[0]["model"]="pythia-1b"
    elif change=="revision": current["config"]["model_revision"]="other"
    elif change=="data": current["config"]["validation_paths"]["target"]="main_data.pt"
    elif change=="policy": current["provenance"]["sdpa_backends_enabled"]["cudnn"]=False
    elif change=="budget": current["counters"]["target_prediction_tokens"]=999
    elif change=="best": current["best_target_validation"].update(step=10,target_prediction_tokens=1000,nll=current["validation_trajectory"][-1]["metrics"]["target"]["nll"])
    elif change=="seed": rows[0]["seed"]=22
    with pytest.raises(ValueError):
        collect_extended_validation(rows,p,b,s,**kw)


def test_best_pair_explicitly_not_equal_budget():
    rows,p,b,s,kw=fixture()
    selected=next(r for r in rows if r["group"]=="M" and r["seed"]==11)
    current=s[selected["id"]]
    middle=copy.deepcopy(current["validation_trajectory"][0]);middle.update(step=2,target_prediction_tokens=200,replay_prediction_tokens=10)
    for m in middle["metrics"].values():m["nll"]-=.1
    current["validation_trajectory"].insert(1,middle)
    current["best_target_validation"].update(step=2,target_prediction_tokens=200,nll=middle["metrics"]["target"]["nll"])
    tables,_=collect_extended_validation(rows,p,b,s,**kw)
    pair=next(r for r in tables["best_paired_differences"] if r["comparison"]=="M-B4" and r["seed"]==11)
    assert (pair["checkpoint_target_tokens_a"],pair["checkpoint_target_tokens_b"])==(200,100)
    assert not pair["equal_checkpoint_target_exposure"]


def activation_fixture(tmp_path):
    oldcfg=tmp_path/"oldcfg.json";newcfg=tmp_path/"newcfg.json"
    write_json(oldcfg,dict(deadline_utc="old",seed=11,replay_weight=.05))
    write_json(newcfg,dict(deadline_utc="new",seed=11,replay_weight=.05))
    source=tmp_path/"source.json";runtime=tmp_path/"runtime.json";activation=tmp_path/"activation.json";prep=tmp_path/"prep.json"
    original=dict(id="draw1001",phase="conditional_bank_variance",active=False,status="prepared_only",training_seed=11,generation_seed=11,selection_seed=1001,config_path=str(oldcfg))
    actual={**original,"active":True,"status":"activated","config_path":str(newcfg),"prepared_config_path":str(oldcfg),"prepared_config_sha256":sha256(oldcfg),"config_sha256":sha256(newcfg)}
    write_json(source,[original]);write_json(runtime,[actual]);write_json(prep,{})
    act=dict(phase="conditional_bank_variance",source_artifacts_modified=False,source_registry=str(source),source_registry_sha256=sha256(source),
        execution_registry=str(runtime),execution_registry_sha256=sha256(runtime),analysis_registry=str(source),preparation_manifest=str(prep),preparation_manifest_sha256=sha256(prep),
        config_overrides={"draw1001":dict(prepared_config_path=str(oldcfg),prepared_config_sha256=sha256(oldcfg),runtime_config_path=str(newcfg),runtime_config_sha256=sha256(newcfg),operational_override=dict(deadline_utc="new"))})
    write_json(activation,act)
    return source,runtime,activation


def test_activation_retains_draw_identity_and_rejects_scientific_mutation(tmp_path):
    source,runtime,activation=activation_fixture(tmp_path)
    old,new,sources=validate_bank_activation(source,runtime,activation)
    assert old[0]["selection_seed"]==new[0]["selection_seed"]==1001
    assert "seed" not in new[0]
    actual=json.loads(runtime.read_text());actual[0]["selection_seed"]=1002;write_json(runtime,actual)
    act=json.loads(activation.read_text());act["execution_registry_sha256"]=sha256(runtime);write_json(activation,act)
    with pytest.raises(ValueError,match="scientific"):
        validate_bank_activation(source,runtime,activation)


def test_activation_hash_and_undeclared_override_refused(tmp_path):
    source,runtime,activation=activation_fixture(tmp_path)
    cfg=tmp_path/"newcfg.json";value=json.loads(cfg.read_text());value["replay_weight"]=1;write_json(cfg,value)
    with pytest.raises(ValueError,match="Frozen input"):
        validate_bank_activation(source,runtime,activation)


def test_even_rehashed_activation_cannot_override_scientific_config(tmp_path):
    source,runtime,activation=activation_fixture(tmp_path)
    cfg=tmp_path/"newcfg.json";value=json.loads(cfg.read_text());value["replay_weight"]=1;write_json(cfg,value)
    act=json.loads(activation.read_text());act["config_overrides"]["draw1001"]["runtime_config_sha256"]=sha256(cfg)
    rows=json.loads(runtime.read_text());rows[0]["config_sha256"]=sha256(cfg);write_json(runtime,rows)
    act["execution_registry_sha256"]=sha256(runtime);write_json(activation,act)
    with pytest.raises(ValueError,match="beyond"):
        validate_bank_activation(source,runtime,activation)


def test_gate_never_reads_partial_summaries_and_duplicate_ids_rejected(tmp_path):
    rows=[dict(id="one",summary_path=str(tmp_path/"does_not_exist")),dict(id="two")]
    gate=cohort_gate(rows,dict(jobs={"one":dict(status="completed"),"two":dict(status="running")}))
    assert not gate["ready"] and gate["completed_runs"]==1
    with pytest.raises(ValueError,match="unique"):
        cohort_gate([rows[0],rows[0]],{})


@pytest.mark.parametrize("mutation",["nonzero_exit","wrong_definition","missing_output","master_changed"])
def test_complete_scheduler_gate_binds_definition_exit_and_outputs(tmp_path,mutation):
    path=tmp_path/"validation_summary.json";write_json(path,{})
    job=dict(id="one",argv=["synthetic"],outputs=[str(path)])
    state=dict(jobs={"one":dict(status="completed",returncode=0,missing_outputs=[],
        definition_sha256=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest())})
    assert cohort_gate([dict(id="one")],state,[job],[job])["ready"]
    master=copy.deepcopy(job)
    if mutation=="nonzero_exit":state["jobs"]["one"]["returncode"]=1
    elif mutation=="wrong_definition":state["jobs"]["one"]["definition_sha256"]="wrong"
    elif mutation=="missing_output":path.unlink()
    elif mutation=="master_changed":master["argv"]=["other"]
    with pytest.raises(ValueError):
        cohort_gate([dict(id="one")],state,[job],[master])


def test_writer_phase_and_source_identity_no_cross_model_pooling(tmp_path):
    rows,p,b,s,kw=fixture()
    tables,execution=collect_extended_validation(rows,p,b,s,**kw)
    source=tmp_path/"source.json";write_json(source,{"synthetic":True})
    result=write_extended_analysis(tmp_path/"out",tables,execution,**kw,sources=[proof(source)],target=1000,plots=False)
    assert result["split"]=="validation" and result["model"]=="pythia-410m"
    assert result["logical_endpoint_count"]==48 and result["test_or_behavior_scores_read"] is False
    assert result["sources"]==[proof(source)]


def test_conditional_table_discussion_uses_group_column_and_train_seed_scope(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from watch_auxiliary_validation import bank_discussion
    from replay_study.analysis import write_csv
    write_csv(tmp_path/"endpoint_statistics.csv",[dict(group="B4",bank_chunks=20,metric="g_new",mean=1,std_ddof1=.5,n=10)])
    # Existing conditional aggregation uses group for the comparison identity.
    write_csv(tmp_path/"paired_validation_statistics.csv",[dict(group="M-B4",bank_chunks=20,metric="g_new",mean=.2,std_ddof1=.1,n=10)])
    bank_discussion(tmp_path)
    text=(tmp_path/"条件选库验证讨论.md").read_text()
    assert "M-B4" in text and "不是十个独立训练 seed" in text


def test_print_plots_keep_exact_seven_inch_canvas(tmp_path):
    from replay_study.aux_validation import plot_extended
    import xml.etree.ElementTree as ET
    rows,p,b,s,kw=fixture()
    tables,_=collect_extended_validation(rows,p,b,s,**kw)
    plot_extended(tables,tmp_path,model=kw["model"],target=1000)
    files=list(tmp_path.glob("*.svg"))
    assert len(files)==5
    for path in files:
        assert ET.parse(path).getroot().attrib["width"]=="504pt"  # 7 * 72 points.
        assert path.with_suffix(".png").exists() and path.with_suffix(".pdf").exists()


def completion_fixture(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    rows=[]
    for index in range(2):
        path=tmp_path/f"summary{index}.json";write_json(path,{"synthetic_validation_only":index})
        rows.append(dict(id=str(index),summary_path=str(path)))
    registry=tmp_path/"registry.json";snapshot=tmp_path/"snapshot.json";audit=tmp_path/"audit.json"
    write_json(registry,rows);write_json(snapshot,{})
    write_json(audit,dict(status="passed",source_violations=[],completed_runs=2,numerical_null_control_status="passed",
        numerical_null_control_diagnostics=[],phase_audit=dict(runs=[dict(id=r["id"],summary_sha256=sha256(r["summary_path"])) for r in rows])))
    completion=tmp_path/"completion.json"
    write_json(completion,dict(status="passed",phase="extended_synthetic",model="pythia-410m",completed_runs=2,expected_runs=2,
        source_registry=proof(registry),input_snapshot=proof(snapshot),audit=proof(audit),proofs={},source_inputs=[],all_pairs_bitwise_equal=True))
    phase=dict(phase="extended_synthetic",model="pythia-410m",registry=proof(registry),snapshot=proof(snapshot),completion=str(completion))
    return phase,rows


def test_independent_completion_binds_exact_audited_summary_bytes(tmp_path):
    from watch_auxiliary_validation import validate_completion
    phase,rows=completion_fixture(tmp_path)
    sources,null=validate_completion(phase,rows)
    assert null["status"]=="passed" and null["all_pairs_bitwise_equal"]
    write_json(rows[0]["summary_path"],{"synthetic_validation_only":"changed"})
    with pytest.raises(ValueError,match="audited summaries"):
        validate_completion(phase,rows)


def test_independent_completion_rejects_unbound_or_failed_audit(tmp_path):
    from watch_auxiliary_validation import validate_completion
    phase,rows=completion_fixture(tmp_path)
    path=Path(phase["completion"]);record=json.loads(path.read_text());record["model"]="pythia-1b";write_json(path,record)
    with pytest.raises(ValueError,match="identity"):
        validate_completion(phase,rows)
    record["model"]="pythia-410m";write_json(path,record)
    audit=Path(record["audit"]["path"]);value=json.loads(audit.read_text());value["status"]="failed";write_json(audit,value)
    with pytest.raises(ValueError,match="Frozen input"):
        validate_completion(phase,rows)
