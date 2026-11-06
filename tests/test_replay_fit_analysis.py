"""Hand-computed signs, seed variance, exposure and ordered identity invariants."""
import copy
import math

import pytest

from replay_study.replay_fit_analysis import (PHASE, METRICS, collect, contrast, summarize,
                                             validate_corpus, validate_scope)
from replay_study.nll_v2_analysis import MODEL_REVISIONS


def corpus(nlls, kls, docs):
    chunks = [dict(chunk_index=i, doc_id=d, valid_prediction_tokens=511, nll=n, sum_nll=n*511, kl=k, sum_kl=k*511)
              for i, (n,k,d) in enumerate(zip(nlls,kls,docs))]
    return dict(chunks=chunks, complete_corpus=True, chunks_evaluated=len(docs), chunks_available=len(docs),
                raw_input_tokens=len(docs)*512, valid_prediction_tokens=len(docs)*511, prediction_tokens=len(docs)*511,
                nll=sum(nlls)/len(docs), sum_nll=sum(nlls)*511, kl=sum(kls)/len(docs), sum_kl=sum(kls)*511,
                precision="bf16", model_weight_dtype="torch.float32", kl_direction="teacher||student",
                kl_vocabulary="exact_full_vocabulary", kl_arithmetic="float32")


def test_gap_subtracts_each_partitions_own_initial_not_the_raw_nll_gap():
    # Initial raw gap is 3.0. Adapted raw gap is 3.3; genuine differential fit is only +0.3.
    initial = {"bank": {"nll":2., "kl":.01}, "complement": {"nll":5., "kl":.02}}
    adapted = {"bank": {"nll":1.5, "kl":.1}, "complement": {"nll":4.8, "kl":.13}}
    r = contrast(adapted,initial)
    assert r["bank_delta_nll"] == -.5
    assert r["complement_delta_nll"] == pytest.approx(-.2)
    assert r["nll_gap"] == pytest.approx(.3)
    assert r["kl_gap"] == pytest.approx(.02)  # Subtract measured initial KL, not an assumed exact zero.


def test_seed_pairing_and_sample_sd_are_not_chunk_or_endpoint_pseudoreplication():
    rows=[]
    for endpoint in ("final","best"):
        for i,seed in enumerate((11,22,33)):
            for method in ("B3","B4"):
                # Paired B3-B4 values [1,3,5], mean 3, sample SD 2.
                value = [11.,23.,35.][i] if method == "B3" else [10.,20.,30.][i]
                rows.append(dict(seed=seed,group=method,endpoint=endpoint,**{k:value for k in METRICS}))
    _,paired,stats=summarize(rows)
    s=next(s for s in stats if s["endpoint"]=="final" and s["metric"]=="nll_gap")
    assert s["n"]==3 and s["mean"]==3 and s["std_ddof1"]==2
    assert [p["nll_gap"] for p in paired if p["endpoint"]=="final"]==[1,3,5]
    with pytest.raises(ValueError,match="partial"):
        summarize(rows[:-1])


def test_equal_document_sets_do_not_permit_swapped_chunk_identities():
    c=corpus([1.,3.],[.1,.3],["docA","docB"])
    validate_corpus(c,["docA","docB"],precision="bf16",weight_dtype="torch.float32")
    with pytest.raises(ValueError,match="ordered chunk"):
        validate_corpus(c,["docB","docA"],precision="bf16",weight_dtype="torch.float32")
    c["sum_nll"]+=511
    with pytest.raises(ValueError,match="chunk sums"):
        validate_corpus(c,["docA","docB"],precision="bf16",weight_dtype="torch.float32")


def scope_fixture():
    m=dict(schema_version=2,mode="nll",phase=PHASE,role="exploratory_diagnostic",primary_evaluation=False,
           independent_test=False,specified_after_main_validation=True,jobs=[],records=[],baseline_groups={})
    for seed in (11,22,33):
        gid=f"base{seed}"; initial=f"initial{seed}"
        m["baseline_groups"][gid]=dict(id=gid,seed=seed,model_tag="pythia-410m",initial_job_id=initial,
            initial_output=initial+".json",test_paths={"bank":"bank","complement":"comp"},
            inference_policy=dict(revision=MODEL_REVISIONS["pythia-410m"],deterministic_algorithms=True,
                                  deterministic_warn_only=False,CUBLAS_WORKSPACE_CONFIG=":4096:8"))
        m["jobs"].append(dict(id=initial,kind="initial",baseline_group=gid,output=initial+".json"))
        for method in ("B3","B4"):
            for endpoint in ("final","best"):
                rid=f"{method}{seed}{endpoint}"; ck=f"{rid}.pt"; ch="sha"+rid
                r=dict(id=rid,run_id=f"{method}{seed}",actual_run_id=f"{method}{seed}",is_training_alias=False,
                    phase=PHASE,phase_name=PHASE,role="exploratory_diagnostic",data_family="same_pool_bank_complement",
                    model_tag="pythia-410m",training_seed=seed,group=method,kind=endpoint,baseline_group=gid,
                    initial_job_id=initial,inference_job_id=rid,ratio=.05,weight=.05,learning_rate=3e-5,
                    checkpoint=ck,checkpoint_sha256=ch,weights_identity="checkpoint:"+ch,output=rid+".json",
                    target_prediction_tokens_at_checkpoint=10_000_000 if endpoint=="final" else 3_270_400,
                    replay_prediction_tokens_at_checkpoint=500_000 if endpoint=="final" else 163_520,
                    full_training_committed_path=dict(target_prediction_tokens=10_000_000,replay_prediction_tokens=500_000))
                m["records"].append(r)
                m["jobs"].append(dict(id=rid,kind="adapted",baseline_group=gid,record_ids=[rid],
                    **{k:r[k] for k in ("checkpoint","checkpoint_sha256","weights_identity","output")}))
    return m


@pytest.mark.parametrize("mutation",["partial","best_exposure","role","foreign_seed"])
def test_exact_scope_refuses_partial_foreign_and_best_as_full_budget(mutation):
    m=scope_fixture(); assert len(validate_scope(m))==15
    if mutation=="partial": m["jobs"].pop()
    elif mutation=="best_exposure": m["records"][1]["target_prediction_tokens_at_checkpoint"]=10_000_000
    elif mutation=="role": m["role"]="primary"
    else: m["records"][0]["training_seed"]=101
    with pytest.raises(ValueError): validate_scope(m)


def test_scores_remain_sealed_without_replay_specific_opt_in(tmp_path):
    with pytest.raises(PermissionError): collect(tmp_path/"does_not_exist.json","ignored",tmp_path/"timing.json")
