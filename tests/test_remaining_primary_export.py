"""Frozen remaining endpoint selection and split packaging; no score reader."""
import copy
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from export_remaining_primary_endpoints import (PACKAGES,REVISION,build_verified_tar,checkpoint_metadata,file_sha256,
    same_metadata,select_remaining_records,validate_alias)


def manifest_fixture():
    matrix=[(g,s,0. if g=="B0" else .05,0. if g=="B0" else .05)
            for g in ("B0","B1","B2","B3","B4","D","M","C") for s in (11,22,33)]
    matrix += [(g,s,b,.05) for g in ("B4","M") for s in (11,22,33) for b in (.01,.1)]
    records=[]
    for kind in ("final","best"):
        for group,seed,ratio,weight in matrix:
            run=f"formal_{group}_{seed}_{ratio}"
            records.append(dict(id=run+"__"+kind,run_id=run,actual_run_id=run,kind=kind,group=group,training_seed=seed,ratio=ratio,weight=weight,
                role="primary",phase="formal",phase_name="primary",model_tag="pythia-410m",data_family="main_v3",
                selection_seed=None,generation_seed=None,is_training_alias=False,output="SEALED_SCORE_MUST_NOT_OPEN"))
    return dict(records=records)


def test_remaining_factorial_has_48_logical_45_weights_and_first12_sensitivity_finals():
    selected=select_remaining_records(manifest_fixture())
    assert len(selected)==48 and len({r["weight_endpoint_id"] for r in selected})==45
    assert sum(r["kind"]=="best" for r in selected)==36
    finals=selected[:12]
    assert all(r["kind"]=="final" for r in finals)
    assert {r["group"] for r in finals}=={"B4","M"} and {r["ratio"] for r in finals}=={.01,.1}
    aliases=[r for r in selected if r["alias_of"]]
    assert len(aliases)==3 and all(r["group"]=="D" and r["kind"]=="best" for r in aliases)
    assert {r["training_seed"] for r in aliases}=={11,22,33}
    assert PACKAGES["final"]["logical"]==PACKAGES["final"]["unique"]==12
    assert PACKAGES["best"]["logical"]==36 and PACKAGES["best"]["unique"]==33


@pytest.mark.parametrize("change",["missing","duplicate","phase","model","weight"])
def test_reject_incomplete_duplicate_or_mixed_scientific_identity(change):
    manifest=manifest_fixture()
    if change=="missing":manifest["records"].pop()
    elif change=="duplicate":manifest["records"][1]=copy.deepcopy(manifest["records"][0])
    elif change=="phase":manifest["records"][0]["role"]="supplement_A"
    elif change=="model":manifest["records"][0]["model_tag"]="pythia-1b"
    elif change=="weight":manifest["records"][0]["weight"]=5
    with pytest.raises(ValueError):select_remaining_records(manifest)


def test_selection_does_not_consult_scores_or_existing_outputs():
    m=manifest_fixture();initial=[r["id"] for r in select_remaining_records(m)]
    for index,r in enumerate(m["records"]):r.update(test_score=10000-index,validation_score=index)
    assert [r["id"] for r in select_remaining_records(m)]==initial


def test_alias_requires_exact_weight_proof_and_checkpoint_sources():
    selected=select_remaining_records(manifest_fixture());r=next(r for r in selected if r["alias_of"])
    canonical=next(x for x in selected if x["id"]==r["weight_source_record_id"])
    fingerprint=dict(sha256_model_state="all-tensor-bytes",state_elements=12)
    for row in (r,canonical):row.update(checkpoint="/synthetic/"+row["id"],checkpoint_sha256="sha_"+row["id"])
    pair=dict(id="seed11best",kind="identical_bank_null_control",seed=11,bitwise_model_state_equal=True)
    for name,row in (("left",canonical),("right",r)):
        pair[name]=dict(run_id=row["run_id"],checkpoint=row["checkpoint"],checkpoint_sha256=row["checkpoint_sha256"],
            kind="best",model_revision=REVISION,model_state_fingerprint=fingerprint)
    payload=dict(status="passed",comparisons=[pair])
    r["source_model_equivalence"]=dict(proof_sha256="proof",comparison_index=0,
        algorithm="sha256_sorted_key_dtype_shape_contiguous_uint8_v1",fingerprint=fingerprint)
    assert validate_alias(r,canonical,payload,"proof")["bitwise_model_state_equal"]
    payload["comparisons"][0]["right"]["checkpoint_sha256"]="changed"
    with pytest.raises(ValueError,match="source checkpoint"):validate_alias(r,canonical,payload,"proof")


def test_metadata_preserves_unknown_rng_stream_and_best_fields_exactly(tmp_path):
    special=torch.from_numpy(np.array([0x80000000,0x7fc00001],dtype=np.uint32).view(np.float32))
    saved=dict(model_state=dict(x=torch.ones(2)),optimizer_state={"large":"excluded"},format_version=1,
        selection=dict(step=3,target_prediction_tokens=100),counters=dict(target_prediction_tokens=100,replay_prediction_tokens=5),
        rng_state=dict(torch=special,numpy=np.array([1,2],dtype=np.uint32)),
        target_stream=dict(position=77),new_unknown_metadata=dict(a=(1,-0.0,float("nan"))))
    metadata=checkpoint_metadata(saved)
    assert set(metadata)==set(saved)-{"model_state","optimizer_state"}
    path=tmp_path/"metadata.pt";torch.save(metadata,path)
    restored=torch.load(path,weights_only=False)
    assert same_metadata(metadata,restored)
    restored["target_stream"]["position"]=78
    assert not same_metadata(metadata,restored)


def test_outer_tar_every_member_sha_and_source_files_retained(tmp_path):
    source=tmp_path/"weights.zip";source.write_bytes(b"synthetic exact byte payload")
    metadata=tmp_path/"metadata.pt";metadata.write_bytes(b"metadata")
    members=[(source,"weights/x.zip",file_sha256(source)),(metadata,"metadata/x.pt",file_sha256(metadata))]
    tar=tmp_path/"complete.tar";receipt=build_verified_tar(tar,members,20<<20)
    assert receipt["verified_tar_members"]==2 and receipt["tar_sha256"]==file_sha256(tar)
    assert receipt["file_fsync_completed"] and receipt["parent_directory_fsync_completed"]
    assert source.read_bytes()==b"synthetic exact byte payload" and metadata.exists()
    with tarfile.open(tar) as package:assert package.extractfile("weights/x.zip").read()==source.read_bytes()
    with pytest.raises(FileExistsError):build_verified_tar(tar,members,20<<20)


def test_outer_tar_bad_source_sha_and_insufficient_ceiling_cannot_commit(tmp_path):
    source=tmp_path/"source";source.write_bytes(b"original");tar=tmp_path/"reject.tar"
    with pytest.raises(ValueError,match="exceeds"):build_verified_tar(tar,[(source,"source",file_sha256(source))],100)
    assert not tar.exists() and not tar.with_name(tar.name+".partial").exists()
    with pytest.raises(ValueError,match="input changed"):build_verified_tar(tar,[(source,"source","wrong")],20<<20)
    assert not tar.exists() and source.read_bytes()==b"original"


def test_complete_package_metadata_base_restore_and_receipt_assembly(tmp_path,monkeypatch):
    import export_remaining_primary_endpoints as exporter
    root=tmp_path/"root";delivery=tmp_path/"delivery";base=tmp_path/"base"
    delivery.mkdir();base.mkdir()
    for name in ("model.safetensors","config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","README.md"):
        (base/name).write_bytes(b"synthetic base artifact")
    for name in ("src/replay_study/weight_archive.py","src/replay_study/__init__.py","scripts/archive_model_weights.py",
                 "results/formal_run_audit.json","results/formal_best_model_equivalence_proofs.json","manifest.json","plan.json"):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("{}")
    for key,value in dict(ROOT=root,DELIVERY=delivery,BASE=base,ENDPOINTS=delivery/"endpoints",SOURCE_MANIFEST=root/"manifest.json",PLAN=root/"plan.json").items():
        monkeypatch.setattr(exporter,key,value)
    cfg=dict(seed=11,replay_ratio=.01,replay_weight=.05,learning_rate=3e-5,run_label="synthetic",model_revision=REVISION)
    counters=dict(step=306,target_prediction_tokens=10_000_000,replay_prediction_tokens=100_000)
    selected=dict(step=1,target_prediction_tokens=100,nll=1.)
    summary=dict(status="completed",config=cfg,counters=counters,validation_trajectory=[],best_target_validation=selected)
    checkpoint=tmp_path/"original.pt";torch.save(dict(model_state=dict(x=torch.ones(2)),optimizer_state={},config=cfg,counters=counters,
        validation_trajectory=[],best_target_validation=selected,rng_state=dict(cpu=torch.tensor([1,2],dtype=torch.uint8))),checkpoint)
    cp=tmp_path/"config.json";sp=tmp_path/"summary.json";exporter.write(cp,cfg);exporter.write(sp,summary)
    archive=tmp_path/"weight.zip";archive.write_bytes(b"already verified weight payload")
    row=dict(id="record",run_id="synthetic",kind="final",group="B4",training_seed=11,ratio=.01,weight=.05,
        checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),frozen_source_stat=exporter.signature(checkpoint),
        config_path=str(cp),summary_path=str(sp),step=306,target_prediction_tokens_at_checkpoint=10_000_000,replay_prediction_tokens_at_checkpoint=100_000,
        delivery_endpoint_id="synthetic__final",weight_endpoint_id="synthetic__final",alias_of=None,alias_evidence=None,
        source_config_sha256=file_sha256(cp),source_summary_sha256=file_sha256(sp),actual_exposure=dict(target_prediction_tokens=10_000_000))
    record=dict(delivery_endpoint_id="synthetic__final",archive_path=str(archive),archive_member="weights/synthetic.zip",archive_bytes=archive.stat().st_size,
        archive_sha256=file_sha256(archive),model_fingerprint=dict(sha256_model_state="synthetic"),encode=dict(model_raw_bytes=100,wall_seconds=1),
        encode_process_cpu_seconds=.8,decode_process_cpu_seconds=.1,verification_process_cpu_seconds=.1)
    receipt=exporter.assemble_package([row],[record],[],dict(selection_rule="synthetic frozen"),{},
        dict(name="synthetic_budget",logical=1,unique=1,limit_bytes=20<<20),lambda *args:None,lambda:None,time.monotonic(),time.process_time(),[])
    assert receipt["logical_endpoints"]==receipt["verified_unique_weights"]==1
    assert receipt["alias_count"]==0 and receipt["all_original_checkpoints_retained"]
    assert Path(receipt["tar_path"]).exists()
    manifest=exporter.read(delivery/"synthetic_budget_manifest.json")
    assert manifest["logical_runs"][0]["all_nonweight_metadata_roundtrip_verified"]
