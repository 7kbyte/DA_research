#!/usr/bin/env python3
"""Prepare, or analyze only complete, authorized 67-job primary NLL results."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from replay_study.analysis import write_csv
from replay_study.nll_v2_analysis import collect_scores,plot_frontier,prepare_plan,read,sha,write_bundle
from monitor_primary_nll_readiness import MANIFEST_SHA,inspect,validate_primary_scope


def first_read_receipt(path,journal,*,manifest_sha,readiness_receipt,authorization_evidence=None):
    """Persist the first semantic access time immediately before json.load."""
    journal=Path(journal)
    if journal.exists():
        previous=read(journal)
        if (previous.get("manifest_sha256"),previous.get("role"),previous.get("phase"),previous.get("heldout_scope"))!=(
                manifest_sha,"primary","formal","primary NLL only; behavior and all other phases remain sealed"):
            raise ValueError("existing first-read journal belongs to another scope")
        return previous
    receipt=dict(first_real_test_semantic_read_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        first_score_path=str(path),manifest_sha256=manifest_sha,role="primary",phase="formal",phase_name="primary",
        model_tag="pythia-410m",data_family="main_v3",heldout_scope="primary NLL only; behavior and all other phases remain sealed",
        before_first_score_json_parse=True,all_67_jobs_completed_and_integrity_verified=True,
        readiness_receipt=str(readiness_receipt),readiness_receipt_sha256=sha(readiness_receipt),
        authorization="Root conditional authorization: only after successful enqueue, all67 complete, source/attestation/SHA checks pass",
        methods_grids_horizons_and_validation_selection_rules_frozen_before_first_test_access=True,
        test_used_for_model_or_hyperparameter_selection=False)
    if authorization_evidence is not None:receipt["authorization_evidence"]=authorization_evidence
    journal.parent.mkdir(parents=True,exist_ok=True)
    temporary=journal.with_name(journal.name+f".{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        stream.write(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n");stream.flush();os.fsync(stream.fileno())
    try:
        os.link(temporary,journal)  # Atomic first writer wins; never reset the first time.
    except FileExistsError:
        return first_read_receipt(path,journal,manifest_sha=manifest_sha,readiness_receipt=readiness_receipt,
                                  authorization_evidence=authorization_evidence)
    finally:
        temporary.unlink()
    fd=os.open(journal.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=ROOT/"docs/primary_evaluation_v2_execution_matched_nll_manifest.json")
    p.add_argument("--manifest-sha256",default=MANIFEST_SHA)
    p.add_argument("--state",type=Path,default=ROOT/"logs/supplemental_job_state.json")
    p.add_argument("--output",type=Path,default=ROOT/"results/analysis/primary_nll_v2")
    p.add_argument("--readiness",type=Path,default=ROOT/"results/primary_nll_readiness.json")
    p.add_argument("--first-read-journal",type=Path,default=ROOT/"docs/primary_nll_first_test_read.json")
    p.add_argument("--enqueue-confirmation",type=Path)
    p.add_argument("--unseal-primary-nll-after-complete",action="store_true")
    p.add_argument("--wait-seconds",type=float,default=0)
    a=p.parse_args()
    if a.manifest_sha256!=MANIFEST_SHA:raise ValueError("this authorization binds the exact root-approved primary manifest")
    manifest=read(a.manifest)
    validate_primary_scope(manifest)
    if not a.unseal_primary_nll_after_complete:
        a.output.mkdir(parents=True,exist_ok=True)
        plans=[]
        for endpoint in ("final","best"):
            plan=prepare_plan(a.manifest,a.manifest_sha256,role="primary",phase_name="primary",model_tag="pythia-410m",data_family="main_v3",checkpoint=endpoint)
            plans.append({k:plan[k] for k in ("manifest_sha256","identity","sources_sha256","score_files_read","expected_conditions")})
        (a.output/"prepared_only.json").write_text(json.dumps(dict(state="waiting_root_enqueue_confirmation_and_all67_complete",plans=plans),ensure_ascii=False,indent=2)+"\n")
        print("prepared only; no result files opened")
        return
    if a.enqueue_confirmation is None:raise ValueError("explicit successful enqueue confirmation is required")
    confirmation=read(a.enqueue_confirmation)
    if (confirmation.get("successful_enqueue_confirmed_by_root"),confirmation.get("manifest_sha256"),confirmation.get("phase"),confirmation.get("physical_jobs"))!=(
            True,a.manifest_sha256,"primary",67):raise ValueError("missing exact successful primary-only enqueue confirmation")
    if set(confirmation["job_ids"])!={j["id"] for j in manifest["jobs"]}:
        raise ValueError("root enqueue confirmation does not cover the exact manifest job IDs")
    authority_path=Path(confirmation["authority"])
    authority=read(authority_path)
    if (authority.get("authorized_by"),authority.get("phase"),authority.get("physical_jobs"),authority.get("manifest_sha256"),
            authority.get("training_methods_grids_and_horizons_frozen"),authority.get("behavior_and_other_phases_remain_sealed"))!=(
            "root","primary",67,a.manifest_sha256,True,True):raise ValueError("root primary-only authority differs")
    authorization_evidence=dict(enqueue_confirmation=str(a.enqueue_confirmation),enqueue_confirmation_sha256=sha(a.enqueue_confirmation),
        authority=str(authority_path),authority_sha256=sha(authority_path),frozen_training_metadata_sha256=authority["source_sha256"],
        analysis_sources_sha256={str(p):sha(p) for p in (Path(__file__),ROOT/"scripts/monitor_primary_nll_readiness.py",
            ROOT/"src/replay_study/nll_v2_analysis.py",ROOT/"src/replay_study/analysis.py")})
    deadline=time.monotonic()+a.wait_seconds
    while True:
        readiness=inspect(a.manifest,a.manifest_sha256,a.state)
        a.readiness.parent.mkdir(parents=True,exist_ok=True)
        tmp=a.readiness.with_name(a.readiness.name+".tmp");tmp.write_text(json.dumps(readiness,ensure_ascii=False,indent=2)+"\n");tmp.replace(a.readiness)
        print(json.dumps({k:readiness[k] for k in ("status","statuses","all_physical_jobs_complete")}),flush=True)
        if readiness["status"]=="passed_complete_primary_integrity":break
        if time.monotonic()>=deadline:raise RuntimeError("still incomplete; primary score access remains sealed")
        time.sleep(min(20,max(0,deadline-time.monotonic())))
    first_seen=False
    def before_read(path):
        nonlocal first_seen
        if not first_seen:
            first_read_receipt(path,a.first_read_journal,manifest_sha=a.manifest_sha256,readiness_receipt=a.readiness,
                               authorization_evidence=authorization_evidence)
            first_seen=True
    for endpoint in ("final","best"):
        plan=prepare_plan(a.manifest,a.manifest_sha256,role="primary",phase_name="primary",model_tag="pythia-410m",data_family="main_v3",checkpoint=endpoint)
        bundle=collect_scores(plan,read_heldout_scores=True,allow_partial=False,n_bootstrap=5000,before_score_read=before_read)
        bundle["first_real_test_read_receipt"]=read(a.first_read_journal)
        bundle["full67_readiness_receipt_sha256"]=sha(a.readiness)
        write_bundle(bundle,a.output/endpoint);plot_frontier(bundle,a.output/endpoint,figure_width=7)
        print(endpoint,"complete",bundle["completed_conditions"],"records; 5000 paired document bootstrap replicates",flush=True)
    # One global cost table includes initial once; endpoint tables must not be summed.
    costs=[]
    for job in manifest["jobs"]:
        path=Path(job["output"]);att=read(path.with_name(path.stem+"_execution.json"))
        costs.append(dict(inference_job_id=job["id"],kind=job["kind"],
                          inference_subprocess_wall_seconds=att["inference_subprocess_wall_seconds"],
                          input_verification_wall_seconds=att["verification_wall_seconds"],
                          teacher_forward_timing_is_nested=True))
    write_csv(a.output/"all67_physical_inference_costs.csv",costs)
    (a.output/"scope_and_costs.json").write_text(json.dumps(dict(status="complete_primary_only",manifest_sha256=a.manifest_sha256,
        logical_endpoints=72,physical_jobs=67,initial_inferences=1,adapted_inferences=66,avoided_alias_forward_calls=6,
        inference_subprocess_wall_seconds=sum(r["inference_subprocess_wall_seconds"] for r in costs),
        input_verification_wall_seconds=sum(r["input_verification_wall_seconds"] for r in costs),
        first_real_test_read_receipt=read(a.first_read_journal),bootstrap_replicates=5000,
        interval_scope="paired document bootstrap conditional on trained checkpoints; does not replace three-seed sample SD",
        selection_policy="All methods, grids, horizons and validation selection rules frozen; no new selection from test",
        behavior_and_other_phase_scores_still_sealed=True),ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":main()
