#!/usr/bin/env python3
"""Score-blind primary-NLL completion gate; no partial score JSON is parsed."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replay_study.nll_v2_analysis import prepare_plan, read, sha

MANIFEST_SHA = "d1502bcaff5cdcd75d984a4b8776f1cb4027b596f474835bfe9ed18875d9fd18"


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def validate_primary_scope(manifest, *, expected_runs=36, expected_jobs=67):
    if manifest.get("schema_version") != 2 or manifest.get("mode") != "nll":
        raise ValueError("explicit schema2 NLL identity required")
    phases = manifest["phases"]
    if len(phases) != 1 or any(phases[0].get(k) != v for k,v in dict(name="primary",role="primary",model_tag="pythia-410m",
            expected_phase="formal",data_family="main_v3",rows=expected_runs).items()):
        raise ValueError("only the frozen primary/410M/main_v3 phase is authorized")
    jobs = {j["id"]:j for j in manifest["jobs"]}
    records = manifest["records"]
    if (len(jobs) != expected_jobs or len(jobs) != len(manifest["jobs"]) or len(records) != expected_runs*2 or
            len({r["run_id"] for r in records}) != expected_runs or len(manifest["baseline_groups"]) != 1):
        raise ValueError("complete primary physical/logical job inventory required")
    for r in records:
        if any(r.get(k) != v for k,v in dict(role="primary",phase_name="primary",phase="formal",model_tag="pythia-410m",data_family="main_v3").items()):
            raise ValueError("foreign phase/model/data record forbidden")
    if len({(r["run_id"],r["kind"]) for r in records}) != expected_runs*2 or {r["kind"] for r in records} != {"final","best"}:
        raise ValueError("every primary run must have exactly final and best")
    used = {r["inference_job_id"] for r in records} | {g["initial_job_id"] for g in manifest["baseline_groups"].values()}
    if used != set(jobs) or any(j["kind"] not in {"initial","adapted"} for j in jobs.values()):
        raise ValueError("foreign, unused or imported physical inference forbidden in this primary gate")
    return jobs


def inspect(manifest_path, manifest_sha, state_path, *, expected_runs=36, expected_jobs=67):
    if sha(manifest_path) != manifest_sha:
        raise ValueError("primary frozen manifest SHA differs")
    manifest = read(manifest_path)
    jobs = validate_primary_scope(manifest,expected_runs=expected_runs,expected_jobs=expected_jobs)
    raw_state = Path(state_path).read_bytes()
    state = json.loads(raw_state)
    statuses = {jid:state["jobs"].get(jid,{}) for jid in jobs}
    complete = all(s.get("status")=="completed" and s.get("returncode")==0 and not s.get("missing_outputs") for s in statuses.values())
    result = dict(status="pending",checked_at_utc=utc(),manifest=str(manifest_path),manifest_sha256=manifest_sha,
        state_path=str(state_path),state_sha256=__import__("hashlib").sha256(raw_state).hexdigest(),expected_physical_jobs=expected_jobs,
        expected_logical_records=expected_runs*2,phase="primary",role="primary",model_tag="pythia-410m",data_family="main_v3",
        statuses=dict(Counter(s.get("status","not_queued") for s in statuses.values())),
        all_physical_jobs_complete=complete,score_json_semantically_read=False,result_files_hashed=False,
        behavior_or_other_phase_results_opened=False)
    if not complete:
        # No existence checks, attestation reads, hashes or parsing of any result.
        return result
    sources={str(manifest_path):manifest_sha}
    def verify(path,expected):
        observed=sha(path)
        if observed!=expected:raise ValueError("frozen source/result SHA differs: "+str(path))
        sources[str(path)]=observed
    for name,expected in manifest.get("common_input_sha256",{}).items():verify(name,expected)
    for phase in manifest["phases"]:
        verify(phase["protocol"],phase["protocol_sha256"])
        verify(phase["registry"],phase["registry_sha256"])
    for group in manifest["baseline_groups"].values():
        for name,expected in group["evaluation_input_sha256"].items():verify(name,expected)
        for name,expected in group["base_model_assets_sha256"].items():verify(name,expected)
    # Planning checks registry/config/alias proofs without opening score paths.
    for endpoint in ("final","best"):
        prepare_plan(manifest_path,manifest_sha,role="primary",phase_name="primary",model_tag="pythia-410m",data_family="main_v3",checkpoint=endpoint)
    attestations=[]
    logical=[dict(job=j,path=j["output"],record=None) for j in jobs.values()]
    for record in manifest["records"]:
        job=jobs[record["inference_job_id"]]
        if record["output"]!=job["output"]:logical.append(dict(job=job,path=record["output"],record=record))
    for case in logical:
        job,path,record=case["job"],Path(case["path"]),case["record"]
        ap=path.with_name(path.stem+"_execution.json")
        att=read(ap);sources[str(ap)]=sha(ap)
        checkpoint_sha=record["checkpoint_sha256"] if record else job.get("checkpoint_sha256")
        if (att.get("manifest_sha256"),att.get("job_id"),att.get("verified_frozen_inputs"),att.get("baseline_group"),att.get("checkpoint_sha256"))!=(
                manifest_sha,job["id"],True,job["baseline_group"],checkpoint_sha):
            raise ValueError("primary execution attestation source differs")
        if att.get("weights_identity")!=(record["weights_identity"] if record else job.get("weights_identity")):
            raise ValueError("primary complete model-state identity differs")
        if record:
            if (att.get("inference_alias") is not True or att.get("logical_record_id")!=record["id"] or
                    att.get("inference_subprocess_wall_seconds")!=0):
                raise ValueError("logical alias lacks explicit zero-forward attestation")
        elif att.get("inference_alias") is not False or att.get("actual_argv")!=job["argv"]:
            raise ValueError("primary physical inference argv/alias differs")
        if set(att["output_sha256"])!={str(path)}:
            raise ValueError("NLL attestation cannot include behavioral/other result output")
        verify(path,att["output_sha256"][str(path)])
        attestations.append(dict(job_id=job["id"],logical_record_id=record["id"] if record else None,
                                 attestation=str(ap),attestation_sha256=sources[str(ap)],output=str(path),output_sha256=sources[str(path)]))
    for path,expected in sources.items():
        if sha(path)!=expected:raise ValueError("readiness source changed during verification")
    result.update(status="passed_complete_primary_integrity",result_files_hashed=True,score_json_semantically_read=False,
                  attestation_records=len(attestations),attestations=attestations,sources_sha256=sources,
                  checker_source_sha256=sha(__file__),verified_at_utc=utc())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=ROOT/"docs/primary_evaluation_v2_execution_matched_nll_manifest.json")
    p.add_argument("--manifest-sha256",default=MANIFEST_SHA)
    p.add_argument("--state",type=Path,default=ROOT/"logs/supplemental_job_state.json")
    p.add_argument("--output",type=Path,default=ROOT/"results/primary_nll_readiness.json")
    p.add_argument("--watch-seconds",type=float,default=0)
    p.add_argument("--poll-seconds",type=float,default=20)
    a=p.parse_args();end=time.monotonic()+a.watch_seconds
    while True:
        value=inspect(a.manifest,a.manifest_sha256,a.state)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        tmp=a.output.with_suffix(a.output.suffix+".tmp");tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n");tmp.replace(a.output)
        print(json.dumps({k:value[k] for k in ("status","statuses","all_physical_jobs_complete","score_json_semantically_read")}),flush=True)
        if value["all_physical_jobs_complete"] or time.monotonic()>=end:break
        time.sleep(min(a.poll_seconds,30,max(0,end-time.monotonic())))


if __name__=="__main__":main()
