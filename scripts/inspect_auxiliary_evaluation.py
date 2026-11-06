#!/usr/bin/env python3
"""Score-blind authorized initial timing; replay-fit integrity only after all 15 finish."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluation_plan import encoded, sha

FIELDS=("status","gpu","pid","started_at_utc","ended_at_utc","elapsed_seconds","returncode","missing_outputs")


def inspect(authority_path,state_path):
    authority=json.loads(Path(authority_path).read_text());raw=Path(state_path).read_bytes();state=json.loads(raw)
    result={"recorded_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"authority":str(authority_path),"authority_sha256":sha(authority_path),
            "source_state_sha256":hashlib.sha256(raw).hexdigest(),"initials":[],"replay_fit":{},"scores_or_predictions_parsed":False,"new_inference_started":False}
    for permit in authority["jobs"]:
        if permit["phase"] not in {"planned_model_initial_timing_v1","replay_fit_diagnostic_v2"}:continue
        manifest_path=Path(permit["manifest"])
        if sha(manifest_path)!=permit["manifest_sha256"]:raise ValueError("Authority/manifest SHA mismatch")
        manifest=json.loads(manifest_path.read_text());jobs=manifest["jobs"]
        if len(jobs)!=permit["jobs"]:raise ValueError("Authority/job count mismatch")
        complete=all(state["jobs"].get(j["id"],{}).get("status")=="completed" and state["jobs"][j["id"]].get("returncode")==0 and not state["jobs"][j["id"]].get("missing_outputs") for j in jobs)
        replay=permit["phase"]=="replay_fit_diagnostic_v2"
        rows=[]
        for job in jobs:
            s=state["jobs"].get(job["id"],{});row={"id":job["id"],"state":{k:s[k] for k in FIELDS if k in s}}
            if not replay and job["kind"]!="initial":raise ValueError("Planned-model timing permits only initial jobs")
            ap=Path(job["output"]).with_name(Path(job["output"]).stem+"_execution.json")
            if ap.exists() and (not replay or complete):
                a=json.loads(ap.read_text())
                if a.get("manifest_sha256")!=permit["manifest_sha256"] or a.get("job_id")!=job["id"] or a.get("verified_frozen_inputs") is not True or a.get("checkpoint_sha256")!=job.get("checkpoint_sha256"):
                    raise ValueError("Attestation differs from frozen job")
                expected={job["output"]}
                if not replay:
                    expected.update(str(Path(job["output"]).with_name(Path(job["output"]).stem+"_"+name+".jsonl")) for name in manifest["baseline_groups"][job["baseline_group"]]["tasks"])
                if set(a["output_sha256"])!=expected or any(sha(p)!=v for p,v in a["output_sha256"].items()):raise ValueError("Attested output integrity failed")
                row.update(attestation=str(ap),attestation_sha256=sha(ap),output_hashes_valid=True,
                           execution={k:a[k] for k in ("verification_wall_seconds","inference_subprocess_wall_seconds","started_at_utc","finished_at_utc","preflight_environment") if k in a})
            elif complete:raise ValueError("Completed job lacks its execution attestation")
            rows.append(row)
        if replay:result["replay_fit"]={"all_15_completed":complete,"integrity_check_deferred_until_all_completed":not complete,"jobs":rows,"results_semantically_parsed":False}
        else:
            result["initials"]=rows;result["initial_manifest"]=str(manifest_path);result["initial_manifest_sha256"]=permit["manifest_sha256"]
            result["both_initials_completed"]=complete
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--authority",type=Path,default=Path("docs/auxiliary_evaluation_execution_20261106.json"));p.add_argument("--state",type=Path,default=Path("logs/supplemental_job_state.json"));p.add_argument("--output",type=Path,required=True);p.add_argument("--watch-seconds",type=float,default=0);p.add_argument("--poll-seconds",type=float,default=20);a=p.parse_args()
    end=time.monotonic()+a.watch_seconds
    while True:
        v=inspect(a.authority,a.state);a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix(a.output.suffix+".tmp");tmp.write_bytes(encoded(v));tmp.replace(a.output)
        print(json.dumps({"initial_states":[{"id":r["id"],"status":r["state"].get("status"),"scorer_wall_seconds":r.get("execution",{}).get("inference_subprocess_wall_seconds")} for r in v["initials"]],"replay_fit_all_completed":v["replay_fit"].get("all_15_completed"),"scores_parsed":False}),flush=True)
        if (v.get("both_initials_completed") and v["replay_fit"].get("all_15_completed")) or time.monotonic()>=end:break
        time.sleep(min(a.poll_seconds,30,max(0,end-time.monotonic())))


if __name__=="__main__":main()
