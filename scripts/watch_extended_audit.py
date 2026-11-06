#!/usr/bin/env python3
"""Wait on the outer scheduler and run the final extended CPU audit once."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_formal_runs import sha256
from make_experiments import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'results/extended_formal_freeze.json')
    p.add_argument('--state',type=Path,default=ROOT/'logs/supplemental_job_state.json')
    p.add_argument('--progress',type=Path,default=ROOT/'logs/extended_final_audit_watch.json')
    p.add_argument('--audit-output',type=Path,default=ROOT/'results/extended_run_audit.json')
    p.add_argument('--proof-output',type=Path,default=ROOT/'results/extended_final_model_equivalence_proofs.json')
    p.add_argument('--until',default='2026-11-07T00:10:00Z')
    p.add_argument('--poll-seconds',type=float,default=60.)
    args=p.parse_args()
    if not 1<=args.poll_seconds<=60:p.error('Polling must be between1 and60 seconds')
    deadline=dt.datetime.fromisoformat(args.until.replace('Z','+00:00'))
    manifest_sha=sha256(args.manifest);manifest=json.loads(args.manifest.read_text());ids=[]
    for phase in manifest['phases']:
        path=Path(phase['active_registry'])
        if sha256(path)!=phase['active_registry_sha256']:raise ValueError('Frozen active registry changed')
        ids.extend(row['id'] for row in json.loads(path.read_text()))
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate extended formal run IDs')
    audit_script=ROOT/'scripts/audit_extended_runs.py';audit_sha=sha256(audit_script)
    while True:
        if sha256(args.manifest)!=manifest_sha or sha256(audit_script)!=audit_sha:
            raise ValueError('Watch authority or audit implementation changed')
        state=json.loads(args.state.read_text());now=dt.datetime.now(dt.timezone.utc)
        complete=[name for name in ids if state['jobs'].get(name,{}).get('status')=='completed']
        failed=[name for name in ids if state['jobs'].get(name,{}).get('status') in {'failed','blocked'}]
        record={'updated_at_utc':now.isoformat(),'status':'waiting','expected_runs':len(ids),
            'completed_runs':len(complete),'failed_or_blocked_runs':failed,'manifest_sha256':manifest_sha,
            'audit_source_sha256':audit_sha,'pid':os.getpid(),'gpu_work_launched':False,
            'test_and_behavior_scores_read':False}
        if failed:
            record['status']='failed_job_requires_operator_review';atomic_json(args.progress,record);return 1
        if len(complete)==len(ids) or now>=deadline:
            record['status']='auditing';atomic_json(args.progress,record)
            command=[sys.executable,str(audit_script),'--freeze-manifest',str(args.manifest),
                '--state',str(args.state),'--output',str(args.audit_output),'--compare-weights',
                '--equivalence-proof-output',str(args.proof_output)]
            result=subprocess.run(command,cwd=ROOT,check=False)
            record.update(updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                status='passed' if result.returncode==0 else 'audit_failed',returncode=result.returncode,
                audit_output=str(args.audit_output),proof_output=str(args.proof_output),command=command)
            atomic_json(args.progress,record);return result.returncode
        atomic_json(args.progress,record);time.sleep(args.poll_seconds)


if __name__=='__main__':raise SystemExit(main())
