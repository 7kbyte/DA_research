#!/usr/bin/env python3
"""One-shot CPU watcher for the explicitly authorized full A-NLL phase only."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from replay_study import supplemental_analysis_guard_v2 as gate


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:f.write(gate.encoded(value));f.flush();os.fsync(f.fileno())
    temp.replace(path)


def readiness(plan,state,master):
    definitions={j['id']:j for j in master};counts=Counter();pending=[]
    for item in plan['closure_jobs']:
        name=item['job_id'];expected=item['definition'];actual=state['jobs'].get(name)
        if definitions.get(name)!=expected:raise ValueError('Frozen master definition changed or missing: '+name)
        digest=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
        if actual is None or actual.get('definition_sha256')!=digest:raise ValueError('Missing/mismatched closure scheduler record: '+name)
        status=actual.get('status');counts[status]+=1
        if status in ('failed','blocked') or (status=='completed' and (actual.get('returncode')!=0 or actual.get('missing_outputs') or actual.get('stop_requested_at'))):
            raise ValueError('Closure job unsuccessful or incomplete: '+name)
        if status!='completed':pending.append(name)
    if not pending:
        for item in plan['closure_jobs']:
            missing=[p for p in item['definition'].get('outputs',[]) if not Path(p).is_file()]
            if missing:raise FileNotFoundError('Completed job declared output missing: '+item['job_id'])
    return {'ready':not pending,'closure_jobs':len(plan['closure_jobs']),'status_counts':dict(counts),'pending_job_ids':pending,
        'score_outputs_parsed_or_hashed':False,'watcher_reads_state_and_file_metadata_only':True}


def watch(config_proof):
    config=gate.Verifier().proof(config_proof)
    if config['scope']!='ONLY_A_NLL_full126_and_validation_selected18':raise ValueError('Only authorized A NLL scope supported')
    if gate.sha(__file__)!=config['watcher_source']['sha256']:raise ValueError('Watcher source changed')
    plan,_=gate.authorize(config['plan'],config['authority'])
    if plan['contract_id']!='supplement_A_nll' or plan['counts']['logical_endpoints']!=126 or plan['closure_job_count']!=145:raise ValueError('Complete A126/prior19 closure required')
    root=Path(config['watch_root']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    launch=root/'ANALYSIS_LAUNCH.json'
    if launch.exists() or Path(plan['first_read_receipt_path']).exists() or Path(config['analysis_output']).exists():raise FileExistsError('Analysis attempt/first-read/output already exists; no automatic retry or timestamp reset')
    started=gate.now()
    try:
        while True:
            for p in (config_proof,config['plan'],config['authority']):
                if gate.sha(p['path'])!=p['sha256']:raise ValueError('Watcher authority/config/plan changed: '+p['path'])
            status=readiness(plan,gate.read(config['state']),gate.read(config['master_queue']))
            atomic(root/'status.json',{'status':'waiting_full_A_closure' if not status['ready'] else 'ready_to_launch_once','updated_at_utc':gate.now(),
                'started_at_utc':started,'config':config_proof,'plan':config['plan'],'authority':config['authority'],**status})
            if status['ready']:break
            if gate.now()>='2026-11-07T03:26:32':raise TimeoutError('Hard deadline reached before full A completion')
            time.sleep(min(30,config['poll_seconds']))
        gate.authorize(config['plan'],config['authority'])
        gate.immutable(launch,{'launched_at_utc':gate.now(),'config':config_proof,'argv':config['analysis_argv'],'scope':config['scope'],
            'whole_runtime_gate_and_immutable_first_read_remain_in_frozen_v2_analyze':True,'automatic_attempt_number':1})
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        begin=time.perf_counter()
        with (root/'analysis.log').open('xb') as log:
            completed=subprocess.run(config['analysis_argv'],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=False)
        elapsed=time.perf_counter()-begin
        if completed.returncode:raise RuntimeError('Frozen analyze failed with return code '+str(completed.returncode)+'; inspect analysis.log, no retry permitted')
        result=gate.read(Path(config['analysis_output'])/'COMPLETE.json')
        if result.get('complete') is not True or result['plan']!=config['plan'] or result['authority']!=config['authority']:raise ValueError('Unexpected final analysis identity/completion')
        final={'status':'completed','completed_at_utc':gate.now(),'config':config_proof,'analysis_wall_seconds':elapsed,'returncode':0,
            'analysis_complete':gate.proof(Path(config['analysis_output'])/'COMPLETE.json'),'first_read_receipt':result['first_read_receipt'],
            'all_A_NLL_results_now_authorized_and_parsed':True,'other_behavior_B_C_scores_remain_sealed':True,
            'A_main18_reused_results_were_previously_seen':True,'GPU_inference_started':False,'scope':config['scope']}
        gate.immutable(root/'COMPLETED.json',final);atomic(root/'status.json',final)
    except Exception as error:
        atomic(root/'status.json',{'status':'failed_stopped_no_retry','failed_at_utc':gate.now(),'config':config_proof,'error':str(error),
            'analysis_attempt_exists':launch.exists(),'first_read_receipt_exists':Path(plan['first_read_receipt_path']).exists(),
            'instruction':'Stop and report; do not reset journal, overwrite first analysis, or retry automatically'})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--config-sha256',required=True);a=p.parse_args()
    watch({'path':str(a.config.resolve()),'sha256':a.config_sha256})
