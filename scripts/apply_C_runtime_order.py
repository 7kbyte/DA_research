#!/usr/bin/env python3
"""Apply only the root-authorized C-order bytes, with strict live preconditions."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_C_runtime_order_review import job_segments
from schedule_jobs import digest_job,validate_queue
from stat_bound_digest_cache import atomic,fsync_directory,small_proof

REVIEW_SHA='ce22651f60609ee40eeb59824fb2829fd5eea1336a6cef93cc2c3b311c1604c8'
CANDIDATE_SHA='7cb38a7cb952557a7f48d656ea0615e787fdc75b5ff9abebe1ce28971fda975c'


def read(path):return json.loads(Path(path).read_bytes())


def master_pids():
    found=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:argv=path.read_bytes().decode().split('\0')
        except (OSError,UnicodeDecodeError):continue
        if any(Path(arg).name=='schedule_jobs.py' for arg in argv if arg):found.append(int(path.parent.name))
    return sorted(found)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root-authorized-execution',action='store_true',required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    folder=ROOT/'results/C_runtime_order_review_v1';review_path=folder/'review.json';candidate_path=folder/'candidate_queue.json'
    if small_proof(review_path)['sha256']!=REVIEW_SHA or small_proof(candidate_path)['sha256']!=CANDIDATE_SHA:raise ValueError('Only the explicit reviewed/root-authorized candidate is accepted')
    review=read(review_path);candidate_raw=candidate_path.read_bytes();candidate=validate_queue(json.loads(candidate_raw))
    source=ROOT/'configs/supplemental_v1_queue.json';state_path=ROOT/'logs/supplemental_job_state.json'
    if a.output_dir.exists():raise FileExistsError('Execution evidence must use a new immutable namespace')
    with source.with_suffix('.mutation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        before=source.read_bytes();state_raw=state_path.read_bytes();state=json.loads(state_raw);queue=json.loads(before)
        if hashlib.sha256(before).hexdigest()!=review['source_queue_snapshot']['sha256']:raise ValueError('Live source queue no longer matches the review')
        ids=set(review['old_C_order']);lookup={j['id']:j for j in queue};new={j['id']:j for j in candidate}
        if len(queue)!=504 or len(candidate)!=504 or set(lookup)!=set(new):raise ValueError('All504 job identities must remain present')
        if len(ids)!=63 or any(state['jobs'][n]['status']!='pending' for n in ids):raise ValueError('All63 C jobs must still be pending')
        B=[j['id'] for j in queue if j['id'].startswith('bankvar_')]
        pendingB=[n for n in B if state['jobs'][n]['status']=='pending'];runningB=[n for n in B if state['jobs'][n]['status']=='running']
        if len(pendingB)<8 or len(runningB)!=4 or set(state['active_jobs'])!=set(runningB):raise ValueError('Require at least8 pending B plus4 running B as a safe dispatch lead-in')
        pid=state['scheduler_pid']
        if master_pids()!=[pid]:raise ValueError('Exactly the original single master must be live')
        old_order,old_bytes=job_segments(before);new_order,new_bytes=job_segments(candidate_raw)
        for name in lookup:
            if lookup[name]!=new[name] or old_bytes[name]!=new_bytes[name] or digest_job(new[name])!=state['jobs'][name]['definition_sha256']:
                raise ValueError('A definition or serialized job byte differs')
        if [n for n in old_order if n not in ids]!=[n for n in new_order if n not in ids]:raise ValueError('Non-C order changed')
        if any((old_order[i] not in ids or new_order[i] not in ids) and old_order[i]!=new_order[i] for i in range(504)):
            raise ValueError('A non-C position changed')
        v5=read(ROOT/'docs/supplemental_evaluation_preparation_v5_execution.json')
        for path,wanted in {**v5['sources'],**v5['frozen_scorer_input_sha256']}.items():
            if small_proof(path)['sha256']!=wanted:raise ValueError('Frozen execution source changed')
        core=read(ROOT/'artifacts/provenance/formal_deterministic_repository_freeze.json')
        for path,entry in core['core_sources'].items():
            if small_proof(ROOT/path)['sha256']!=entry['current']:raise ValueError('Frozen scientific core changed')
        a.output_dir.mkdir(parents=True)
        (a.output_dir/'before_queue.json').write_bytes(before);(a.output_dir/'before_state.json').write_bytes(state_raw)
        (a.output_dir/'authorized_candidate.json').write_bytes(candidate_raw)
        temp=source.with_name(source.name+'.root_authorized_reorder_'+str(os.getpid())+'.tmp')
        with temp.open('xb') as stream:
            os.chmod(temp,source.stat().st_mode&0o777);stream.write(candidate_raw);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,source);fsync_directory(source.parent)
        if source.read_bytes()!=candidate_raw:raise ValueError('Published queue bytes differ from the candidate')
        publication=dt.datetime.now(dt.timezone.utc).isoformat()
    end=time.monotonic()+35
    while True:
        after_raw=state_path.read_bytes();after=json.loads(after_raw)
        if after['updated_at_utc']!=state['updated_at_utc']:break
        if time.monotonic()>end:raise RuntimeError('Published order; scheduler reload confirmation timed out, review before further changes')
        time.sleep(1)
    if after['scheduler_pid']!=pid or master_pids()!=[pid]:raise ValueError('Master identity changed after publication')
    if source.read_bytes()!=candidate_raw:raise ValueError('Another queue mutation followed publication')
    if any(after['jobs'][n]['definition_sha256']!=digest_job(new[n]) for n in new):raise ValueError('Reloaded state definition mismatch')
    if any(after['jobs'][n]['status']!='pending' for n in ids):raise ValueError('C dispatch began before post-publication review')
    if any(after['jobs'][n]['status'] in ('failed','blocked') for n in new):raise ValueError('A job failure was present after reload')
    (a.output_dir/'after_queue.json').write_bytes(source.read_bytes());(a.output_dir/'after_state.json').write_bytes(after_raw)
    record={'schema_version':1,'status':'published_reload_confirmed','published_at_utc':publication,'confirmed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'root_authorization':'Explicit parent message authorizing candidate SHA '+CANDIDATE_SHA,
        'review':small_proof(review_path),'candidate':small_proof(candidate_path),'before_queue':small_proof(a.output_dir/'before_queue.json'),
        'after_queue':small_proof(a.output_dir/'after_queue.json'),'before_state':small_proof(a.output_dir/'before_state.json'),'after_state':small_proof(a.output_dir/'after_state.json'),
        'master_pid_before_and_after':pid,'single_master_verified':True,'all504_raw_job_bytes_and_definition_SHAs_equal':True,
        'all63_C_still_pending_after_reload':True,'B_pending_before_publication':len(pendingB),'B_running_before_publication':len(runningB),
        'B_order_and_non_C_positions_unchanged':True,'first_four_C_ids':review['fixed_first_four_C_ids'],
        'file_and_parent_fsync_atomic_replace':True,'scheduler_or_children_paused_or_restarted':False,
        'new_GPU_launcher_started':False,'scientific_core10_and_frozen8_and_v5_sources_unchanged':True,'scores_read':False,'producer_source':small_proof(__file__)}
    atomic(a.output_dir/'execution_receipt.json',record)
    print(json.dumps({'receipt':small_proof(a.output_dir/'execution_receipt.json'),'master_pid':pid,'after_queue_sha256':small_proof(source)['sha256'],'status':record['status']}))


if __name__=='__main__':main()
