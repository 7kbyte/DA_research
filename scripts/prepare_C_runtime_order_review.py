#!/usr/bin/env python3
"""Read-only C-order proposal; retain every original serialized job byte."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from schedule_jobs import digest_job,validate_queue
from stat_bound_digest_cache import atomic,small_proof


def job_segments(raw):
    text=raw.decode('utf-8');position=0;decoder=json.JSONDecoder();segments={};order=[]
    while position<len(text) and text[position].isspace():position+=1
    if text[position]!='[':raise ValueError('A flat queue list is required')
    position+=1
    while True:
        while position<len(text) and (text[position].isspace() or text[position]==','):position+=1
        if text[position]==']':break
        value,end=decoder.raw_decode(text,position);name=value['id']
        if name in segments:raise ValueError('Duplicate job ID')
        segments[name]=text[position:end].encode('utf-8');order.append(name);position=end
    if json.loads(raw)!=[json.loads(segments[n]) for n in order]:raise ValueError('Raw segment decoder disagrees with the queue')
    return order,segments


def propose(queue,state,rows):
    ids={r['id'] for r in rows}
    if len(ids)!=63 or len(rows)!=63:raise ValueError('All63 formal C jobs required')
    current={j['id']:j for j in queue}
    if not ids<=set(current):raise ValueError('C registry is not fully present in the master')
    if any(state['jobs'].get(name,{}).get('status')!='pending' for name in ids):raise ValueError('Every C job must still be pending')
    selected=[]
    for model,group in [('pythia-410m','B4'),('pythia-1b','B4'),('pythia-2.8b','B4'),('pythia-410m','B2')]:
        choices=[r['id'] for r in rows if r['model']==model and r['group']==group and r['seed']==11]
        if len(choices)!=1:raise ValueError('Exactly one fixed seed11 representative per requested model/objective')
        selected+=choices
    old=[j['id'] for j in queue if j['id'] in ids];new=selected+[n for n in old if n not in selected]
    iterator=iter(new);candidate=[current[next(iterator)] if j['id'] in ids else j for j in queue]
    validate_queue(candidate)
    if [j for j in candidate if j['id'] not in ids]!=[j for j in queue if j['id'] not in ids]:raise ValueError('A non-C job changed position/order')
    if any(current[j['id']]!=j or digest_job(current[j['id']])!=digest_job(j) for j in candidate):raise ValueError('A job definition changed')
    return candidate,selected,old,new


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError('Use a new review namespace')
    queue_path=ROOT/'configs/supplemental_v1_queue.json';state_path=ROOT/'logs/supplemental_job_state.json'
    raw=queue_path.read_bytes();state_raw=state_path.read_bytes();queue=json.loads(raw);state=json.loads(state_raw)
    freeze_path=ROOT/'results/extended_formal_freeze.json';freeze=json.loads(freeze_path.read_bytes());rows=[]
    for phase in freeze['phases']:
        proof={'path':phase['active_registry'],'sha256':phase['active_registry_sha256']}
        if small_proof(proof['path'])!=proof:raise ValueError('Frozen C registry SHA changed')
        rows+=json.loads(Path(proof['path']).read_bytes())
    candidate,first4,old,new=propose(queue,state,rows);source_order,segments=job_segments(raw)
    candidate_raw=b'[\n  '+b',\n  '.join(segments[j['id']] for j in candidate)+b'\n]\n'
    new_order,new_segments=job_segments(candidate_raw)
    if json.loads(candidate_raw)!=candidate:raise ValueError('Candidate raw bytes changed a parsed job')
    if queue_path.read_bytes()!=raw:raise ValueError('Master was modified during the read-only review')
    proofrows=[];positions={n:i for i,n in enumerate(source_order)}
    for i,name in enumerate(new_order):
        before=segments[name];after=new_segments[name]
        if before!=after:raise ValueError('Job serialized bytes changed')
        proofrows.append({'id':name,'old_position':positions[name],'new_position':i,
            'serialized_job_sha256_before':hashlib.sha256(before).hexdigest(),'serialized_job_sha256_after':hashlib.sha256(after).hexdigest(),
            'serialized_job_bytes_equal':True,'definition_sha256_before':digest_job(queue[positions[name]]),
            'definition_sha256_after':digest_job(candidate[i])})
    a.output_dir.mkdir(parents=True);(a.output_dir/'source_master_queue.json').write_bytes(raw)
    (a.output_dir/'source_scheduler_state.json').write_bytes(state_raw);(a.output_dir/'candidate_queue.json').write_bytes(candidate_raw)
    B=[j['id'] for j in queue if j['id'].startswith('bankvar_')];pendingB=[n for n in B if state['jobs'][n]['status']=='pending']
    record={'schema_version':1,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'review_only_not_applied',
        'source_live_master_path':str(queue_path),'source_queue_snapshot':small_proof(a.output_dir/'source_master_queue.json'),
        'source_state_snapshot':small_proof(a.output_dir/'source_scheduler_state.json'),'candidate_queue':small_proof(a.output_dir/'candidate_queue.json'),
        'master_jobs':len(queue),'C_jobs':63,'C_pending':63,'fixed_first_four_C_ids':first4,
        'old_C_order':old,'proposed_C_order':new,'non_C_positions_and_relative_order_unchanged':True,
        'B_order_unchanged':B==[j['id'] for j in candidate if j['id'].startswith('bankvar_')],
        'B_pending_at_review':len(pendingB),'all_job_serialized_bytes_equal':True,'all_job_scheduler_definition_SHAs_equal':True,
        'rows':proofrows,'root_execution_authorization_required':True,'live_queue_or_state_modified':False,
        'any_GPU_work_started_or_stopped':False,'scores_read':False,'producer_source':small_proof(__file__),
        'compatibility':{'scheduler':'Reloads queue every loop; enforces per-ID definition hashes, not global order SHA.',
            'mutation_lock':'Future root-authorized publication must hold configs/supplemental_v1_queue.mutation.lock used by other mutators; scheduler itself does not take it.',
            'observers_and_cohort_gates':'Use per-job ID/definition and phase complete counts, preserving all24/24/15-run gates.',
            'prewarm':'v5 compares frozen job objects by ID; registered per-run final/best paths cover all models independent of order.',
            'phase_queues_and_registries':'Remain byte-identical; only the mutable master list positions may change.'},
        'future_publication_preconditions':['Root explicitly authorizes this order and records a new operational queue SHA.',
            'Under the mutation lock, require exact current master bytes/SHA match the reviewed source, every C job still pending, and abundant pending B work ahead of C.',
            'Write the reviewed candidate to a sibling temporary file, flush/fsync, atomic replace, fsync the directory; do not change state or restart/pause scheduler.',
            'Immediately verify all504 individual definitions, all63 C pending or subsequently launched under the new order, and unchanged B order.',
            'If another append occurred, regenerate a new review; if C dispatch is imminent/already started, do not apply this proposal blindly.'],
        'limitations':['The scheduler can finish an in-memory old queue iteration before its next reload; mutation lock alone is not a dispatch barrier.',
            'Current pending B work provides a safe lead-in, but this condition must be freshly verified during any later authorized publication.',
            'The four entries are first C launch order, not a guarantee of physical GPU0/1/2/3 mapping or simultaneous start.',
            'Concurrent large-model CPU/I/O setup and new cohort finish times may change operational timing; report measured steady steps, not a causal model-size speed claim.']}
    atomic(a.output_dir/'review.json',record)
    print(json.dumps({'review':small_proof(a.output_dir/'review.json'),'candidate':record['candidate_queue'],'first_four':first4,
        'all_job_bytes_equal':True,'B_pending':len(pendingB),'applied':False}))


if __name__=='__main__':main()
