#!/usr/bin/env python3
"""Prepare/start a score-blind evaluation-only successor after the old master exits."""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_formal_runs import sha256
from make_experiments import atomic_json
from run_frozen_evaluation import attestation_path,verify_outputs
from schedule_jobs import digest_job,validate_queue

HARD_DEADLINE='2026-11-07T03:26:32Z'
WORKER=ROOT/'scripts/run_frozen_evaluation.py'


def read(path):return json.loads(Path(path).read_text())


def checked(path,expected):
    if not expected or sha256(path)!=expected:raise ValueError(f'Frozen source SHA mismatch: {path}')
    return read(path)


def process_groups():
    """Include wrapper descendants that survive in a recorded child's process group."""
    processes={}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=path.read_text().rsplit(')',1)[1].split()
            processes[int(path.parent.name)]={'ppid':int(fields[1]),'pgrp':int(fields[2]),'state':fields[0]}
        except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError,IndexError):continue
    return processes


def assert_quiescent(state,processes=None):
    processes=process_groups() if processes is None else processes
    scheduler=state.get('scheduler_pid')
    if scheduler in processes:raise RuntimeError(f'Old master is still alive/unreaped: {scheduler}')
    groups={record['pid'] for record in state.get('jobs',{}).values() if isinstance(record.get('pid'),int)}
    remaining=[pid for pid,record in processes.items() if pid in groups or record['pgrp'] in groups or record['ppid']==scheduler]
    if remaining:raise RuntimeError(f'Old master children/process groups have not fully exited: {remaining}')


@contextlib.contextmanager
def old_master_lease(state_path):
    lock_path=Path(state_path).with_suffix('.lock')
    with lock_path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Old master still holds its scheduler lock') from error
        state=read(state_path);assert_quiescent(state)
        yield state,lock


def paths_for_output(output,mode,group):
    path=Path(output);paths={str(path),str(attestation_path(path))}
    if mode=='behavior':paths.update(str(path.with_name(path.stem+'_'+name+'.jsonl')) for name in group['tasks'])
    return paths


def job_definition(job,authorities,hashes):
    """Only the exact frozen single-GPU dispatcher grammar is accepted."""
    argv=job['argv']
    if (len(argv)!=10 or Path(argv[0]).resolve()!=Path(sys.executable).resolve() or Path(argv[1]).resolve()!=WORKER.resolve()
            or argv[2]!='--manifest' or argv[4]!='--manifest-sha256' or argv[6]!='--job-id'
            or argv[7]!=job['id'] or argv[8:]!=['--device','cuda:0']):
        raise ValueError(f'Evaluation-only queue rejects non-frozen worker argv: {job["id"]}')
    if job.get('env')!={'CUBLAS_WORKSPACE_CONFIG':':4096:8'}:raise ValueError('Only the frozen cuBLAS worker environment is accepted')
    duration=float(job.get('expected_seconds',0))
    if not math.isfinite(duration) or duration<0:raise ValueError('Expected evaluation duration must be finite and nonnegative')
    path=str(Path(argv[3]).resolve());key=(path,argv[5])
    if key not in authorities:raise ValueError('Job manifest must be explicitly listed with its SHA')
    manifest=authorities[key]
    common={str(Path(source).resolve()):wanted for source,wanted in manifest['common_input_sha256'].items()}
    if common.get(str(WORKER.resolve()))!=sha256(WORKER):raise ValueError('The frozen dispatcher source must be SHA-bound')
    matches=[entry for entry in manifest['jobs'] if entry['id']==job['id']]
    if len(matches)!=1:raise ValueError('Exactly one manifest job must bind each queue ID')
    entry=matches[0]
    if entry['kind'] not in {'initial','adapted','reuse'}:raise ValueError('Unsupported frozen evaluation kind')
    if entry['kind']!='reuse':
        scorer=ROOT/'scripts'/('run_evaluate.py' if manifest['mode']=='nll' else 'evaluate_tasks.py')
        if not entry.get('argv') or Path(entry['argv'][1]).resolve()!=scorer.resolve():
            raise ValueError('Manifest scorer must be the frozen NLL/behavior entrypoint')
        if common.get(str(scorer.resolve()))!=sha256(scorer):raise ValueError('The frozen scorer source must be SHA-bound')
    if job.get('depends_on',[])!=entry.get('depends_on',[]):raise ValueError('Queue dependencies differ from its frozen manifest')
    group=manifest['baseline_groups'][entry['baseline_group']]
    records=[row for row in manifest.get('records',[]) if row['inference_job_id']==job['id']]
    expected=set() if entry['kind']=='reuse' else paths_for_output(entry['output'],manifest['mode'],group)
    for row in records:expected.update(paths_for_output(row['output'],manifest['mode'],group))
    if set(job.get('outputs',[]))!=expected:raise ValueError('Queue outputs differ from its complete frozen output set')
    for source,wanted in {**manifest['common_input_sha256'],**entry['input_sha256']}.items():
        if source in hashes and hashes[source]!=wanted:raise ValueError('Conflicting input SHA authorities')
        hashes[source]=wanted
    return {'manifest':manifest,'entry':entry,'manifest_path':path,'manifest_sha256':argv[5],'records':records}


def evaluation_plan(spec,input_verifier=None):
    """Merge explicit evaluation sources and the full prior-baseline dependency closure."""
    authorities={};sources=[];hashes={};all_jobs={};bindings={};wanted=[]
    for role in ['evaluation_sources','prior_sources']:
        for source in spec.get(role,[]):
            queue=checked(source['queue_path'],source['queue_sha256'])
            if not isinstance(queue,list):raise ValueError('Evaluation source must be a flat job list')
            if len({j['id'] for j in queue})!=len(queue):raise ValueError('Duplicate job IDs within one source queue')
            for proof in source['manifests']:
                path=str(Path(proof['path']).resolve());manifest=checked(path,proof['sha256'])
                if manifest.get('schema_version')!=2 or manifest.get('mode') not in {'nll','behavior'}:
                    raise ValueError('Only explicit v2 NLL/behavior manifests are allowed')
                authorities[(path,proof['sha256'])]=manifest
            sources.append((role,queue))
    for role,queue in sources:
        for job in queue:
            binding=job_definition(job,authorities,hashes)
            if job['id'] in all_jobs and all_jobs[job['id']]!=job:raise ValueError(f'Immutable job definition differs: {job["id"]}')
            all_jobs[job['id']]=job;bindings[job['id']]=binding
            if role=='evaluation_sources':wanted.append(job['id'])
    selected=[];seen=set();visiting=set()
    def include(name):
        if name in seen:return
        if name in visiting:raise ValueError('Dependency cycle')
        if name not in all_jobs:raise ValueError(f'Explicit prior evaluation source required for dependency: {name}')
        visiting.add(name)
        for dependency in all_jobs[name].get('depends_on',[]):include(dependency)
        visiting.remove(name);seen.add(name);selected.append(copy.deepcopy(all_jobs[name]))
    for name in wanted:include(name)
    if not selected:raise ValueError('At least one explicit evaluation job is required')
    validate_queue(selected)
    selected_bindings={job['id']:bindings[job['id']] for job in selected}
    ancestors={}
    def dependency_ancestors(name):
        if name not in ancestors:
            direct=set(all_jobs[name].get('depends_on',[]));result=set(direct)
            for parent in direct:result.update(dependency_ancestors(parent))
            ancestors[name]=result
        return ancestors[name]
    # A reused baseline must be represented by its original explicit initial job.
    for job in selected:
        binding=selected_bindings[job['id']];entry=binding['entry'];manifest=binding['manifest']
        group=manifest['baseline_groups'][entry['baseline_group']]
        if entry['kind']=='initial':continue
        initial=group['initial_job_id']
        if initial not in selected_bindings or selected_bindings[initial]['entry']['kind']!='initial':
            raise ValueError('Complete original baseline must be present in the dependency closure')
        if initial not in dependency_ancestors(job['id']):raise ValueError('Initial baseline must precede its dependent inference in the DAG')
        origin=group.get('initial_origin')
        if origin and (selected_bindings[initial]['manifest_sha256']!=origin['manifest_sha256'] or
                       selected_bindings[initial]['entry']['output']!=origin['output']):
            raise ValueError('Prior baseline origin differs from its explicit frozen authority')
    # Only selected inputs are hashed; the many unrelated prior endpoints are excluded.
    selected_hashes={}
    for binding in selected_bindings.values():
        selected_hashes.update(binding['manifest']['common_input_sha256']);selected_hashes.update(binding['entry']['input_sha256'])
    for path,wanted_sha in selected_hashes.items():
        if input_verifier is None:
            if sha256(path)!=wanted_sha:raise ValueError(f'Frozen evaluation input changed: {path}')
        else:
            evidence=input_verifier(path,wanted_sha)
            if evidence!=(str(Path(path).resolve()),wanted_sha):raise ValueError('Explicit input verifier must return the checked canonical path and expected SHA')
    return selected,selected_bindings,selected_hashes


class InputVerification:
    """An explicit operational verifier; default behavior remains fresh full SHA."""
    def __init__(self,spec):
        from stat_bound_digest_cache import StatBoundHashes
        proof=spec.get('digest_cache')
        self.hashes=StatBoundHashes(snapshot=proof['path'],snapshot_sha256=proof['sha256']) if proof else None
        self.full_reads=0;self.full_read_bytes=0

    def verify(self,path,expected):
        if self.hashes is not None:return self.hashes.verify(path,expected)
        if sha256(path)!=expected:raise ValueError(f'Frozen evaluation input changed: {path}')
        self.full_reads+=1;self.full_read_bytes+=Path(path).stat().st_size
        return str(Path(path).resolve()),expected

    def receipt(self):
        counters=self.hashes.counters if self.hashes is not None else {'full_reads':self.full_reads,'full_read_bytes':self.full_read_bytes,'cache_hits':0,'invalidations':0}
        return {'mode':'stat_bound_historical_digest_with_fresh_fallback' if self.hashes else 'fresh_full_file_hash',
                'scope':'selected inference input files; small manifest/queue/output-attestation verification remains fresh',
                'counters':dict(counters),'cache_snapshot':self.hashes.snapshot_proof if self.hashes else None,
                'cache_reader_source':self.hashes.source if self.hashes else None,
                'cache_hits_are_current_byte_reads':False,'frozen_worker_always_fresh_full_input_hash':True}


def completion_proof(job,binding):
    """Read execution attestations and stream hashes, never deserialize scored outputs."""
    for output in job['outputs']:
        if not Path(output).is_file():raise ValueError(f'Missing completed output: {output}')
    manifest=binding['manifest'];entry=binding['entry'];group=manifest['baseline_groups'][entry['baseline_group']]
    endpoints=[]
    if entry['kind']!='reuse':endpoints.append((entry['output'],entry.get('checkpoint_sha256'),None))
    endpoints.extend((row['output'],row.get('checkpoint_sha256'),row) for row in binding['records']
                     if Path(row['output']).resolve()!=Path(entry['output']).resolve())
    if not endpoints:raise ValueError('Completed reuse requires materialized logical endpoint attestations')
    proofs={}
    for output,checkpoint,logical in endpoints:
        attestation=verify_outputs(output,binding['manifest_sha256'],job['id'])
        expected=paths_for_output(output,manifest['mode'],group)-{str(attestation_path(output))}
        if set(attestation.get('output_sha256',{}))!=expected or attestation.get('checkpoint_sha256')!=checkpoint:
            raise ValueError('Completed output set/checkpoint is not the frozen complete endpoint')
        if logical is None and (attestation.get('inference_alias') is True or attestation.get('actual_argv')!=entry['argv']):
            raise ValueError('Canonical completion does not bind the exact frozen scorer command')
        if logical is not None and (attestation.get('inference_alias') is not True or
                                   attestation.get('logical_record_id')!=logical['id'] or
                                   attestation.get('inference_subprocess_wall_seconds')!=0):
            raise ValueError('Logical alias lacks its own zero-inference attestation')
        proofs[str(attestation_path(output))]=sha256(attestation_path(output))
    if entry['kind']=='reuse':
        origin=entry['inference_origin'];verify_outputs(origin['output'],origin['manifest_sha256'],origin['job_id'])
    return proofs


def inherited_state(jobs,bindings,old_queue,old_state):
    lookup={job['id']:job for job in old_queue};records={};evidence={}
    for job in jobs:
        name=job['id'];digest=digest_job(job);old=old_state.get('jobs',{}).get(name)
        if name in lookup and lookup[name]!=job:raise ValueError(f'Old/new same-ID definition differs: {name}')
        if old and old.get('definition_sha256')!=digest:raise ValueError(f'Old state definition differs: {name}')
        record={'status':'pending','definition_sha256':digest};proof={'inherited':False,'reason':'not_completed_in_old_state'}
        if old and old.get('status')=='completed' and old.get('returncode')==0 and 'stop_requested_at' not in old and not old.get('missing_outputs'):
            if name not in lookup:raise ValueError('Completed state has no matching original queue definition')
            try:attestations=completion_proof(job,bindings[name])
            except (OSError,ValueError,KeyError) as error:proof['reason']=str(error)
            else:
                record=copy.deepcopy(old);record['inherited_completed_from_old_master']=True
                proof={'inherited':True,'definition_sha256':digest,'attestation_sha256':attestations}
        records[name]=record;evidence[name]=proof
    return {'jobs':records},evidence


def deadline_settings(spec):
    deadline=spec.get('hard_deadline_utc',HARD_DEADLINE)
    if deadline!=HARD_DEADLINE:raise ValueError('This handoff is bound to the authorized hard deadline')
    tail=float(spec.get('tail_seconds',1800));grace=90.;poll=10.
    if not math.isfinite(tail) or tail<1800:raise ValueError('At least30 minutes of finite final non-inference reserve is required')
    stop=tail+grace+2*poll
    return {'hard_deadline_utc':deadline,'tail_seconds':tail,'terminate_grace_seconds':grace,
        'poll_seconds':poll,'stop_margin_seconds':stop,'launch_margin_seconds':stop+300.,'gpus':'0,1,2,3'}


def prepare(args):
    spec=checked(args.spec,args.spec_sha256)
    if spec.get('schema_version')!=1:raise ValueError('Handoff spec schema_version1 required')
    if spec.get('example_only'):raise ValueError('Copy and explicitly finalize the example spec before preparing a handoff')
    settings=deadline_settings(spec)
    with old_master_lease(spec['old_master_state']) as (old_state,_):
        verifier=InputVerification(spec)
        old_queue=read(spec['old_master_queue']);jobs,bindings,inputs=evaluation_plan(spec,verifier.verify)
        new_state,evidence=inherited_state(jobs,bindings,old_queue,old_state)
        folder=args.output_dir.resolve()
        if folder.exists():raise FileExistsError('Use a new handoff namespace; original results are never overwritten')
        folder.mkdir(parents=True)
        for source,name in [(args.spec,'source_spec.json'),(Path(spec['old_master_state']),'old_state_snapshot.json'),
                            (Path(spec['old_master_queue']),'old_queue_snapshot.json')]:shutil.copyfile(source,folder/name)
        atomic_json(folder/'queue.json',jobs);atomic_json(folder/'state.json',new_state)
        record={'schema_version':1,'status':'prepared_not_started','prepared_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'source_spec':str(args.spec.resolve()),'source_spec_sha256':args.spec_sha256,'settings':settings,
            'old_master_state':spec['old_master_state'],'old_master_queue':spec['old_master_queue'],
            'old_state_sha256':sha256(folder/'old_state_snapshot.json'),'old_queue_sha256':sha256(folder/'old_queue_snapshot.json'),
            'queue':str(folder/'queue.json'),'queue_sha256':sha256(folder/'queue.json'),
            'state':str(folder/'state.json'),'prepared_state_sha256':sha256(folder/'state.json'),
            'inherited_completed_jobs':sum(p['inherited'] for p in evidence.values()),'total_evaluation_jobs':len(jobs),
            'completion_transfer':evidence,'input_sha256':inputs,'producer_source_sha256':sha256(__file__),
            'input_verification':verifier.receipt(),
            'scores_or_predictions_deserialized':False,'new_gpu_work_started':False}
        atomic_json(folder/'handoff.json',record)
    print(json.dumps({k:record[k] for k in ['status','total_evaluation_jobs','inherited_completed_jobs']}))


def start(args):
    record=checked(args.handoff,args.handoff_sha256);folder=args.handoff.resolve().parent
    with (folder/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'launch.json').exists():raise FileExistsError('Handoff already launched; review its state instead of duplicating it')
        if sha256(__file__)!=record['producer_source_sha256']:raise ValueError('Reviewed handoff helper changed after preparation')
        if sha256(folder/'old_state_snapshot.json')!=record['old_state_sha256'] or sha256(folder/'old_queue_snapshot.json')!=record['old_queue_sha256']:
            raise ValueError('Archived old-master snapshots changed')
        if sha256(record['queue'])!=record['queue_sha256'] or sha256(record['state'])!=record['prepared_state_sha256']:
            raise ValueError('Prepared queue/state changed before launch')
        spec=checked(record['source_spec'],record['source_spec_sha256']);settings=deadline_settings(spec)
        with old_master_lease(record['old_master_state']) as (old_state,lease):
            if sha256(record['old_master_state'])!=record['old_state_sha256'] or sha256(record['old_master_queue'])!=record['old_queue_sha256']:
                raise ValueError('Old master inputs/state changed after quiescent preparation')
            verifier=InputVerification(spec)
            jobs,bindings,_=evaluation_plan(spec,verifier.verify)
            if jobs!=read(record['queue']):raise ValueError('Source evaluation definitions changed')
            for job in jobs:
                if record['completion_transfer'][job['id']]['inherited']:
                    proof=completion_proof(job,bindings[job['id']])
                    if proof!=record['completion_transfer'][job['id']]['attestation_sha256']:
                        raise ValueError('Inherited output attestation changed; prepare a new handoff bundle')
            now=dt.datetime.now(dt.timezone.utc).timestamp()
            deadline=dt.datetime.fromisoformat(settings['hard_deadline_utc'].replace('Z','+00:00')).timestamp()
            if now>=deadline-settings['stop_margin_seconds']:raise RuntimeError('The protected final reserve has begun; do not start inference')
            command=[sys.executable,str(ROOT/'scripts/schedule_jobs.py'),'--queue',record['queue'],'--state',record['state'],
                '--gpus',settings['gpus'],'--deadline',settings['hard_deadline_utc'],'--launch-margin-seconds',str(settings['launch_margin_seconds']),
                '--stop-margin-seconds',str(settings['stop_margin_seconds']),'--terminate-grace-seconds',str(settings['terminate_grace_seconds']),
                '--poll-seconds',str(settings['poll_seconds']),'--exit-when-idle']
            with (folder/'scheduler.log').open('ab') as output:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                    start_new_session=True,pass_fds=(lease.fileno(),))
            launch={'pid':child.pid,'launched_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'argv':command,
                'handoff_sha256':args.handoff_sha256,'old_master_lock_inherited_by_new_scheduler':True,
                'input_verification':verifier.receipt(),
                'training_jobs_present':False,'score_files_deserialized':False}
            atomic_json(folder/'launch.json',launch)
    print(json.dumps(launch))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--spec',type=Path,required=True);p.add_argument('--spec-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.set_defaults(run=prepare)
    p=sub.add_parser('start');p.add_argument('--handoff',type=Path,required=True);p.add_argument('--handoff-sha256',required=True);p.set_defaults(run=start)
    args=parser.parse_args();args.run(args)


if __name__=='__main__':main()
