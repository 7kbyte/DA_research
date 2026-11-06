#!/usr/bin/env python3
"""Prepare and extend one supplemental scheduler; individual jobs inherit one GPU."""
from __future__ import annotations
import argparse,copy,datetime as dt,fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from make_experiments import atomic_json
from make_extended_experiments import build_extended
from schedule_jobs import validate_queue

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def payload_sha(payload):
    return hashlib.sha256((json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()).hexdigest()

def compatible_job(job):
    if 'CUDA_VISIBLE_DEVICES' in job.get('env',{}):raise ValueError('Individual jobs must inherit outer CUDA_VISIBLE_DEVICES')
    if any(Path(arg).name in {'calibrate_extended_experiments.py','schedule_jobs.py'} for arg in job['argv']):
        raise ValueError('Nested schedulers are excluded from the supplemental master queue')
    if any(Path(arg).name=='benchmark_extended_models.py' for arg in job['argv']) and '--one' not in job['argv']:
        raise ValueError('Benchmark jobs use --one and inherit the outer GPU assignment')

def merge_jobs(existing,incoming,priority=False):
    if len({j['id'] for j in existing})!=len(existing) or len({j['id'] for j in incoming})!=len(incoming):
        raise ValueError('Duplicate job IDs within a queue')
    lookup={job['id']:job for job in existing};fresh=[]
    for job in incoming:
        compatible_job(job)
        if job['id'] in lookup:
            if job!=lookup[job['id']]:raise ValueError(f'Immutable job definition changed: {job["id"]}')
        else:fresh.append(job);lookup[job['id']]=job
    combined=fresh+existing if priority else existing+fresh
    validate_queue(combined);return combined

def write_immutable(path,payload):
    if Path(path).exists() and json.loads(Path(path).read_text())!=payload:raise FileExistsError(path)
    atomic_json(path,payload)

def prepare(args):
    namespace=args.namespace;queue=[];bench_records=[]
    for model,micro in [('pythia-410m',64),('pythia-1b',32)]:
        configs,_,_=build_extended(model,'lambda',ROOT,args.scratch,3e-5,.05,micro,
                                  training_deadline=args.deadline)
        cfg=next(c for c in configs.values() if c['replay_objective']=='kl' and 'generated_random' in c['replay_path'] and c['replay_weight']==.05)
        run_id=f"{namespace}_benchmark_{model.replace('.','p').replace('-','_')}_m{micro}"
        output=args.scratch/'runs'/run_id;config_path=ROOT/f'configs/{namespace}/benchmarks/{run_id}.json'
        report=ROOT/f'results/benchmarks/{run_id}.json'
        cfg.update(output_dir=str(output),run_label=run_id,max_steps=10,evaluate_at_start=False,
            evaluate_at_end=False,evaluate_teacher_kl=False,validation_paths={},eval_every_steps=0)
        write_immutable(config_path,cfg)
        job={'id':run_id,'argv':[sys.executable,str(ROOT/'scripts/benchmark_extended_models.py'),'--one',str(config_path),'--output',str(report)],
            'env':{'CUBLAS_WORKSPACE_CONFIG':':4096:8'},'expected_seconds':180,'outputs':[str(report)]}
        queue.append(job);bench_records.append({'id':run_id,'model':model,'microbatch':micro,'phase':'batch_benchmark',
            'config_path':str(config_path),'output_dir':str(output),'report_path':str(report),'config_sha256':sha(config_path)})
    source=json.loads(args.A_registry.read_text());source_jobs=json.loads(args.A_queue.read_text())
    lookup={job['id']:job for job in source_jobs};records=[]
    for row in source:
        record=copy.deepcopy(row)
        if row['execution_kind']=='alias':records.append(record);continue
        original_path=Path(row['config_path']);original=json.loads(original_path.read_text())
        if row['config_sha256']!=sha(original_path):raise ValueError('Preregistered A config changed')
        if original['microbatch_size']!=16 or original['gradient_accumulation_steps']!=4:
            raise ValueError('A preserves main micro16 accum4; extended batch choices are separate')
        config=copy.deepcopy(original);config['deadline_utc']=args.deadline
        path=ROOT/f'configs/{namespace}/A/{row["id"]}.json';write_immutable(path,config)
        record.update(config_path=str(path),config_sha256=sha(path),active=True,
            preregistered_config_path=str(original_path),preregistered_config_sha256=sha(original_path),
            operational_override={'deadline_utc':args.deadline},launch_status='prepared_not_started')
        job=copy.deepcopy(lookup[row['id']]);index=job['argv'].index('--config')+1;job['argv'][index]=str(path)
        job['env']={**job.get('env',{}),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        queue.append(job);records.append(record)
    queue=merge_jobs([],queue)
    master=ROOT/f'configs/{namespace}_queue.json';index_path=ROOT/f'configs/{namespace}_registry_index.json'
    A_registry=ROOT/f'configs/{namespace}_A_registry.json';bench_registry=ROOT/f'configs/{namespace}_benchmark_registry.json'
    write_immutable(A_registry,records);write_immutable(bench_registry,bench_records)
    write_immutable(master,queue)
    write_immutable(index_path,{'training_deadline_utc':args.deadline,'master_queue':str(master),
        'phases':{'batch_benchmark':{'registry':str(bench_registry),'job_ids':[r['id'] for r in bench_records]},
            'A_lambda':{'registry':str(A_registry),'source_registry':str(args.A_registry),'source_registry_sha256':sha(args.A_registry),
                        'job_ids':[r['id'] for r in records if r['execution_kind']=='train']}},
        'status':'prepared_not_started','policy':'One outer scheduler, individual single-GPU jobs, CPU-only phase selection'})
    print(json.dumps({'status':'prepared_not_started','jobs':len(queue),'benchmark_jobs':2,'A_training_jobs':len(source_jobs),
        'master_queue':str(master),'phase_index':str(index_path)}))

def activate_prepared(args,payload):
    if not getattr(args,'activate_prepared_phase',False):
        raise ValueError('Prepared-only queue requires explicit --activate-prepared-phase')
    audit=json.loads(args.main_audit.read_text())
    if audit['status']!='passed' or audit['completed_runs']!=36:
        raise ValueError('Main36 full audit must pass before prepared-phase activation')
    if payload.get('prepared_only') is not True or payload.get('launch_authorized') is not False:
        raise ValueError('Expected an explicitly prepared-only source queue')
    if not args.phase or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.' for c in args.phase):
        raise ValueError('Invalid phase name')
    rows=json.loads(args.phase_registry.read_text());incoming=copy.deepcopy(payload['jobs']);validate_queue(incoming)
    if len({r['id'] for r in rows})!=len(rows) or {r['id'] for r in rows}!={j['id'] for j in incoming}:
        raise ValueError('Prepared registry and queue identities differ')
    if any(r.get('phase')!='conditional_bank_variance' or 'seed' in r or
           r.get('training_seed')!=11 or r.get('generation_seed')!=11 or
           not isinstance(r.get('selection_seed'),int) for r in rows):
        raise ValueError('Prepared bank draws must retain conditional phase and distinct training/selection seeds')
    folder=args.runtime_root/args.phase;runtime_rows=[];writes=[];proof={}
    lookup={r['id']:r for r in rows}
    preparation_path=args.phase_registry.parent/'preparation_manifest.json'
    preparation=json.loads(preparation_path.read_text())
    prepared_proof={r['id']:r for r in preparation['banks']}
    for job in incoming:
        compatible_job(job);row=lookup[job['id']];original_path=Path(row['config_path'])
        if sha(original_path)!=prepared_proof[row['id']]['config_sha256']:
            raise ValueError('Prepared training config changed')
        original=json.loads(original_path.read_text())
        if original['seed']!=row['training_seed']:raise ValueError('Selection draw must not become a training seed')
        config=copy.deepcopy(original);config['deadline_utc']=args.activation_deadline
        destination=folder/f'{row["id"]}.json';writes.append((destination,config))
        position=job['argv'].index('--config')+1
        if Path(job['argv'][position])!=original_path:raise ValueError('Prepared job config differs from registry')
        job['argv'][position]=str(destination)
        runtime=copy.deepcopy(row);runtime.update(config_path=str(destination),active=True,status='activated',
            prepared_config_path=str(original_path),prepared_config_sha256=sha(original_path),config_sha256=payload_sha(config))
        runtime_rows.append(runtime)
        proof[row['id']]={'prepared_config_path':str(original_path),'prepared_config_sha256':sha(original_path),
                         'runtime_config_path':str(destination),'runtime_config_sha256':payload_sha(config),
                         'operational_override':{'deadline_utc':args.activation_deadline}}
    runtime_registry=folder/'executor_registry.json';writes.append((runtime_registry,runtime_rows))
    activation={'phase':'conditional_bank_variance','source_registry':str(args.phase_registry),
        'source_registry_sha256':sha(args.phase_registry),'source_queue':str(args.phase_queue),
        'source_queue_sha256':sha(args.phase_queue),'preparation_manifest':str(preparation_path),
        'preparation_manifest_sha256':sha(preparation_path),'execution_registry':str(runtime_registry),
        'execution_registry_sha256':payload_sha(runtime_rows),
        'analysis_registry':str(args.phase_registry),'launch_activation':'Explicit master append with --activate-prepared-phase',
        'source_artifacts_modified':False,'config_overrides':proof}
    activation_path=folder/'activation.json';writes.append((activation_path,activation))
    return incoming,writes,{'execution_registry':str(runtime_registry),'execution_registry_sha256':payload_sha(runtime_rows),
                            'activation_path':str(activation_path),'activation_sha256':payload_sha(activation),
                            'analysis_registry':str(args.phase_registry)}

def append(args):
    with args.master_queue.with_suffix('.mutation.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        return append_locked(args)

def append_locked(args):
    existing=json.loads(args.master_queue.read_text());payload=json.loads(args.phase_queue.read_text())
    if isinstance(payload,dict):incoming,writes,activation=activate_prepared(args,payload)
    else:incoming,writes,activation=payload,[],{}
    for job in incoming:
        if args.depends_on:job['depends_on']=sorted(set(job.get('depends_on',[])+args.depends_on))
    merged=merge_jobs(existing,incoming,args.priority)
    index=json.loads(args.phase_index.read_text())
    record={'registry':str(args.phase_registry),'queue_source':str(args.phase_queue),
        'registry_sha256':sha(args.phase_registry),'queue_source_sha256':sha(args.phase_queue),
        'job_ids':[job['id'] for job in incoming],'active':True,'execution_status':'scheduled',**activation}
    if args.phase in index['phases'] and index['phases'][args.phase]!=record:raise ValueError('Phase already registered differently')
    index['phases'][args.phase]=record
    for path,value in writes:write_immutable(path,value)
    atomic_json(args.master_queue,merged);atomic_json(args.phase_index,index)
    print(json.dumps({'master_jobs':len(merged),'phase':args.phase,'priority':args.priority}))

def start(args):
    audit=json.loads(args.main_audit.read_text())
    if audit['status']!='passed' or audit['completed_runs']!=36:raise ValueError('Main36 full audit must pass before supplemental launch')
    validate_queue(json.loads(args.master_queue.read_text()))
    for job in json.loads(args.master_queue.read_text()):compatible_job(job)
    args.state.parent.mkdir(parents=True,exist_ok=True);log=args.state.with_suffix('.scheduler.log')
    with args.state.with_suffix('.lock').open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Supplemental scheduler already holds this state lock')
        fcntl.flock(handle,fcntl.LOCK_UN)
    index_path=getattr(args,'phase_index',None)
    index=json.loads(index_path.read_text()) if index_path else None
    if index and Path(index['master_queue']).resolve()!=args.master_queue.resolve():
        raise ValueError('Phase index belongs to another master queue')
    with log.open('ab',buffering=0) as stream:
        proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/schedule_jobs.py'),'--queue',str(args.master_queue),
            '--state',str(args.state),'--gpus',args.gpus,'--deadline',args.deadline],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    if index is not None:
        index.update(status='running',scheduler_pid=proc.pid,scheduler_state=str(args.state),
                     started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),gpus=args.gpus.split(','))
        for phase in index['phases'].values():phase.update(active=True,execution_status='scheduled')
        atomic_json(index_path,index)
    print(json.dumps({'status':'started_detached','pid':proc.pid,'state':str(args.state),'log':str(log)}))

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--namespace',default='supplemental_v1')
    p.add_argument('--scratch',type=Path,default=Path('/tmp/grad-research-20261106'))
    p.add_argument('--deadline',default='2026-11-07T00:00:00Z')
    p.add_argument('--A-registry',type=Path,default=ROOT/'configs/lambda_sweep_deterministic_registry.json')
    p.add_argument('--A-queue',type=Path,default=ROOT/'configs/lambda_sweep_deterministic_queue.json');p.set_defaults(function=prepare)
    a=sub.add_parser('append');a.add_argument('--master-queue',type=Path,required=True)
    a.add_argument('--phase-index',type=Path,required=True);a.add_argument('--phase',required=True)
    a.add_argument('--phase-queue',type=Path,required=True);a.add_argument('--phase-registry',type=Path,required=True)
    a.add_argument('--depends-on',action='append');a.add_argument('--priority',action='store_true')
    a.add_argument('--activate-prepared-phase',action='store_true')
    a.add_argument('--main-audit',type=Path,default=ROOT/'results/formal_run_audit.json')
    a.add_argument('--runtime-root',type=Path,default=ROOT/'configs/supplemental_v1/activated')
    a.add_argument('--activation-deadline',default='2026-11-07T00:00:00Z');a.set_defaults(function=append)
    s=sub.add_parser('start');s.add_argument('--master-queue',type=Path,required=True)
    s.add_argument('--state',type=Path,default=ROOT/'logs/supplemental_job_state.json')
    s.add_argument('--main-audit',type=Path,default=ROOT/'results/formal_run_audit.json')
    s.add_argument('--phase-index',type=Path,default=ROOT/'configs/supplemental_v1_registry_index.json')
    s.add_argument('--gpus',default='0,1,2,3');s.add_argument('--deadline',default='2026-11-07T00:00:00Z');s.set_defaults(function=start)
    args=parser.parse_args();args.function(args)

if __name__=='__main__':main()
