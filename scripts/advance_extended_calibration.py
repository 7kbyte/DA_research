#!/usr/bin/env python3
"""Advance CPU selectors and append LR/lambda jobs to an existing outer master.

This driver never launches a scheduler, training process, or model forward. It
cannot generate/append formal extended runs. All GPU children belong to master.
"""
from __future__ import annotations
import argparse,datetime as dt,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from make_experiments import atomic_json
from make_extended_experiments import experiment_prefix,digest

MODELS=['pythia-410m','pythia-1b','pythia-2.8b']

def read(path):return json.loads(Path(path).read_text())

def cpu_command(script,*arguments):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,arguments)],check=True,cwd=ROOT,
                   env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})

def phase_paths(model,micro,phase):
    namespace=f'{experiment_prefix(model,micro)}_{phase}'
    return {'namespace':namespace,'registry':ROOT/f'configs/{namespace}_registry.json',
            'queue':ROOT/f'configs/{namespace}_queue.json','protocol':ROOT/f'docs/{namespace}_protocol.json'}

def require_registered_phase(paths,index):
    record=index['phases'][paths['namespace']]
    if record['registry_sha256']!=digest(paths['registry']) or record['queue_source_sha256']!=digest(paths['queue']):
        raise ValueError('Registered calibration phase artifacts changed')

def completion(paths,state):
    statuses={job['id']:state['jobs'].get(job['id'],{}).get('status','pending') for job in read(paths['queue'])}
    failed=[name for name,status in statuses.items() if status in {'failed','blocked'}]
    return all(status=='completed' for status in statuses.values()),failed

def ensure_phase(args,model,micro,phase,batch_record,lr_record=None):
    paths=phase_paths(model,micro,phase);index=read(args.phase_index)
    if paths['namespace'] not in index['phases']:
        command=['--model',model,'--microbatch',micro,'--phase',phase,
                 '--training-deadline',args.deadline,'--batch-selection-record',batch_record]
        if lr_record:command+=['--learning-rate-selection',lr_record]
        cpu_command('make_extended_experiments.py',*command)
        cpu_command('manage_supplemental_queue.py','append','--master-queue',args.master_queue,
                    '--phase-index',args.phase_index,'--phase',paths['namespace'],
                    '--phase-queue',paths['queue'],'--phase-registry',paths['registry'],'--priority')
    require_registered_phase(paths,read(args.phase_index));return paths

def select_phase(args,model,micro,phase,paths):
    destination=ROOT/f'results/extended/{model}/m{micro}'/('learning_rate_selection.json' if phase=='lr' else 'lambda_selection.json')
    if not destination.exists():
        cpu_command('select_extended_phase.py','--phase',phase,'--model',model,
            '--registry',paths['registry'],'--protocol',paths['protocol'],'--phase-queue',paths['queue'],
            '--master-state',args.master_state,'--output',destination)
    result=read(destination)
    if (result['model']!=model or result['microbatch_size']!=micro or
            result['source_protocol_sha256']!=digest(paths['protocol']) or
            result['source_registry_sha256']!=digest(paths['registry'])):
        raise ValueError('Saved calibration selection has different frozen sources')
    return destination,result

def advance(args):
    report={'updated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'models':{},
            'driver_source_sha256':digest(__file__),'gpu_work_launched_directly':False,
            'formal_extended_runs_authorized_by_driver':False,'master_state':str(args.master_state)}
    if not args.main_audit.exists() or read(args.main_audit).get('status')!='passed' or read(args.main_audit).get('completed_runs')!=36:
        return {**report,'status':'waiting_main36_audit'}
    if not args.master_state.exists():return {**report,'status':'waiting_existing_master'}
    state=read(args.master_state)
    if state.get('stopping'):return {**report,'status':'outer_master_stopping'}
    benchmarks=read(ROOT/'configs/supplemental_v1_benchmark_registry.json')
    # Let both initial benchmark jobs start before any priority insertion.
    if any(state['jobs'].get(row['id'],{}).get('status','pending')=='pending' for row in benchmarks):
        return {**report,'status':'waiting_initial_benchmark_launch'}
    for model in MODELS:
        if model=='pythia-2.8b':
            batch_record=ROOT/'results/benchmarks/pythia-2.8b_batch_recommendation.json'
        else:
            bench=next(row for row in benchmarks if row['model']==model)
            if state['jobs'].get(bench['id'],{}).get('status') not in {'completed','failed'}:
                report['models'][model]={'status':'waiting_batch_benchmark'};continue
            candidate=Path(bench['report_path'])
            if not candidate.exists():raise FileNotFoundError('Benchmark failed without its auditable report')
            batch_record=ROOT/f'results/benchmarks/{experiment_prefix(model,16).rsplit("_m",1)[0]}_batch_selection.json'
            if not batch_record.exists():
                command=['--model',model,'--candidate',candidate,'--output',batch_record]
                if model=='pythia-1b':command+=['--baseline',ROOT/'results/benchmarks/benchmark_extended_pythia_1b_m16_s101.json']
                cpu_command('select_extended_batch.py',*command)
            if read(batch_record)['candidate']['sha256']!=digest(candidate):raise ValueError('Batch benchmark changed')
        batch=read(batch_record);micro=batch['selected_microbatch']
        record={'status':'lr_running','microbatch':micro,'accumulation':64//micro,'batch_selection_path':str(batch_record)}
        paths=ensure_phase(args,model,micro,'lr',batch_record)
        done,failed=completion(paths,state)
        if failed:record.update(status='lr_failed',failed_jobs=failed)
        elif done:
            lr_path,lr=select_phase(args,model,micro,'lr',paths)
            record.update(status='lambda_running',learning_rate_selection_path=str(lr_path),selected_learning_rate=lr['selected_learning_rate'])
            paths=ensure_phase(args,model,micro,'lambda',batch_record,lr_path)
            done,failed=completion(paths,state)
            if failed:record.update(status='lambda_failed',failed_jobs=failed)
            elif done:
                weight_path,weight=select_phase(args,model,micro,'lambda',paths)
                record.update(status='calibration_complete',lambda_selection_path=str(weight_path),
                              selected_shared_lambda=weight['selected_shared_lambda'])
        report['models'][model]=record
    statuses=[r['status'] for r in report['models'].values()]
    report['status']='complete' if len(statuses)==3 and all(s=='calibration_complete' for s in statuses) else (
        'failed' if any(s.endswith('_failed') for s in statuses) else 'calibrating')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master-queue',type=Path,default=ROOT/'configs/supplemental_v1_queue.json')
    parser.add_argument('--phase-index',type=Path,default=ROOT/'configs/supplemental_v1_registry_index.json')
    parser.add_argument('--master-state',type=Path,default=ROOT/'logs/supplemental_job_state.json')
    parser.add_argument('--main-audit',type=Path,default=ROOT/'results/formal_run_audit.json')
    parser.add_argument('--output',type=Path,default=ROOT/'logs/extended_calibration_master_progress.json')
    parser.add_argument('--deadline',default='2026-11-07T00:00:00Z');parser.add_argument('--watch',action='store_true')
    parser.add_argument('--poll-seconds',type=float,default=30)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                report=advance(args);atomic_json(args.output,report)
                if not args.watch or report['status'] in {'complete','failed','outer_master_stopping'}:break
                if time.time()>=dt.datetime.fromisoformat(args.deadline.replace('Z','+00:00')).timestamp():break
                time.sleep(min(max(args.poll_seconds,1),60))
        except BaseException as error:
            atomic_json(args.output,{'status':'driver_error','exception':type(error).__name__,'message':str(error)})
            traceback.print_exc();raise
    print(json.dumps(report))

if __name__=='__main__':main()
