#!/usr/bin/env python3
"""Prepare or explicitly execute one model's preregistered extended calibration."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from make_experiments import atomic_json
from make_extended_experiments import MODELS,experiment_prefix
from select_hyperparameters import select_learning_rate,select_shared_lambda

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(args):
    prefix=experiment_prefix(args.model,args.microbatch)
    microbatch=args.microbatch or MODELS[args.model]['microbatch']
    destination=ROOT/f'results/extended/{args.model}/m{microbatch}'
    def generate(phase,extra=()):
        argv=[sys.executable,str(ROOT/'scripts/make_extended_experiments.py'),'--model',args.model,'--phase',phase]
        if args.microbatch:argv+=['--microbatch',str(args.microbatch)]
        argv+=['--training-deadline',args.training_deadline]
        subprocess.run(argv+list(extra),check=True,cwd=ROOT)
    def schedule(phase):
        namespace=f'{prefix}_{phase}'
        subprocess.run([sys.executable,str(ROOT/'scripts/schedule_jobs.py'),'--queue',
            str(ROOT/f'configs/{namespace}_queue.json'),'--state',str(ROOT/f'logs/{namespace}_job_state.json'),
            '--gpus',args.gpus,'--deadline',args.training_deadline,'--exit-when-idle'],check=True,cwd=ROOT)
        state=json.loads((ROOT/f'logs/{namespace}_job_state.json').read_text())
        if any(row['status']!='completed' for row in state['jobs'].values()):raise RuntimeError(namespace)
        return state
    generate('lr')
    if not args.execute:
        print('Prepared only; no GPU jobs started. --execute is an explicit launch action.');return
    if not args.benchmark_record:
        raise ValueError('Execution requires the model-specific measured batch selection record')
    if args.benchmark_record:
        measured=json.loads(args.benchmark_record.read_text())
        if measured.get('model')!=args.model or measured.get('selected_microbatch')!=microbatch:
            raise ValueError('Benchmark must freeze this model and microbatch')
        protocol_path=ROOT/f'docs/{prefix}_lr_protocol.json'
        protocol=json.loads(protocol_path.read_text())
        protocol.update(batch_selection_record=str(args.benchmark_record),batch_selection_record_sha256=sha(args.benchmark_record))
        atomic_json(protocol_path,protocol)
    schedule('lr')
    protocol=json.loads((ROOT/f'docs/{prefix}_lr_protocol.json').read_text())
    records=json.loads((ROOT/f'configs/{prefix}_lr_registry.json').read_text())
    initial_path=Path(protocol['initial_validation_path']);initial=json.loads(initial_path.read_text())
    evaluations={}
    for row in records:
        summary=json.loads(Path(row['summary_path']).read_text())
        evaluation=destination/f"lr_{row['label']}_validation.json"
        atomic_json(evaluation,{'corpora':summary['validation_trajectory'][-1]['metrics'],
            'source_summary':row['summary_path'],'checkpoint':row['checkpoint_path']})
        evaluations[f"lr_{row['label']}"]={'evaluation_path':str(evaluation)}
    lr=select_learning_rate(records,initial,protocol,evaluations)
    lr.update(model=args.model,data_version='extended_v1',deterministic_algorithms=True,
        microbatch_size=protocol['microbatch_size'],gradient_accumulation_steps=protocol['gradient_accumulation_steps'],
        initial_validation_path=str(initial_path),initial_validation_sha256=sha(initial_path),
        source_protocol_path=str(ROOT/f'docs/{prefix}_lr_protocol.json'),
        source_protocol_sha256=sha(ROOT/f'docs/{prefix}_lr_protocol.json'))
    lr_path=destination/'learning_rate_selection.json';atomic_json(lr_path,lr)
    generate('lambda',['--learning-rate-selection',str(lr_path)])
    protocol_path=ROOT/f'docs/{prefix}_lambda_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    protocol.update(initial_validation_path=str(initial_path),initial_validation_sha256=sha(initial_path),
        learning_rate_selection_record=str(lr_path),learning_rate_selection_record_sha256=sha(lr_path))
    atomic_json(protocol_path,protocol)
    state=schedule('lambda')
    records=json.loads((ROOT/f'configs/{prefix}_lambda_registry.json').read_text())
    summaries={r['id']:json.loads(Path(r['summary_path']).read_text()) for r in records}
    result=select_shared_lambda(records,initial,protocol,summaries,state)
    result.update(model=args.model,data_version='extended_v1',deterministic_algorithms=True,
        microbatch_size=protocol['microbatch_size'],gradient_accumulation_steps=protocol['gradient_accumulation_steps'],
        source_protocol_path=str(protocol_path),source_protocol_sha256=sha(protocol_path))
    atomic_json(destination/'lambda_selection.json',result)
    print(json.dumps({'model':args.model,'selected_learning_rate':lr['selected_learning_rate'],
        'selected_shared_lambda':result['selected_shared_lambda'],'formal_started':False}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--microbatch',type=int)
    parser.add_argument('--benchmark-record',type=Path)
    parser.add_argument('--gpus',default='0,2,3')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--training-deadline',default='2026-11-07T00:00:00Z')
    run(parser.parse_args())
