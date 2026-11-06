#!/usr/bin/env python3
"""CPU-only selection from completed jobs in one external supplemental scheduler."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
from make_experiments import atomic_json
from select_hyperparameters import select_learning_rate,select_shared_lambda
from replay_study.analysis import verify_recorded_execution,training_validation_execution

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require_complete_jobs(queue,state):
    ids=[job['id'] for job in queue]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate phase job IDs')
    missing=[name for name in ids if state['jobs'].get(name,{}).get('status')!='completed']
    if missing:raise ValueError(f'Phase selection requires every job completed: {missing}')

def select(args):
    protocol=json.loads(args.protocol.read_text());registry=json.loads(args.registry.read_text())
    queue=json.loads(args.phase_queue.read_text());state=json.loads(args.master_state.read_text())
    require_complete_jobs(queue,state)
    if protocol['model']!=args.model:raise ValueError('Wrong model protocol')
    for name,path in protocol['validation_paths'].items():
        if sha(path)!=protocol['validation_sha256'][name]:raise ValueError('Validation data changed')
    initial_path=Path(protocol['initial_validation_path']);initial=json.loads(initial_path.read_text())
    if protocol.get('initial_validation_sha256') and sha(initial_path)!=protocol['initial_validation_sha256']:
        raise ValueError('Initial baseline changed')
    policy=initial['execution_provenance']
    if not (policy['deterministic_algorithms'] and not policy['deterministic_warn_only'] and
            policy['cudnn_sdp_enabled'] and policy['model_weight_dtype']=='torch.float32' and policy['precision']=='bf16'):
        raise ValueError('Model baseline execution policy does not match strict training')
    for row in registry:
        cfg=json.loads(Path(row['config_path']).read_text())
        if row.get('config_sha256') and sha(row['config_path'])!=row['config_sha256']:raise ValueError('Phase config changed')
        for field in ['microbatch_size','gradient_accumulation_steps']:
            if cfg[field]!=protocol[field]:raise ValueError('Phase batch mismatch')
        summary=json.loads(Path(row['summary_path']).read_text())
        verify_recorded_execution(policy,training_validation_execution(summary))
        if initial.get('revision')!=cfg['model_revision']:raise ValueError('Baseline teacher revision mismatch')
        for field in cfg:
            if field not in {'output_dir','device','resume_from','max_steps','deadline_utc','deadline_margin_seconds',
                             'save_every_steps','keep_step_checkpoints'} and summary['config'].get(field)!=cfg[field]:
                raise ValueError(f'Phase summary disagrees with configuration: {field}')
    if args.phase=='lr':
        evaluations={}
        for row in registry:
            summary=json.loads(Path(row['summary_path']).read_text())
            output=args.output.parent/f"lr_{row['label']}_validation.json"
            atomic_json(output,{'checkpoint':row['checkpoint_path'],'corpora':summary['validation_trajectory'][-1]['metrics'],
                'source_summary':row['summary_path']})
            evaluations[f"lr_{row['label']}"]={'evaluation_path':str(output)}
        result=select_learning_rate(registry,initial,protocol,evaluations)
    else:
        summaries={r['id']:json.loads(Path(r['summary_path']).read_text()) for r in registry}
        result=select_shared_lambda(registry,initial,protocol,summaries,state)
    result.update(model=args.model,data_version=protocol['data_version'],microbatch_size=protocol['microbatch_size'],
        gradient_accumulation_steps=protocol['gradient_accumulation_steps'],deterministic_algorithms=True,
        initial_validation_path=str(initial_path),initial_validation_sha256=sha(initial_path),
        source_protocol_path=str(args.protocol),source_protocol_sha256=sha(args.protocol),
        source_registry_sha256=sha(args.registry),master_state_path=str(args.master_state),
        batch_selection_record=protocol.get('batch_selection_record'),
        selection_computation_device='cpu_only_no_model_forward')
    atomic_json(args.output,result)
    print(json.dumps({k:result[k] for k in ['model','selected_learning_rate','selected_shared_lambda'] if k in result}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['lr','lambda'],required=True);parser.add_argument('--model',required=True)
    for name in ['registry','protocol','phase-queue','master-state','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    select(parser.parse_args())
