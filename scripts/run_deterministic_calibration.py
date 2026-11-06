#!/usr/bin/env python3
"""Freeze and execute strict deterministic LR and shared-lambda calibration."""
from __future__ import annotations
import argparse, dataclasses, datetime as dt, hashlib, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from replay_study.train import TrainConfig
from make_experiments import atomic_json,build_experiments
from select_hyperparameters import select_learning_rate,select_shared_lambda
SCRATCH=Path('/tmp/grad-research-20261106')
LR_NS='lr_deterministic_v3'
LAMBDA_NS='lambda_deterministic_v3'
BASELINE=ROOT/f'results/pilots/{LR_NS}/initial.json'
LR_RESULT=ROOT/'results/pilot_lr_selection_deterministic_v3.json'
LAMBDA_RESULT=ROOT/'results/pilot_lambda_selection_deterministic_v3.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def status(stage,registry,**updates):
    path=ROOT/'docs/protocol_status.json'
    payload=json.loads(path.read_text())
    payload.update(stage=stage,active_registry=str(registry.relative_to(ROOT)),
        updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),**updates)
    atomic_json(path,payload)

def save_configs(namespace,configs,registry,queue):
    for path,config in configs.items():
        destination=Path(path)
        if destination.exists() and json.loads(destination.read_text())!=config:raise FileExistsError(destination)
        atomic_json(destination,config)
    atomic_json(ROOT/f'configs/{namespace}_registry.json',registry)
    atomic_json(ROOT/f'configs/{namespace}_queue.json',queue)

def schedule(namespace,gpus):
    subprocess.run([sys.executable,str(ROOT/'scripts/schedule_jobs.py'),'--queue',
        str(ROOT/f'configs/{namespace}_queue.json'),'--state',str(ROOT/f'logs/{namespace}_job_state.json'),
        '--gpus',gpus,'--deadline','2026-11-07T03:26:32Z','--exit-when-idle'],check=True,cwd=ROOT)
    state=json.loads((ROOT/f'logs/{namespace}_job_state.json').read_text())
    if any(row['status']!='completed' for row in state['jobs'].values()):raise RuntimeError(namespace)
    return state

def prepare_lr():
    frozen=ROOT/'docs/hyperparameter_selection_protocol_deterministic_v3_lr.json'
    protocol=json.loads((ROOT/'docs/hyperparameter_selection_protocol_v3_lr.json').read_text())
    protocol.update(phase='strict_deterministic_lr_calibration',deterministic_algorithms=True,
        deterministic_warn_only=False,attention_backend='auto',cublas_workspace_config=':4096:8',
        calibration_action='Retrain all LR and lambda pilots after strict CUDA backward determinism fix; old pilots provisional',
        initial_validation_path=str(BASELINE),shared_learning_rate=None,
        learning_rate_selection_record=str(LR_RESULT),active=True,
        numerical_amendment='Identical-bank B4/D diverged under nondeterministic SDPA backward. No test scores accessed.')
    if frozen.exists():protocol=json.loads(frozen.read_text())
    else:
        protocol['frozen_at_utc']=dt.datetime.now(dt.timezone.utc).isoformat();atomic_json(frozen,protocol)
    for name,path in protocol['validation_paths'].items():
        if sha(path)!=protocol['validation_sha256'][name]:raise ValueError(name)
    records,configs,queue=[],{},[]
    argv=[sys.executable,str(ROOT/'scripts/run_evaluate.py'),'--model',str(SCRATCH/'models/pythia-410m-final'),
        '--revision',protocol['model_revision'],'--device','cuda:0','--batch-size','16','--precision','bf16',
        '--teacher-kl','--include-chunks','--deterministic-algorithms','--attention-backend','auto','--output',str(BASELINE)]
    for name,path in protocol['validation_paths'].items():argv+=['--corpus',f'{name}={path}']
    queue.append({'id':f'{LR_NS}_initial','argv':argv,'env':{'CUBLAS_WORKSPACE_CONFIG':':4096:8'},
        'expected_seconds':120,'outputs':[str(BASELINE)]})
    for lr in protocol['learning_rate_grid_completed']:
        label=format(lr,'.0e');run_id=f'pilot_{LR_NS}_{label}'
        cfg=TrainConfig.from_dict(json.loads((ROOT/f'configs/pilots/lr_{label}.json').read_text()))
        cfg=dataclasses.replace(cfg,output_dir=str(SCRATCH/'runs'/run_id),run_label=run_id,
            deterministic_algorithms=True,attention_backend='auto',evaluate_at_end=True,
            evaluate_teacher_kl=True,evaluation_batch_size=16,replay_weight=0.,validation_paths=protocol['validation_paths'])
        cfg.validate();config_path=ROOT/f'configs/{LR_NS}/{run_id}.json'
        summary_path=ROOT/f'results/pilots/{run_id}_summary.json'
        configs[str(config_path)]=dataclasses.asdict(cfg)
        records.append({'id':run_id,'label':label,'group':'B0','seed':101,'ratio':0.,'weight':0.,
            'learning_rate':lr,'config_path':str(config_path),'output_dir':cfg.output_dir,
            'checkpoint_path':str(Path(cfg.output_dir)/'checkpoints/latest.pt'),'summary_path':str(summary_path),'active':True})
        queue.append({'id':run_id,'argv':[sys.executable,str(ROOT/'scripts/make_experiments.py'),
            'run','--config',str(config_path),'--summary-copy',str(summary_path)],
            'env':{'CUBLAS_WORKSPACE_CONFIG':':4096:8'},'expected_seconds':180,
            'outputs':[str(summary_path),str(Path(cfg.output_dir)/'summary.json')]})
    save_configs(LR_NS,configs,records,queue)
    status('严格确定性：基线与三点LR重新校准',ROOT/f'configs/{LR_NS}_registry.json',
        active_lr_selection=None,active_lambda_selection=None)
    return protocol,records

def execute(gpus):
    protocol,records=prepare_lr()
    schedule(LR_NS,gpus)
    baseline_path=BASELINE.with_name('initial_cudnn_matched.json')
    if not baseline_path.exists():baseline_path=BASELINE
    initial=json.loads(baseline_path.read_text());evaluations={}
    if not initial.get('execution_provenance',{}).get('cudnn_sdp_enabled'):
        raise ValueError('Initial evaluation must match training auto SDPA including cudnn_sdp_enabled=True')
    for record in records:
        summary=json.loads(Path(record['summary_path']).read_text())
        output=ROOT/f"results/pilots/{LR_NS}/lr_{record['label']}.json"
        atomic_json(output,{'checkpoint':record['checkpoint_path'],
            'corpora':summary['validation_trajectory'][-1]['metrics'],
            'evaluation_source':'training final fixed-exposure validation','source_summary':record['summary_path']})
        evaluations[f"lr_{record['label']}"]={'evaluation_path':str(output)}
    selected=select_learning_rate(records,initial,protocol,evaluations)
    selected.update(data_version='v3',result_role='active_strict_deterministic_calibration',
        deterministic_algorithms=True,attention_backend='auto',
        protocol_sha256=sha(ROOT/'docs/hyperparameter_selection_protocol_deterministic_v3_lr.json'))
    atomic_json(LR_RESULT,selected)
    protocol.update(phase='strict_deterministic_lambda_calibration',
        shared_learning_rate=selected['selected_learning_rate'],initial_validation_path=str(baseline_path),initial_validation_sha256=sha(baseline_path),
        learning_rate_selection_record_sha256=sha(LR_RESULT),
        frozen_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    lambda_protocol=ROOT/'docs/hyperparameter_selection_protocol_deterministic_v3_lambda.json'
    atomic_json(lambda_protocol,protocol)
    configs,registry,queue=build_experiments('lambda',ROOT,SCRATCH,
        selected['selected_learning_rate'],protocol=protocol,namespace=LAMBDA_NS)
    save_configs(LAMBDA_NS,configs,registry,queue)
    status('严格确定性：共享λ完整16格校准',ROOT/f'configs/{LAMBDA_NS}_registry.json',
        active_lr_selection=str(LR_RESULT.relative_to(ROOT)))
    state=schedule(LAMBDA_NS,gpus)
    protocol=json.loads(lambda_protocol.read_text())
    initial=json.loads(Path(protocol['initial_validation_path']).read_text())
    summaries={r['id']:json.loads(Path(r['summary_path']).read_text()) for r in registry}
    result=select_shared_lambda(registry,initial,protocol,summaries,state)
    result.update(data_version='v3',result_role='active_strict_deterministic_calibration',
        deterministic_algorithms=True,attention_backend='auto',protocol_sha256=sha(lambda_protocol))
    atomic_json(LAMBDA_RESULT,result)
    status('严格确定性校准完成；待正式B4/D全程数值空对照',ROOT/f'configs/{LAMBDA_NS}_registry.json',
        active_lambda_selection=str(LAMBDA_RESULT.relative_to(ROOT)))
    print(json.dumps({'selected_learning_rate':selected['selected_learning_rate'],
        'selected_shared_lambda':result['selected_shared_lambda'],'lr_result':str(LR_RESULT),
        'lambda_result':str(LAMBDA_RESULT)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--gpus',default='0,2,3')
    args=parser.parse_args()
    execute(args.gpus) if args.execute else prepare_lr()
