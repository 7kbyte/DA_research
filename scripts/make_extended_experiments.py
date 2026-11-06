#!/usr/bin/env python3
"""Generate model-specific extended-data experiments without launching GPU work."""
from __future__ import annotations
import argparse,dataclasses,datetime as dt,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from replay_study.train import TrainConfig
from make_experiments import GROUPS,atomic_json,number_label

MODELS={
 'pythia-410m':{'revision':'9879c9b5f8bea9051dcb0e68dff21493d67e9d4f','microbatch':16,'eval_batch':16,'target_tokens':100_000_000},
 'pythia-1b':{'revision':'f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2','microbatch':16,'eval_batch':16,'target_tokens':100_000_000},
 'pythia-2.8b':{'revision':'2a259cdd96a4beb1cdf467512e3904197345f6a9','microbatch':8,'eval_batch':8,'target_tokens':50_000_000},
}
LR_GRID=[1e-5,3e-5,1e-4]
LAMBDA_GRID=[.05,.5,1.,5.]

def experiment_prefix(model,microbatch=None):
    microbatch=microbatch or MODELS[model]['microbatch']
    return f"extended_v2_{model.replace('.','p').replace('-','_')}_m{microbatch}"

def formal_namespace(model,microbatch=None,target_prediction_tokens=None):
    target=target_prediction_tokens or MODELS[model]['target_tokens']
    suffix='' if target==MODELS[model]['target_tokens'] else f'_h{number_label(target/1_000_000)}m'
    return f'{experiment_prefix(model,microbatch)}{suffix}_formal'

def corpus_paths(root,split):
    return {name:str(root/'artifacts/data_extended'/f'{stem}_{split}.pt')
        for name,stem in [('target','finemath'),('c4','c4'),('wikitext','wikitext103')]}

def build_extended(model,phase,root,scratch,learning_rate=None,weight=None,microbatch=None,namespace=None,
                   training_deadline='2026-11-07T00:00:00Z',target_prediction_tokens=None):
    spec=MODELS[model];microbatch=microbatch or spec['microbatch']
    if microbatch<1 or 64%microbatch:raise ValueError('Microbatch must divide effective batch 64')
    if target_prediction_tokens is not None and (phase!='formal' or target_prediction_tokens<=0):
        raise ValueError('Explicit horizon applies only to a positive formal target budget')
    formal_target=target_prediction_tokens or spec['target_tokens']
    namespace=namespace or (formal_namespace(model,microbatch,formal_target) if phase=='formal' else f'{experiment_prefix(model,microbatch)}_{phase}')
    if phase=='lr':cells=[('B0',101,lr,0.) for lr in LR_GRID]
    elif phase=='lambda':
        if learning_rate is None:raise ValueError('Lambda requires this model\'s selected learning rate')
        cells=[(g,101,learning_rate,w) for w in LAMBDA_GRID for g in ['B1','B2','B3','B4']]
    elif phase=='formal':
        if learning_rate is None or weight is None:raise ValueError('Formal requires this model\'s selected LR and lambda')
        groups=['B0','B1','B2','B4','M'] if model=='pythia-2.8b' else list(GROUPS)
        cells=[(g,seed,learning_rate,0. if g=='B0' else weight) for seed in [11,22,33] for g in groups]
    else:raise ValueError(phase)
    configs,registry,queue={},[],[]
    for group,seed,lr,w in cells:
        suffix=f'lr{number_label(lr)}' if phase=='lr' else f'w{number_label(w)}'
        run_id=f'{namespace}_{group}_s{seed}_{suffix}'
        bank,objective=GROUPS[group]
        output=scratch/'runs'/run_id;path=root/f'configs/{namespace}/{run_id}.json'
        summary=root/f'results/extended/{model}/{run_id}_summary.json'
        cfg=TrainConfig(model_name_or_path=str(scratch/'models'/f'{model}-final'),model_revision=spec['revision'],
            target_path=str(root/'artifacts/data_extended/finemath_train.pt'),output_dir=str(output),seed=seed,
            target_prediction_tokens=formal_target if phase=='formal' else 2_000_000,
            replay_path=str(root/'artifacts/extended'/model/'banks'/f'{bank}_seed{seed}.pt') if bank else None,
            replay_objective=objective,replay_ratio=0. if group=='B0' else .05,replay_weight=w,
            microbatch_size=microbatch,gradient_accumulation_steps=64//microbatch,learning_rate=lr,
            precision='bf16',device='cuda:0',attention_backend='auto',deterministic_algorithms=True,
            eval_every_steps=153 if phase=='formal' else 0,save_every_steps=0,evaluate_at_start=False,
            evaluate_at_end=True,evaluate_teacher_kl=True,evaluation_batch_size=spec['eval_batch'],
            validation_paths=corpus_paths(root,'validation'),keep_step_checkpoints=False,
            deadline_utc=training_deadline,run_label=run_id)
        cfg.validate();configs[str(path)]=dataclasses.asdict(cfg)
        registry.append({'id':run_id,'model':model,'phase':phase,'group':group,'seed':seed,'ratio':cfg.replay_ratio,
            'weight':w,'learning_rate':lr,'config_path':str(path),'output_dir':str(output),'summary_path':str(summary),
            'checkpoint_path':str(output/'checkpoints/latest.pt'),'label':format(lr,'.0e') if phase=='lr' else None,
            'test_paths':corpus_paths(root,'test'),'active':False,'launch_status':'prepared_not_authorized',
            'effective_batch_chunks':64,'eval_every_target_tokens_nominal':5_000_000 if phase=='formal' else None,
            'eval_every_target_tokens_actual':5_003_712 if phase=='formal' else None})
        if phase=='formal':registry[-1].update(target_prediction_tokens=formal_target,learning_rate_horizon_prediction_tokens=formal_target)
        queue.append({'id':run_id,'argv':[sys.executable,str(root/'scripts/make_experiments.py'),'run','--config',
            str(path),'--summary-copy',str(summary)],'env':{'CUBLAS_WORKSPACE_CONFIG':':4096:8'},
            'expected_seconds':3600 if phase=='formal' else 300,'outputs':[str(summary),str(output/'summary.json')]})
    return configs,registry,queue

def digest(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):hasher.update(block)
    return hasher.hexdigest()

def protocol_record(model,root,microbatch):
    spec=MODELS[model]
    provenance_path=root/('artifacts/assets/model_provenance.json' if model=='pythia-410m' else 'artifacts/assets/extended_model_provenance.json')
    provenance=json.loads(provenance_path.read_text())
    if model!='pythia-410m':provenance=next(row for row in provenance['models'] if row['repo'].endswith('/'+model))
    code_paths=[*root.glob('src/replay_study/*.py'),root/'scripts/run_train.py',root/'scripts/run_evaluate.py',root/'scripts/make_experiments.py',
                root/'scripts/make_extended_experiments.py',root/'scripts/calibrate_extended_experiments.py']
    paths=[root/'artifacts/data_extended/finemath_train.pt',*map(Path,corpus_paths(root,'validation').values()),
           *map(Path,corpus_paths(root,'test').values()),root/'artifacts/data_extended/c4_candidate.pt']
    return {'model':model,'model_revision':spec['revision'],'data_version':'extended_v1',
        'model_provenance':provenance,'model_provenance_manifest_path':str(provenance_path),
        'model_provenance_manifest_sha256':digest(provenance_path),
        'source_code_sha256':{str(path.relative_to(root)):digest(path) for path in code_paths},
        'seed':101,'learning_rate_grid_completed':LR_GRID,'groups':['B1','B2','B3','B4'],'lambda_grid':LAMBDA_GRID,
        'target_prediction_tokens':2_000_000,'replay_prediction_tokens':100_000,'replay_ratio':.05,
        'microbatch_size':microbatch,'gradient_accumulation_steps':64//microbatch,
        'learning_rate_selection_rule':'Minimum target validation NLL after exactly 2M prediction tokens; strict ties smaller LR',
        'lambda_group_score':'(Ltarget-Ltarget_initial)+0.5*((Lc4-Lc4_initial)+(Lwikitext-Lwikitext_initial))',
        'lambda_aggregate':'unweighted arithmetic mean across B1,B2,B3,B4',
        'lambda_selection_rule':'minimum aggregate score; strict equal-score ties smaller lambda',
        'validation_paths':corpus_paths(root,'validation'),'test_paths':corpus_paths(root,'test'),
        'validation_sha256':{name:digest(path) for name,path in corpus_paths(root,'validation').items()},
        'data_sha256':{str(path.relative_to(root)):digest(path) for path in paths},
        'deterministic_algorithms':True,'deterministic_warn_only':False,'attention_backend':'auto',
        'allow_tf32':True,'cudnn_deterministic':True,'cudnn_benchmark':False,
        'model_weight_precision':'FP32','precision':'bf16','effective_batch_chunks':64,
        'evaluation_batch_size':spec['eval_batch'],'formal_target_prediction_tokens':spec['target_tokens'],
        'formal_eval_every_steps':153,'formal_eval_every_target_tokens_actual':5_003_712,
        'benchmark_status':f'candidate microbatch{microbatch} accum{64//microbatch}; matched throughput/batch selection record required before execution',
        'source_pool_rule':'Frozen teacher of the same model/revision; extended C4 candidates; all banks model-specific',
        'relationship_to_primary':'Extended target includes primary prefix; correlated robustness extension, not an independent replication',
        'evaluation_rule':'Use matched data_extended val/test; main v3 cannot evaluate extended training contamination-free',
        'test_scores_seen':False,'launch_status':'prepared_not_authorized',
        'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--phase',choices=['lr','lambda','formal'],required=True)
    parser.add_argument('--scratch',type=Path,default=Path('/tmp/grad-research-20261106'))
    parser.add_argument('--microbatch',type=int)
    parser.add_argument('--learning-rate-selection',type=Path)
    parser.add_argument('--lambda-selection',type=Path)
    parser.add_argument('--namespace')
    parser.add_argument('--training-deadline',default='2026-11-07T00:00:00Z')
    parser.add_argument('--batch-selection-record',type=Path)
    parser.add_argument('--target-prediction-tokens',type=int,help='Formal-only complete LR horizon; a changed budget gets its own h<M>m namespace')
    args=parser.parse_args();spec=MODELS[args.model];micro=args.microbatch or spec['microbatch']
    choices={};lr=weight=None;lr_record=None
    for name,path in [('learning_rate',args.learning_rate_selection),('lambda',args.lambda_selection)]:
        if path:
            record=json.loads(path.read_text())
            if record.get('model')!=args.model or record.get('data_version')!='extended_v1':
                raise ValueError('Selection must come from the same model and extended evaluation data')
            if record.get('microbatch_size')!=micro or record.get('gradient_accumulation_steps')!=64//micro:
                raise ValueError('Selection must use the same microbatch and accumulation as this run')
            choices[name]={'path':str(path),'sha256':digest(path)}
            if name=='learning_rate':lr=record['selected_learning_rate'];lr_record=record
            else:
                weight=record['selected_shared_lambda']
                if lr is not None and record['shared_learning_rate']!=lr:raise ValueError('Lambda calibration used another learning rate')
    namespace=args.namespace or (formal_namespace(args.model,micro,args.target_prediction_tokens) if args.phase=='formal' else f'{experiment_prefix(args.model,micro)}_{args.phase}')
    configs,registry,queue=build_extended(args.model,args.phase,ROOT,args.scratch,lr,weight,micro,namespace,args.training_deadline,args.target_prediction_tokens)
    frozen=protocol_record(args.model,ROOT,micro);frozen.update(shared_learning_rate=lr,shared_lambda=weight,choices=choices,
        configuration_version=2,training_deadline_utc=args.training_deadline)
    if args.target_prediction_tokens is not None:
        frozen.update(original_formal_target_prediction_tokens=spec['target_tokens'],formal_target_prediction_tokens=args.target_prediction_tokens,
                      formal_learning_rate_horizon_prediction_tokens=args.target_prediction_tokens)
        amendment=ROOT/'docs/extended_2p8b_horizon_amendment.json'
        if args.model=='pythia-2.8b' and args.target_prediction_tokens==25_000_000 and amendment.exists():
            frozen['resource_horizon_amendment']={'path':str(amendment),'sha256':digest(amendment)}
    if args.batch_selection_record:
        selected=json.loads(args.batch_selection_record.read_text())
        if selected['model']!=args.model or selected['selected_microbatch']!=micro:raise ValueError('Batch selection record mismatch')
        frozen['batch_selection_record']={'path':str(args.batch_selection_record),'sha256':digest(args.batch_selection_record)}
    elif lr_record and lr_record.get('batch_selection_record'):
        frozen['batch_selection_record']=lr_record['batch_selection_record']
    if lr_record:
        initial=Path(lr_record['initial_validation_path'])
        if digest(initial)!=lr_record['initial_validation_sha256']:raise ValueError('Model initial baseline changed')
        frozen.update(initial_validation_path=str(initial),initial_validation_sha256=digest(initial))
    frozen['bank_sha256']={}
    for cfg in configs.values():
        if not cfg['replay_path']:continue
        bank=Path(cfg['replay_path'])
        if not bank.exists():raise FileNotFoundError(f'Model-specific bank not ready: {bank}')
        meta=json.loads(bank.with_suffix('.json').read_text())
        if 'generated_' in bank.name and meta.get('revision')!=spec['revision']:
            raise ValueError('Generated bank teacher revision mismatch')
        if 'c4_' in bank.name and meta.get('source_pool_sha256')!=frozen['data_sha256']['artifacts/data_extended/c4_candidate.pt']:
            raise ValueError('C4 bank must use the extended candidate pool')
        frozen['bank_sha256'][str(bank)]={'tensor':digest(bank),'manifest':digest(bank.with_suffix('.json'))}
    for path,cfg in configs.items():
        destination=Path(path)
        if destination.exists() and json.loads(destination.read_text())!=cfg:raise FileExistsError(destination)
        atomic_json(destination,cfg)
    if args.phase=='lr':
        batch_folder=f'/m{micro}' if micro!=spec['microbatch'] else ''
        baseline=ROOT/f'results/extended/{args.model}{batch_folder}/initial_validation.json'
        argv=[sys.executable,str(ROOT/'scripts/run_evaluate.py'),'--model',str(args.scratch/'models'/f'{args.model}-final'),
            '--revision',spec['revision'],'--device','cuda:0','--batch-size',str(spec['eval_batch']),
            '--precision','bf16','--teacher-kl','--include-chunks','--deterministic-algorithms','--output',str(baseline)]
        for name,path in corpus_paths(ROOT,'validation').items():argv+=['--corpus',f'{name}={path}']
        queue.insert(0,{'id':namespace+'_initial','argv':argv,'env':{'CUBLAS_WORKSPACE_CONFIG':':4096:8'},
            'expected_seconds':180,'outputs':[str(baseline)]})
        frozen['initial_validation_path']=str(baseline)
    for record in registry:record['config_sha256']=digest(record['config_path'])
    atomic_json(ROOT/f'configs/{namespace}_registry.json',registry)
    atomic_json(ROOT/f'configs/{namespace}_queue.json',queue)
    atomic_json(ROOT/f'docs/{namespace}_protocol.json',frozen)
    print(json.dumps({'model':args.model,'phase':args.phase,'configs':len(configs),'jobs':len(queue),'launch_status':'prepared_not_authorized'}))

if __name__=='__main__':main()
