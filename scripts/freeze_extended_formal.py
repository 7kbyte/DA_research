#!/usr/bin/env python3
"""Audit completed extended calibration and freeze a complete formal cohort."""
import argparse,datetime as dt,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from audit_formal_runs import CORES,sha256
from make_experiments import atomic_json,GROUPS
from make_extended_experiments import MODELS,experiment_prefix,formal_namespace
from reproduce_experiments import model_catalog
from select_extended_phase import require_complete_jobs
from select_hyperparameters import select_shared_lambda
from replay_study.analysis import _check_summary,training_validation_execution,verify_recorded_execution

def read(path):return json.loads(Path(path).read_text())

def freeze(args):
    prefix=experiment_prefix(args.model,args.microbatch);state=read(args.master_state)
    formal_id=formal_namespace(args.model,args.microbatch,args.target_prediction_tokens)
    target=args.target_prediction_tokens or MODELS[args.model]['target_tokens']
    destination=ROOT/f'artifacts/provenance/{formal_id}_input_snapshot.json'
    active_path=ROOT/f'configs/{formal_id}_active_registry.json'
    if destination.exists() or active_path.exists():raise FileExistsError('Formal freeze already exists; never overwrite a frozen cohort')
    choices={phase:ROOT/f'results/extended/{args.model}/m{args.microbatch}'/filename for phase,filename in
             [('lr','learning_rate_selection.json'),('lambda','lambda_selection.json')]}
    evidence={};common_stream=None;actual_lr=None
    for phase in ['lr','lambda']:
        choice=read(choices[phase]);protocol_path=ROOT/f'docs/{prefix}_{phase}_protocol.json';protocol=read(protocol_path)
        registry_path=ROOT/f'configs/{prefix}_{phase}_registry.json';rows=read(registry_path)
        if (choice['source_protocol_sha256']!=sha256(protocol_path) or choice['source_registry_sha256']!=sha256(registry_path)
                or choice['model']!=args.model or choice['data_version']!='extended_v1' or choice['microbatch_size']!=args.microbatch):
            raise ValueError('Calibration source/model/batch identity changed')
        require_complete_jobs(read(ROOT/f'configs/{prefix}_{phase}_queue.json'),state)
        initial_path=Path(choice['initial_validation_path']);initial=read(initial_path)
        if sha256(initial_path)!=choice['initial_validation_sha256']:raise ValueError('Calibration baseline changed')
        summaries={}
        for row in rows:
            cfg=read(row['config_path']);summary=read(row['summary_path']);summaries[row['id']]=summary
            if sha256(row['config_path'])!=row['config_sha256']:raise ValueError('Calibration config changed')
            _check_summary(summary,row)
            verify_recorded_execution(initial['execution_provenance'],training_validation_execution(summary))
            if (summary['status']!='completed' or summary['counters']['target_prediction_tokens']!=2_000_000 or
                    summary['counters']['replay_prediction_tokens']!=(0 if phase=='lr' else 100_000)):
                raise ValueError('Calibration exposure mismatch')
            if common_stream is None:common_stream=summary['target_stream']
            elif summary['target_stream']!=common_stream:raise ValueError('LR/lambda target streams differ')
        if phase=='lr':
            actual_lr=min(rows,key=lambda r:(summaries[r['id']]['validation_trajectory'][-1]['metrics']['target']['nll'],r['learning_rate']))['learning_rate']
            if actual_lr!=choice['selected_learning_rate']:raise ValueError('LR selection cannot be reproduced')
        else:
            recomputed=select_shared_lambda(rows,initial,protocol,summaries,state)
            if recomputed['selected_shared_lambda']!=choice['selected_shared_lambda'] or recomputed['selected_mean_score']!=choice['selected_mean_score']:
                raise ValueError('Lambda selection cannot be reproduced')
        evidence[phase]={'selection_path':str(choices[phase]),'selection_sha256':sha256(choices[phase]),
                        'protocol_path':str(protocol_path),'protocol_sha256':sha256(protocol_path),
                        'registry_path':str(registry_path),'registry_sha256':sha256(registry_path),
                        'summaries_sha256':{r['id']:sha256(r['summary_path']) for r in rows}}
    protocol_path=ROOT/f'docs/{formal_id}_protocol.json';protocol=read(protocol_path)
    for label,phase in [('learning_rate','lr'),('lambda','lambda')]:
        if protocol['choices'][label]['sha256']!=sha256(choices[phase]):raise ValueError('Formal hyperparameter authority changed')
    selected_weight=read(choices['lambda'])['selected_shared_lambda']
    registry_path=ROOT/f'configs/{formal_id}_registry.json';registry=read(registry_path)
    queue_path=ROOT/f'configs/{formal_id}_queue.json';queue=read(queue_path)
    model=MODELS[args.model];groups=['B0','B1','B2','B4','M'] if args.model=='pythia-2.8b' else ['B0','B1','B2','B3','B4','D','M','C']
    expected={(g,s) for g in groups for s in [11,22,33]}
    if len(registry)!=len(expected) or {(r['group'],r['seed']) for r in registry}!=expected:raise ValueError('Incomplete formal three-seed cohort')
    code={}
    core_freeze=read(ROOT/'artifacts/provenance/formal_deterministic_repository_freeze.json')
    for path in CORES:
        actual=sha256(ROOT/path)
        if actual!=core_freeze['core_sources'][path]['current']:raise ValueError('Frozen scientific implementation changed')
        code[path]={'sha256':actual,'bytes':(ROOT/path).stat().st_size}
    configs={};data={};model_path=None
    for row in registry:
        cfg=read(row['config_path']);key=str(Path(row['config_path']).relative_to(ROOT))
        if sha256(row['config_path'])!=row['config_sha256']:raise ValueError('Formal config changed')
        for field,expected_value in [('target_prediction_tokens',target),('microbatch_size',args.microbatch),
                ('gradient_accumulation_steps',64//args.microbatch),('learning_rate',actual_lr),('eval_every_steps',153),
                ('precision','bf16'),('deterministic_algorithms',True),('attention_backend','auto'),
                ('seed',row['seed']),('replay_ratio',0. if row['group']=='B0' else .05),
                ('replay_weight',0. if row['group']=='B0' else selected_weight),
                ('replay_objective',GROUPS[row['group']][1]),('evaluate_teacher_kl',True),
                ('evaluate_at_end',True),('save_every_steps',0),('validation_paths',protocol['validation_paths'])]:
            if cfg[field]!=expected_value:raise ValueError(f'Formal configuration mismatch: {field}')
        if cfg['model_revision']!=model['revision'] or '/data_extended/' not in cfg['target_path']:
            raise ValueError('Formal teacher/data mismatch')
        configs[key]={'sha256':sha256(row['config_path']),'bytes':Path(row['config_path']).stat().st_size};model_path=Path(cfg['model_name_or_path'])
    for relative,expected_sha in protocol['data_sha256'].items():
        path=ROOT/relative
        if sha256(path)!=expected_sha:raise ValueError('Formal data content changed')
        data[relative]={'sha256':expected_sha,'bytes':path.stat().st_size}
        manifest=path.with_suffix('.json')
        if manifest.exists():data[str(manifest.relative_to(ROOT))]={'sha256':sha256(manifest),'bytes':manifest.stat().st_size}
    for raw,hashes in protocol['bank_sha256'].items():
        path=Path(raw)
        for file,expected_sha in [(path,hashes['tensor']),(path.with_suffix('.json'),hashes['manifest'])]:
            if sha256(file)!=expected_sha:raise ValueError('Formal bank content/manifest changed')
            data[str(file.relative_to(ROOT))]={'sha256':expected_sha,'bytes':file.stat().st_size}
    catalog=model_catalog()[model_path.name];model_files={}
    if catalog['revision']!=model['revision']:raise ValueError('Pinned model revision mismatch')
    for filename,expected_sha in catalog['files'].items():
        path=model_path/filename
        if sha256(path)!=expected_sha:raise ValueError('Pinned model file changed')
        model_files[str(path)]={'sha256':expected_sha,'bytes':path.stat().st_size}
    for row in registry:row.update(active=True,launch_status='frozen_authorized_pending_master_append')
    atomic_json(active_path,registry)
    snapshot={'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'passed','model':args.model,
        'configuration_prefix':formal_id,'formal_runs':len(registry),'target_prediction_tokens':target,'code':code,'data':data,'configs':configs,
        'model_files':model_files,'calibration_audit':evidence,'active_registry':str(active_path),
        'active_registry_sha256':sha256(active_path),'prepared_registry_path':str(registry_path),
        'prepared_registry_sha256':sha256(registry_path),'queue_path':str(queue_path),'queue_sha256':sha256(queue_path),
        'protocol_path':str(protocol_path),'protocol_sha256':sha256(protocol_path),'test_scores_accessed':False,
        'training_started_by_freeze':False,'source_repository_commit':core_freeze['git_commit']}
    atomic_json(destination,snapshot)
    print(json.dumps({'status':'passed','formal_runs':len(registry),'active_registry':str(active_path),'snapshot':str(destination)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--microbatch',type=int,required=True)
    p.add_argument('--target-prediction-tokens',type=int)
    p.add_argument('--master-state',type=Path,default=ROOT/'logs/supplemental_job_state.json');freeze(p.parse_args())
