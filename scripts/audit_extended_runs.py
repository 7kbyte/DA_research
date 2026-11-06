#!/usr/bin/env python3
"""Read-only audit of frozen extended training, without test or behavior scores."""
from __future__ import annotations
import argparse,datetime as dt,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from audit_formal_runs import events_at,expected_replay,sha256
from make_experiments import atomic_json
from make_model_equivalence_proofs import ALGORITHM,checkpoint_identity
from replay_study.analysis import _check_summary,training_validation_execution,verify_recorded_execution
CORPORA={'target','c4','wikitext'}
STEP_TOKENS=64*511


def audit_steps(steps,config):
    """Check exact target and replay causal-token budgets at every prefix."""
    problems=[];target=replay=0
    for index,event in enumerate(steps,1):
        target+=event['step_counts']['target_prediction_tokens']
        replay+=event['step_counts']['replay_prediction_tokens']
        checks=[(event['step']==index,'contiguous_optimizer_steps'),
            (target==min(index*STEP_TOKENS,config['target_prediction_tokens']),'target_prefix_budget'),
            (event['counters']['target_prediction_tokens']==target,'target_counter_sum'),
            (event['counters']['replay_prediction_tokens']==replay,'replay_counter_sum'),
            (replay==expected_replay(target,config['replay_ratio']),'exact_replay_prefix_floor'),
            (math.isfinite(event['target_ce']) and math.isfinite(event['grad_norm']),'finite_target_and_gradient'),
            (event['replay_loss'] is None or math.isfinite(event['replay_loss']),'finite_replay_loss')]
        problems.extend({'step':index,'check':label} for valid,label in checks if not valid)
    return {'violations':problems,'steps':len(steps),'target_prediction_tokens':target,'replay_prediction_tokens':replay}


def audit_validation(summary,config):
    problems=[];trajectory=summary['validation_trajectory']
    last_step=math.ceil(config['target_prediction_tokens']/STEP_TOKENS)
    expected=list(range(config['eval_every_steps'],last_step+1,config['eval_every_steps']))
    if not expected or expected[-1]!=last_step:expected.append(last_step)
    if [p['step'] for p in trajectory]!=expected:problems.append('complete_fixed_exposure_validation_trajectory')
    for point in trajectory:
        if set(point['metrics'])!=CORPORA:problems.append('validation_corpora_only')
        target=min(point['step']*STEP_TOKENS,config['target_prediction_tokens'])
        if point['target_prediction_tokens']!=target or point['replay_prediction_tokens']!=expected_replay(target,config['replay_ratio']):
            problems.append('validation_exposure_accounting')
        for metrics in point['metrics'].values():
            if not (metrics['complete_corpus'] and math.isfinite(metrics['nll']) and math.isfinite(metrics['kl'])):
                problems.append('complete_finite_validation_nll_and_teacher_kl')
            if metrics['precision']!='bf16' or metrics['model_weight_dtype']!='torch.float32':problems.append('validation_precision')
    if trajectory:
        best=min(trajectory,key=lambda p:p['metrics']['target']['nll']);selected=summary['best_target_validation']
        if selected['step']!=best['step'] or selected['nll']!=best['metrics']['target']['nll']:
            problems.append('best_selected_by_target_validation_only')
    return problems


def audit_phase(snapshot_path,state,allow_incomplete=False,compare_weights=False):
    snapshot=json.loads(Path(snapshot_path).read_text());registry_path=Path(snapshot['active_registry'])
    registry=json.loads(registry_path.read_text());problems=[];summaries={};traces={};per_seed={};runs=[]
    def require(valid,check,run_id=None):
        if not valid:problems.append({'run_id':run_id,'check':check})
    require(sha256(registry_path)==snapshot['active_registry_sha256'],'frozen_active_registry_hash')
    require(len(registry)==snapshot['formal_runs'],'complete_registered_cohort')
    for name in ['protocol','queue']:
        require(sha256(snapshot[name+'_path'])==snapshot[name+'_sha256'],f'frozen_{name}')
    for phase,evidence in snapshot['calibration_audit'].items():
        for name in ['selection','protocol','registry']:
            require(sha256(evidence[name+'_path'])==evidence[name+'_sha256'],f'frozen_calibration_{phase}_{name}')
    for section in ['code','data','model_files']:
        for path,recorded in snapshot[section].items():
            actual_path=Path(path) if section=='model_files' else ROOT/path
            require(sha256(actual_path)==recorded['sha256'],f'frozen_{section}:{path}')
    choice_path=snapshot['calibration_audit']['lr']['selection_path'];choice=json.loads(Path(choice_path).read_text())
    initial=json.loads(Path(choice['initial_validation_path']).read_text())
    require(sha256(choice_path)==snapshot['calibration_audit']['lr']['selection_sha256'],'frozen_lr_selection')
    require(sha256(choice['initial_validation_path'])==choice['initial_validation_sha256'],'frozen_initial_validation')
    for row in registry:
        run_id=row['id'];config_path=Path(row['config_path']);config=json.loads(config_path.read_text())
        relative=str(config_path.relative_to(ROOT));require(sha256(config_path)==snapshot['configs'][relative]['sha256'],'frozen_config',run_id)
        status=state['jobs'].get(run_id,{}).get('status','missing')
        require(status not in {'failed','blocked','missing'},f'scheduler_status:{status}',run_id)
        event_path=Path(row['output_dir'])/'events.jsonl';events=events_at(event_path,allow_incomplete) if event_path.exists() else []
        require(not any(e['event']=='error' for e in events),'no_error_events',run_id)
        steps=[e for e in events if e['event']=='train_step'];traces[run_id]=steps;accounting=audit_steps(steps,config)
        problems.extend({'run_id':run_id,**item} for item in accounting['violations'])
        reference=per_seed.setdefault(row['seed'],steps) if steps else []
        for left,right in zip(reference,steps):
            require(left['target_stream_hash']==right['target_stream_hash'],'same_seed_target_stream',run_id)
            require(left['learning_rate']==right['learning_rate'],'same_seed_lr_schedule',run_id)
        if status!='completed':
            runs.append({'id':run_id,'scheduler_status':status,'observed_steps':len(steps)});continue
        summary=json.loads(Path(row['summary_path']).read_text());summaries[run_id]=summary
        try:
            _check_summary(summary,row)
            verify_recorded_execution(initial['execution_provenance'],training_validation_execution(summary))
        except ValueError as error:require(False,str(error),run_id)
        require(summary['config']==config,'runtime_equals_frozen_config',run_id)
        require(summary['status']=='completed','fixed_horizon_completed',run_id)
        require(summary['provenance']['teacher_weight_dtype']=='torch.float32','teacher_fp32_weights',run_id)
        target=snapshot['target_prediction_tokens'];replay=expected_replay(target,row['ratio']);counters=summary['counters']
        require(counters['target_prediction_tokens']==target==accounting['target_prediction_tokens'],'final_target_budget',run_id)
        require(counters['replay_prediction_tokens']==replay==accounting['replay_prediction_tokens'],'final_replay_budget',run_id)
        require(counters['step']==len(steps)==math.ceil(target/STEP_TOKENS),'final_optimizer_steps',run_id)
        for check in audit_validation(summary,config):require(False,check,run_id)
        for name,exposure in [('target',target),('replay',replay)]:
            coverage=summary['coverage_accounting'][name]
            if coverage is None:
                require(name=='replay' and exposure==0,'absent_bank_only_with_zero_replay',run_id)
                continue
            expected_unique=min(exposure,coverage['unique_capacity_prediction_tokens'])
            require(coverage['no_duplicate_chunks_verified'],f'{name}_exact_unique_chunks_verified',run_id)
            require(coverage['unique_prediction_tokens_exposed']==expected_unique,f'{name}_unique_exposure',run_id)
            require(coverage['repeated_prediction_token_exposures']==exposure-expected_unique,f'{name}_repeated_exposure',run_id)
        capacity=0 if row['group']=='B0' else 100156
        require(summary['replay_bank_capacity_prediction_tokens']==capacity,'bank_capacity_K',run_id)
        require(summary['replay_bank_unique_prediction_tokens']==capacity,'bank_unique_capacity_K',run_id)
        for path in [summary['checkpoint_path'],summary['best_target_validation']['checkpoint_path']]:require(Path(path).exists(),'final_and_best_checkpoints_exist',run_id)
        timing=summary['timing'];require(timing['training_teacher_forward_seconds_complete'],'teacher_forward_timing_complete',run_id)
        require(not timing['nested_timers_are_additive_costs'] and timing['training_teacher_forward_seconds_nested_in']=='training_seconds','nested_teacher_cost_not_added_twice',run_id)
        require(0<=counters['training_teacher_forward_seconds']<=counters['training_seconds']+.01,'teacher_training_subcost',run_id)
        require(0<=counters['validation_teacher_forward_seconds']<=counters['validation_seconds']+.01,'teacher_validation_subcost',run_id)
        runs.append({'id':run_id,'scheduler_status':status,'target_prediction_tokens':target,'replay_prediction_tokens':replay,
            'optimizer_steps':counters['step'],'best_target_step':summary['best_target_validation']['step'],
            'summary_sha256':sha256(row['summary_path']),'costs':{k:v for k,v in counters.items() if k.endswith('_seconds')}})
    controls=[]
    for seed in [11,22,33]:
        pair={r['group']:r for r in registry if r['seed']==seed and r['group'] in {'B4','D'}}
        if set(pair)!={'B4','D'} or not all(r['id'] in summaries for r in pair.values()):continue
        left,right=(summaries[pair[g]['id']] for g in ['B4','D'])
        same_bank=left['provenance']['replay']['sha256_tensors']==right['provenance']['replay']['sha256_tensors']
        require(same_bank,'D_B4_identical_bank_tensor',pair['D']['id']);differences=[]
        for a,b in zip(traces[pair['B4']['id']],traces[pair['D']['id']]):
            differences.extend(abs(a[k]-b[k]) for k in ['target_ce','replay_loss','grad_norm'])
        for a,b in zip(left['validation_trajectory'],right['validation_trajectory']):
            differences.extend(abs(a['metrics'][c][k]-b['metrics'][c][k]) for c in sorted(CORPORA) for k in ['nll','kl'])
        require(max(differences,default=math.inf)==0,'D_B4_exact_numerical_trajectories',pair['D']['id'])
        control={'seed':seed,'B4':pair['B4']['id'],'D':pair['D']['id'],'same_bank':same_bank,
            'max_numeric_difference':max(differences,default=None)}
        if compare_weights:
            identities={g:checkpoint_identity(summaries[pair[g]['id']]['checkpoint_path']) for g in ['B4','D']}
            fingerprints={g:identities[g]['model_state_fingerprint'] for g in ['B4','D']}
            equal=fingerprints['B4']==fingerprints['D'];require(equal,'D_B4_bitwise_final_model_state',pair['D']['id'])
            control.update(model_state_fingerprints=fingerprints,bitwise_model_state_equal=equal)
            for field,group in [('left','B4'),('right','D')]:
                control[field]={**identities[group],'run_id':pair[group]['id'],'kind':'final',
                    'config_path':pair[group]['config_path'],'config_sha256':sha256(pair[group]['config_path']),
                    'model_revision':summaries[pair[group]['id']]['config']['model_revision'],
                    'model_name_or_path':summaries[pair[group]['id']]['config']['model_name_or_path']}
        controls.append(control)
    complete=len(summaries)==len(registry);require(allow_incomplete or complete,'all_registered_jobs_completed')
    return {'phase':snapshot['configuration_prefix'],'model':snapshot['model'],'status':'failed' if problems else ('passed' if complete else 'in_progress'),
        'completed_runs':len(summaries),'expected_runs':len(registry),'violations':problems,'snapshot':str(snapshot_path),
        'snapshot_sha256':sha256(snapshot_path),'runs':runs,'D_B4_null_controls':controls,'model_weights_compared':compare_weights,
        'test_and_behavior_scores_read':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-manifest',type=Path,default=ROOT/'results/extended_formal_freeze.json')
    p.add_argument('--state',type=Path,default=ROOT/'logs/supplemental_job_state.json')
    p.add_argument('--output',type=Path,default=ROOT/'results/extended_run_audit.json')
    p.add_argument('--allow-incomplete',action='store_true');p.add_argument('--compare-weights',action='store_true')
    p.add_argument('--equivalence-proof-output',type=Path);args=p.parse_args()
    if args.equivalence_proof_output and not args.compare_weights:p.error('Equivalence proof requires --compare-weights')
    manifest=json.loads(args.freeze_manifest.read_text());state=json.loads(args.state.read_text());phases=[]
    for row in manifest['phases']:
        if sha256(ROOT/row['snapshot'])!=row['snapshot_sha256']:raise ValueError('Formal snapshot changed after phase activation')
        phases.append(audit_phase(ROOT/row['snapshot'],state,args.allow_incomplete,args.compare_weights))
    result={'audited_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'status':'failed' if any(x['status']=='failed' for x in phases) else ('passed' if all(x['status']=='passed' for x in phases) else 'in_progress'),
        'completed_runs':sum(x['completed_runs'] for x in phases),'expected_runs':sum(x['expected_runs'] for x in phases),
        'phases':phases,'test_and_behavior_scores_read':False,
        'freeze_manifest_sha256':sha256(args.freeze_manifest),
        'audit_source_sha256':sha256(Path(__file__)),
        'model_fingerprint_source_sha256':sha256(ROOT/'scripts/audit_formal_runs.py')}
    atomic_json(args.output,result);print(json.dumps({k:result[k] for k in ['status','completed_runs','expected_runs']}))
    if args.equivalence_proof_output and result['status']=='passed':
        comparisons=[]
        for phase in phases:
            for control in phase['D_B4_null_controls']:
                comparisons.append({'id':f"{phase['phase']}_D_B4_s{control['seed']}_final",
                    'kind':'identical_bank_null_control','seed':control['seed'],
                    'left':control['left'],'right':control['right'],'bitwise_model_state_equal':True})
        atomic_json(args.equivalence_proof_output,{'schema_version':1,'status':'passed',
            'computed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'algorithm':ALGORITHM,
            'algorithm_source':{'path':'scripts/audit_formal_runs.py','sha256':sha256(ROOT/'scripts/audit_formal_runs.py'),'function':'model_fingerprint'},
            'source_audit':{'path':str(args.output),'sha256':sha256(args.output)},
            'producer_source':{'path':str(Path(__file__).resolve()),'sha256':sha256(__file__)},
            'comparisons':comparisons,'test_scores_accessed':False,'checkpoint_files_modified':False})
    return 1 if result['status']=='failed' else 0


if __name__=='__main__':raise SystemExit(main())
