#!/usr/bin/env python3
"""Read-only audit of formal accounting, frozen inputs, validation and null controls."""
from __future__ import annotations
import argparse
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CORES=['src/replay_study/train.py','src/replay_study/objectives.py','src/replay_study/evaluate.py',
       'src/replay_study/execution.py',
       'src/replay_study/data.py','src/replay_study/generate.py','src/replay_study/selection.py',
       'scripts/run_train.py','scripts/run_evaluate.py','scripts/make_experiments.py']


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def events_at(path,allow_partial):
    text=Path(path).read_text()
    lines=text.splitlines()
    if allow_partial and text and not text.endswith('\n'):lines=lines[:-1]
    return [json.loads(line) for line in lines if line]


def expected_replay(target,ratio):
    rate=Fraction(str(ratio));return target*rate.numerator//rate.denominator


def compare_values(a,b,atol=1e-5,rtol=1e-5):
    return {'absolute_difference':abs(a-b),'within_tolerance':math.isclose(a,b,abs_tol=atol,rel_tol=rtol)}

def logical_key(path):
    for base in [ROOT,Path('/data/202610/7kb/research')]:
        try:return str(Path(path).relative_to(base))
        except ValueError:pass
    raise ValueError(f'Input outside the logical workspace: {path}')


def model_fingerprint(path):
    import torch
    saved=torch.load(path,map_location='cpu',weights_only=False)
    digest=hashlib.sha256()
    count=0
    for name,tensor in sorted(saved['model_state'].items()):
        digest.update(json.dumps([name,str(tensor.dtype),list(tensor.shape)]).encode())
        raw=tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
        count+=tensor.numel()
    return {'sha256_model_state':digest.hexdigest(),'state_elements':count}


def audit(registry,state,snapshot,allow_incomplete=False,compare_weights=True,
          expected_lr=3e-5,expected_lambda=.05,require_deterministic=False):
    problems=[]
    def require(condition,message,run_id=None):
        if not condition:problems.append({'run_id':run_id,'check':message})
    require(len(registry)==36,'registry has 36 runs')
    expected={(g,s,0.0 if g=='B0' else .05) for s in [11,22,33] for g in ['B0','B1','B2','B3','B4','D','M','C']}
    expected|={(g,s,b) for s in [11,22,33] for g in ['B4','M'] for b in [.01,.1]}
    require({(r['group'],r['seed'],r['ratio']) for r in registry}==expected,'formal group/seed/ratio grid')
    frozen_code={}
    for key in CORES:
        actual=sha256(ROOT/key);wanted=snapshot['code'].get(key,{}).get('sha256')
        frozen_code[key]={'actual_sha256':actual,'frozen_sha256':wanted,'matches':actual==wanted}
        require(actual==wanted,f'frozen scientific source unchanged: {key}')
    referenced=set()
    summaries={};steps={};runs=[]
    per_seed={}
    all_complete=True
    for record in registry:
        name=record['id'];status=state['jobs'].get(name,{}).get('status','missing')
        require(status not in {'failed','blocked','missing'},f'scheduler status {status}',name)
        config=json.loads(Path(record['config_path']).read_text())
        relative=logical_key(record['config_path'])
        require(sha256(record['config_path'])==snapshot['configs'][relative]['sha256'],'frozen input config hash',name)
        referenced.update([config['target_path'],*config['validation_paths'].values(),*record.get('test_paths',{}).values()])
        if config['replay_path']:referenced.add(config['replay_path'])
        require(config['target_prediction_tokens']==10_000_000,'planned 10M target tokens',name)
        require(config['learning_rate']==expected_lr,f'shared LR {expected_lr}',name)
        require(config['replay_weight']==(0 if record['group']=='B0' else expected_lambda),f'shared lambda {expected_lambda}',name)
        if require_deterministic:
            require(config['deterministic_algorithms'] and config.get('attention_backend')=='auto','strict deterministic SDPA auto configuration',name)
            require(record.get('active') is True,'active strict registry',name)
        require(config['microbatch_size']==16 and config['gradient_accumulation_steps']==4,'microbatch16 accum4',name)
        require(config['precision']=='bf16' and config['evaluate_teacher_kl'],'FP32 weights/BF16 autocast with teacher validation KL',name)
        require(config['eval_every_steps']==50 and config['save_every_steps']==0,'frozen validation/save schedule',name)
        require(set(config['validation_paths'])=={'target','c4','wikitext'},'validation corpus names only',name)
        require(all('evaluation_v3' in path and 'validation' in Path(path).name for path in config['validation_paths'].values()),'v3 validation paths only',name)
        event_path=Path(record['output_dir'])/'events.jsonl'
        events=events_at(event_path,allow_incomplete) if event_path.exists() else []
        require(not any(event['event']=='error' for event in events),'no training error events',name)
        current_steps=[event for event in events if event['event']=='train_step']
        steps[name]=current_steps
        if current_steps:
            target_sum=replay_sum=0
            for index,event in enumerate(current_steps,1):
                target_sum+=event['step_counts']['target_prediction_tokens']
                replay_sum+=event['step_counts']['replay_prediction_tokens']
                require(event['step']==index,'contiguous active step audit trail',name)
                require(event['counters']['target_prediction_tokens']==target_sum,'per-step cumulative target sum',name)
                require(event['counters']['replay_prediction_tokens']==replay_sum,'per-step cumulative replay sum',name)
                require(replay_sum==expected_replay(target_sum,record['ratio']),'exact cumulative replay floor budget',name)
                require(math.isfinite(event['target_ce']) and math.isfinite(event['grad_norm']),'finite training losses/gradients',name)
                if event['replay_loss'] is not None:require(math.isfinite(event['replay_loss']),'finite replay loss',name)
            # Every comparable prefix must share the exact target data stream and LR.
            seed=record['seed']
            reference=per_seed.setdefault(seed,current_steps)
            for left,right in zip(reference,current_steps):
                require(left['target_stream_hash']==right['target_stream_hash'],'same-seed per-step target stream',name)
                require(left['learning_rate']==right['learning_rate'],'same-seed per-step LR',name)
        if status!='completed':
            all_complete=False
            runs.append({'id':name,'scheduler_status':status,'last_step':current_steps[-1]['step'] if current_steps else 0})
            continue
        summary_path=Path(record['summary_path'])
        require(summary_path.exists(),'synced summary exists',name)
        if not summary_path.exists():continue
        summary=json.loads(summary_path.read_text());summaries[name]=summary
        if require_deterministic:
            provenance=summary['provenance']
            require(provenance['deterministic_algorithms'] and provenance['deterministic_warn_only'] is False,'strict runtime determinism (no warn-only)',name)
            require(provenance['cublas_workspace_config']==':4096:8','fixed cuBLAS deterministic workspace',name)
            require(provenance['cudnn_deterministic'] and not provenance['cudnn_benchmark'],'fixed cuDNN execution policy',name)
            require(provenance['sdpa_backends_enabled'].get('cudnn') is True,'cuDNN SDP backend explicitly enabled under auto',name)
            require(provenance['model_weight_dtype']=='torch.float32' and provenance['teacher_weight_dtype']=='torch.float32','actual FP32 student and teacher weights',name)
        counters=summary['counters'];budget=expected_replay(10_000_000,record['ratio'])
        require(summary['status']=='completed','training completed fixed budget',name)
        require(summary['config']==config,'runtime configuration equals frozen input',name)
        require(counters['target_prediction_tokens']==10_000_000,'achieved 10M target prediction tokens',name)
        require(counters['replay_prediction_tokens']==budget,'achieved exact replay prediction budget',name)
        require(sum(event['step_counts']['target_prediction_tokens'] for event in current_steps)==10_000_000,'event target exposures equal summary',name)
        require(sum(event['step_counts']['replay_prediction_tokens'] for event in current_steps)==budget,'event replay exposures equal summary',name)
        require(counters['step']==306 and len(current_steps)==306,'306 optimizer steps for the frozen target batch',name)
        bank_capacity=0 if record['group']=='B0' else 100156
        require(summary['replay_bank_capacity_prediction_tokens']==bank_capacity,'replay bank capacity K',name)
        require(summary['replay_bank_unique_prediction_tokens']==bank_capacity,'replay bank unique capacity K',name)
        unique_expected=min(budget,bank_capacity)
        require(summary['replay_unique_prediction_tokens_exposed']==unique_expected,'actual replay unique exposure coverage',name)
        target_capacity=summary['provenance']['target']['unique_prediction_tokens']
        require(summary['target_unique_prediction_tokens_exposed']==min(10_000_000,target_capacity),'actual target unique exposure coverage',name)
        if bank_capacity:
            coverage=summary['coverage_accounting']['replay']
            require(coverage['no_duplicate_chunks_verified'],'no-duplicate assumption is verified',name)
            require(coverage['repeated_prediction_token_exposures']==budget-unique_expected,'repeated replay exposures separated',name)
        trajectory=summary['validation_trajectory']
        require([v['step'] for v in trajectory]==[50,100,150,200,250,300,306],'complete validation trajectory',name)
        for point in trajectory:
            require(set(point['metrics'])=={'target','c4','wikitext'},'no test corpus used in checkpoint selection',name)
            for corpus,metrics in point['metrics'].items():
                require(metrics['complete_corpus'] and math.isfinite(metrics['nll']),'complete finite validation NLL',name)
                require('kl' in metrics and math.isfinite(metrics['kl']),'complete finite teacher KL',name)
                require(metrics['model_weight_dtype']=='torch.float32' and metrics['precision']=='bf16','uniform evaluation precision',name)
        selected=min(trajectory,key=lambda point:point['metrics']['target']['nll'])
        best=summary['best_target_validation']
        require(best['step']==selected['step'] and best['nll']==selected['metrics']['target']['nll'],'best checkpoint selected by target validation only',name)
        require(Path(best['checkpoint_path']).exists(),'best model-only checkpoint exists',name)
        require(Path(summary['checkpoint_path']).exists(),'final full checkpoint exists',name)
        timing=summary['timing']
        require(timing['training_teacher_forward_seconds_complete'],'teacher training timing complete',name)
        require(timing['training_teacher_forward_seconds_nested_in']=='training_seconds' and not timing['nested_timers_are_additive_costs'],'nested teacher timers not double counted',name)
        require(0<=counters['training_teacher_forward_seconds']<=counters['training_seconds']+.01,'teacher time is training subcost',name)
        require(0<=counters['validation_teacher_forward_seconds']<=counters['validation_seconds']+.01,'validation teacher is validation subcost',name)
        require(all(counters[field]>=0 for field in ['setup_seconds','training_seconds','validation_seconds','checkpoint_seconds']),'nonnegative disjoint major costs',name)
        runs.append({'id':name,'scheduler_status':status,'target_tokens':counters['target_prediction_tokens'],
            'replay_tokens':budget,'bank_capacity':bank_capacity,'actual_unique_replay_tokens':unique_expected,
            'repeated_replay_tokens':budget-unique_expected,'best_target_step':best['step'],
            'final_target_nll':trajectory[-1]['metrics']['target']['nll'],'peak_gpu_memory_bytes':summary['peak_gpu_memory_bytes'],
            'costs':{k:counters[k] for k in ['setup_seconds','training_seconds','validation_seconds','checkpoint_seconds',
                    'training_teacher_forward_seconds','validation_teacher_forward_seconds','validation_kl_arithmetic_seconds']}})
    frozen_inputs={}
    for path in sorted(referenced):
        key=logical_key(path)
        actual=sha256(path);wanted=snapshot['data'][key]['sha256']
        frozen_inputs[key]={'actual_sha256':actual,'frozen_sha256':wanted,'matches':actual==wanted}
        require(actual==wanted,f'frozen data/bank tensor hash: {key}')
    null_controls=[]
    null_atol,null_rtol=(1e-7,0.) if require_deterministic else (1e-5,1e-5)
    for seed in [11,22,33]:
        bank={r['group']:r for r in registry if r['seed']==seed and r['ratio']==.05 and r['group'] in ['B4','D']}
        b4,d=bank['B4'],bank['D']
        if b4['id'] not in summaries or d['id'] not in summaries:continue
        left,right=summaries[b4['id']],summaries[d['id']]
        same_bank=left['provenance']['replay']['sha256_tensors']==right['provenance']['replay']['sha256_tensors']
        require(same_bank,'D/B4 use identical bank tensors for the null control',d['id'])
        comparisons=[]
        for a,b in zip(left['validation_trajectory'],right['validation_trajectory']):
            for corpus in ['target','c4','wikitext']:
                for metric in ['nll','kl']:
                    comparison=compare_values(a['metrics'][corpus][metric],b['metrics'][corpus][metric],null_atol,null_rtol)
                    comparisons.append({'step':a['step'],'corpus':corpus,'metric':metric,**comparison})
        per_step=[]
        for a,b in zip(steps[b4['id']],steps[d['id']]):
            for metric in ['target_ce','replay_loss','grad_norm']:
                per_step.append(compare_values(a[metric],b[metric],null_atol,null_rtol))
        equivalent=all(item['within_tolerance'] for item in comparisons+per_step)
        require(equivalent,f'D/B4 numerical trajectories agree (atol={null_atol},rtol={null_rtol})',d['id'])
        null={'seed':seed,'B4':b4['id'],'D':d['id'],'same_bank':same_bank,'numerical_equivalent':equivalent,
              'atol':null_atol,'rtol':null_rtol,'max_validation_absolute_difference':max(c['absolute_difference'] for c in comparisons),
              'max_training_absolute_difference':max(c['absolute_difference'] for c in per_step),'validation_comparisons':comparisons}
        if all_complete and compare_weights:
            left_hash=model_fingerprint(left['checkpoint_path']);right_hash=model_fingerprint(right['checkpoint_path'])
            null.update(B4_model_fingerprint=left_hash,D_model_fingerprint=right_hash,
                        bitwise_model_state_equal=left_hash==right_hash)
            if require_deterministic:require(left_hash==right_hash,'D/B4 final model state bitwise equal',d['id'])
        null_controls.append(null)
    if not allow_incomplete:require(all_complete,'all 36 scheduler jobs completed')
    status='failed' if problems else ('passed' if all_complete else 'in_progress')
    return {'audited_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'status':status,
        'completed_runs':len(summaries),'expected_runs':36,'violations':problems,
        'frozen_scientific_code':frozen_code,'frozen_input_tensors':frozen_inputs,'runs':runs,
        'D_B4_null_controls':null_controls,'model_weights_compared':all_complete and compare_weights,
        'cost_note':'teacher forward and KL arithmetic timers are nested subcosts and must not be added to their parent totals',
        'test_likelihood_accessed_by_this_audit':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',type=Path,default=ROOT/'configs/formal_registry.json')
    parser.add_argument('--state',type=Path,default=ROOT/'logs/formal_job_state.json')
    parser.add_argument('--snapshot',type=Path,default=ROOT/'artifacts/provenance/formal_input_snapshot.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/formal_run_audit.json')
    parser.add_argument('--allow-incomplete',action='store_true')
    parser.add_argument('--skip-model-weights',action='store_true')
    parser.add_argument('--learning-rate',type=float,default=3e-5)
    parser.add_argument('--shared-lambda',type=float,default=.05)
    parser.add_argument('--require-deterministic',action='store_true')
    args=parser.parse_args()
    result=audit(json.loads(args.registry.read_text()),json.loads(args.state.read_text()),
                 json.loads(args.snapshot.read_text()),args.allow_incomplete,not args.skip_model_weights,
                 args.learning_rate,args.shared_lambda,args.require_deterministic)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(args.output.suffix+'.tmp')
    temporary.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(temporary,args.output)
    print(json.dumps({'status':result['status'],'completed_runs':result['completed_runs'],'violations':len(result['violations']),'output':str(args.output)}))
    if result['violations']:raise SystemExit(1)


if __name__=='__main__':main()
