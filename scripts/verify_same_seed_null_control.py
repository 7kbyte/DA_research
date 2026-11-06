#!/usr/bin/env python3
"""Require complete B4/D numerical identity before releasing remaining formal jobs."""
from __future__ import annotations
import argparse,datetime as dt,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_formal_runs import model_fingerprint,events_at
from make_experiments import atomic_json

def verify(registry,seed):
    pair={r['group']:r for r in registry if r['group'] in ['B4','D'] and r['seed']==seed and r['ratio']==.05}
    if set(pair)!={'B4','D'}:raise ValueError('Requires a unique B4/D matched seed pair')
    summaries={g:json.loads(Path(r['summary_path']).read_text()) for g,r in pair.items()}
    failures=[]
    def check(condition,label):
        if not condition:failures.append(label)
    left,right=summaries['B4'],summaries['D']
    for g,s in summaries.items():
        check(s['status']=='completed' and s['counters']['target_prediction_tokens']==10_000_000,f'{g} complete 10M')
        check(s['counters']['replay_prediction_tokens']==500_000,f'{g} replay500k')
        check(s['config']['deterministic_algorithms'] and not s['provenance']['deterministic_warn_only'],f'{g} strict determinism')
    omitted={'output_dir','replay_path','run_label'}
    check({k:v for k,v in left['config'].items() if k not in omitted}==
          {k:v for k,v in right['config'].items() if k not in omitted},'matched scientific configurations')
    check(left['provenance']['replay']['sha256_tensors']==right['provenance']['replay']['sha256_tensors'],'identical replay tensor')
    check(left['target_stream']==right['target_stream'],'final target stream')
    check(left['replay_stream']==right['replay_stream'],'final replay stream')
    events={g:[e for e in events_at(Path(r['output_dir'])/'events.jsonl',False) if e['event']=='train_step'] for g,r in pair.items()}
    check(len(events['B4'])==len(events['D'])==306,'complete306-step trajectories')
    max_train=0.
    for a,b in zip(events['B4'],events['D']):
        for field in ['target_stream_hash','replay_stream_hash','learning_rate','step_counts']:
            check(a[field]==b[field],f'step{a["step"]} {field}')
        for field in ['target_ce','replay_loss','grad_norm']:
            diff=abs(a[field]-b[field]);max_train=max(max_train,diff)
            check(diff<=1e-7,f'step{a["step"]} {field}')
    check([v['step'] for v in left['validation_trajectory']]==[v['step'] for v in right['validation_trajectory']]==[50,100,150,200,250,300,306],'validation schedule')
    max_validation=0.
    for a,b in zip(left['validation_trajectory'],right['validation_trajectory']):
        for corpus in ['target','c4','wikitext']:
            for metric in ['nll','kl']:
                diff=abs(a['metrics'][corpus][metric]-b['metrics'][corpus][metric]);max_validation=max(max_validation,diff)
                check(diff<=1e-7,f'validation{a["step"]} {corpus} {metric}')
    fingerprints={g:model_fingerprint(s['checkpoint_path']) for g,s in summaries.items()}
    check(fingerprints['B4']==fingerprints['D'],'final model state bitwise equality')
    return {'status':'failed' if failures else 'passed','seed':seed,'run_ids':{g:r['id'] for g,r in pair.items()},
        'violations':failures,'max_training_absolute_difference':max_train,'max_validation_absolute_difference':max_validation,
        'model_fingerprints':fingerprints,'bitwise_model_state_equal':fingerprints['B4']==fingerprints['D'],
        'tolerance_absolute':1e-7,'tolerance_relative':0.,'audited_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'test_likelihood_accessed':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',type=Path,required=True);parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=verify(json.loads(args.registry.read_text()),args.seed);atomic_json(args.output,result)
    print(json.dumps(result));raise SystemExit(0 if result['status']=='passed' else 1)
