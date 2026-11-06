#!/usr/bin/env python3
"""Choose an extended-data microbatch from frozen throughput and memory rules."""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from make_experiments import atomic_json

POLICY={'minimum_throughput_ratio':1.05,'minimum_reserved_headroom_bytes':8*1024**3,
        'effective_batch_chunks':64,'steady_steps':'2-10',
        'selection_inputs':'throughput and memory only; no validation or test losses',
        'tie_rule':'retain the previously measured microbatch'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def choose(model,baseline,candidate,policy=POLICY):
    if baseline['model']!=model or candidate['model']!=model:raise ValueError('Benchmark model mismatch')
    for report in (baseline,candidate):
        if report['status']!='succeeded':continue
        if report['microbatch']*report['accumulation']!=policy['effective_batch_chunks']:
            raise ValueError('Benchmark effective batch mismatch')
        speed=report['steady_target_prediction_tokens_per_second']
        if not math.isfinite(speed) or speed<=0:raise ValueError('Invalid benchmark throughput')
    if baseline['status']!='succeeded':raise ValueError('A successful baseline is required')
    ratio=None;headroom=None;accepted=False
    if candidate['status']=='succeeded':
        ratio=candidate['steady_target_prediction_tokens_per_second']/baseline['steady_target_prediction_tokens_per_second']
        headroom=candidate['gpu_total_memory_bytes']-candidate['peak_gpu_reserved_bytes']
        accepted=ratio>=policy['minimum_throughput_ratio'] and headroom>=policy['minimum_reserved_headroom_bytes']
    selected=candidate if accepted else baseline
    return {'model':model,'selected_microbatch':selected['microbatch'],
        'gradient_accumulation_steps':selected['accumulation'],'effective_batch_chunks':64,
        'candidate_adopted':accepted,'candidate_throughput_ratio':ratio,'candidate_reserved_headroom_bytes':headroom,
        'selection_policy':policy,'failed_attempts':int(candidate['status']!='succeeded'),
        'failed_attempt_seconds':candidate.get('failed_wall_seconds',0) if candidate['status']!='succeeded' else 0,
        'formal_configs_changed':False,'root_controls_formal_budget':True}

def primary_baseline(registry_path,run_id):
    record=next(r for r in json.loads(registry_path.read_text()) if r['id']==run_id)
    config_path=Path(record['config_path']);config=json.loads(config_path.read_text())
    if config['microbatch_size']!=16 or config['gradient_accumulation_steps']!=4:
        raise ValueError('Expected the main micro16 accumulated baseline')
    event_path=Path(record['output_dir'])/'events.jsonl'
    steps=[r for r in map(json.loads,event_path.read_text().splitlines()) if r['event']=='train_step'][1:10]
    if len(steps)!=9:raise ValueError('Need completed steps 2 through 10')
    tokens=sum(r['step_counts']['target_prediction_tokens'] for r in steps)
    seconds=sum(r['step_seconds'] for r in steps)
    return {'status':'succeeded','model':'pythia-410m','microbatch':16,'accumulation':4,
        'steady_steps':'2-10','steady_target_tokens':tokens,'steady_seconds':seconds,
        'steady_target_prediction_tokens_per_second':tokens/seconds,
        'config_path':str(config_path),'config_sha256':sha(config_path),
        'event_path':str(event_path),'event_sha256':sha(event_path),
        'comparison_limit':'Baseline uses primary target and training seed11; candidate uses extended target and seed101. Both use identical sequence length, effective batch and strict BF16 FP32 execution; timing is a short operational estimate.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['pythia-410m','pythia-1b'],required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--baseline',type=Path)
    p.add_argument('--primary-registry',type=Path,default=ROOT/'configs/formal_deterministic_registry.json')
    p.add_argument('--primary-run-id',default='formal_deterministic_B4_s11_b0p05_w0p05')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.model=='pythia-410m' and not args.baseline:
        baseline=primary_baseline(args.primary_registry,args.primary_run_id)
        baseline_ref={'registry_path':str(args.primary_registry),'registry_sha256':sha(args.primary_registry),
                      'derived_benchmark':baseline}
    else:
        if not args.baseline:p.error('--baseline is required for this model')
        baseline=json.loads(args.baseline.read_text());baseline_ref={'path':str(args.baseline),'sha256':sha(args.baseline)}
    candidate=json.loads(args.candidate.read_text())
    result=choose(args.model,baseline,candidate)
    result.update(baseline=baseline_ref,candidate={'path':str(args.candidate),'sha256':sha(args.candidate)})
    if args.output.exists() and json.loads(args.output.read_text())!=result:raise FileExistsError(args.output)
    atomic_json(args.output,result);print(json.dumps(result))

if __name__=='__main__':main()
