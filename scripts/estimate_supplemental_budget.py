#!/usr/bin/env python3
"""Estimate the requested supplemental matrix from measured nested wall timers."""
import argparse,datetime as dt,json,math,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from make_experiments import atomic_json

def read(path):return json.loads(Path(path).read_text())
def mean(values):return statistics.mean(values)

def estimate(deadline,contingency,secondary_target_tokens=50_000_000,include_secondary=True):
    primary=[]
    for row in read(ROOT/'configs/formal_deterministic_registry.json'):
        path=Path(row['summary_path'])
        if path.exists():
            summary=read(path)
            if summary['status']=='completed' and row['ratio'] in {0,.05}:primary.append((row,summary))
    groups={g:[s['counters'] for r,s in primary if r['group']==g] for g in ['B0','B1','B2','B3','B4','D','M','C']}
    base=mean(c['training_seconds'] for c in groups['B4']);speed410=10_000_000/base
    training_factors={g:mean(c['training_seconds'] for c in cs)/base for g,cs in groups.items()}
    validation410=mean(s['counters']['validation_seconds']/len(s['validation_trajectory']) for _,s in primary)
    primary_tokens=sum(next(s for _,s in primary)['validation_trajectory'][-1]['metrics'][c]['valid_prediction_tokens'] for c in ['target','c4','wikitext'])
    extended_tokens=sum(read(ROOT/f'artifacts/data_extended/{c}_validation.json')['prediction_tokens'] for c in ['finemath','c4','wikitext103'])
    master=read(ROOT/'logs/supplemental_job_state.json') if (ROOT/'logs/supplemental_job_state.json').exists() else {'jobs':{}}
    requested=[];model_estimates={}
    benchmarks=read(ROOT/'configs/supplemental_v1_benchmark_registry.json')
    remaining_benchmarks=[r for r in benchmarks if master['jobs'].get(r['id'],{}).get('status')!='completed']
    requested.append({'phase':'batch_benchmark','requested_runs':2,'remaining_runs':len(remaining_benchmarks),
                      'remaining_gpu_seconds':180*len(remaining_benchmarks),'estimate_basis':'Queue conservative allowance180s each until the probes finish; completed probes are removed'})
    A=read(ROOT/'configs/supplemental_v1_A_registry.json')
    A_new=[r for r in A if r['execution_kind']=='train']
    A_remaining=[r for r in A_new if master['jobs'].get(r['id'],{}).get('status')!='completed']
    A_seconds=sum(mean(c['total_accounted_wall_seconds'] for c in groups[r['group']]) for r in A_remaining)
    requested.append({'phase':'A','requested_new_runs':108,'remaining_runs':len(A_remaining),'remaining_gpu_seconds':A_seconds,'estimate_basis':'Matched complete main runs by group; all 7 validation points and checkpoint cost included'})
    B_path=ROOT/'artifacts/bank_variance_410m_v1/registry.json'
    B_remaining=60
    if B_path.exists():B_remaining=sum(master['jobs'].get(r['id'],{}).get('status')!='completed' for r in read(B_path))
    B_seconds=B_remaining*mean(c['total_accounted_wall_seconds'] for g in ['B4','M','C'] for c in groups[g])
    requested.append({'phase':'B','requested_runs':60,'remaining_runs':B_remaining,'remaining_gpu_seconds':B_seconds,'estimate_basis':'Complete KL main runs; identical exposure and sequence length, conditional bank selection CPU cost separate'})
    for model,count,target in [('pythia-410m',24,100_000_000),('pythia-1b',24,100_000_000),('pythia-2.8b',15 if include_secondary else 0,secondary_target_tokens)]:
        steps=math.ceil(target/32704);npoints=steps//153+int(steps%153!=0)
        if model=='pythia-410m':
            baseline_speed=speed410;speed=speed410;micro=16
            counters=[s['counters'] for r,s in primary if r['group']=='B4']
            latest=mean(c['elapsed_seconds']-c['prior_elapsed_seconds'] for c in counters)
            setup=mean(c['setup_seconds'] for c in counters);source='Completed main B4 runs'
        else:
            slug=model.replace('.','p').replace('-','_');default=16 if model=='pythia-1b' else 8
            bench_path=ROOT/f'results/benchmarks/benchmark_extended_{slug}_m{default}_s101.json'
            bench=read(bench_path);baseline_speed=bench['steady_target_prediction_tokens_per_second']
            speed=baseline_speed;micro=default;latest=bench['counters']['checkpoint_seconds'];setup=bench['counters']['setup_seconds'];source=str(bench_path)
        choice=ROOT/f'results/benchmarks/extended_v2_{model.replace(".","p").replace("-","_")}_batch_selection.json'
        if choice.exists():
            chosen=read(choice);micro=chosen['selected_microbatch']
            if chosen.get('candidate_adopted'):
                candidate=read(chosen['candidate']['path']);speed=candidate['steady_target_prediction_tokens_per_second'];source=chosen['candidate']['path']
        validation=validation410*(extended_tokens/primary_tokens)*(speed410/baseline_speed)
        best_save=latest/3
        phase_index=ROOT/'configs/supplemental_v1_registry_index.json'
        phases=read(phase_index)['phases'] if phase_index.exists() else {}
        pilots_done=0;measured=[];measured_groups={};formal_done=0
        prefix=f'extended_v2_{model.replace(".","p").replace("-","_")}_m{micro}'
        for name,phase in phases.items():
            if not name.startswith(prefix+'_'):continue
            rows=read(phase['registry'])
            if name.endswith('_formal'):
                formal_done+=sum(master['jobs'].get(r['id'],{}).get('status')=='completed' for r in rows)
            elif name.endswith(('_lr','_lambda')):
                for row in rows:
                    if master['jobs'].get(row['id'],{}).get('status')=='completed':
                        pilots_done+=1;s=read(row['summary_path']);measured.append(s)
                        measured_groups.setdefault(row['group'],[]).append(s['counters']['training_seconds'])
        if measured:
            validation=mean(s['counters']['validation_seconds']/len(s['validation_trajectory']) for s in measured)
            latest=mean(s['counters']['elapsed_seconds']-s['counters']['prior_elapsed_seconds'] for s in measured)
            best_save=mean(s['counters']['checkpoint_seconds']-(s['counters']['elapsed_seconds']-s['counters']['prior_elapsed_seconds']) for s in measured)
            setup=mean(s['counters']['setup_seconds'] for s in measured)
        cohort=['B0','B1','B2','B4','M'] if model=='pythia-2.8b' else list(groups)
        model_factors=dict(training_factors)
        if measured_groups.get('B4'):
            denominator=mean(measured_groups['B4'])
            model_factors.update({g:mean(seconds)/denominator for g,seconds in measured_groups.items()})
            model_factors.update({g:1. for g in ['D','M','C']})
        factor=mean(model_factors[g] for g in cohort)
        formal_per_run=target/speed*factor+npoints*(validation+best_save)+latest+setup
        pilot_factor=(3*model_factors['B0']+4*sum(model_factors[g] for g in ['B1','B2','B3','B4']))/19
        pilot_per_run=2_000_000/speed*pilot_factor+validation+best_save+latest+setup
        pilot_seconds=(19-pilots_done)*pilot_per_run+(0 if measured else validation/2+setup)
        model_estimates[model]={'microbatch':micro,'steady_target_prediction_tokens_per_second':speed,'throughput_source':source,
            'validation_seconds_per_point':validation,'validation_cost_source':'completed extended pilots' if measured else 'main measured validation scaled by model baseline training throughput and corpus tokens',
            'latest_checkpoint_seconds':latest,'model_only_checkpoint_seconds_estimate':best_save,'setup_seconds':setup,
            'formal_validation_points':npoints,'best_saves_assumed':npoints,'formal_seconds_per_run':formal_per_run,
            'formal_target_prediction_tokens':target,'pilot_seconds_per_run':pilot_per_run,'pilots_completed':pilots_done,'formal_completed':formal_done}
        model_estimates[model].update(training_time_factors=model_factors,
            training_factor_source='same model/micro pilot mean relative to B4; D/M/C use equal-shape KL factor1' if measured_groups.get('B4') else 'primary method ratios until matching pilot B4 is measured')
        requested.append({'phase':model+'_pilot','requested_training_runs':19,'remaining_runs':19-pilots_done,'remaining_gpu_seconds':pilot_seconds})
        requested.append({'phase':model+'_formal','requested_runs':count,'remaining_runs':count-formal_done,'target_tokens_per_run':target,'remaining_gpu_seconds':(count-formal_done)*formal_per_run})
    now=dt.datetime.now(dt.timezone.utc);remaining=(dt.datetime.fromisoformat(deadline.replace('Z','+00:00'))-now).total_seconds()
    gpu_seconds=sum(r['remaining_gpu_seconds'] for r in requested)
    return {'estimated_at_utc':now.isoformat(),'training_deadline_utc':deadline,'remaining_wall_hours':remaining/3600,
        'requested_secondary_target_prediction_tokens':secondary_target_tokens if include_secondary else None,
        'available_gpu_hours':4*remaining/3600,'requested_remaining_gpu_hours':gpu_seconds/3600,
        'ideal_four_gpu_wall_hours':gpu_seconds/4/3600,'contingency_fraction':contingency,
        'four_gpu_wall_hours_with_contingency':gpu_seconds*(1+contingency)/4/3600,
        'fits_without_contingency':gpu_seconds<=4*remaining,'fits_with_contingency':gpu_seconds*(1+contingency)<=4*remaining,
        'requested_matrix':requested,'models':model_estimates,'main_validation_seconds_per_point':validation410,
        'main_validation_tokens':primary_tokens,'extended_validation_tokens':extended_tokens,
        'limitations':['Short benchmark throughput extrapolates to long runs; validation for larger models becomes measured after pilots finish.',
            'Assumes all validation points improve target NLL and save model-only best; this is a conservative checkpoint bound.',
            'Independent test and behavioral scoring, CPU bank preparation and archive work are excluded; the final 3h26 reserve is separate.',
            'Before main36 completes this timestamp includes its remaining tail and audit; refresh after master launch for the actual supplemental window.',
            'Teacher forward and KL arithmetic are nested in training/validation totals and never added twice.'],
        'priority_policy':'Preserve preregistered comparisons and all 3 training seeds; A prioritizes B4/M; 2.8B is the secondary model-size extension. Root reviews compute-only reductions before changing the requested matrix.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deadline',default='2026-11-07T00:00:00Z')
    p.add_argument('--contingency',type=float,default=.1);p.add_argument('--secondary-target-tokens',type=int,choices=[25_000_000,50_000_000])
    p.add_argument('--output',type=Path,default=ROOT/'results/supplemental_compute_budget.json');args=p.parse_args()
    selected=args.secondary_target_tokens;enabled=True
    decision=ROOT/'docs/extended_2p8b_horizon_selection.json'
    if selected is None and decision.exists():
        selected=read(decision)['selected_target_prediction_tokens'];enabled=selected is not None
    result=estimate(args.deadline,args.contingency,selected or 50_000_000,enabled);atomic_json(args.output,result)
    print(json.dumps({k:result[k] for k in ['remaining_wall_hours','requested_remaining_gpu_hours','ideal_four_gpu_wall_hours','four_gpu_wall_hours_with_contingency','fits_with_contingency']}))
