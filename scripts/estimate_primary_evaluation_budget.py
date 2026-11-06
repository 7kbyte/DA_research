#!/usr/bin/env python3
"""Score-blind timing sensitivity for primary NLL/behavior before training cutoff."""
import argparse
import collections
import datetime as dt
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_formal_runs import sha256
from make_experiments import atomic_json


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-budget',type=Path,default=ROOT/'results/supplemental_training_budget_before_primary_evaluation.json')
    p.add_argument('--output',type=Path,default=ROOT/'results/primary_early_evaluation_budget.json')
    args=p.parse_args()
    sources={'training_budget':args.training_budget,'auxiliary_timing':ROOT/'results/auxiliary_evaluation_timing_completed_v1.json',
        'replay_manifest':ROOT/'docs/replay_fit_diagnostic_v2_manifest.json','nll_budget':ROOT/'results/nll_scope_budget_v1.json',
        'behavior_budget':ROOT/'results/behavioral_timing_budget_v1.json',
        'primary_nll_manifest':ROOT/'docs/primary_evaluation_v2_execution_matched_nll_manifest.json',
        'primary_behavior_manifest':ROOT/'docs/primary_evaluation_v2_execution_matched_behavior_manifest.json'}
    data={key:read(path) for key,path in sources.items()};training=data['training_budget'];timing=data['auxiliary_timing']
    replay=timing['replay_fit']
    if not replay['all_15_completed'] or any(not row['output_hashes_valid'] or row['state']['status']!='completed' or row['state']['returncode']!=0 for row in replay['jobs']):
        raise ValueError('Complete integrity-checked score-blind diagnostic timing is required')
    pools=collections.defaultdict(list)
    for row in replay['jobs']:
        kind=row['id'].split('_')[-1]
        pools[kind].append(row['state']['elapsed_seconds'])
    main=next(group for group in data['nll_budget']['groups'] if group['model']=='pythia-410m' and group['data_family']=='main_v3')
    source_tokens=data['replay_manifest']['workload']['prediction_tokens_per_case']
    target_tokens=main['prediction_tokens_per_inference']
    added_loops=max(0,target_tokens-source_tokens)*main['timing_proxy']['median_seconds_per_prediction_token']
    proxy={kind:max(values)+added_loops for kind,values in pools.items()}
    counts=collections.Counter()
    for job in data['primary_nll_manifest']['jobs']:
        if job['kind']=='initial':kind='initial'
        else:
            kinds={r['kind'] for r in data['primary_nll_manifest']['records'] if r['inference_job_id']==job['id']}
            kind=max(kinds,key=lambda k:proxy[k])
        counts[kind]+=1
    nll_seconds=sum(proxy[kind]*number for kind,number in counts.items())
    now=dt.datetime.now(dt.timezone.utc)
    deadline=dt.datetime.fromisoformat(training['training_deadline_utc'].replace('Z','+00:00'))
    available_hours=(deadline-now).total_seconds()/3600
    train_gpu_hours=training['requested_remaining_gpu_hours'];contingency=training['contingency_fraction']
    cases=[]
    for behavior_case,factor in zip(data['behavior_budget']['primary']['sensitivity_scenarios'],[1.,1.25,1.5]):
        if behavior_case['adapted_scorer_factor_assumption']!=factor:raise ValueError('Sensitivity grid changed')
        behavior_seconds=behavior_case['remaining_scorer_job_wall_seconds_proxy']
        evaluation_slot_hours=(nll_seconds*factor+behavior_seconds)/3600
        evaluation_wall_hours=evaluation_slot_hours/4
        total_wall_hours=(train_gpu_hours+evaluation_slot_hours)*(1+contingency)/4
        before_midnight=available_hours-total_wall_hours
        cases.append({'adapted_behavior_scorer_factor':factor,'nll_whole_job_factor':factor,
            'nll_job_seconds_proxy':nll_seconds*factor,'behavior_job_seconds_proxy':behavior_seconds,
            'additional_four_slot_hours_proxy':evaluation_wall_hours,
            'additional_four_slot_hours_with_10pct':evaluation_wall_hours*(1+contingency),
            'all_training_plus_primary_evaluation_four_slot_hours_with_10pct':total_wall_hours,
            'margin_before_midnight_minutes':before_midnight*60,
            'margin_before_master_2350_stop_minutes':before_midnight*60-10,
            'fits_before_midnight':before_midnight>=0,'fits_before_master_stop':before_midnight>=1/6})
    result={'recorded_at_utc':now.isoformat(),'status':'timing_sensitivity_only_no_execution',
        'sources':{k:{'path':str(path),'sha256':sha256(path)} for k,path in sources.items()},
        'requested_primary_nll_jobs':len(data['primary_nll_manifest']['jobs']),
        'remaining_primary_behavior_jobs':sum(job['kind']!='initial' for job in data['primary_behavior_manifest']['jobs']),
        'completed_primary_behavior_initial_not_charged_again':True,'nll_job_kind_counts':dict(counts),
        'replay_fit_full_job_scheduler_elapsed_pools':dict(pools),
        'source_diagnostic_prediction_tokens_per_case':source_tokens,'primary_prediction_tokens_per_case':target_tokens,
        'additional_validation_rate_corpus_loop_seconds_proxy':added_loops,
        'nll_elapsed_proxy_seconds_by_kind':proxy,'nll_elapsed_proxy_method':'Max observed same-kind complete replay-fit whole-job wall plus additional token-scaled validation loop wall; startup/SHA/polling already included, teacher timing nested.',
        'remaining_training_four_slot_hours_with_10pct':train_gpu_hours*(1+contingency)/4,
        'remaining_window_before_midnight_hours':available_hours,'master_stop_margin_seconds':600,
        'new_initial_scorer_seconds_context_only':{row['id']:row['execution']['inference_subprocess_wall_seconds'] for row in timing['initials']},
        'sensitivity_scenarios':cases,'central_fits_before_master_stop':cases[1]['fits_before_master_stop'],
        'all_scenarios_fit_before_master_stop':all(case['fits_before_master_stop'] for case in cases),
        'limitations':['Four-slot division assumes continuously busy independent GPUs; not a runtime guarantee.',
            'Whole long-horizon training remains extrapolated from short measured probes; its10% contingency is preserved.',
            'Replay-fit and primary NLL data differ; token-rate transfer and shared CPU/I/O contention are sensitivity assumptions.',
            'Behavior adapted checkpoint startup is unmeasured; existing1/1.25/1.5 scorer sensitivity is retained.',
            'This includes primary67 NLL and66 remaining behavior only; supplemental NLL/behavior still need their independent later budget.'],
        'scores_or_predictions_deserialized':False,'gpu_work_started':False,'training_or_scientific_conditions_changed':False,
        'producer_source_sha256':sha256(__file__)}
    atomic_json(args.output,result)
    print(json.dumps({k:result[k] for k in ['requested_primary_nll_jobs','remaining_primary_behavior_jobs','central_fits_before_master_stop','all_scenarios_fit_before_master_stop','sensitivity_scenarios']}))


if __name__=='__main__':main()
