#!/usr/bin/env python3
"""Replace model-size timing extrapolation using score-blind initial attestations."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def update(first_path,aux_path,nll_path):
    first,aux,nll=map(read,(first_path,aux_path,nll_path))
    if first.get('complete_timing_available') is not True or aux.get('both_initials_completed') is not True or any(x.get('scores_or_predictions_parsed') is not False for x in (first,aux)):
        raise ValueError('Three complete score-blind initial timing observations required')
    source_sha={str(p):sha(p) for p in (first_path,aux_path,nll_path)}
    main_manifest=read(first['manifest']);aux_manifest=read(aux['initial_manifest'])
    if sha(first['manifest'])!=first['manifest_sha256'] or sha(aux['initial_manifest'])!=aux['initial_manifest_sha256']:raise ValueError('Timing source manifest changed')
    counts={'pythia-410m':109,'pythia-1b':25,'pythia-2.8b':16}
    rows=[('pythia-410m',first,main_manifest),*[(aux_manifest['baseline_groups'][j['baseline_group']]['model_tag'],next(r for r in aux['initials'] if r['id']==j['id']),aux_manifest) for j in aux_manifest['jobs']]]
    measured={}
    for model,row,manifest in rows:
        jid=row.get('id',row.get('job_id'));job=next(j for j in manifest['jobs'] if j['id']==jid)
        if job['kind']!='initial' or row.get('output_hashes_valid') is not True:raise ValueError('Initial integrity proof required')
        state=row.get('state',row.get('state_snapshot'));e=row['execution']
        if state.get('status')!='completed' or state.get('returncode')!=0:raise ValueError('Completed initial scheduler state required')
        scorer=e['inference_subprocess_wall_seconds'];verify=e['verification_wall_seconds'];elapsed=state['elapsed_seconds']
        input_bytes=sum(Path(p).stat().st_size for p in {**manifest['common_input_sha256'],**job['input_sha256']})
        if model=='pythia-410m':
            cp_bytes=max(r['checkpoint_bytes'] for r in main_manifest['records'] if r['kind']=='final')
            cp_size_source='Largest existing frozen primary final checkpoint; all extra 410M cells conservatively use final-size SHA proxy'
        else:
            proof=next(p for p in aux_manifest['source_execution_evidence'] if p['model']==model)
            registry=read(proof['source_registry']);sizes=[]
            for r in registry:
                summary=read(r['summary_path']);cp=Path(summary['checkpoint_path'])
                if summary['status']!='completed':raise ValueError('Only completed LR checkpoint-size proxy')
                sizes.append(cp.stat().st_size);source_sha[r['summary_path']]=sha(r['summary_path'])
            cp_bytes=max(sizes);cp_size_source='Largest completed same-model LR latest checkpoint file size; future formal file bytes unmeasured'
        measured[model]={'scorer_subprocess_wall_seconds':scorer,'SHA_manifest_verification_wall_seconds':verify,
                         'scheduler_elapsed_seconds':elapsed,'scheduler_unclassified_residual_wall_seconds':elapsed-scorer-verify,
                         'inference_policy':manifest['baseline_groups'][job['baseline_group']]['inference_policy'],
                         'initial_verified_input_bytes':input_bytes,'future_final_checkpoint_byte_size_proxy':cp_bytes,
                         'checkpoint_size_proxy_source':cp_size_source,'adapted_input_byte_ratio_proxy':(input_bytes+cp_bytes)/input_bytes,
                         'single_initial_observation_not_a_timing_distribution':True}
    if set(measured)!=set(counts):raise ValueError('One initial per frozen model required')
    scenarios=[]
    for scorer_factor,extra_load,nll_factor,nll_startup in [(1.,0.,.75,5.),(1.25,10.,1.,15.),(1.5,30.,1.5,30.)]:
        by_model={m:(counts[m]-1)*(v['scorer_subprocess_wall_seconds']*scorer_factor+v['SHA_manifest_verification_wall_seconds']*v['adapted_input_byte_ratio_proxy']+v['scheduler_unclassified_residual_wall_seconds']+extra_load) for m,v in measured.items()}
        behavioral=sum(by_model.values());nll_remaining=nll['summed_validation_rate_corpus_loop_seconds_proxy']*nll_factor+nll['all_NLL_inference_jobs_before_further_proofs']*nll_startup
        scenarios.append({'adapted_scorer_factor_assumption':scorer_factor,'extra_checkpoint_loading_allowance_per_behavior_job_seconds':extra_load,
                          'NLL_test_validation_rate_factor_assumption':nll_factor,'NLL_nonloop_startup_SHA_allowance_per_job_seconds':nll_startup,
                          'remaining_behavior_job_wall_seconds_proxy_by_model':by_model,'remaining_behavior_job_wall_seconds_proxy':behavioral,
                          'remaining_NLL_job_wall_seconds_proxy':nll_remaining,'combined_ideal_four_uncontended_slots_seconds_proxy':(behavioral+nll_remaining)/4,
                          'interpretation':'Sensitivity assumptions only, not confidence intervals, measured adapted costs or runtime guarantees'})
    return {'schema_version':2,'source_sha256':source_sha,'measured_initials':measured,
            'behavior_scope_before_further_proofs':{'total_inferences':150,'model_counts':counts,'completed_initials':3,'remaining_adapted_inferences_upper_count':147,
                                                    'primary_remaining':66,'unproven_future_A_C_reuse_not_deducted':True},
            'NLL_scope_before_further_proofs':{'logical_endpoints':nll['logical_endpoints'],'inferences':nll['all_NLL_inference_jobs_before_further_proofs']},
            'replay_fit':{'all_15_complete_integrity_verified':aux['replay_fit']['all_15_completed'] and all(r.get('output_hashes_valid') for r in aux['replay_fit']['jobs']),
                          'recorded_scorer_subprocess_wall_seconds_sum':sum(r['execution']['inference_subprocess_wall_seconds'] for r in aux['replay_fit']['jobs']),
                          'recorded_SHA_manifest_verification_wall_seconds_sum':sum(r['execution']['verification_wall_seconds'] for r in aux['replay_fit']['jobs']),
                          'remaining_inferences':0,'results_semantically_parsed':False},
            'sensitivity_scenarios':scenarios,'evaluation_window_seconds_after_training':12392,
            'limitations':['Three measured initial wall observations replace all model-parameter scaling assumptions',
                           'Adapted checkpoint loading, baseline transitions and different caching/contention are not yet measured',
                           'SHA byte scaling and future checkpoint sizes are proxies; model CPU deserialization cost is a separate allowance',
                           'Initial observations on available GPUs do not estimate an adapted runtime distribution',
                           'No wall component is labeled pure GPU time; nested result/teacher timers are not added',
                           'Four-slot arithmetic excludes ongoing training, retries, packaging and unavailable slots'],
            'test_or_behavior_scores_parsed':False,'new_inference_started':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--first',type=Path,default=Path('results/first_behavior_timing_monitor.json'));p.add_argument('--aux',type=Path,default=Path('results/auxiliary_evaluation_timing_completed_v1.json'));p.add_argument('--nll',type=Path,default=Path('results/nll_scope_budget_v1.json'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new immutable budget snapshot')
    value=update(a.first,a.aux,a.nll);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'source_initial_seconds':{m:v['scorer_subprocess_wall_seconds'] for m,v in value['measured_initials'].items()},'combined_ideal4_scenario_seconds':[v['combined_ideal_four_uncontended_slots_seconds_proxy'] for v in value['sensitivity_scenarios']],'scores_parsed':False}))


if __name__=='__main__':main()
