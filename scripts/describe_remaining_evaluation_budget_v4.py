#!/usr/bin/env python3
"""Immutable, score-blind A completion receipt and remaining evaluation v4."""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from describe_remaining_evaluation_budget import dispatch,quantiles,utc
from reconcile_A_nll_runtime import reconcile
from stat_bound_digest_cache import StatBoundHashes,atomic,signature,small_proof


def read(path):return json.loads(Path(path).read_bytes())


def bind(proof):
    if small_proof(proof['path'])!=proof:raise ValueError('Frozen metadata SHA mismatch')
    return read(proof['path'])


def cache_inventory(catalog,progress):
    """Stat-check historical full-file receipts without rereading model bytes."""
    h=StatBoundHashes(snapshot=progress['cache_snapshot']['path'],snapshot_sha256=progress['cache_snapshot']['sha256'])
    registered={str(Path(r['output_dir']).resolve()/'checkpoints'/name)
                for r in catalog['prewarm_runs'] for name in ('latest.pt','checkpoint_best.pt')}
    found=[]
    for key in h.entries:
        if '/checkpoints/' not in key or not key.endswith('.pt'):continue
        if key not in registered:raise ValueError('Cached checkpoint is outside the frozen prewarm cohort')
        receipt=h._receipt(key)
        if receipt['source_authority'].get('kind')!='completed_training_checkpoint':raise ValueError('Missing completed-run full-read authority')
        if signature(Path(key).stat())!=receipt['stat_before']:raise ValueError('Completed checkpoint stat changed since prewarm')
        h.verify(key,receipt['sha256'])
        found.append({'path':key,'receipt':h.entries[key],'bytes':receipt['read_bytes'],'full_read_seconds':receipt['full_read_seconds']})
    if h.counters['full_reads'] or h.counters['full_read_bytes']:raise ValueError('Inventory must not reread checkpoint bytes')
    return {'registered_actual_training_runs':len(catalog['prewarm_runs']),'registered_final_and_best_checkpoint_paths':len(registered),
        'completed_checkpoint_full_read_receipts_stat_verified':len(found),'not_yet_cached_registered_paths':len(registered)-len(found),
        'historical_checkpoint_bytes_fully_read':sum(r['bytes'] for r in found),
        'historical_checkpoint_full_read_seconds_sum':sum(r['full_read_seconds'] for r in found),
        'this_inventory_checkpoint_full_byte_reads':0,'this_inventory_stat_bound_cache_hits':h.counters['cache_hits'],
        'cache_snapshot':progress['cache_snapshot'],'preparation_progress_statistics':progress['counters_this_process'],
        'future_checkpoint_coverage_is_registered_not_already_read':True,'checkpoint_receipts':found,
        'handoff_condition':'Root must explicitly bind the latest immutable digest_cache snapshot path+SHA in the finalized handoff spec; omitted means default fresh full hashing.',
        'prepare_and_start_can_reuse_same_snapshot_after_current_stat_checks':True,
        'worker_always_fresh_full_input_sha_before_each_inference_unchanged':True,
        'two_2p261TB_fresh_handoff_passes_not_assumed_in_this_budget':True,
        'residual_preflight_allowance_seconds_unmeasured':300,
        'residual_risks':['Missing/changed/uncovered files require fresh full verification or rejection, never an unconditional cache hit.',
            'Five-minute preflight allowance is not measured worst-case; last-run readiness, proof production, I/O contention or missed prewarm can exceed it.']}


def A_behavior_scope(manifest,primary,primary_sha):
    if len(manifest['records'])!=18:raise ValueError('Exactly18 complete validation-selected A behavior endpoints required')
    initial=[j for j in manifest['jobs'] if j['kind']=='initial']
    if initial:raise ValueError('The actual selected A plan must reuse the existing initial')
    old={j['id']:j for j in primary['jobs']};adapted=[];aliases=[]
    for job in manifest['jobs']:
        if job['kind']=='adapted':adapted.append(job)
        elif job['kind']=='reuse':
            origin=job['inference_origin'];canonical=old.get(origin['job_id'])
            if (canonical is None or canonical['kind']!='adapted' or origin['manifest_sha256']!=primary_sha or
                    origin['output']!=canonical['output'] or canonical['baseline_group']!=job['baseline_group'] or
                    canonical['weights_identity']!=job['weights_identity']):raise ValueError('A behavior alias differs from its frozen canonical main inference')
            aliases.append(job)
        else:raise ValueError('Unexpected A behavior job kind')
    if len(adapted)+len(aliases)!=18:raise ValueError('Selected behavior queue is not complete')
    return adapted,aliases


def future_cases(v3,completion,A_behavior,primary_behavior,primary_sha):
    primary_jobs=[j for j in primary_behavior['jobs'] if j['kind']=='adapted']
    if len(primary_jobs)!=66:raise ValueError('The full66 main behavior jobs remain in scope')
    new_behavior,aliases=A_behavior_scope(A_behavior,primary_behavior,primary_sha)
    A_times=completion['whole_slot_statistics']['adapted'];alias_times=completion['whole_slot_statistics']['reuse']
    cases=[]
    for old in v3['sensitivity_scenarios']:
        name=old['name'];factor=old['whole_wall_sensitivity_factor'];stat={'low':'min','central':'median','high':'max'}[name]
        measured_final=A_times[stat]*factor;alias_wall=max(old['A_alias_materialization_wall_allowance_per_job_seconds'],alias_times[stat])
        jobs=[];per_phase=[]
        def append_rows(name,count,seconds,model=None,initial=False,dependency=None,actual=None,forward=True):
            per_phase.append({'id':name,'remaining_forward_jobs':count if forward else 0,'materialization_jobs':0 if forward else count,
                'per_job_whole_slot_seconds_proxy':seconds,'sum_slot_seconds_proxy':count*seconds,'model':model})
            for i in range(count):
                record=actual[i] if actual else None;job_id=record['id'] if record else name if initial else name+'_'+str(i)
                depends=([record['inference_origin']['job_id']] if not forward and record else dependency) or []
                jobs.append({'id':job_id,'seconds':seconds,'initial':initial,'depends_on':depends,'forward':forward})
        for row in old['per_phase']:
            name_id=row['id'];count=row['remaining_forward_jobs'];seconds=row['per_job_whole_slot_seconds_proxy']
            if name_id in ('A_nll','A_selected_behavior_max'):continue
            if name_id=='B_nll':seconds=measured_final
            if name_id=='C_pythia_410m_nll_final':
                seconds=measured_final*v3['timing_transfer']['NLL']['pythia-410m:extended_v1']['whole_wall_transfer_ratio_assumption']
            initial=name_id.endswith('_nll_initial')
            dependency=[name_id.rsplit('_',1)[0]+'_initial'] if name_id.startswith('C_') and '_nll_' in name_id and not initial else []
            append_rows(name_id,count,seconds,initial=initial,dependency=dependency,
                actual=primary_jobs if name_id=='main_behavior_adapted' else None)
        behavior_seconds=next(r['per_job_whole_slot_seconds_proxy'] for r in old['per_phase'] if r['id']=='A_selected_behavior_max')
        append_rows('A_selected_behavior',len(new_behavior),behavior_seconds,actual=new_behavior)
        append_rows('A_selected_behavior_materialization',len(aliases),alias_wall,actual=aliases,forward=False)
        full=dispatch(jobs)
        if full['job_count']!=336:raise ValueError('Only the completed126 A-NLL queue jobs may leave the old462-job future envelope')
        duration=full['baseline_first_LPT_wall_seconds_proxy']+300.
        finish_limit=utc('2026-11-07T03:26:32Z')-dt.timedelta(seconds=1800)
        starts={}
        for label,start in [('from_2350',utc('2026-11-06T23:50:00Z')),('from_midnight',utc('2026-11-07T00:00:00Z'))]:
            finish=start+dt.timedelta(seconds=duration)
            starts[label]={'start_utc':start.isoformat(),'finish_utc':finish.isoformat(),'margin_before_30min_delivery_tail_minutes':(finish_limit-finish).total_seconds()/60}
        cases.append({'name':name,'whole_wall_sensitivity_factor':factor,'A_observed_410m_final_reference_seconds_before_factor':A_times[stat],
            'future_410m_NLL_final_whole_slot_seconds_proxy':measured_final,'alias_materialization_whole_slot_seconds_proxy':alias_wall,
            'remaining_forward_jobs':sum(j['forward'] for j in jobs),'remaining_materialization_jobs':sum(not j['forward'] for j in jobs),
            'per_phase':per_phase,'four_slot_dispatch':full,'handoff_preflight_allowance_seconds_unmeasured':300.,
            'wall_seconds_with_dispatch_tail_and_handoff_proxy':duration,'wall_minutes_with_dispatch_tail_and_handoff_proxy':duration/60,
            'starts':starts,'latest_feasible_start_utc_proxy':(finish_limit-dt.timedelta(seconds=duration)).isoformat(),
            'C_future_D_B4_equality_discount_spent':False,'already_completed_A_actual_slot_seconds_separate':completion['actual_whole_slot_seconds_sum'],
            'counterfactual_slot_work_if_A_not_done_early':full['sum_whole_job_slot_seconds_proxy']+completion['actual_whole_slot_seconds_sum'],
            'total_compute_reduction_due_to_early_A_seconds':0.})
    return cases


def produce(output,sources):
    root=ROOT
    if output.exists() or sources.exists():raise FileExistsError('Choose new immutable budget and source namespaces')
    sources.mkdir(parents=True)
    capture={}
    for name,path in {'scheduler_state.json':root/'logs/supplemental_job_state.json','master_queue.json':root/'configs/supplemental_v1_queue.json',
        'raw_runtime.json':root/'results/runtime_budget_latest.json','prehash_progress.json':root/'artifacts/evaluation_ready_v5/cache_progress.json',
        'ready_index.json':root/'artifacts/evaluation_ready_v5/ready_index.json'}.items():
        dest=sources/name;dest.write_bytes(path.read_bytes());capture[name]=dest
    advice=root/'results/early_A_nll_timing_v1/advice.json';advice_sha='56b7b93b934b0ba8f89d28e03d373b69bf6c82cff805fa0a2160e3e50778d587'
    reconciled=reconcile(advice,advice_sha,capture['raw_runtime.json'],capture['scheduler_state.json'],capture['master_queue.json'])
    if reconciled['A_jobs_present_in_master']!=126 or reconciled['A_jobs_remaining']!=0 or reconciled['A_status_counts_by_kind']['completed']!={'adapted':108,'reuse':18}:
        raise ValueError('All126 successful A eval jobs required before this budget revision')
    state=read(capture['scheduler_state.json']);jobs=[]
    for sample in reconciled['matched_completed_A_job_timing_samples']:
        actual=state['jobs'][sample['id']]
        jobs.append({**sample,'status':actual['status'],'returncode':actual['returncode'],
            'started_at_utc':actual['started_at_utc'],'ended_at_utc':actual['ended_at_utc']})
    first=min(utc(j['started_at_utc']) for j in jobs);last=max(utc(j['ended_at_utc']) for j in jobs)
    completion={'schema_version':1,'status':'passed','utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'manifest':reconciled['manifest'],'queue':reconciled['queue'],'advice':small_proof(advice),'expected_jobs':126,'completed_jobs':126,
        'new_forward_jobs_completed':108,'materialization_only_jobs_completed':18,'new_initial_jobs':0,
        'earlier_primary_canonical_forwards_and_baseline_deducted_again':False,
        'actual_whole_slot_seconds_sum':sum(j['whole_slot_seconds'] for j in jobs),
        'whole_slot_statistics':{k:quantiles([j['whole_slot_seconds'] for j in jobs if j['kind']==k]) for k in ('adapted','reuse')},
        'first_started_at_utc':first.isoformat(),'last_ended_at_utc':last.isoformat(),
        'mixed_master_first_start_to_last_exit_seconds':(last-first).total_seconds(),'phase_had_exclusive_four_GPUs_claimed':False,
        'scheduler_definition_and_output_integrity_verified':True,'source_snapshots':{n:small_proof(p) for n,p in capture.items()},'jobs':jobs,
        'nested_teacher_scorer_verification_timers_added_to_slots':False,'scores_or_predictions_deserialized':False,
        'same_work_moved_earlier_not_total_compute_saved':True,'producer_source':small_proof(__file__)}
    completion_path=sources/'A_runtime_completion.json';atomic(completion_path,completion)
    v3_path=root/'results/evaluation_timing_budget_v3.json';v3=read(v3_path)
    index=read(capture['ready_index.json']);A_ready=next(p for p in index['phases'] if p['id']=='supplement_A_behavior')
    if A_ready.get('ready') is not True:raise ValueError('Actual frozen selected A behavior workload required')
    A_behavior=bind(A_ready['manifest']);primary_path=root/'docs/primary_evaluation_v2_execution_matched_behavior_manifest.json';primary_sha=small_proof(primary_path)['sha256'];primary=read(primary_path)
    for job in primary['jobs']:
        if job['kind']=='adapted' and state['jobs'].get(job['id'],{}).get('status')=='completed':raise ValueError('Main behavior already ran; use a new actual-completion accounting revision')
    catalog_proof=read(root/'docs/supplemental_evaluation_preparation_v5_execution.json')['catalog'];catalog=bind(catalog_proof)
    inventory=cache_inventory(catalog,read(capture['prehash_progress.json']));inventory_path=sources/'prehash_inventory.json';atomic(inventory_path,inventory)
    cases=future_cases(v3,completion,A_behavior,primary,primary_sha);runtime=read(capture['raw_runtime.json'])
    forecast_at=utc(reconciled['utc'])
    remaining_training=(runtime['remaining_slot_hours']*3600-reconciled['raw_generic_A_remaining_slot_seconds_removed'])*1.1/4
    training_eta=forecast_at+dt.timedelta(seconds=remaining_training)
    finish_limit=utc('2026-11-07T02:56:32Z')
    for case in cases:
        finish=training_eta+dt.timedelta(seconds=case['wall_seconds_with_dispatch_tail_and_handoff_proxy'])
        case['from_current_training_forecast']={'training_finish_utc':training_eta.isoformat(),'evaluation_finish_utc':finish.isoformat(),
            'margin_before_30min_delivery_tail_minutes':(finish_limit-finish).total_seconds()/60}
    result={'schema_version':4,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'score_blind_remaining_evaluation_forecast',
        'prior_v3':small_proof(v3_path),'A_completion':small_proof(completion_path),'A_selected_behavior_ready':A_ready['manifest'],
        'prehash_inventory':small_proof(inventory_path),'frozen_catalog':catalog_proof,
        'accounting':{'v3_remaining_NLL_forward_jobs':297,'actual_new_A_NLL_forward_jobs_removed':108,
            'remaining_NLL_forward_jobs':189,'v3_A_NLL_materialization_jobs':18,'actual_A_NLL_materialization_jobs_removed':18,
            'earlier_primary18_forwards_removed_again':0,'earlier_primary_NLL_initial_removed_again':0,
            'v3_remaining_behavior_forward_upper_count':147,'actual_frozen_A_behavior_forward_count':15,
            'actual_frozen_A_behavior_reuse_count':3,'remaining_behavior_forward_jobs':144,'remaining_forward_jobs':333,
            'remaining_materialization_jobs':3,'remaining_queue_jobs':336,'completed_A_slot_seconds_separate':completion['actual_whole_slot_seconds_sum'],
            'total_compute_reduction_due_to_moving_A_earlier':0.,'future_unproved_C_avoided_forwards_not_deducted':{'NLL_max':12,'behavior_max':6}},
        'sensitivity_scenarios':cases,'training_forecast':{'updated_at_utc':runtime['updated_at_utc'],'finish_utc_with_existing_10pct':training_eta.isoformat(),
            'raw_generic_A_allowance_removed_once_slot_seconds':reconciled['raw_generic_A_remaining_slot_seconds_removed'],
            'master_jobs':runtime['master_queue_jobs'],'master_status_counts':runtime['master_status_counts'],
            'raw_observer_A_jobs_still_in_other_queued_jobs':[n for n in next(p for p in runtime['requested_matrix'] if p['phase']=='other_queued_jobs')['job_ids'] if n in {j['id'] for j in jobs}]},
        'time_window':{'master_2350_stop_utc':'2026-11-06T23:50:00Z','midnight_utc':'2026-11-07T00:00:00Z','hard_deadline_utc':'2026-11-07T03:26:32Z','delivery_tail_seconds':1800,'latest_inference_finish_utc':finish_limit.isoformat()},
        'source_proofs':{'primary_behavior_manifest':small_proof(primary_path),'producer_source':small_proof(__file__),**{n:small_proof(p) for n,p in capture.items()}},
        'limitations':['The108 A NLL forwards and18 materializations are actual same-definition successful work; the18 earlier canonical main forwards and all completed baselines were already absent from v3 remaining work.',
            'A NLL whole-slot min/median/max improve only410M final-NLL proxies; selected-best/initial costs and behavior rates are separate, while1B/2.8B v3 extrapolations remain unchanged.',
            'Three selected A behavior aliases are proved in the frozen READY manifest and still depend on their remaining main behavior canonical forwards; their materialization is charged, not assumed free.',
            'No future C D/B4 equality discount is spent before complete state proofs/manifests exist; any aliases still require fresh input SHA and materialization overhead.',
            'Factors1/1.25/1.5 and alias floors10/20/40s are sensitivity assumptions; extrapolation, shared CPU/I/O and real scheduling may exceed them.',
            'All four-slot evaluation schedules require the old master and children fully exited and all required source/proof gates satisfied; no second scheduler is started.',
            'Five-minute handoff/preflight allowance assumes explicit cache snapshot binding and stable covered inputs; it is unmeasured and excludes the reserved30-minute delivery tail.'],
        'no_score_fields_read':True,'no_science_or_queue_changes':True,'no_new_GPU_work_started':True}
    atomic(output,result)
    import csv
    with (sources/'scenarios.csv').open('w',newline='') as stream:
        fields=['scenario','evaluation_minutes','from_2350_margin_minutes','from_midnight_margin_minutes','latest_start_utc','from_training_forecast_margin_minutes']
        writer=csv.DictWriter(stream,fields);writer.writeheader()
        for c in cases:writer.writerow({'scenario':c['name'],'evaluation_minutes':c['wall_minutes_with_dispatch_tail_and_handoff_proxy'],
            'from_2350_margin_minutes':c['starts']['from_2350']['margin_before_30min_delivery_tail_minutes'],
            'from_midnight_margin_minutes':c['starts']['from_midnight']['margin_before_30min_delivery_tail_minutes'],
            'latest_start_utc':c['latest_feasible_start_utc_proxy'],'from_training_forecast_margin_minutes':c['from_current_training_forecast']['margin_before_30min_delivery_tail_minutes']})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);a=p.parse_args()
    value=produce(a.output,a.source_root)
    print(json.dumps({'output':small_proof(a.output),'completion':value['A_completion'],'remaining':value['accounting'],
        'scenario_minutes':[c['wall_minutes_with_dispatch_tail_and_handoff_proxy'] for c in value['sensitivity_scenarios']]}))


if __name__=='__main__':main()
