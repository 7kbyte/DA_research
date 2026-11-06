#!/usr/bin/env python3
"""Replace only manifest-bound A-NLL generic allowances in an independent forecast."""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from estimate_early_A_nll import bound,count_jobs,fifo_schedule,read
from handoff_evaluation import job_definition,completion_proof
from schedule_jobs import digest_job
from stat_bound_digest_cache import atomic,small_proof

EXPECTED_RAW_OBSERVER_SOURCE_SHA256='aaf4c774b47d00bef8794ac4151f1194ad04c6d6d6dce830ceceb337f331d0f4'


def numbers(raw,queue,entries,present,remaining,measured,completed,reference):
    """Keep raw non-A work and its10% margin; reprice only actual A IDs."""
    lookup={j['id']:j for j in queue};other=next(p for p in raw['requested_matrix'] if p['phase']=='other_queued_jobs')
    charged=set(other['job_ids'])&set(lookup)
    allowance=sum(lookup[n]['expected_seconds'] for n in charged)
    if allowance>other['remaining_gpu_seconds']+.01:raise ValueError('Raw other-queued subtotal is inconsistent with A allowance')
    non_A=raw['remaining_slot_hours']*3600-allowance
    if non_A<0:raise ValueError('Negative remaining non-A work')
    forward=[r['whole_slot_seconds'] for r in measured if r['kind']=='adapted']
    reuse=[r['whole_slot_seconds'] for r in measured if r['kind']=='reuse']
    forward_base=statistics.median(forward) if len(forward)>=3 else max([reference,*forward])
    reuse_base=statistics.median(reuse) if len(reuse)>=3 else max(reuse,default=0.)
    cases=[]
    for name,factor,alias in [('low',1.,10.),('central',1.25,20.),('high',1.5,40.)]:
        rates={'adapted':forward_base*factor,'reuse':max(alias,reuse_base)}
        if any(entries[j['id']]['kind'] not in rates for j in remaining):raise ValueError('A only supports frozen adapted/reuse jobs')
        durations={j['id']:rates[entries[j['id']]['kind']] for j in remaining}
        schedule=fifo_schedule(remaining,durations,completed);A=sum(durations.values())
        ideal=(non_A+A)*1.1/4;tail=schedule['isolated_phase_dispatch_tail_seconds'];cost=ideal+tail
        # A progress may be newer than the raw non-A estimate. Start at now,
        # retaining stale non-A work rather than granting unmeasured progress.
        eta=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=cost)
        stop=dt.datetime(2026,11,6,23,50,tzinfo=dt.timezone.utc)
        cases.append({'name':name,'per_remaining_forward_whole_slot_seconds':rates['adapted'],
            'per_remaining_reuse_whole_slot_seconds':rates['reuse'],'remaining_A_slot_seconds':A,
            'remaining_A_ideal_four_slot_minutes':A/4/60,'A_isolated_dispatch_tail_seconds':tail,
            'all_queued_remaining_four_gpu_hours_with_10pct_plus_A_tail':cost/3600,
            'forecast_finish_utc':eta.isoformat(),'margin_before_master_2350_stop_minutes':(stop-eta).total_seconds()/60})
    return {'A_ids_charged_in_raw_other_queued_jobs':sorted(charged),'raw_generic_A_remaining_slot_seconds_removed':allowance,
            'raw_generic_A_four_gpu_minutes_with_10pct_removed':allowance*1.1/4/60,
            'remaining_non_A_slot_seconds_kept_from_raw_forecast':non_A,
            'matched_completed_timing_sample_counts':{'adapted':len(forward),'reuse':len(reuse)},
            'forward_reference_seconds':forward_base,'reference_rule':'At least3 complete matched samples: median; otherwise max(frozen main final median, observations). Alias scenarios never fall below10/20/40s.',
            'sensitivity_scenarios':cases,'A_jobs_present_in_master':len(present),'A_jobs_remaining':len(remaining),
            'A_completed_whole_slot_seconds_sum':sum(r['whole_slot_seconds'] for r in measured),
            'completed_and_unqueued_A_jobs_charged_to_remaining':False,'running_jobs_receive_partial_progress_credit':False}


def reconcile(advice_path,advice_sha,raw_path,state_path,master_path):
    advice=bound({'path':str(Path(advice_path).resolve()),'sha256':advice_sha})
    queue=bound(advice['queue']);manifest=bound(advice['manifest']);cat=bound(advice['catalog'])
    plan=next(p for p in cat['plans'] if p['id']=='supplement_A_nll')
    if advice['new_kind_counts']!={'adapted':108,'reuse':18} or advice['new_initials']!=0:raise ValueError('Expected the complete frozen A126 phase')
    authorities={(str(Path(advice['manifest']['path']).resolve()),advice['manifest']['sha256']):manifest}
    for proof in plan['prior_plans']:authorities[(str(Path(proof['path']).resolve()),proof['sha256'])]=bound(proof)
    captured={k:Path(p).read_bytes() for k,p in {'raw_runtime':raw_path,'current_state':state_path,'master_queue':master_path}.items()}
    raw=json.loads(captured['raw_runtime']);state=json.loads(captured['current_state']);master=json.loads(captured['master_queue'])
    if raw.get('observer_source_sha256')!=EXPECTED_RAW_OBSERVER_SOURCE_SHA256:
        raise ValueError('Raw observer policy changed; review A recognition before avoiding double charging')
    if raw['status']!='monitoring' or raw['failed_or_blocked_jobs']:raise ValueError('Independent observer reports a failed job or frozen-source violation')
    age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(raw['updated_at_utc'])).total_seconds()
    if age>120:raise ValueError('The raw runtime observer is stale')
    expected={j['id']:j for j in queue};current={j['id']:j for j in master}
    present=[j for j in queue if j['id'] in current]
    if any(current[j['id']]!=j for j in present):raise ValueError('An existing A job definition differs from the frozen incoming queue')
    # Validate the full source A DAG plus existing non-A priors even after an
    # append. Thus its original initial baseline remains explicit and verified.
    counts=count_jobs(queue,manifest,[j for j in master if j['id'] not in expected],state,authorities)
    if present and len(present)!=126:raise ValueError('Only a complete frozen A126 append is supported')
    entries=counts['entries'];remaining=[];measured=[];completed={r['id'] for r in counts['completed_prior_integrity']}
    for job in present:
        record=state['jobs'].get(job['id'],{})
        if record.get('definition_sha256') not in (None,digest_job(job)):raise ValueError('Actual A scheduler definition differs')
        if record.get('status') in ('failed','blocked') or ('stop_requested_at' in record):raise ValueError('An A evaluation job needs failure/deadline review')
        if record.get('status')=='completed':
            if record.get('returncode')!=0 or record.get('missing_outputs'):raise ValueError('A completion lacks successful output integrity')
            binding=job_definition(job,authorities,{})
            attestations=completion_proof(job,binding)
            elapsed=record['elapsed_seconds']
            if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid completed A whole-slot time')
            measured.append({'id':job['id'],'kind':entries[job['id']]['kind'],'whole_slot_seconds':elapsed,
                'definition_sha256':digest_job(job),'execution_attestation_sha256':attestations})
            completed.add(job['id'])
        else:remaining.append(job)
    result=numbers(raw,queue,entries,present,remaining,measured,completed,advice['timing_reference']['main_final_median_whole_slot_seconds'])
    return {'schema_version':1,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'status':'active_A_runtime_reconciled' if present else 'prepared_only_A_not_in_master',
        'advice':{'path':str(Path(advice_path).resolve()),'sha256':advice_sha},'manifest':advice['manifest'],'queue':advice['queue'],
        'source_proofs':{k:{'path':str(Path(p).resolve()),'sha256_of_exact_bytes_read':hashlib.sha256(captured[k]).hexdigest(),
            'bytes_read':len(captured[k]),'mutable_operational_snapshot_not_a_frozen_source':True}
            for k,p in {'raw_runtime':raw_path,'current_state':state_path,'master_queue':master_path}.items()},
        'source_runtime_updated_at_utc':raw['updated_at_utc'],'source_runtime_age_seconds':age,
        'source_runtime_master_job_count':raw['master_queue_jobs'],'current_master_job_count':len(master),
        'A_status_counts_by_kind':{'completed':dict(collections.Counter(r['kind'] for r in measured)),
            'remaining':dict(collections.Counter(entries[j['id']]['kind'] for j in remaining))},
        'A_measured_whole_slot_seconds_by_kind':{kind:{'count':len(values),'sum':sum(values),'min':min(values),'median':statistics.median(values),'max':max(values)}
            for kind in ('adapted','reuse') if (values:=[r['whole_slot_seconds'] for r in measured if r['kind']==kind])},
        'matched_completed_A_job_timing_samples':measured,**result,
        'limitations':['The raw observer remains unchanged; only the A IDs actually charged under other_queued_jobs are removed.',
            'Other training forecasts remain as of the raw observer timestamp; its age and queue-count lag are explicit.',
            'Forecast ETA starts at this reconciliation time while retaining the older non-A remainder; no unmeasured non-A progress credit is given.',
            'No source job expected_seconds/argv/outputs/definition is changed; the GPU scheduler never consumes this forecast.',
            'Forward factors1/1.25/1.5 and minimum reuse10/20/40s are sensitivity assumptions, not confidence intervals.',
            'Running A jobs retain a full-job charge; only successful definition/output-attested completions have zero remaining cost.'],
        'scores_or_prediction_fields_accessed':False,'output_hashes_streamed_without_score_deserialization':True,
        'new_gpu_work_started':False,'master_queue_or_state_modified':False,'producer_source':small_proof(__file__)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--advice',type=Path,required=True);p.add_argument('--advice-sha256',required=True)
    p.add_argument('--raw-runtime',type=Path,default=ROOT/'results/runtime_budget_latest.json')
    p.add_argument('--state',type=Path,default=ROOT/'logs/supplemental_job_state.json');p.add_argument('--master-queue',type=Path,default=ROOT/'configs/supplemental_v1_queue.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--watch',action='store_true');p.add_argument('--poll-seconds',type=float,default=60.)
    a=p.parse_args()
    if not 5<=a.poll_seconds<=60:p.error('Poll between5 and60seconds')
    while True:
        value=reconcile(a.advice,a.advice_sha256,a.raw_runtime,a.state,a.master_queue);atomic(a.output,value)
        print(json.dumps({k:value[k] for k in ['status','A_jobs_present_in_master','A_jobs_remaining','matched_completed_timing_sample_counts','raw_generic_A_remaining_slot_seconds_removed']}),flush=True)
        if not a.watch or value['A_jobs_present_in_master'] and not value['A_jobs_remaining']:break
        time.sleep(a.poll_seconds)


if __name__=='__main__':main()
