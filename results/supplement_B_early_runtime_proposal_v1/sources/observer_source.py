#!/usr/bin/env python3
"""CPU-only runtime observer; reads timing/exposure, never test or behavior scores."""
import argparse,collections,datetime as dt,hashlib,json,math,os,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_formal_runs import sha256
from estimate_supplemental_budget import estimate
from make_experiments import atomic_json


def read(path):return json.loads(Path(path).read_text())


def event_snapshot(path):
    path=Path(path)
    if not path.exists():return [],None
    raw=path.read_bytes();lines=raw.splitlines()
    if raw and not raw.endswith(b'\n'):lines=lines[:-1]
    return [json.loads(line) for line in lines if line],hashlib.sha256(raw).hexdigest()


def steady_sample(events,minimum_steps=100):
    steps=[event for event in events if event['event']=='train_step']
    if len(steps)<minimum_steps:return None
    sampled=steps[10:];tokens=sum(e['step_counts']['target_prediction_tokens'] for e in sampled)
    seconds=sum(e['step_seconds'] for e in sampled)
    return {'steps_observed':len(steps),'warmup_steps_excluded':10,'sampled_target_prediction_tokens':tokens,
        'sampled_training_seconds':seconds,'seconds_per_target_prediction_token':seconds/tokens}


def remaining_formal(config,events,status,seconds_per_token,validation,best_save,latest_save,setup):
    if status=='completed':return {'total_seconds':0.,'remaining_target_prediction_tokens':0}
    steps=[event for event in events if event['event']=='train_step']
    done=steps[-1]['counters']['target_prediction_tokens'] if steps else 0
    remaining=max(0,config['target_prediction_tokens']-done);final_step=math.ceil(config['target_prediction_tokens']/32704)
    points=list(range(config['eval_every_steps'],final_step+1,config['eval_every_steps']))
    if not points or points[-1]!=final_step:points.append(final_step)
    paid={event['step'] for event in events if event['event']=='validation'};unpaid=sum(step not in paid for step in points)
    latest_done=any(e['event']=='checkpoint' and e['step']==final_step for e in events)
    costs={'training_seconds':remaining*seconds_per_token,'validation_seconds':unpaid*validation,
        'best_model_save_seconds_bound':unpaid*best_save,'final_checkpoint_seconds':0. if latest_done else latest_save,
        'setup_seconds':setup if not events else 0.}
    return {'total_seconds':sum(costs.values()),'remaining_target_prediction_tokens':remaining,'remaining_validation_points':unpaid,'costs':costs}


def extended_runtime(baseline,state,phases):
    rows=[];known=set();traces={};samples=collections.defaultdict(list);complete=collections.defaultdict(list)
    for name,phase in phases.items():
        if not name.startswith('extended_v2_') or not name.endswith(('_formal','_lr','_lambda')):continue
        known.update(phase['job_ids'])
        if name.endswith('_formal'):rows.extend(read(phase['registry']))
    for row in rows:
        path=Path(row['output_dir'])/'events.jsonl';events,digest=event_snapshot(path);traces[row['id']]=events
        sample=steady_sample(events)
        if sample:
            settings=baseline['models'][row['model']];predicted=settings['training_time_factors'][row['group']]/settings['steady_target_prediction_tokens_per_second']
            samples[row['model']].append({'id':row['id'],'group':row['group'],'sample':sample,
                'observed_to_probe_seconds_ratio':sample['seconds_per_target_prediction_token']/predicted,
                'event_path':str(path),'trace_sha256':digest})
        if state['jobs'].get(row['id'],{}).get('status')=='completed':complete[row['model']].append((read(row['summary_path']),events))
    result={}
    for model,settings in baseline['models'].items():
        measured=samples[model];ratio=statistics.median(s['observed_to_probe_seconds_ratio'] for s in measured) if measured else 1.
        val=settings['validation_seconds_per_point'];best=settings['model_only_checkpoint_seconds_estimate']
        latest=settings['latest_checkpoint_seconds'];setup=settings['setup_seconds']
        if complete[model]:
            val=statistics.mean(s['counters']['validation_seconds']/len(s['validation_trajectory']) for s,_ in complete[model])
            latest=statistics.mean(s['counters']['elapsed_seconds']-s['counters']['prior_elapsed_seconds'] for s,_ in complete[model])
            best=statistics.mean((s['counters']['checkpoint_seconds']-(s['counters']['elapsed_seconds']-s['counters']['prior_elapsed_seconds']))/
                max(1,sum(e['event']=='best_target_validation' for e in events)) for s,events in complete[model])
            setup=statistics.mean(s['counters']['setup_seconds'] for s,_ in complete[model])
        jobs=[]
        for row in [r for r in rows if r['model']==model]:
            rate=settings['training_time_factors'][row['group']]/settings['steady_target_prediction_tokens_per_second']*ratio
            own=[s['sample']['seconds_per_target_prediction_token'] for s in measured if s['group']==row['group']]
            if own:rate=statistics.median(own)
            status=state['jobs'].get(row['id'],{}).get('status','pending')
            jobs.append({'id':row['id'],'scheduler_status':status,**remaining_formal(read(row['config_path']),traces[row['id']],status,rate,val,best,latest,setup)})
        result[model]={'reliable_long_run_samples':measured,'minimum_optimizer_steps':100,'warmup_steps_excluded':10,
            'median_observed_to_probe_seconds_ratio':ratio,'complete_long_runs':len(complete[model]),
            'validation_seconds_per_point':val,'model_only_best_save_seconds':best,'latest_checkpoint_seconds':latest,
            'remaining_seconds':sum(j['total_seconds'] for j in jobs),'jobs':jobs}
    return result,known


def observe():
    decision=read(ROOT/'docs/extended_2p8b_horizon_selection.json')
    baseline=estimate('2026-11-07T00:00:00Z',.1,decision['selected_target_prediction_tokens'] or 50_000_000,
                      decision['selected_target_prediction_tokens'] is not None)
    state=read(ROOT/'logs/supplemental_job_state.json');queue=read(ROOT/'configs/supplemental_v1_queue.json')
    phases=read(ROOT/'configs/supplemental_v1_registry_index.json')['phases'];updates,known=extended_runtime(baseline,state,phases)
    for phase in ['batch_benchmark','A_lambda','B_conditional_bank_variance']:
        if phase in phases:known.update(phases[phase]['job_ids'])
    matrix=[]
    for phase in baseline['requested_matrix']:
        record=dict(phase)
        if record['phase'].endswith('_formal'):record['remaining_gpu_seconds']=updates[record['phase'].removesuffix('_formal')]['remaining_seconds']
        matrix.append(record)
    lookup={j['id']:j for j in queue};nll_queue=read(ROOT/'configs/primary_evaluation_v2_execution_matched_nll_queue.json')
    manifest_path=ROOT/'docs/primary_evaluation_v2_execution_matched_nll_manifest.json';manifest=read(manifest_path)
    if sha256(manifest_path)!=read(ROOT/'docs/primary_nll_enqueue_confirmation.json')['manifest_sha256']:raise ValueError('Queued NLL manifest changed')
    proxy=read(ROOT/'results/primary_early_evaluation_budget.json')['nll_elapsed_proxy_seconds_by_kind']
    observed=collections.defaultdict(list);pending=[]
    for job in nll_queue:
        if lookup.get(job['id'])!=job:raise ValueError('Master primary NLL definition differs from its frozen source')
        known.add(job['id']);entry=next(r for r in manifest['jobs'] if r['id']==job['id'])
        kind='initial' if entry['kind']=='initial' else max({r['kind'] for r in manifest['records'] if r['inference_job_id']==job['id']},key=lambda k:proxy[k])
        status=state['jobs'].get(job['id'],{})
        if status.get('status')=='completed' and status.get('returncode')==0:observed[kind].append(status['elapsed_seconds'])
        else:pending.append((job,kind))
    rates={k:statistics.median(observed[k]) if len(observed[k])>=3 else max([v,*observed[k]]) for k,v in proxy.items()}
    nll_seconds=sum(rates[kind] for _,kind in pending)
    matrix.append({'phase':'queued_primary_nll','remaining_runs':len(pending),'remaining_gpu_seconds':nll_seconds,
        'per_kind_whole_job_seconds':rates,'completed_job_elapsed_samples':dict(observed),'score_fields_read':False})
    others=[j for j in queue if j['id'] not in known and state['jobs'].get(j['id'],{}).get('status')!='completed']
    matrix.append({'phase':'other_queued_jobs','remaining_runs':len(others),'job_ids':[j['id'] for j in others],
        'remaining_gpu_seconds':sum(float(j.get('expected_seconds',0)) for j in others),'basis':'Frozen expected_seconds allowance, not a measured cost'})
    seconds=sum(r['remaining_gpu_seconds'] for r in matrix);now=dt.datetime.now(dt.timezone.utc)
    remaining=(dt.datetime(2026,11,7,tzinfo=dt.timezone.utc)-now).total_seconds();cost=seconds*1.1/4
    core=read(ROOT/'artifacts/provenance/formal_deterministic_repository_freeze.json')
    matches={path:sha256(ROOT/path)==h['current'] for path,h in core['core_sources'].items()}
    failures=[name for name,r in state['jobs'].items() if r['status'] in {'failed','blocked'}]
    return {'updated_at_utc':now.isoformat(),'status':'frozen_core_mismatch' if not all(matches.values()) else ('failed_job_detected' if failures else 'monitoring'),
        'pid':os.getpid(),'master_queue_jobs':len(queue),'master_status_counts':dict(collections.Counter(r['status'] for r in state['jobs'].values())),
        'failed_or_blocked_jobs':failures,'frozen_scientific_core_matches':matches,'requested_matrix':matrix,'extended_model_runtime_updates':updates,
        'remaining_slot_hours':seconds/3600,'ideal_four_gpu_remaining_wall_hours':seconds/4/3600,
        'four_gpu_remaining_wall_hours_with_10pct':cost/3600,'remaining_wall_hours_until_midnight':remaining/3600,
        'margin_until_midnight_minutes':(remaining-cost)/60,'margin_before_master_2350_stop_minutes':(remaining-600-cost)/60,
        'fits_before_master_stop_with_10pct':cost<=remaining-600,'unfinished_primary_nll_jobs':len(pending),
        'remaining_primary_nll_slot_seconds':nll_seconds,
        'limitations':['Slot division is a runtime forecast, not a completion guarantee.',
            'All future validation points conservatively pay a best-model save.',
            'Running A/B and NLL receive no progress credit; C observed training/validation exposure is subtracted.',
            'C requires100 steps, excludes first10; own-method rate first, otherwise model median normalized by fixed pilot method factor.',
            'Only timing forecasts change; all grids/batches/horizons/selection rules stay frozen.',
            'Teacher and KL timers are nested and never added twice.'],
        'scores_or_predictions_read':False,'new_gpu_work_launched':False,'scientific_conditions_changed':False,'observer_source_sha256':sha256(__file__)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'results/runtime_budget_latest.json')
    p.add_argument('--watch',action='store_true');p.add_argument('--poll-seconds',type=float,default=60.);args=p.parse_args()
    if not 1<=args.poll_seconds<=60:p.error('Poll at most60 seconds')
    while True:
        report=observe();atomic_json(args.output,report)
        print(json.dumps({k:report[k] for k in ['updated_at_utc','status','master_status_counts','unfinished_primary_nll_jobs','four_gpu_remaining_wall_hours_with_10pct','margin_before_master_2350_stop_minutes']}),flush=True)
        if not args.watch or report['status']!='monitoring' or dt.datetime.now(dt.timezone.utc)>=dt.datetime(2026,11,7,tzinfo=dt.timezone.utc):break
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
