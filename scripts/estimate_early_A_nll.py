#!/usr/bin/env python3
"""Score-blind timing advice for a real READY A-NLL plan; never enqueue jobs."""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import heapq
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from handoff_evaluation import job_definition,completion_proof
from schedule_jobs import digest_job,validate_queue
from stat_bound_digest_cache import atomic,small_proof
from watch_lambda_selection import complete_grid
from describe_remaining_evaluation_budget import dispatch


def read(path):return json.loads(Path(path).read_bytes())


def bound(proof):
    if small_proof(proof['path'])!=proof:raise ValueError('Frozen metadata SHA mismatch: '+str(proof['path']))
    return read(proof['path'])


def fifo_schedule(jobs,durations,completed,slots=4):
    """Simulate the master's ready-first FIFO on otherwise available slots."""
    pending=list(jobs);completed=set(completed);free=list(range(slots));running=[];now=0.;rows=[]
    while pending or running:
        for slot in list(free):
            eligible=next((i for i,j in enumerate(pending) if set(j.get('depends_on',[]))<=completed),None)
            if eligible is None:continue
            job=pending.pop(eligible);end=now+durations[job['id']];free.remove(slot)
            heapq.heappush(running,(end,slot,job['id']));rows.append({'id':job['id'],'slot':slot,'start':now,'end':end})
        if not running:
            if pending:raise ValueError('An incomplete prior baseline/dependency blocks the timing plan')
            break
        now=running[0][0]
        while running and running[0][0]<=now:
            end,slot,name=heapq.heappop(running);completed.add(name);free.append(slot)
        free.sort()
    seconds=sum(durations[j['id']] for j in jobs);span=max((r['end'] for r in rows),default=0.)
    return {'sum_slot_seconds':seconds,'ideal_four_slot_seconds':seconds/slots,
            'ready_first_FIFO_makespan_seconds':span,'isolated_phase_dispatch_tail_seconds':span-seconds/slots,
            'jobs':rows,'assumes_four_available_slots':True}


def count_jobs(queue,manifest,existing,state,authorities):
    """Verify job grammar, immutable overlap and completed prior origin integrity."""
    current={j['id']:j for j in existing};entries={j['id']:j for j in manifest['jobs']}
    if len(entries)!=len(queue) or {j['id'] for j in queue}!=set(entries):raise ValueError('READY queue/manifest job set differs')
    bindings={};input_authorities={};new=[];already=[]
    for job in queue:
        bindings[job['id']]=job_definition(job,authorities,input_authorities)
        if job['id'] in current:
            if current[job['id']]!=job:raise ValueError('Same-ID immutable master definition differs')
            already.append(job)
        else:new.append(job)
    validate_queue([*existing,*new])
    required={d for j in new for d in j.get('depends_on',[]) if d not in entries}
    # Reused forwards must be actual already-completed canonical origin jobs.
    for job in queue:
        entry=entries[job['id']]
        if entry['kind']=='reuse':required.add(entry['inference_origin']['job_id'])
    prior=[]
    for name in sorted(required):
        if name not in current:raise ValueError('A prior dependency is absent from the current master')
        job=current[name];binding=job_definition(job,authorities,input_authorities);entry=binding['entry']
        record=state['jobs'].get(name,{})
        if (record.get('status')!='completed' or record.get('returncode')!=0 or record.get('missing_outputs') or
                'stop_requested_at' in record or record.get('definition_sha256')!=digest_job(job)):
            raise ValueError('A complete definition-matched prior baseline/origin is required')
        if any(not Path(p).is_file() for p in job['outputs']):raise ValueError('Completed prior output is missing')
        # Frozen verify_outputs streams output SHA; it never deserializes scores.
        attestations=completion_proof(job,binding)
        prior.append({'id':name,'kind':entry['kind'],'manifest':binding['manifest_path'],
                      'manifest_sha256':binding['manifest_sha256'],'definition_sha256':digest_job(job),
                      'verified_execution_attestation_sha256':attestations,
                      'status':'completed','returncode':0,'integrity_verified_without_score_parse':True})
    for job in queue:
        entry=entries[job['id']]
        if entry['kind']=='reuse':
            origin=entry['inference_origin'];source=next((r for r in prior if r['id']==origin['job_id']),None)
            canonical=job_definition(current[origin['job_id']],authorities,input_authorities)['entry'] if source else {}
            if (source is None or source['manifest_sha256']!=origin['manifest_sha256'] or canonical.get('output')!=origin['output'] or
                    canonical.get('baseline_group')!=entry.get('baseline_group') or canonical.get('weights_identity')!=entry.get('weights_identity')):
                raise ValueError('Reuse does not bind the same completed canonical prior output')
    return {'new_jobs':new,'already_in_master':already,'entries':entries,'completed_prior_integrity':prior,
            'new_kind_counts':dict(collections.Counter(entries[j['id']]['kind'] for j in new)),
            'required_existing_initial_count':sum(p['kind']=='initial' for p in prior),
            'required_existing_adapted_origin_count':sum(p['kind']=='adapted' for p in prior)}


def future_without_A(case):
    """Keep all conservative non-A forward work; remove only the completed phase."""
    jobs=[]
    for row in case['per_phase']:
        if row['id']=='A_nll':continue
        name=row['id'];is_initial=name.endswith('_nll_initial')
        # Existing v3 rows have one matched extended NLL initial per model.
        baseline=name.rsplit('_',1)[0]+'_initial'
        for i in range(row['remaining_forward_jobs']):
            jobs.append({'id':name if is_initial else name+'_'+str(i),'seconds':row['per_job_whole_slot_seconds_proxy'],
                'initial':is_initial,'depends_on':[baseline] if name.startswith('C_') and '_nll_' in name and not is_initial else []})
    return dispatch(jobs)


def timing_cases(counts,runtime,v3,reference_seconds):
    rows=[];entries=counts['entries'];new=counts['new_jobs'];completed={r['id'] for r in counts['completed_prior_integrity']}
    completed.update(j['id'] for j in counts['already_in_master'] if j['id'] not in {r['id'] for r in new})
    for name,factor,reuse in [('low',1.,10.),('central',1.25,20.),('high',1.5,40.)]:
        durations={j['id']:(reuse if entries[j['id']]['kind']=='reuse' else reference_seconds*factor) for j in new}
        schedule=fifo_schedule(new,durations,completed);case=next(c for c in v3['sensitivity_scenarios'] if c['name']==name)
        increment=schedule['ideal_four_slot_seconds']*1.1+schedule['isolated_phase_dispatch_tail_seconds']
        original_A=sum(r['sum_slot_seconds_proxy'] for r in case['per_phase'] if r['id']=='A_nll')+case['materialization_only_jobs']*case['A_alias_materialization_wall_allowance_per_job_seconds']
        future=future_without_A(case);old_total=case['four_slot_dispatch']['sum_whole_job_slot_seconds_proxy']
        # New A estimates replace the old A proxy before moving the same work.
        rebased=old_total-original_A+schedule['sum_slot_seconds']
        if abs(rebased-(future['sum_whole_job_slot_seconds_proxy']+schedule['sum_slot_seconds']))>.01:
            raise ValueError('Early/final slot accounting does not conserve total work')
        remaining=runtime['four_gpu_remaining_wall_hours_with_10pct']*3600
        at=dt.datetime.fromisoformat(runtime['updated_at_utc']);eta=at+dt.timedelta(seconds=remaining+increment)
        rows.append({'name':name,'forward_whole_slot_seconds_assumption':reference_seconds*factor,'reuse_materialization_seconds_assumption':reuse,
            'whole_slot_factor':factor,'additional_slot_seconds':schedule['sum_slot_seconds'],
            'additional_four_gpu_ideal_minutes':schedule['ideal_four_slot_seconds']/60,
            'isolated_phase_FIFO_makespan_minutes':schedule['ready_first_FIFO_makespan_seconds']/60,
            'isolated_phase_dispatch_tail_seconds':schedule['isolated_phase_dispatch_tail_seconds'],
            'additional_four_gpu_minutes_with_10pct_plus_tail':increment/60,
            'training_forecast_four_gpu_hours_with_existing_10pct':remaining/3600,
            'training_plus_early_A_hours_with_10pct_plus_tail':(remaining+increment)/3600,
            'forecast_finish_utc':eta.isoformat(),'margin_before_master_2350_stop_minutes':runtime['margin_before_master_2350_stop_minutes']-increment/60,
            'fits_before_master_2350_stop':runtime['margin_before_master_2350_stop_minutes']>=increment/60,
            'final_evaluation_slot_seconds_before_A_retimed':rebased,
            'final_evaluation_slot_seconds_if_A_completes_early':future['sum_whole_job_slot_seconds_proxy'],
            'final_evaluation_debit_slot_seconds_if_A_completes_early':schedule['sum_slot_seconds'],
            'total_slot_work_reduction_from_moving_A':0.,'final_evaluation_dispatch_if_A_completes_early':future,
            'final_evaluation_wall_plus_original_300s_handoff_if_A_completes_early_minutes':(future['baseline_first_LPT_wall_seconds_proxy']+case['extra_unmeasured_handoff_preflight_allowance_seconds'])/60,
            'schedule':schedule})
    return rows


def estimate(root):
    root=Path(root);ready_path=root/'artifacts/evaluation_ready_v5/supplement_A_nll/READY.json'
    if not ready_path.is_file():raise FileNotFoundError('Wait for a real v5 A-NLL READY receipt')
    ready=read(ready_path)
    if ready.get('ready') is not True or ready.get('id')!='supplement_A_nll' or ready.get('mode')!='nll':raise ValueError('Actual A-NLL ready authority required')
    catalog_path=root/'configs/supplemental_evaluation_preparation_v5.json';cat=read(catalog_path)
    authority=read(root/'docs/supplemental_evaluation_preparation_v5_execution.json');bound(authority['catalog'])
    index_raw=(root/'artifacts/evaluation_ready_v5/ready_index.json').read_bytes();index=json.loads(index_raw)
    selected=next(p for p in index['phases'] if p['id']=='supplement_A_nll')
    if (index['catalog']!=authority['catalog'] or selected.get('ready') is not True or
            any(selected[k]!=ready[k] for k in ('queue','manifest','phase_spec'))):
        raise FileNotFoundError('Wait for A-NLL READY to be committed in the authoritative v5 index')
    plan=next(p for p in cat['plans'] if p['id']=='supplement_A_nll');registry=bound(plan['phase']['registry'])
    master_path=root/'configs/supplemental_v1_queue.json';state_path=root/'logs/supplemental_job_state.json'
    # Capture mutable metadata once; the output stores these exact bytes separately.
    state_raw=state_path.read_bytes();queue_raw=master_path.read_bytes();runtime_raw=(root/'results/runtime_budget_latest.json').read_bytes()
    state=json.loads(state_raw);existing=json.loads(queue_raw);runtime=json.loads(runtime_raw)
    parent_raw=(root/'logs/formal_deterministic_job_state.json').read_bytes();parent=json.loads(parent_raw)
    if not complete_grid(registry,state,parent)['ready']:raise ValueError('Full A108 physical plus18 parent aliases required')
    selection_progress_path=root/'logs/supplement_A_selection_watch.json';selection=read(selection_progress_path)
    if selection.get('status')!='selection_frozen' or selection.get('gate',{}).get('ready') is not True or len(selection.get('choices',{}))!=6:
        raise FileNotFoundError('Wait for the existing complete validation-only selector receipt')
    for key in ['selection','analysis_manifest']:
        proof={'path':selection['selection_path' if key=='selection' else key],
               'sha256':selection['selection_sha256' if key=='selection' else key+'_sha256']}
        if small_proof(proof['path'])!=proof:raise ValueError('Complete selection proof changed')
    queue=bound(ready['queue']);manifest=bound(ready['manifest']);bound(ready['phase_spec'])
    if manifest.get('mode')!='nll' or len(manifest['records'])!=126:raise ValueError('All126 A final logical endpoints required')
    authorities={(str(Path(ready['manifest']['path']).resolve()),ready['manifest']['sha256']):manifest}
    for proof in plan['prior_plans']:
        prior=bound(proof);authorities[(str(Path(proof['path']).resolve()),proof['sha256'])]=prior
    counts=count_jobs(queue,manifest,existing,state,authorities)
    if counts['already_in_master']:raise ValueError('A jobs are already in the master; this pre-enqueue advice requires a new timing review')
    v3path=root/'results/evaluation_timing_budget_v3.json';v3=read(v3path)
    reference=v3['measured_primary_NLL']['by_endpoint_whole_slot_seconds']['final']['median']
    if abs(reference-40.078)>.001 or runtime['status']!='monitoring' or runtime['failed_or_blocked_jobs']:
        raise ValueError('Expected complete measured reference and healthy runtime forecast required')
    if (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(runtime['updated_at_utc'])).total_seconds()>120:
        raise ValueError('Refresh the independent runtime observer before making timing advice')
    cases=timing_cases(counts,runtime,v3,reference)
    record={'schema_version':1,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'timing_advice_only_no_enqueue',
        'ready':small_proof(ready_path),'catalog':small_proof(catalog_path),'queue':ready['queue'],'manifest':ready['manifest'],
        'complete_training_grid':{'physical':108,'parent_aliases':18,'logical':126,'complete':True},
        'selection_frozen_receipt':small_proof(selection_progress_path),'selection_sha256':selection['selection_sha256'],
        'logical_endpoints':len(manifest['records']),'incoming_queue_jobs':len(queue),'new_jobs':len(counts['new_jobs']),
        'already_in_master':len(counts['already_in_master']),'new_kind_counts':counts['new_kind_counts'],
        'completed_existing_initials':counts['required_existing_initial_count'],
        'completed_existing_adapted_reuse_origins':counts['required_existing_adapted_origin_count'],
        'completed_prior_integrity':counts['completed_prior_integrity'],'new_initials':counts['new_kind_counts'].get('initial',0),
        'runtime_forecast_updated_at_utc':runtime['updated_at_utc'],'runtime_forecast_age_seconds':(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(runtime['updated_at_utc'])).total_seconds(),
        'timing_reference':{'source':small_proof(v3path),'main_final_median_whole_slot_seconds':reference,'whole_slot_nested_components_not_added_again':True},
        'sensitivity_scenarios':cases,'all_scenarios_fit_before_master_stop':all(c['fits_before_master_2350_stop'] for c in cases),
        'limitations':['An early A phase shifts the same measured/proxy slot work out of final evaluation; it does not reduce total inference work.',
            'Future debit is conditional on actual completed integrity-verified jobs; no unfinished inference is marked completed now.',
            'Four-slot FIFO tail assumes slots are available after current nonpreemptive children; full mixed training makespan and shared CPU/I/O contention are not a guarantee.',
            'The current training forecast retains its10% contingency; the additional phase also receives10% plus its isolated FIFO imbalance tail.',
            'Whole-slot40.078s includes scorer/SHA/startup/polling; factors1/1.25/1.5 and alias10/20/40s are sensitivity assumptions.',
            'C long-horizon training and final evaluation transfer remain unmeasured/partly measured; no scientific condition or score affects this advice.'],
        'prior_output_sha_verified_without_score_parse':True,'score_or_prediction_fields_accessed':False,
        'enqueue_authorized_by_this_report':False,'queue_or_state_modified':False,'gpu_work_started':False,
        'producer_source':small_proof(__file__)}
    return record,{'scheduler_state.json':state_raw,'master_queue.json':queue_raw,'runtime_budget.json':runtime_raw,'parent_state.json':parent_raw,
                   'ready_index.json':index_raw,'selection_frozen_receipt.json':selection_progress_path.read_bytes()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    value,sources=estimate(ROOT)
    if a.output_dir.exists():raise FileExistsError('Use a new immutable timing snapshot directory')
    a.output_dir.mkdir(parents=True)
    source_proofs={}
    for name,raw in sources.items():
        path=a.output_dir/name;path.write_bytes(raw);source_proofs[name]=small_proof(path)
    value['captured_source_snapshots']=source_proofs;atomic(a.output_dir/'advice.json',value)
    import csv
    fields=['name','forward_whole_slot_seconds_assumption','reuse_materialization_seconds_assumption','additional_four_gpu_ideal_minutes',
            'isolated_phase_dispatch_tail_seconds','additional_four_gpu_minutes_with_10pct_plus_tail','training_plus_early_A_hours_with_10pct_plus_tail',
            'margin_before_master_2350_stop_minutes','final_evaluation_debit_slot_seconds_if_A_completes_early','total_slot_work_reduction_from_moving_A']
    with (a.output_dir/'scenarios.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(value['sensitivity_scenarios'])
    lines=['# A-NLL 提前排程的纯时间评估', '',
        f"生成时间：{value['utc']}。仅供 root 决策；未授权、未追加或启动任何任务。", '',
        f"冻结 manifest：`{value['manifest']['path']}`，SHA256 `{value['manifest']['sha256']}`。", '',
        f"真实完整网格126条件（108实际训练、18主运行别名），选择已冻结。新增 queue jobs {value['new_jobs']}："
        f"forward {value['new_kind_counts'].get('adapted',0)}、reuse/materialization {value['new_kind_counts'].get('reuse',0)}、initial {value['new_initials']}。"
        f"复用已完成的既有 initial {value['completed_existing_initials']} 个和 canonical adapted origin {value['completed_existing_adapted_reuse_origins']} 个，定义及完整输出 SHA 已核对。", '',
        f"训练时间口径取 observer {value['runtime_forecast_updated_at_utc']}，保留原10%周转余量。参考主 final 实测 whole-slot中位数 {value['timing_reference']['main_final_median_whole_slot_seconds']:.6f}秒；SHA/scorer/polling均已包含，不重复相加。", '',
        '|场景|forward/reuse秒|四卡理想增量min|独立A队列尾部秒|含10%+尾部增量min|23:50前剩余min|',
        '|---|---:|---:|---:|---:|---:|']
    for row in value['sensitivity_scenarios']:
        lines.append(f"|{row['name']}|{row['forward_whole_slot_seconds_assumption']:.3f}/{row['reuse_materialization_seconds_assumption']:.0f}|{row['additional_four_gpu_ideal_minutes']:.2f}|{row['isolated_phase_dispatch_tail_seconds']:.2f}|{row['additional_four_gpu_minutes_with_10pct_plus_tail']:.2f}|{row['margin_before_master_2350_stop_minutes']:.2f}|")
    lines.extend(['', '提前执行只将相同工作移到当前 master；最终评价的同额 slot 工作仅在这些任务真实完成且完整性通过后扣除。总推理工作减少为0。上述扣除目前为条件预算，不是假定任务已经完成。', '',
        '独立A队列尾部按四卡均可用、原依赖已完成的FIFO模拟；现有非抢占children、长程C吞吐、共享CPU/I/O以及全master调度尾部仍有不确定性，不能当作按时完成保证。科学范围、选择规则和归档QA均不变。'])
    (a.output_dir/'advice.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:value[k] for k in ['status','manifest','new_jobs','new_kind_counts','completed_existing_initials','completed_existing_adapted_reuse_origins','all_scenarios_fit_before_master_stop']}))


if __name__=='__main__':main()
