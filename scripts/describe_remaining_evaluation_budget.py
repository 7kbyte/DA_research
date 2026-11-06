#!/usr/bin/env python3
"""CPU-only remaining evaluation budget from whole-slot scheduler receipts.

Only frozen manifests, scheduler state, execution attestations, preparation
metadata and earlier score-blind structural/validation timing receipts are read.
No score output, choice logprob, prediction, or checkpoint tensor is parsed.
"""
import argparse
import collections
import datetime as dt
import hashlib
import heapq
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from schedule_jobs import digest_job

MODELS=('pythia-410m','pythia-1b','pythia-2.8b')
SCENARIOS=(('low',1.,0.,10.),('central',1.25,1.,20.),('high',1.5,3.,40.))


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def utc(value):return dt.datetime.fromisoformat(value.replace('Z','+00:00'))
def quantiles(values):return {'n':len(values),'min':min(values),'median':statistics.median(values),'max':max(values),'sum':sum(values)}


def measure(manifest_path,state,definitions,kinds=None):
    """Extract scheduler wall and attestation subwalls without opening outputs."""
    path=Path(manifest_path);manifest=read(path);manifest_sha=sha(path);rows=[]
    for job in manifest['jobs']:
        if kinds and job['kind'] not in kinds:continue
        current=state['jobs'].get(job['id'],{})
        if (current.get('status')!='completed' or current.get('returncode')!=0 or current.get('missing_outputs') or
                job['id'] not in definitions or current.get('definition_sha256')!=digest_job(definitions[job['id']])):
            raise ValueError('A complete, definition-matched measured job is required: '+job['id'])
        output=Path(job['output']);attestation_path=output.with_name(output.stem+'_execution.json');attestation=read(attestation_path)
        if (attestation.get('job_id')!=job['id'] or attestation.get('manifest_sha256')!=manifest_sha or
                attestation.get('checkpoint_sha256')!=job.get('checkpoint_sha256') or attestation.get('verified_frozen_inputs') is not True or
                str(output) not in attestation.get('output_sha256',{})):
            raise ValueError('Measured job lacks a matching frozen execution attestation: '+job['id'])
        elapsed=current['elapsed_seconds'];verification=attestation['verification_wall_seconds'];scorer=attestation['inference_subprocess_wall_seconds']
        if any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in (elapsed,verification,scorer)) or elapsed+.01<verification+scorer:
            raise ValueError('Invalid whole-slot/subwall timing')
        if job['kind']=='initial':kind='initial'
        else:
            endpoints={r['kind'] for r in manifest['records'] if r['inference_job_id']==job['id']}
            if len(endpoints)!=1:raise ValueError('Measured mixed endpoint needs explicit timing attribution')
            kind=next(iter(endpoints))
        group=manifest['baseline_groups'][job['baseline_group']]
        rows.append({'id':job['id'],'kind':kind,'model':group['model_tag'],'whole_slot_wall_seconds':elapsed,
            'verification_wall_seconds_nested':verification,'scorer_subprocess_wall_seconds_nested':scorer,
            'unclassified_scheduler_wall_seconds':elapsed-verification-scorer,'policy':group['inference_policy'],
            'input_bytes':sum(Path(p).stat().st_size for p in {**manifest['common_input_sha256'],**job['input_sha256']}),
            'attestation':{'path':str(attestation_path),'sha256':sha(attestation_path)},
            'started_at_utc':current['started_at_utc'],'ended_at_utc':current['ended_at_utc']})
    return manifest,rows


def physical_scope():
    """Forward counts are distinct from logical endpoints and materialization."""
    rows=[]
    def add(id,mode,model,family,logical,physical,completed,aliases=0,completed_logical=0,endpoint=None):
        rows.append({'id':id,'mode':mode,'model':model,'data_family':family,'endpoint':endpoint,
            'logical_adapted_endpoints':logical,'physical_forward_jobs':physical,'completed_forward_jobs':completed,
            'remaining_forward_jobs':physical-completed,'already_proven_aliases':aliases,'completed_logical_endpoints':completed_logical})
    add('main_nll_adapted','nll','pythia-410m','main_v3',72,66,66,6,72,'final+best')
    add('main_nll_initial','nll','pythia-410m','main_v3',0,1,1,endpoint='initial')
    add('main_behavior_adapted','behavior','pythia-410m','behavior_tasks',72,66,0,6,endpoint='final+best')
    add('behavior_initials_410m','behavior','pythia-410m','behavior_tasks',0,1,1,endpoint='initial')
    add('A_nll','nll','pythia-410m','main_v3',126,108,0,18,18,'final')
    add('A_selected_behavior_max','behavior','pythia-410m','behavior_tasks',18,18,0,endpoint='final')
    add('B_nll','nll','pythia-410m','main_v3',60,60,0,endpoint='final')
    for model,count in zip(MODELS,(24,24,15)):
        name=model.replace('-','_').replace('.','p')
        add('C_'+name+'_nll_final','nll',model,'extended_v1',count,count,0,endpoint='final')
        add('C_'+name+'_nll_best','nll',model,'extended_v1',count,count,0,endpoint='best')
        add('C_'+name+'_nll_initial','nll',model,'extended_v1',0,1,0,endpoint='initial')
        add('C_'+name+'_behavior_final','behavior',model,'behavior_tasks',count,count,0,endpoint='final')
        if model!='pythia-410m':add('behavior_initials_'+name,'behavior',model,'behavior_tasks',0,1,1,endpoint='initial')
    return rows


def dispatch(jobs,slots=4):
    """A feasible post-training baseline-first list schedule, no GPU execution."""
    clocks=[(0.,i) for i in range(slots)];heapq.heapify(clocks);finished={};schedule=[]
    initials=[j for j in jobs if j.get('initial')];others=[j for j in jobs if not j.get('initial')]
    for job in [*sorted(initials,key=lambda j:-j['seconds']),*sorted(others,key=lambda j:-j['seconds'])]:
        free,slot=heapq.heappop(clocks);start=max(free,max((finished[p] for p in job.get('depends_on',[])),default=0.))
        end=start+job['seconds'];finished[job['id']]=end;heapq.heappush(clocks,(end,slot))
        schedule.append({'id':job['id'],'slot':slot,'start':start,'end':end})
    total=sum(j['seconds'] for j in jobs);makespan=max((t for t,_ in clocks),default=0.)
    return {'sum_whole_job_slot_seconds_proxy':total,'ideal_four_slot_wall_seconds_proxy':total/slots,
        'baseline_first_LPT_wall_seconds_proxy':makespan,'LPT_imbalance_and_dependency_tail_seconds_proxy':makespan-total/slots,
        'largest_job_seconds_proxy':max((j['seconds'] for j in jobs),default=0.),'job_count':len(jobs)}


def build(root,state,runtime,source_paths):
    root=Path(root);definitions={j['id']:j for j in read(root/'configs/supplemental_v1_queue.json')}
    nll_manifest,nll_rows=measure(root/'docs/primary_evaluation_v2_execution_matched_nll_manifest.json',state,definitions)
    if len(nll_rows)!=67:raise ValueError('All 67 primary NLL jobs must be complete before this v3 budget')
    _,main_initial=measure(root/'docs/primary_evaluation_v2_execution_matched_behavior_manifest.json',state,definitions,{'initial'})
    _,planned_initial=measure(root/'docs/planned_model_behavioral_timing_v1_manifest.json',state,definitions,{'initial'})
    behavior_rows={r['model']:r for r in [*main_initial,*planned_initial]}
    if set(behavior_rows)!=set(MODELS):raise ValueError('Exactly three measured behavior initial models required')
    nll_reference={k:quantiles([r['whole_slot_wall_seconds'] for r in nll_rows if r['kind']==k]) for k in ('initial','final','best')}
    old=read(root/'results/evaluation_timing_budget_v2.json');nll_structure=read(root/'results/nll_scope_budget_v1.json')
    task_structure=read(root/'results/behavioral_budget_preregistered_v2.json');scope=physical_scope()
    frozen_nll=read(root/'configs/nll_scope_execution_freeze_v1.json');frozen_behavior=read(root/'configs/behavioral_scope_execution_freeze_v2.json')
    if (frozen_nll['logical_endpoints']!=384 or frozen_behavior['scope']['primary']['logical_endpoints']!=72 or
            frozen_behavior['scope']['supplement_A']['maximum_endpoints']!=18 or frozen_behavior['scope']['extended']['logical_endpoints_by_model']!=dict(zip(MODELS,(24,24,15)))):
        raise ValueError('Frozen scope differs from the explicit v3 runtime accounting')
    rates={(g['model'],g['data_family']):g for g in nll_structure['groups']}
    source_group=rates[('pythia-410m','main_v3')]
    source_loop=source_group['prediction_tokens_per_inference']*source_group['timing_proxy']['median_seconds_per_prediction_token']
    transfers={}
    for (model,family),group in rates.items():
        target_loop=group['prediction_tokens_per_inference']*group['timing_proxy']['median_seconds_per_prediction_token']
        transfers[model+':'+family]={'prediction_tokens':group['prediction_tokens_per_inference'],
            'token_count_ratio_to_main':group['prediction_tokens_per_inference']/source_group['prediction_tokens_per_inference'],
            'matched_validation_seconds_per_token':group['timing_proxy']['median_seconds_per_prediction_token'],
            'whole_wall_transfer_ratio_assumption':target_loop/source_loop,
            'batch_size':group['inference_policy']['batch_size'],'context_limit':group['inference_policy']['max_length']}
    checkpoint_sizes={m:old['measured_initials'][m]['future_final_checkpoint_byte_size_proxy'] for m in MODELS}
    # Only an already serialized checkpoint size is used, without loading tensors.
    # Final size also bounds the unknown selected-best loading allowance here.
    cases=[]
    for name,factor,load_per_gib,alias_wall in SCENARIOS:
        stat={'low':'min','central':'median','high':'max'}[name];jobs=[];per_row=[];future_savings=[]
        def nll_seconds(model,family,kind):
            return nll_reference[kind][stat]*transfers[model+':'+family]['whole_wall_transfer_ratio_assumption']*factor
        def behavior_seconds(model):
            observed=behavior_rows[model];size=checkpoint_sizes[model]
            extra_sha=observed['verification_wall_seconds_nested']*size/observed['input_bytes']
            # Add only the new checkpoint input; original SHA/scorer/polling are
            # already nested inside the measured whole-slot initial observation.
            extra_load=size/(1<<30)*load_per_gib
            return observed['whole_slot_wall_seconds']*factor+extra_sha+extra_load
        for row in scope:
            count=row['remaining_forward_jobs']
            if not count:continue
            baseline='C_'+row['model']+'_nll_initial'
            if row['mode']=='nll':
                one=nll_seconds(row['model'],row['data_family'],row['endpoint'])
                initial=row['endpoint']=='initial';dependency=[] if initial or row['data_family']=='main_v3' else [baseline]
            else:one=behavior_seconds(row['model']);initial=False;dependency=[]
            for i in range(count):
                jobs.append({'id':baseline if initial else row['id']+'_'+str(i),'seconds':one,'initial':initial,'depends_on':dependency})
            per_row.append({'id':row['id'],'remaining_forward_jobs':count,'per_job_whole_slot_seconds_proxy':one,'sum_slot_seconds_proxy':one*count})
        # Cross-manifest A aliases need bounded CPU verification/materialization,
        # although their already-completed forward is not charged a second time.
        for i in range(18):jobs.append({'id':'A_prior_materialization_'+str(i),'seconds':alias_wall,'forward':False})
        for model in ('pythia-410m','pythia-1b'):
            for kind in ('final','best'):
                future_savings.append({'model':model,'mode':'nll','endpoint':kind,'max_avoided_forwards':3,'slot_seconds_proxy':3*nll_seconds(model,'extended_v1',kind)})
            future_savings.append({'model':model,'mode':'behavior','endpoint':'final','max_avoided_forwards':3,'slot_seconds_proxy':3*behavior_seconds(model)})
        full=dispatch(jobs)
        # An explicit proof-success scenario removes only D's matching endpoint
        # jobs. It is never deducted from the conservative main projection.
        savings_keys={f'C_{m.replace("-","_").replace(".","p")}_{mode}_{kind}':3 for m in ('pythia-410m','pythia-1b') for mode,kind in (('nll','final'),('nll','best'),('behavior','final'))}
        after=[];removed=collections.Counter()
        for job in jobs:
            key=job['id'].rsplit('_',1)[0]
            if key in savings_keys and removed[key]<savings_keys[key]:removed[key]+=1
            else:after.append(job)
        proof_success=dispatch(after)
        cases.append({'name':name,'whole_wall_sensitivity_factor':factor,'new_behavior_checkpoint_deserialization_allowance_seconds_per_GiB':load_per_gib,
            'A_alias_materialization_wall_allowance_per_job_seconds':alias_wall,'remaining_forward_jobs':sum(r['remaining_forward_jobs'] for r in scope),
            'materialization_only_jobs':18,'per_phase':per_row,'four_slot_dispatch':full,
            'extra_unmeasured_handoff_preflight_allowance_seconds':300.,'wall_with_dispatch_tail_and_handoff_allowance_seconds_proxy':full['baseline_first_LPT_wall_seconds_proxy']+300.,
            'future_C_all_D_B4_equal_savings':future_savings,'future_C_all_D_B4_equal_dispatch':proof_success,
            'future_C_all_D_B4_equal_slot_seconds_saved_proxy':full['sum_whole_job_slot_seconds_proxy']-proof_success['sum_whole_job_slot_seconds_proxy'],
            'future_C_all_D_B4_equal_LPT_wall_seconds_saved_proxy':full['baseline_first_LPT_wall_seconds_proxy']-proof_success['baseline_first_LPT_wall_seconds_proxy'],
            'interpretation':'Sensitivity assumptions, not confidence intervals, observed adapted-task runtimes, GPU kernel time, or guarantees'})
    now=dt.datetime.now(dt.timezone.utc);midnight=utc('2026-11-07T00:00:00Z');deadline=utc('2026-11-07T03:26:32Z');delivery_start=deadline-dt.timedelta(seconds=1800)
    training_at=utc(runtime['updated_at_utc']);training_eta=training_at+dt.timedelta(hours=runtime['four_gpu_remaining_wall_hours_with_10pct'])
    for case in cases:
        duration=case['wall_with_dispatch_tail_and_handoff_allowance_seconds_proxy']
        case['margin_from_midnight_to_delivery_tail_minutes_proxy']=((delivery_start-midnight).total_seconds()-duration)/60
        case['margin_from_training_forecast_to_delivery_tail_minutes_proxy']=((delivery_start-training_eta).total_seconds()-duration)/60
        case['latest_start_before_delivery_tail_utc_proxy']=(delivery_start-dt.timedelta(seconds=duration)).isoformat()
    return {'schema_version':3,'recorded_at_utc':now.isoformat(),'prepared_only':True,'scope_changed':False,
        'source_sha256':{str(p):sha(p) for p in source_paths},'source_state_canonical_json_sha256':hashlib.sha256(encoded(state)).hexdigest(),
        'measured_primary_NLL':{'all_67_completed':True,'remaining_forward_jobs':0,'by_endpoint_whole_slot_seconds':nll_reference,'jobs':nll_rows,
            'subwall_accounting':'scorer and input verification are nested in whole_slot_wall_seconds; never add them again'},
        'measured_behavior_initials':behavior_rows,'scope_accounting':scope,
        'totals':{'NLL_logical_adapted_endpoints':384,'NLL_physical_forward_jobs_including_baselines':364,'NLL_completed_forward_jobs':67,'NLL_remaining_forward_jobs':297,
            'behavior_logical_adapted_endpoints_max':153,'behavior_physical_forward_jobs_including_baselines_max':150,'behavior_completed_forward_jobs':3,'behavior_remaining_forward_jobs_max':147,
            'remaining_forward_jobs_max':444,'already_proven_primary_forward_aliases_each_mode':6,'A_NLL_already_completed_prior_forward_reuses':18,
            'A_remaining_materialization_only_jobs_allowance':18,'C_future_aliases_not_deducted_NLL_max':12,'C_future_aliases_not_deducted_behavior_max':6,
            'A_validation_selected_behavior_same_checkpoint_reuse_not_deducted_max':18},
        'refresh_policy':'Preserve this immutable v3 snapshot. A later score-blind snapshot may substitute complete READY manifest workloads and their frozen actual equivalence evidence; missing/incomplete manifests never authorize scope changes or guessed aliases.',
        'timing_transfer':{'NLL':transfers,'NLL_formula':'observed main whole-slot wall by endpoint × matched validation cost-per-token-and-corpus-size ratio × explicit sensitivity factor; no subwall added',
            'behavior_formula':'observed same-model initial whole-slot wall × sensitivity factor + only new-checkpoint SHA byte-transfer proxy + explicit new-checkpoint deserialization allowance',
            'checkpoint_size_upper_proxies':checkpoint_sizes,'selected_best_uses_final_size_proxy':True,'parameter_count_used_to_predict_seconds':False},
        'structural_behavior_work':{m:{'parameter_count':v['parameter_count'],'work_per_pass':v['work_per_pass']} for m,v in task_structure['scope'].items()},
        'sensitivity_scenarios':cases,'time_window':{'midnight_utc':midnight.isoformat(),'final_deadline_utc':deadline.isoformat(),'delivery_tail_seconds':1800,
            'latest_inference_finish_utc':delivery_start.isoformat(),'midnight_window_seconds_before_delivery_tail':(delivery_start-midnight).total_seconds(),
            'training_forecast_source_updated_at_utc':training_at.isoformat(),'training_forecast_age_seconds_at_budget':(now-training_at).total_seconds(),
            'current_master_jobs':runtime['master_queue_jobs'],'current_master_status_counts':runtime['master_status_counts'],
            'forecast_training_finish_utc_with_existing_10pct':training_eta.isoformat(),'training_forecast_includes_remaining_primary_NLL':runtime['remaining_primary_nll_slot_seconds'],
            'training_forecast_is_not_a_guarantee':True},
        'limitations':['All counts assume complete frozen cohorts; A behavior remains at most18 until full validation-only selection exists.',
            'C410M/1B require each model audit and all six actual endpoint comparisons before v4 readiness; no future equality is used in the conservative projection.',
            'C uses extended_v1 heldout, not main_v3; NLL token counts and batch16/16/8 are explicitly transferred from matched validation timing, not parameter counts.',
            'Whole-wall token-rate transfer also scales fixed overhead and is an assumption; initial NLL and large-checkpoint deserialization remain unmeasured on the extended heldout.',
            'Behavior has43844 short candidates, sequence-length sorting, 2741 or5481 calls; scored continuation tokens do not measure padded/context/attention or kernel-launch work.',
            'The three behavior initials are single observations, not runtime distributions; observed1B wall is below410M and cannot justify monotonic parameter-size scaling.',
            'Four-slot dispatch assumes all training and all required source proofs are complete and four independent GPUs are available; shared CPU/I/O, retries and readiness delays can exceed the scenarios.',
            'Scheduler whole-slot observations already contain polling/child startup; the dispatch tail models load imbalance/dependencies, not another per-job polling charge.',
            'The300s handoff/preflight allowance is explicitly unmeasured and excludes the separate1800s delivery tail and potentially slow CPU model-equivalence proof production.',
            'Alias forwarding is zero additional model inference; cross-manifest verification/materialization is not assumed to have zero operational wall.'],
        'test_or_behavior_scores_parsed':False,'new_inference_started':False,'frozen_modules_modified':False,'producer_source_sha256':sha(__file__)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--runtime-budget',type=Path,default=ROOT/'results/runtime_budget_latest.json');a=p.parse_args()
    if a.output.exists() or a.source_root.exists():raise FileExistsError('Use new immutable v3 output and source snapshot paths')
    a.source_root.mkdir(parents=True);state_path=a.source_root/'scheduler_state.json';state_path.write_bytes((ROOT/'logs/supplemental_job_state.json').read_bytes())
    runtime_path=a.source_root/'runtime_budget.json';runtime_path.write_bytes(a.runtime_budget.read_bytes())
    paths=[state_path,runtime_path,ROOT/'configs/supplemental_v1_queue.json',ROOT/'docs/primary_evaluation_v2_execution_matched_nll_manifest.json',
        ROOT/'docs/primary_evaluation_v2_execution_matched_behavior_manifest.json',ROOT/'docs/planned_model_behavioral_timing_v1_manifest.json',
        ROOT/'results/evaluation_timing_budget_v2.json',ROOT/'results/nll_scope_budget_v1.json',ROOT/'results/behavioral_budget_preregistered_v2.json',
        ROOT/'configs/nll_scope_execution_freeze_v1.json',ROOT/'configs/behavioral_scope_execution_freeze_v2.json',Path(__file__)]
    value=build(ROOT,read(state_path),read(runtime_path),paths);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(encoded(value))
    print(json.dumps({'output':str(a.output),'sha256':sha(a.output),'remaining_NLL':value['totals']['NLL_remaining_forward_jobs'],
        'remaining_behavior_max':value['totals']['behavior_remaining_forward_jobs_max'],'four_slot_wall_minutes_with_tail_and_handoff':[round(c['wall_with_dispatch_tail_and_handoff_allowance_seconds_proxy']/60,2) for c in value['sensitivity_scenarios']],
        'training_finish_forecast_utc':value['time_window']['forecast_training_finish_utc_with_existing_10pct'],'scores_parsed':False}))


if __name__=='__main__':main()
