#!/usr/bin/env python3
"""One-shot, score-blind C410 completion reconciliation. Never starts GPU jobs."""
import argparse
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from describe_remaining_evaluation_budget import dispatch, quantiles, utc
from describe_remaining_evaluation_budget_v4 import cache_inventory
from handoff_evaluation import attestation_path, completion_proof, job_definition
from preflight_C410_ready_nll import Inputs
from reconcile_main_behavior_runtime import fifo_projection
from schedule_jobs import digest_job
from stat_bound_digest_cache import atomic, small_proof

BASE_SHA = '940a0a0651cc7dd20da16d46321d8d4cac6b4dbbf7dd3e1ba6fbdd787e0b8f7d'
NLL_SHA = '48f8b2235c303211e4f79a92c16c7fb1c8d3757ce5d66a97a555fba7dfcf8f32'
BEHAVIOR_SHA = 'db6bdaf2f631ffc1c97f5d9c0ae1d0521e67ee6556790eb472d74195c5fb66e1'
MASTER_SHA = '37f00a8ac4af9d71aabb6b54f425e3fa222f596e25ce6e049015c0a66185e377'
PLAN = 'extended_pythia_410m_100m'


class WaitForGate(RuntimeError):
    pass


def validate_scope(nll, behavior, comparisons):
    """A49 proxy is43 real forwards plus6 proven avoided, never49 measured."""
    n, b = nll['workload'], behavior['workload']
    if (n['logical_checkpoint_endpoints'], n['initial_baselines'], n['adapted_inference_jobs'], n['total_queue_jobs'], n['avoided_duplicate_inferences']) != (48,1,42,43,6):
        raise ValueError('Require the actual C41043-forward/48-logical NLL scope')
    if (b['logical_checkpoint_endpoints'], b['initial_baselines'], b['adapted_inference_jobs'], b['total_queue_jobs'], b['avoided_duplicate_inferences']) != (24,0,21,21,3):
        raise ValueError('Require the actual C41021-forward/24-logical behavior scope')
    if len(nll['jobs']) != 43 or len(nll['records']) != 48 or len(behavior['jobs']) != 21 or len(behavior['records']) != 24:
        raise ValueError('Manifest rows differ from their actual workload')
    expected={(endpoint,seed) for endpoint in ['final','best'] for seed in [11,22,33]}
    if {(x['endpoint'],x['seed']) for x in comparisons} != expected or len(comparisons)!=6 or not all(x['bitwise_model_state_equal'] for x in comparisons):
        raise ValueError('All six real C410 endpoint identity pairs are required')
    records={(r['run_id'],r['kind']):r for r in nll['records']}
    behavior_records={r['run_id']:r for r in behavior['records']}
    for pair in comparisons:
        left=records[(pair['left_run_id'],pair['endpoint'])];right=records[(pair['right_run_id'],pair['endpoint'])]
        if (left['checkpoint_sha256'],right['checkpoint_sha256']) != (pair['left_checkpoint_sha256'],pair['right_checkpoint_sha256']):
            raise ValueError('Proof does not bind the actual endpoint checkpoints')
        if left['inference_job_id'] != right['inference_job_id']:
            raise ValueError('Actual NLL manifest did not apply the claimed alias')
        if pair['endpoint']=='final':
            a,b=behavior_records[left['run_id']],behavior_records[right['run_id']]
            if a['inference_job_id']!=b['inference_job_id'] or (a['checkpoint_sha256'],b['checkpoint_sha256']) != (pair['left_checkpoint_sha256'],pair['right_checkpoint_sha256']):
                raise ValueError('Behavior alias lacks the same actual final-state binding')
    return {'prior_C410_unaliased_NLL_proxy':49,'actual_completed_NLL_forwards_required':43,
        'proven_NLL_avoided_forwards':6,'future_C410_behavior_forwards':21,'future_C410_behavior_avoided':3}


def completion_rows(queue, bindings, master, state, proof_function=completion_proof):
    """Check the entire43 gate before any output byte hash or attestation parse."""
    lookup={j['id']:j for j in master}
    if len(lookup)!=len(master):raise ValueError('Duplicate master IDs')
    for job in queue:
        rec=state['jobs'].get(job['id'],{})
        if lookup.get(job['id'])!=job or rec.get('definition_sha256')!=digest_job(job):raise ValueError('C410 exact definition mismatch')
        if rec.get('status') not in {'pending','running','completed'} or 'stop_requested_at' in rec:raise ValueError('C410 failed/stopped record')
        if rec.get('status')=='completed' and (rec.get('returncode')!=0 or rec.get('missing_outputs')):raise ValueError('C410 unsuccessful completion')
    if any(state['jobs'][j['id']]['status']!='completed' for j in queue):raise WaitForGate('C410 NLL is not43/43 complete; no output read or budget credit')
    rows=[]
    for job in queue:
        binding=bindings[job['id']];entry=binding['entry'];rec=state['jobs'][job['id']]
        proof=proof_function(job,binding)
        att=json.loads(attestation_path(entry['output']).read_bytes())
        wall=rec['elapsed_seconds'];scorer=att['inference_subprocess_wall_seconds'];verify=att['verification_wall_seconds']
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in [wall,scorer,verify]) or wall+.01<scorer+verify:raise ValueError('Invalid whole-slot/subtimer relation')
        kinds={r['kind'] for r in binding['records']}
        kind='initial' if entry['kind']=='initial' else next(iter(kinds)) if len(kinds)==1 else 'mixed'
        if kind not in {'initial','final','best'}:raise ValueError('Mixed/unsupported C410 timing endpoint')
        rows.append(dict(id=job['id'],kind=kind,definition_sha256=digest_job(job),whole_slot_seconds=wall,
            scorer_subprocess_wall_seconds_nested=scorer,input_verification_wall_seconds_nested=verify,
            output_execution_attestations=proof,checkpoint_sha256=entry.get('checkpoint_sha256'),
            started_at_utc=rec['started_at_utc'],ended_at_utc=rec['ended_at_utc']))
    return rows


def future_cases(v5, primary, A, behavior):
    """Only the49 C410 NLL proxy and3 proven future behavior duplicates leave."""
    actual={'main_behavior_adapted':[j for j in primary['jobs'] if j['kind']=='adapted'],
        'A_selected_behavior':[j for j in A['jobs'] if j['kind']=='adapted'],
        'A_selected_behavior_materialization':[j for j in A['jobs'] if j['kind']=='reuse'],
        'C_pythia_410m_behavior_final':behavior['jobs']}
    if {k:len(v) for k,v in actual.items()} != {'main_behavior_adapted':66,'A_selected_behavior':15,'A_selected_behavior_materialization':3,'C_pythia_410m_behavior_final':21}:raise ValueError('Main/A/C410 frozen behavior scope differs')
    cases=[];limit=utc('2026-11-07T02:56:32Z')
    removed_ids={'C_pythia_410m_nll_final':24,'C_pythia_410m_nll_best':24,'C_pythia_410m_nll_initial':1}
    for old in v5['sensitivity_scenarios']:
        if old['whole_wall_sensitivity_factor']!={'low':1.,'central':1.25,'high':1.5}[old['name']]:raise ValueError('Sensitivity factor changed')
        phases=[];jobs=[];removed=[]
        for source in old['per_phase']:
            if source['id'] in removed_ids:
                if source['remaining_forward_jobs']!=removed_ids[source['id']] or source['materialization_jobs']:raise ValueError('Prior C41049 unaliased proxy differs')
                removed.append(source);continue
            row=copy.deepcopy(source)
            if row['id']=='C_pythia_410m_behavior_final':
                if row['remaining_forward_jobs']!=24:raise ValueError('Prior behavior24 proxy changed')
                row['remaining_forward_jobs']=21;row['sum_slot_seconds_proxy']=21*row['per_job_whole_slot_seconds_proxy']
            phases.append(row);count=row['remaining_forward_jobs']+row['materialization_jobs']
            initial=row['id'].endswith('_nll_initial')
            for index in range(count):
                real=actual[row['id']][index] if row['id'] in actual else None
                identity=real['id'] if real else row['id'] if initial else row['id']+'_'+str(index)
                dependencies=[]
                if row['materialization_jobs']:dependencies=[real['inference_origin']['job_id']]
                elif row['id'].startswith('C_') and '_nll_' in row['id'] and not initial:dependencies=[row['id'].rsplit('_',1)[0]+'_initial']
                jobs.append(dict(id=identity,seconds=row['per_job_whole_slot_seconds_proxy'],initial=initial,depends_on=dependencies,forward=not bool(row['materialization_jobs'])))
        if len(removed)!=3 or len(jobs)!=224 or sum(j['forward'] for j in jobs)!=221:raise ValueError('Only49 NLL proxy and3 proven behavior duplicates may leave276')
        projection=dispatch(jobs);duration=projection['baseline_first_LPT_wall_seconds_proxy']+old['handoff_preflight_allowance_seconds_unmeasured']
        starts={}
        for name,t in [('from_2350','2026-11-06T23:50:00Z'),('from_midnight','2026-11-07T00:00:00Z')]:
            end=utc(t)+dt.timedelta(seconds=duration);starts[name]=dict(start_utc=utc(t).isoformat(),finish_utc=end.isoformat(),margin_before_30min_delivery_tail_minutes=(limit-end).total_seconds()/60)
        cases.append(dict(name=old['name'],whole_wall_sensitivity_factor=old['whole_wall_sensitivity_factor'],remaining_forward_jobs=221,
            remaining_NLL_forward_jobs=80,remaining_behavior_forward_jobs=141,remaining_materialization_jobs=3,per_phase=phases,
            v5_C41049_unaliased_proxy_removed_with43_completed_plus6_proof=removed,four_slot_dispatch=projection,
            handoff_preflight_allowance_seconds_unmeasured=old['handoff_preflight_allowance_seconds_unmeasured'],
            wall_seconds_with_dispatch_tail_and_handoff_proxy=duration,wall_minutes_with_dispatch_tail_and_handoff_proxy=duration/60,
            starts=starts,latest_feasible_start_utc_proxy=(limit-dt.timedelta(seconds=duration)).isoformat(),
            other_model_rates_and_unproved_alias_scope_unchanged=True))
    return cases


def collect(args):
    if args.output.exists() or args.source_root.exists():raise FileExistsError('Choose new immutable output/source paths')
    begun=time.perf_counter();cpu=time.process_time();source=Inputs()
    v5=source.read('prior_v5.json',ROOT/'results/evaluation_timing_budget_v5.json',BASE_SHA)
    plans={};manifests={};queues={}
    for mode,digest in [('nll',NLL_SHA),('behavior',BEHAVIOR_SHA)]:
        folder=ROOT/'artifacts/evaluation_ready_v5'/(PLAN+'_'+mode)
        ready=source.read(mode+'_READY.json',folder/'READY.json')
        if ready.get('ready') is not True or ready.get('status')!='ready' or ready.get('id')!=PLAN+'_'+mode or ready.get('mode')!=mode or ready['manifest']['sha256']!=digest:raise ValueError('Actual READY source differs')
        manifests[mode]=source.read(mode+'_manifest.json',ready['manifest']['path'],digest)
        queues[mode]=source.read(mode+'_queue.json',ready['queue']['path'],ready['queue']['sha256']);plans[mode]=ready
        if len(queues[mode])!=len(manifests[mode]['jobs']) or {j['id'] for j in queues[mode]}!={j['id'] for j in manifests[mode]['jobs']}:raise ValueError('Actual queue must contain every manifest inference exactly once')
    identity=plans['nll']['source_model_equivalence']
    comp=source.read('C410_completion.json',identity['completion']['path'],identity['completion']['sha256'])
    audit=source.read('C410_audit.json',identity['audit']['path'],identity['audit']['sha256'])
    if comp['status']!='passed' or audit['status']!='passed' or audit['source_violations']:raise ValueError('Complete C410 audit required')
    for endpoint in ['final','best']:
        ref=comp['proofs'][endpoint];proof=source.read(endpoint+'_proof.json',ref['path'],ref['sha256'])
        actual=[p for p in proof['comparisons'] if p['bitwise_model_state_equal'] and p['endpoint']==endpoint]
        wanted=[p for p in identity['comparisons'] if p['endpoint']==endpoint]
        if len(actual)!=3 or len(wanted)!=3:raise ValueError('Require3 true actual full-state pairs per endpoint')
        for row in wanted:
            p=next(p for p in actual if p['seed']==row['seed'])
            if {p['left']['checkpoint_sha256'],p['right']['checkpoint_sha256']}!={row['left_checkpoint_sha256'],row['right_checkpoint_sha256']}:raise ValueError('READY pair differs from actual full-state proof')
    scope=validate_scope(manifests['nll'],manifests['behavior'],identity['comparisons'])
    master=source.read('master_queue.json',ROOT/'configs/supplemental_v1_queue.json',MASTER_SHA)
    state=source.read('scheduler_state.json',ROOT/'logs/supplemental_job_state.json')
    if state['scheduler_pid']!=2988506:raise ValueError('Unexpected master PID')
    authorities={(str(Path(plans['nll']['manifest']['path']).resolve()),NLL_SHA):manifests['nll']}
    bindings={j['id']:job_definition(j,authorities,{}) for j in queues['nll']}
    # The only output-byte reads are inside this whole43-completion gate.
    rows=completion_rows(queues['nll'],bindings,master,state)
    runtime=source.read('runtime.json',ROOT/'results/runtime_budget_latest.json')
    if runtime['failed_or_blocked_jobs'] or not all(runtime['frozen_scientific_core_matches'].values()) or runtime['master_queue_jobs']!=len(master):raise ValueError('Failed jobs/source mismatch/unreconciled master')
    if not 0 <= (dt.datetime.now(dt.timezone.utc)-utc(runtime['updated_at_utc'])).total_seconds()<90:raise WaitForGate('Stale runtime snapshot')
    crows=[r for u in runtime['extended_model_runtime_updates'].values() for r in u['jobs']]
    if any(state['jobs'][r['id']]['status']!=r['scheduler_status'] for r in crows):raise WaitForGate('State/runtime C transition race')
    other=next(r for r in runtime['requested_matrix'] if r['phase']=='other_queued_jobs')
    if other['job_ids']:raise WaitForGate('Observer still charges completed C410 or new unaccounted work')
    primary=source.read('primary_behavior_manifest.json',ROOT/'docs/primary_evaluation_v2_execution_matched_behavior_manifest.json','312d8e150030d489593e7e705699ec1da5f7d505e349ecd15a978eca2c2f7934')
    Aref=v5['prior_v4'];v4=source.read('prior_v4.json',Aref['path'],Aref['sha256']);ref=v4['A_selected_behavior_ready']
    A=source.read('A_behavior_manifest.json',ref['path'],ref['sha256'])
    lookup={j['id']:j for j in master}
    if any(j['id'] in lookup for m in [primary,A,manifests['behavior']] for j in m['jobs'] if j['kind']!='initial'):raise ValueError('A/main/C410 behavior already entered master; another scope revision required')
    # Existing behavior baseline is explicit, completed, and its opaque outputs verified.
    group=next(iter(manifests['behavior']['baseline_groups'].values()));base=lookup[group['initial_job_id']]
    st=state['jobs'][base['id']]
    if st.get('status')!='completed' or st.get('returncode')!=0 or st.get('definition_sha256')!=digest_job(base) or st.get('missing_outputs') or 'stop_requested_at' in st:raise ValueError('Existing behavior baseline lost completion')
    pauth={(str((ROOT/'docs/primary_evaluation_v2_execution_matched_behavior_manifest.json').resolve()),'312d8e150030d489593e7e705699ec1da5f7d505e349ecd15a978eca2c2f7934'):primary}
    baseproof=completion_proof(base,job_definition(base,pauth,{}))
    cases=future_cases(v5,primary,A,manifests['behavior'])
    training=fifo_projection(master,state,[],{},runtime,runtime['remaining_slot_hours']*3600,True)
    if training['status']!='complete_time_simulation':raise WaitForGate('C-only FIFO contains unpriced work')
    finish=utc(runtime['updated_at_utc'])+dt.timedelta(seconds=training['makespan_seconds_with_10pct'])
    for case in cases:
        end=finish+dt.timedelta(seconds=case['wall_seconds_with_dispatch_tail_and_handoff_proxy'])
        case['from_current_training_forecast']=dict(training_finish_utc=finish.isoformat(),evaluation_finish_utc=end.isoformat(),margin_before_30min_delivery_tail_minutes=(utc('2026-11-07T02:56:32Z')-end).total_seconds()/60)
    progress=source.read('cache_progress.json',ROOT/'artifacts/evaluation_ready_v5/cache_progress.json')
    catref=v5['frozen_catalog'];cat=source.read('catalog.json',catref['path'],catref['sha256']);cache=cache_inventory(cat,progress)
    source.unchanged()
    args.source_root.mkdir(parents=True)
    for name,raw in source.blobs.items():(args.source_root/name).write_bytes(raw)
    complete=dict(schema_version=1,status='passed',scope=scope,completed_forward_jobs=43,logical_adapted_endpoints=48,proven_avoided_forwards=6,
        whole_slot_statistics={k:quantiles([r['whole_slot_seconds'] for r in rows if r['kind']==k]) for k in ['initial','final','best']},
        actual_whole_slot_seconds_sum=sum(r['whole_slot_seconds'] for r in rows),jobs=rows,existing_behavior_baseline_completion_proof=baseproof,
        scored_outputs_deserialized=False,nested_timers_added=False)
    atomic(args.source_root/'C410_runtime_completion.json',complete);atomic(args.source_root/'prehash_inventory.json',cache)
    result=dict(schema_version=6,status='score_blind_remaining_evaluation_forecast',utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        prior_v5=source.proofs['prior_v5.json'],C410_completion=small_proof(args.source_root/'C410_runtime_completion.json'),
        accounting=dict(prior_v5_NLL_unaliased_proxy=129,actual_C410_completed_forward_jobs=43,C410_proven_avoided_NLL_forwards=6,
            remaining_NLL_forward_jobs=80,remaining_behavior_forward_jobs=141,remaining_forward_jobs=221,remaining_materialization_jobs=3,remaining_queue_jobs=224,
            C410_behavior_actual_forward_jobs=21,C410_behavior_logical_endpoints=24,C410_proven_avoided_behavior_forwards=3,
            main66_still_all_unperformed=66,A_B_baselines_or_prior_completions_removed_again=0,other_C_unproved_alias_discounts=0,
            real_C410_NLL_forwards_are49=False,total_compute_reduction_due_to_early_execution=0),
        sensitivity_scenarios=cases,training_forecast=dict(updated_at_utc=runtime['updated_at_utc'],finish_utc=finish.isoformat(),complete_C_only_FIFO=training),
        prehash_inventory=small_proof(args.source_root/'prehash_inventory.json'),source_proofs=source.proofs,producer_source=small_proof(__file__),
        collector_wall_seconds=time.perf_counter()-begun,collector_process_cpu_seconds=time.process_time()-cpu,
        no_score_fields_read=True,no_GPU_or_scheduler_started=True,no_science_or_queue_changes=True,
        limitations=['C41043 actual completed plus6 proven avoided reconcile49 old unaliased proxy; no phantom6 remain.',
            'C410 future behavior24 logical use21 actual jobs; other model aliases unspent until complete evidence.',
            'All remaining v5 rates and1/1.25/1.5 factors stay unchanged; completed C410 times reported separately, never used selectively to lower large-model extrapolations.',
            'Training FIFO adds10pct exactly once; early evaluation moves existing work and does not save total compute.',
            'No repeated full CP hash assumed in handoff; actual immutable snapshot/stat checks still required; worker fresh SHA unchanged.',
            'This budget is not final scheduler admission or permission; real per-job expected_seconds launch cutoff and30min tail must also pass.'])
    atomic(args.output,result)
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);args=p.parse_args()
    try:result=collect(args)
    except WaitForGate as e:print(json.dumps(dict(status='waiting_no_budget_credit',reason=str(e))));raise SystemExit(3)
    print(json.dumps(dict(output=small_proof(args.output),accounting=result['accounting'])))


if __name__=='__main__':main()
