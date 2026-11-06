#!/usr/bin/env python3
"""Score-blind B60 completion gate and immutable remaining evaluation v5.

This operational collector never starts inference or changes any GPU queue.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from describe_remaining_evaluation_budget import dispatch, quantiles, utc
from describe_remaining_evaluation_budget_v4 import cache_inventory
from handoff_evaluation import completion_proof, job_definition
from reconcile_main_behavior_runtime import fifo_projection
from schedule_jobs import digest_job
from stat_bound_digest_cache import atomic, small_proof

FIXED = {
    'prior_v4.json': ('results/evaluation_timing_budget_v4.json', 'a5154c67177e5d43bc77593c8927ebc06c330989197e8db0dc701a80da381328'),
    'B_queue.json': ('artifacts/evaluation_ready_v5/supplement_B_nll/queue.json', 'd166e2f4aaee69870328952274c471a804a2aec387b9838deaf653a9fe353240'),
    'B_manifest.json': ('artifacts/evaluation_ready_v5/supplement_B_nll/manifest.json', 'ffa3304307004d4e013b7414a841c5cdc54df619dd5f0376b276491ab2c515b4'),
    'B_authority.json': ('docs/supplement_B_nll_execution_authority.json', '1a3780b4922931276134d5b23afe1b101097f7361d74ce758519d9604521d7b4'),
    'B_enqueue_confirmation.json': ('docs/supplement_B_nll_enqueue_confirmation.json', '6ffd167857edb05df29bd41eb475c66e3785adbe915de3c0083c9eaa6e0d3956'),
}
MASTER_SHA = 'f9a05e3687c6d0c282180fa0f142770373120c92801087bc26b08128b6f56c42'


class WaitForGate(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_bytes())


def complete_phase(queue, bindings, master, state):
    """Wait for the entire exact phase before hashing any scored output bytes."""
    lookup = {job['id']: job for job in master}
    if len(lookup) != len(master):
        raise ValueError('Duplicate master job IDs')
    for job in queue:
        record = state['jobs'].get(job['id'], {})
        if lookup.get(job['id']) != job or record.get('definition_sha256') != digest_job(job):
            raise ValueError('B job definition differs from the frozen queue')
        if record.get('status') not in {'pending', 'running', 'completed'} or 'stop_requested_at' in record:
            raise ValueError('B job failed/stopped; explicit review required')
        if record.get('status') == 'completed' and (record.get('returncode') != 0 or record.get('missing_outputs')):
            raise ValueError('B completion is not successful')
    counts = {status: sum(state['jobs'][job['id']]['status'] == status for job in queue)
              for status in ('pending', 'running', 'completed')}
    if counts['completed'] != len(queue):
        raise WaitForGate('B phase not fully complete: ' + json.dumps(counts))
    rows = []
    for job in queue:
        record = state['jobs'][job['id']]
        elapsed = record['elapsed_seconds']
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid whole-slot timing')
        proof = completion_proof(job, bindings[job['id']])
        entry = bindings[job['id']]['entry']
        attestation = read(next(iter(proof)))
        scorer = attestation['inference_subprocess_wall_seconds']
        verification = attestation['verification_wall_seconds']
        if (any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in (scorer, verification)) or elapsed + .01 < scorer + verification):
            raise ValueError('Nested inference/verification timings exceed the whole slot')
        rows.append({'id': job['id'], 'kind': entry['kind'], 'actual_run_id': entry['actual_run_id'],
                     'definition_sha256': digest_job(job), 'checkpoint_sha256': entry['checkpoint_sha256'],
                     'status': 'completed', 'returncode': 0, 'whole_slot_seconds': elapsed,
                     'scorer_subprocess_wall_seconds_nested': scorer,
                     'input_verification_wall_seconds_nested': verification,
                     'execution_attestation_sha256': proof,
                     'started_at_utc': record['started_at_utc'], 'ended_at_utc': record['ended_at_utc']})
    return rows


def future_cases(v4, completion, primary_behavior, A_behavior):
    """Remove only B60; preserve all other scope and uncertainty multipliers."""
    main = [job for job in primary_behavior['jobs'] if job['kind'] == 'adapted']
    adapted = [job for job in A_behavior['jobs'] if job['kind'] == 'adapted']
    aliases = [job for job in A_behavior['jobs'] if job['kind'] == 'reuse']
    if len(main) != 66 or len(adapted) != 15 or len(aliases) != 3:
        raise ValueError('The remaining frozen main/A behavior scope changed')
    finish_limit = utc('2026-11-07T02:56:32Z')
    cases = []
    for old in v4['sensitivity_scenarios']:
        factor = old['whole_wall_sensitivity_factor']
        name = old['name']
        if factor != {'low': 1., 'central': 1.25, 'high': 1.5}[name]:
            raise ValueError('Keep the original conservative sensitivity factors')
        stat = {'low': 'min', 'central': 'median', 'high': 'max'}[name]
        reference = max(old['A_observed_410m_final_reference_seconds_before_factor'],
                        completion['whole_slot_statistics']['adapted'][stat])
        measured_final = max(old['future_410m_NLL_final_whole_slot_seconds_proxy'], reference * factor)
        jobs, phases = [], []
        removed = []
        for source in old['per_phase']:
            if source['id'] == 'B_nll':
                if source['remaining_forward_jobs'] != 60 or source['materialization_jobs'] != 0:
                    raise ValueError('Prior v4 B scope differs')
                removed.append(source)
                continue
            row = copy.deepcopy(source)
            if row['id'] == 'C_pythia_410m_nll_final':
                transfer = row['per_job_whole_slot_seconds_proxy'] / old['future_410m_NLL_final_whole_slot_seconds_proxy']
                row['per_job_whole_slot_seconds_proxy'] = max(row['per_job_whole_slot_seconds_proxy'], measured_final * transfer)
                row['sum_slot_seconds_proxy'] = row['remaining_forward_jobs'] * row['per_job_whole_slot_seconds_proxy']
            phases.append(row)
            count = row['remaining_forward_jobs'] + row['materialization_jobs']
            actual = (main if row['id'] == 'main_behavior_adapted' else adapted if row['id'] == 'A_selected_behavior'
                      else aliases if row['id'] == 'A_selected_behavior_materialization' else None)
            initial = row['id'].endswith('_nll_initial')
            for index in range(count):
                record = actual[index] if actual else None
                identity = record['id'] if record else row['id'] if initial else row['id'] + '_' + str(index)
                dependencies = []
                if row['materialization_jobs']:
                    dependencies = [record['inference_origin']['job_id']]
                elif row['id'].startswith('C_') and '_nll_' in row['id'] and not initial:
                    dependencies = [row['id'].rsplit('_', 1)[0] + '_initial']
                jobs.append({'id': identity, 'seconds': row['per_job_whole_slot_seconds_proxy'],
                             'initial': initial, 'depends_on': dependencies, 'forward': not bool(row['materialization_jobs'])})
        if len(removed) != 1 or len(jobs) != 276 or sum(row['forward'] for row in jobs) != 273:
            raise ValueError('Only 60 B forwards may leave the previous 336-job remainder')
        projection = dispatch(jobs)
        duration = projection['baseline_first_LPT_wall_seconds_proxy'] + old['handoff_preflight_allowance_seconds_unmeasured']
        starts = {}
        for label, start in [('from_2350', utc('2026-11-06T23:50:00Z')), ('from_midnight', utc('2026-11-07T00:00:00Z'))]:
            end = start + dt.timedelta(seconds=duration)
            starts[label] = {'start_utc': start.isoformat(), 'finish_utc': end.isoformat(),
                             'margin_before_30min_delivery_tail_minutes': (finish_limit - end).total_seconds() / 60}
        cases.append({'name': name, 'whole_wall_sensitivity_factor': factor,
                      'observed_B_statistic_before_factor': completion['whole_slot_statistics']['adapted'][stat],
                      'future_410m_NLL_final_whole_slot_seconds_proxy': measured_final,
                      'future_410m_reference_never_lowered_from_v4': True,
                      'remaining_forward_jobs': 273, 'remaining_materialization_jobs': 3,
                      'per_phase': phases, 'four_slot_dispatch': projection,
                      'v4_B60_future_slot_proxy_removed_once': removed[0]['sum_slot_seconds_proxy'],
                      'completed_B_actual_whole_slot_seconds_separate': completion['actual_whole_slot_seconds_sum'],
                      'handoff_preflight_allowance_seconds_unmeasured': old['handoff_preflight_allowance_seconds_unmeasured'],
                      'wall_seconds_with_dispatch_tail_and_handoff_proxy': duration,
                      'wall_minutes_with_dispatch_tail_and_handoff_proxy': duration / 60,
                      'starts': starts, 'latest_feasible_start_utc_proxy': (finish_limit - dt.timedelta(seconds=duration)).isoformat(),
                      'C_future_D_B4_equality_discount_spent': False,
                      'earlier_A_and_baseline_future_work_removed_again': 0,
                      'total_compute_reduction_due_to_early_B_seconds': 0})
    return cases


def capture_and_produce(output, sources):
    started, cpu_started = time.perf_counter(), time.process_time()
    if output.exists() or sources.exists():
        raise FileExistsError('Use a new immutable v5 output/source namespace')
    blobs, bound = {}, {}
    for name, (relative, wanted) in FIXED.items():
        path = ROOT / relative
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != wanted:
            raise ValueError('Frozen metadata changed: ' + relative)
        blobs[name] = raw
        bound[name] = {'path': str(path), 'sha256': wanted}
    for name, relative in {'scheduler_state.json': 'logs/supplemental_job_state.json',
                           'master_queue.json': 'configs/supplemental_v1_queue.json',
                           'raw_runtime.json': 'results/runtime_budget_latest.json',
                           'prehash_progress.json': 'artifacts/evaluation_ready_v5/cache_progress.json',
                           'ready_index.json': 'artifacts/evaluation_ready_v5/ready_index.json'}.items():
        path = ROOT / relative
        blobs[name] = path.read_bytes()
        bound[name] = {'path': str(path), 'sha256': hashlib.sha256(blobs[name]).hexdigest()}
    data = {name: json.loads(raw) for name, raw in blobs.items()}
    v4, queue, manifest = (data[name] for name in ['prior_v4.json', 'B_queue.json', 'B_manifest.json'])
    master, state, runtime = (data[name] for name in ['master_queue.json', 'scheduler_state.json', 'raw_runtime.json'])
    if len(master) != 564 or bound['master_queue.json']['sha256'] != MASTER_SHA or state.get('scheduler_pid') != 2988506:
        raise ValueError('The explicitly authorized master identity changed')
    if len(queue) != 60 or len(manifest['jobs']) != 60 or any(row['kind'] != 'adapted' for row in manifest['jobs']):
        raise ValueError('Require exactly the frozen B60 new-forward phase')
    group = next(iter(manifest['baseline_groups'].values()))
    origin = group['initial_origin']
    origin_raw = Path(origin['manifest']).read_bytes()
    if hashlib.sha256(origin_raw).hexdigest() != origin['manifest_sha256']:
        raise ValueError('The original baseline manifest changed')
    authorities = {(str((ROOT / FIXED['B_manifest.json'][0]).resolve()), FIXED['B_manifest.json'][1]): manifest,
                   (str(Path(origin['manifest']).resolve()), origin['manifest_sha256']): json.loads(origin_raw)}
    input_hashes = {}
    bindings = {job['id']: job_definition(job, authorities, input_hashes) for job in queue}
    # This gate occurs before any scored output is opened or read.
    rows = complete_phase(queue, bindings, master, state)
    if runtime['failed_or_blocked_jobs'] or not all(runtime['frozen_scientific_core_matches'].values()):
        raise ValueError('Training failures or scientific source mismatch require review')
    if runtime['master_queue_jobs'] != len(master):
        raise WaitForGate('Runtime observer has not incorporated the authorized master')
    when = utc(runtime['updated_at_utc'])
    if not 0 <= (dt.datetime.now(dt.timezone.utc) - when).total_seconds() < 90:
        raise WaitForGate('Runtime observer sample is stale')
    c_rows = [row for update in runtime['extended_model_runtime_updates'].values() for row in update['jobs']]
    if any(state['jobs'][row['id']]['status'] != row['scheduler_status'] for row in c_rows):
        raise WaitForGate('Wait for observer to reconcile newly started/completed C jobs')
    phase_ids = {job['id'] for job in queue}
    other = next(row for row in runtime['requested_matrix'] if row['phase'] == 'other_queued_jobs')
    if set(other['job_ids']) - phase_ids:
        raise ValueError('New unrelated future work requires another accounting revision')
    lookup = {job['id']: job for job in master}
    generic_removed = sum(lookup[jid]['expected_seconds'] for jid in other['job_ids'])
    if abs(generic_removed - other['remaining_gpu_seconds']) > .01:
        raise ValueError('Frozen generic B allowance cannot be reconciled exactly')
    primary = read(ROOT / 'docs/primary_evaluation_v2_execution_matched_behavior_manifest.json')
    if any(job['kind'] == 'adapted' and job['id'] in lookup for job in primary['jobs']):
        raise ValueError('Main66 has entered the master; do not describe all66 as remaining')
    A_manifest_path = v4['A_selected_behavior_ready']['path']
    if small_proof(A_manifest_path) != v4['A_selected_behavior_ready']:
        raise ValueError('Frozen selected A behavior manifest changed')
    A_manifest = read(A_manifest_path)
    if any(job['id'] in lookup for job in A_manifest['jobs']):
        raise ValueError('Selected A behavior already entered master; use a new revision')
    baseline = lookup[group['initial_job_id']]
    baseline_state = state['jobs'][baseline['id']]
    if (baseline_state.get('status') != 'completed' or baseline_state.get('returncode') != 0
            or baseline_state.get('definition_sha256') != digest_job(baseline) or baseline_state.get('missing_outputs')):
        raise ValueError('Previously counted initial baseline lost completion evidence')
    baseline_proof = completion_proof(baseline, job_definition(baseline, authorities, input_hashes))
    completion = {'schema_version': 1, 'status': 'passed', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'expected_jobs': 60, 'completed_jobs': 60, 'new_forward_jobs_completed': 60,
                  'new_initial_jobs': 0, 'reuse_materialization_jobs_completed': 0,
                  'queue': bound['B_queue.json'], 'manifest': bound['B_manifest.json'],
                  'execution_authority': bound['B_authority.json'], 'enqueue_confirmation': bound['B_enqueue_confirmation.json'],
                  'actual_whole_slot_seconds_sum': sum(row['whole_slot_seconds'] for row in rows),
                  'whole_slot_statistics': {'adapted': quantiles([row['whole_slot_seconds'] for row in rows])},
                  'first_started_at_utc': min(row['started_at_utc'] for row in rows),
                  'last_ended_at_utc': max(row['ended_at_utc'] for row in rows),
                  'phase_had_exclusive_four_GPUs_claimed': False, 'jobs': rows,
                  'initial_baseline_completion_proof': baseline_proof,
                  'initial_baseline_or_A_work_charged_again': False,
                  'nested_timers_added_to_whole_slots': False, 'same_work_moved_earlier_not_total_compute_saved': True,
                  'all_definitions_rc0_complete_outputs_and_opaque_SHA_verified': True,
                  'scored_outputs_deserialized': False, 'producer_source': small_proof(__file__)}
    completion['mixed_master_first_start_to_last_exit_seconds'] = (utc(completion['last_ended_at_utc']) - utc(completion['first_started_at_utc'])).total_seconds()
    catalog_proof = read(ROOT / 'docs/supplemental_evaluation_preparation_v5_execution.json')['catalog']
    if small_proof(catalog_proof['path']) != catalog_proof:
        raise ValueError('Frozen prewarm catalog changed')
    inventory = cache_inventory(read(catalog_proof['path']), data['prehash_progress.json'])
    cases = future_cases(v4, completion, primary, A_manifest)
    nonphase = runtime['remaining_slot_hours'] * 3600 - generic_removed
    training = fifo_projection(master, state, [], {}, runtime, nonphase, True)
    if training['status'] != 'complete_time_simulation':
        raise WaitForGate('Current C-only FIFO forecast is not fully priced')
    train_finish = when + dt.timedelta(seconds=training['makespan_seconds_with_10pct'])
    for case in cases:
        finish = train_finish + dt.timedelta(seconds=case['wall_seconds_with_dispatch_tail_and_handoff_proxy'])
        case['from_current_training_forecast'] = {'training_finish_utc': train_finish.isoformat(),
            'evaluation_finish_utc': finish.isoformat(), 'margin_before_30min_delivery_tail_minutes': (utc('2026-11-07T02:56:32Z') - finish).total_seconds() / 60}
    sources.mkdir(parents=True)
    for name, raw in blobs.items():
        (sources / name).write_bytes(raw)
    completion_path = sources / 'B_runtime_completion.json'
    inventory_path = sources / 'prehash_inventory.json'
    atomic(completion_path, completion)
    atomic(inventory_path, inventory)
    result = {'schema_version': 5, 'status': 'score_blind_remaining_evaluation_forecast',
              'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'prior_v4': bound['prior_v4.json'],
              'B_completion': small_proof(completion_path), 'original_A_completion': v4['A_completion'],
              'prehash_inventory': small_proof(inventory_path), 'frozen_catalog': catalog_proof,
              'accounting': {'v4_remaining_NLL_forward_jobs': 189, 'actual_new_B_NLL_forward_jobs_removed_once': 60,
                'remaining_NLL_forward_jobs': 129, 'remaining_behavior_forward_jobs': 144,
                'main66_behavior_forward_jobs_all_unperformed': 66, 'remaining_forward_jobs': 273,
                'remaining_materialization_jobs': 3, 'remaining_queue_jobs': 276,
                'A108_new_forwards_removed_again': 0, 'A18_materializations_removed_again': 0,
                'earlier_primary_forwards_or_baselines_removed_again': 0,
                'completed_B_slot_seconds_separate': completion['actual_whole_slot_seconds_sum'],
                'total_compute_reduction_due_to_moving_B_earlier': 0,
                'future_unproved_C_avoided_forwards_not_deducted': {'NLL_max': 12, 'behavior_max': 6}},
              'sensitivity_scenarios': cases,
              'training_forecast': {'updated_at_utc': runtime['updated_at_utc'],
                'finish_utc_with_existing_10pct_and_actual_FIFO_tail': train_finish.isoformat(),
                'raw_generic_completed_B_allowance_removed_once_slot_seconds': generic_removed,
                'master_jobs': len(master), 'master_status_counts': runtime['master_status_counts'],
                'complete_C_only_FIFO': training},
              'time_window': copy.deepcopy(v4['time_window']),
              'source_proofs': {name: small_proof(sources / name) for name in blobs},
              'producer_source': small_proof(__file__),
              'collector_wall_seconds': time.perf_counter() - started,
              'collector_process_cpu_seconds': time.process_time() - cpu_started,
              'limitations': ['Only successful exact B60 physical forwards leave v4 remainder; A and all earlier baselines were already accounted for.',
                'B low/median/high observations may raise, never lower, corresponding v4 410M-final references; factors1/1.25/1.5 remain unchanged.',
                'Only the410M final-NLL transfer can use B timings; selected-best, initial, behavior,1B and2.8B extrapolations remain separate and unchanged.',
                'Potential C inference aliases are not spent in this envelope; completed full-state proof and an actual frozen plan are required.',
                'Actual B slots include nested teacher/scorer/input verification; never add nested subitems again.',
                'C-only ETA includes actual running-slot placement and remaining FIFO imbalance, with10 percent once; it remains a forecast.',
                'No75-minute duplicate checkpoint rehash is assumed: explicit digest-cache snapshot binding and fresh current-stat checks remain required.',
                'Frozen inference workers always fresh-read all input SHA bytes before each task; prewarm cache never skips that.',
                'Five-minute handoff/preflight allowance is unmeasured; preserve the separate30-minute delivery tail.'],
              'no_score_fields_read': True, 'no_science_or_queue_changes': True, 'no_GPU_or_scheduler_started': True}
    atomic(output, result)
    with (sources / 'scenarios.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scenario', 'evaluation_minutes', 'from_2350_margin_minutes', 'from_midnight_margin_minutes', 'latest_start_utc', 'from_C_FIFO_forecast_margin_minutes'])
        for case in cases:
            writer.writerow([case['name'], case['wall_minutes_with_dispatch_tail_and_handoff_proxy'],
                case['starts']['from_2350']['margin_before_30min_delivery_tail_minutes'], case['starts']['from_midnight']['margin_before_30min_delivery_tail_minutes'],
                case['latest_feasible_start_utc_proxy'], case['from_current_training_forecast']['margin_before_30min_delivery_tail_minutes']])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--watch-progress', type=Path)
    parser.add_argument('--poll-seconds', type=float, default=60.)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        raise ValueError('CPU polling interval must be1..60 seconds')
    attempts, attempt_wall, attempt_cpu = 0, 0., 0.
    watch_start = time.perf_counter()
    while True:
        started, cpu_started = time.perf_counter(), time.process_time()
        attempts += 1
        try:
            result = capture_and_produce(args.output, args.source_root)
        except WaitForGate as error:
            attempt_wall += time.perf_counter() - started
            attempt_cpu += time.process_time() - cpu_started
            progress = {'status': 'waiting_for_full_phase_or_consistent_runtime', 'reason': str(error),
                        'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'attempts': attempts,
                        'attempt_wall_seconds_excluding_poll_sleep': attempt_wall,
                        'attempt_process_cpu_seconds': attempt_cpu,
                        'calendar_wall_seconds_including_wait_not_work_cost': time.perf_counter() - watch_start,
                        'CUDA_VISIBLE_DEVICES': __import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
                        'no_GPU_or_enqueue': True, 'no_score_semantics_read': True}
            if not args.watch_progress:
                print(json.dumps(progress))
                raise SystemExit(3)
            atomic(args.watch_progress, progress)
            time.sleep(args.poll_seconds)
            continue
        except Exception as error:
            if args.watch_progress:
                atomic(args.watch_progress, {'status': 'failed_requires_review', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                                            'error': repr(error), 'attempts': attempts})
            raise
        break
    if args.watch_progress:
        atomic(args.watch_progress, {'status': 'completed', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'output': small_proof(args.output), 'attempts': attempts,
            'attempt_wall_seconds_excluding_poll_sleep': attempt_wall + time.perf_counter() - started,
            'attempt_process_cpu_seconds': attempt_cpu + time.process_time() - cpu_started,
            'calendar_wall_seconds_including_wait_not_work_cost': time.perf_counter() - watch_start,
            'no_GPU_or_enqueue': True, 'no_score_semantics_read': True})
    print(json.dumps({'output': small_proof(args.output), 'completion': result['B_completion'],
                      'accounting': result['accounting'], 'scenario_minutes': [case['wall_minutes_with_dispatch_tail_and_handoff_proxy'] for case in result['sensitivity_scenarios']]}))


if __name__ == '__main__':
    main()
