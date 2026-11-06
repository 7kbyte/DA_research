#!/usr/bin/env python3
"""Single-phase main66 timing proposal/reconciliation; never enqueue or read scores."""
import argparse
import collections
import datetime as dt
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from handoff_evaluation import job_definition, completion_proof
from schedule_jobs import digest_job
from stat_bound_digest_cache import atomic, small_proof, fsync_directory

MANIFEST = ROOT / 'docs/primary_evaluation_v2_execution_matched_behavior_manifest.json'
MANIFEST_SHA = '312d8e150030d489593e7e705699ec1da5f7d505e349ecd15a978eca2c2f7934'
QUEUE = ROOT / 'configs/primary_evaluation_v2_execution_matched_behavior_queue.json'
QUEUE_SHA = '393f3df2f0213e2e4e4d2387109b750825c797b909f4b05d16cd3cb739f6aaa9'
BUDGET = ROOT / 'results/evaluation_timing_budget_v4.json'
BUDGET_SHA = 'a5154c67177e5d43bc77593c8927ebc06c330989197e8db0dc701a80da381328'
OBSERVER_SHA = 'aaf4c774b47d00bef8794ac4151f1194ad04c6d6d6dce830ceceb337f331d0f4'


def utc(value):
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))


def checked(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Frozen metadata SHA mismatch: ' + str(path))
    return json.loads(raw)


def nonnegative(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Timing must be finite and nonnegative')
    return value


def inspect_phase(queue, manifest, manifest_path, manifest_sha, master, state, now):
    entries = {entry['id']: entry for entry in manifest['jobs']}
    if (manifest['mode'] != 'behavior' or len(entries) != 67 or len(queue) != 67
            or set(entries) != {job['id'] for job in queue}):
        raise ValueError('Require exactly the frozen main67 manifest/queue')
    initial = [job for job in queue if entries[job['id']]['kind'] == 'initial']
    phase = [job for job in queue if entries[job['id']]['kind'] == 'adapted']
    if len(initial) != 1 or len(phase) != 66:
        raise ValueError('Only main66 adapted forwards plus one original baseline are supported')
    baseline = initial[0]
    authority = {(str(Path(manifest_path).resolve()), manifest_sha): manifest}
    hashes = {}
    bindings = {job['id']: job_definition(job, authority, hashes) for job in queue}
    lookup = {job['id']: job for job in master}
    if len(lookup) != len(master):
        raise ValueError('Duplicate master IDs')
    baseline_state = state['jobs'].get(baseline['id'], {})
    if (lookup.get(baseline['id']) != baseline or baseline_state.get('definition_sha256') != digest_job(baseline)
            or baseline_state.get('status') != 'completed' or baseline_state.get('returncode') != 0
            or baseline_state.get('missing_outputs') or 'stop_requested_at' in baseline_state):
        raise ValueError('Original baseline must already be definition-matched and completed')
    baseline_proof = completion_proof(baseline, bindings[baseline['id']])
    reference = nonnegative(baseline_state['elapsed_seconds'])
    present = [job for job in phase if job['id'] in lookup]
    if len(present) not in (0, 66):
        raise ValueError('Only zero or the full frozen main66 append is supported')
    rows = []
    for job in phase:
        jid = job['id']
        if set(job.get('depends_on', [])) != {baseline['id']}:
            raise ValueError('Main66 dependency must be only the already completed baseline')
        if jid not in lookup:
            if jid in state['jobs']:
                raise ValueError('Main66 state exists without its exact master job')
            rows.append({'id': jid, 'status': 'not_enqueued', 'elapsed_seconds': 0., 'definition_sha256': digest_job(job)})
            continue
        record = state['jobs'].get(jid, {})
        if lookup[jid] != job or record.get('definition_sha256') != digest_job(job):
            raise ValueError('Frozen main66 definition differs in master/state')
        status = record.get('status')
        if status not in {'pending', 'running', 'completed'} or 'stop_requested_at' in record:
            raise ValueError('A main66 job requires failure/deadline review')
        row = {'id': jid, 'status': status, 'definition_sha256': digest_job(job), 'elapsed_seconds': 0.}
        if status == 'completed':
            if record.get('returncode') != 0 or record.get('missing_outputs'):
                raise ValueError('Completed main66 job lacks successful outputs')
            row.update(elapsed_seconds=nonnegative(record['elapsed_seconds']),
                       attestation_sha256=completion_proof(job, bindings[jid]))
        elif status == 'running':
            # This phase's generic allowance is fully removed before crediting
            # elapsed time, so the same elapsed seconds cannot be deducted twice.
            started = record.get('started_at_epoch')
            elapsed = now.timestamp() - nonnegative(started) if started is not None else (now - utc(record['started_at_utc'])).total_seconds()
            row['elapsed_seconds'] = nonnegative(elapsed)
        rows.append(row)
    return phase, rows, {'id': baseline['id'], 'whole_slot_seconds': reference,
                         'definition_sha256': digest_job(baseline), 'attestation_sha256': baseline_proof,
                         'charged_to_new_phase': False}


def calculate(raw, queue, rows, baseline, budget, now):
    lookup = {job['id']: job for job in queue}
    phase_ids = set(lookup)
    other = next(row for row in raw['requested_matrix'] if row['phase'] == 'other_queued_jobs')
    charged = phase_ids & set(other['job_ids'])
    removed = sum(nonnegative(lookup[jid]['expected_seconds']) for jid in charged)
    if removed > other['remaining_gpu_seconds'] + .01:
        raise ValueError('Raw generic subtotal cannot support the exact main66 charge')
    nonphase = raw['remaining_slot_hours'] * 3600 - removed
    if nonphase < 0:
        raise ValueError('Negative remaining non-main66 work')
    present = sum(row['status'] != 'not_enqueued' for row in rows)
    if charged and not present:
        raise ValueError('Raw/current master snapshots disagree about main66 presence')
    complete = [row for row in rows if row['status'] == 'completed']
    running = [row for row in rows if row['status'] == 'running']
    observed = [row['elapsed_seconds'] for row in complete]
    # Frozen v4 already transfers actual baseline whole-slot time to adapted
    # checkpoints and low/central/high scenarios. Preserve those assumptions;
    # actual completed adapted medians may raise, never lower, the reference.
    reference = max(baseline['whole_slot_seconds'], statistics.median(observed) if len(observed) >= 3 else max(observed, default=0.))
    cases = []
    stop = utc('2026-11-06T23:50:00Z')
    for case in budget['sensitivity_scenarios']:
        phase = next(row for row in case['per_phase'] if row['id'] == 'main_behavior_adapted')
        if phase['remaining_forward_jobs'] != 66 or phase['materialization_jobs'] != 0:
            raise ValueError('v4 phase differs from main66')
        prior = nonnegative(phase['per_job_whole_slot_seconds_proxy'])
        rate = max(prior, reference * prior / baseline['whole_slot_seconds'])
        remaining = {row['id']: (0. if row['status'] == 'completed' else max(0., rate - row['elapsed_seconds']) if row['status'] == 'running' else rate)
                     for row in rows}
        beyond = [row['id'] for row in running if row['elapsed_seconds'] >= rate]
        work = sum(remaining.values())
        # A conservative independent phase tail on otherwise available slots.
        durations = sorted((value for value in remaining.values() if value > 0), reverse=True)
        slots = [0.] * 4
        for duration in durations:
            index = min(range(4), key=lambda index: slots[index])
            slots[index] += duration
        tail = max(slots) - work / 4
        total = (nonphase + work) * 1.1 / 4 + tail
        finish = now + dt.timedelta(seconds=total)
        margin = (stop - finish).total_seconds() / 60
        cases.append({'name': case['name'], 'frozen_v4_per_forward_whole_slot_seconds': prior,
                      'baseline_to_v4_whole_slot_factor': prior / baseline['whole_slot_seconds'],
                      'per_forward_whole_slot_seconds_proxy': rate,
                      'remaining_main66_slot_seconds_proxy': work,
                      'running_elapsed_seconds_credited_once_capped_per_job': sum(min(rate, row['elapsed_seconds']) for row in running),
                      'running_beyond_proxy_ids_unknown_extra_remaining': beyond,
                      'isolated_four_slot_dispatch_tail_seconds_proxy': tail,
                      'all_work_four_slot_seconds_with_10pct_plus_phase_tail_proxy': total,
                      'forecast_finish_utc': finish.isoformat(), 'margin_before_2350_minutes': margin,
                      'fits_before_2350_under_timing_proxy': margin >= 0 and not beyond,
                      'completed_same_definition_jobs_removed_from_final_evaluation_budget': len(complete),
                      'completed_same_definition_future_slot_seconds_removed_at_v4_proxy': len(complete) * prior,
                      'main66_still_in_final_evaluation_backlog_slot_seconds_at_v4_proxy': (66 - len(complete)) * prior,
                      'if_remaining_main66_finishes_early_future_slot_work_moves_not_disappears': (66 - len(complete)) * prior})
    return {'mode': 'exact_remaining_main66_reconciliation' if present else 'hypothetical_complete_main66_insertion',
            'main66_status_counts': dict(collections.Counter(row['status'] for row in rows)),
            'main66_jobs_present_in_master': present, 'main66_completed_actual_slot_seconds_once': sum(observed),
            'raw_generic_main66_charged_ids': sorted(charged), 'raw_generic_main66_slot_seconds_removed_once': removed,
            'raw_other_non_main66_job_ids_kept': [jid for jid in other['job_ids'] if jid not in phase_ids],
            'non_main66_remaining_slot_seconds_kept_from_raw': nonphase,
            'A126_generic_or_completed_cost_removed_by_this_tool': 0,
            'reference_whole_slot_seconds': reference, 'sensitivity_scenarios': cases,
            'same_work_moved_earlier_total_compute_saved_seconds': 0,
            'baseline_charged_to_phase_seconds': 0}


def fifo_projection(master, state, phase, phase_remaining, raw, nonphase, priority):
    """Pure time simulation of the existing master; launches no processes."""
    phase_ids = {job['id'] for job in phase}
    c_times = {row['id']: row['total_seconds'] for update in raw['extended_model_runtime_updates'].values()
               for row in update['jobs']}
    completed = {jid for jid, record in state['jobs'].items() if record.get('status') == 'completed' and record.get('returncode') == 0}
    lookup = {job['id']: job for job in master}
    hypothetical = not any(jid in lookup for jid in phase_ids)
    remaining_jobs = [job for job in master if job['id'] not in completed]
    if hypothetical:
        # Running children keep their slots; incoming priority only precedes pending jobs.
        running_jobs = [job for job in remaining_jobs if state['jobs'][job['id']]['status'] == 'running']
        pending_jobs = [job for job in remaining_jobs if state['jobs'][job['id']]['status'] != 'running']
        remaining_jobs = running_jobs + (phase + pending_jobs if priority else pending_jobs + phase)
    durations = {}
    unknown = []
    for job in remaining_jobs:
        jid = job['id']
        if jid in phase_ids:
            durations[jid] = phase_remaining[jid]
        elif jid in c_times:
            durations[jid] = c_times[jid]
        else:
            unknown.append(jid)
    if unknown:
        return {'status': 'unpriced_non_C_non_phase_jobs', 'unknown_job_ids': unknown}
    represented_nonphase = sum(value for jid, value in durations.items() if jid not in phase_ids)
    residual = nonphase - represented_nonphase
    if residual < -.01:
        raise ValueError('FIFO nonphase work exceeds the unchanged raw subtotal')
    pending = []
    running = []
    occupied = set()
    for job in remaining_jobs:
        jid = job['id']
        record = state['jobs'].get(jid, {})
        if record.get('status') == 'running':
            gpu = int(record['gpu'])
            if gpu not in range(4) or gpu in occupied:
                raise ValueError('Four actual running GPU slots must be unique')
            occupied.add(gpu)
            heapq.heappush(running, (durations[jid] * 1.1, gpu, jid))
        else:
            pending.append(job)
    free = [slot for slot in range(4) if slot not in occupied]
    now = 0.
    launches = []
    while pending or running:
        for slot in list(free):
            index = next((index for index, job in enumerate(pending) if set(job.get('depends_on', [])) <= completed), None)
            if index is None:
                continue
            job = pending.pop(index)
            end = now + durations[job['id']] * 1.1
            heapq.heappush(running, (end, slot, job['id']))
            free.remove(slot)
            launches.append({'id': job['id'], 'slot': slot, 'start_seconds': now, 'end_seconds': end})
        if not running:
            if pending:
                raise ValueError('Frozen dependencies block FIFO time simulation')
            break
        now = running[0][0]
        while running and running[0][0] <= now:
            _, slot, jid = heapq.heappop(running)
            completed.add(jid)
            free.append(slot)
        free.sort()
    extra = max(0., residual) * 1.1 / 4
    ideal = (nonphase + sum(phase_remaining.values())) * 1.1 / 4
    return {'status': 'complete_time_simulation', 'hypothetical_phase_priority_before_pending': priority if hypothetical else None,
            'actual_running_children_not_preempted': True, 'all_remaining_job_durations_receive_10pct_once': True,
            'unassigned_conservative_nonphase_slot_seconds': max(0., residual),
            'makespan_seconds_with_10pct': now + extra,
            'ideal_four_slot_seconds_with_10pct': ideal,
            'longrun_FIFO_imbalance_seconds_above_ideal': now + extra - ideal,
            'future_dispatch_timing_rows': launches}


def reconcile(raw_path, state_path, master_path):
    manifest = checked(MANIFEST, MANIFEST_SHA)
    queue = checked(QUEUE, QUEUE_SHA)
    budget = checked(BUDGET, BUDGET_SHA)
    raw_bytes = {key: Path(path).read_bytes() for key, path in
                 [('raw_runtime', raw_path), ('master_state', state_path), ('master_queue', master_path)]}
    raw, state, master = (json.loads(raw_bytes[key]) for key in ('raw_runtime', 'master_state', 'master_queue'))
    now = dt.datetime.now(dt.timezone.utc)
    age = (now - utc(raw['updated_at_utc'])).total_seconds()
    if (raw.get('observer_source_sha256') != OBSERVER_SHA or raw['status'] != 'monitoring'
            or raw['failed_or_blocked_jobs'] or not -1 <= age <= 120
            or not all(raw['frozen_scientific_core_matches'].values())):
        raise ValueError('Original runtime observer is stale, changed or unhealthy')
    phase, rows, baseline = inspect_phase(queue, manifest, MANIFEST, MANIFEST_SHA, master, state, now)
    value = calculate(raw, phase, rows, baseline, budget, now)
    if not value['main66_jobs_present_in_master']:
        value['current_C_only_FIFO_without_new_phase'] = fifo_projection(
            master, state, phase, {job['id']: 0. for job in phase}, raw,
            value['non_main66_remaining_slot_seconds_kept_from_raw'], True)
    for case in value['sensitivity_scenarios']:
        rate = case['per_forward_whole_slot_seconds_proxy']
        remaining = {row['id']: (0. if row['status'] == 'completed' else max(0., rate - row['elapsed_seconds']) if row['status'] == 'running' else rate)
                     for row in rows}
        variants = []
        for priority in ([True, False] if not value['main66_jobs_present_in_master'] else [False]):
            simulation = fifo_projection(master, state, phase, remaining, raw, value['non_main66_remaining_slot_seconds_kept_from_raw'], priority)
            if simulation['status'] == 'complete_time_simulation':
                finish = now + dt.timedelta(seconds=simulation['makespan_seconds_with_10pct'])
                margin = (utc('2026-11-06T23:50:00Z') - finish).total_seconds() / 60
                simulation.update(forecast_finish_utc=finish.isoformat(), margin_before_2350_minutes=margin,
                                  preserves_at_least15min_before_2350=margin >= 15 and not case['running_beyond_proxy_ids_unknown_extra_remaining'])
            variants.append(simulation)
        case['full_master_FIFO_scenarios'] = variants
    qualifying = {model: update['reliable_long_run_samples'] for model, update in raw['extended_model_runtime_updates'].items()}
    all_models = set(qualifying) == {'pythia-410m', 'pythia-1b', 'pythia-2.8b'} and all(any(row['sample']['steps_observed'] >= 100 and row['sample']['warmup_steps_excluded'] == 10
                                                for row in samples) for samples in qualifying.values())
    meets15 = all_models and all(all(simulation.get('preserves_at_least15min_before_2350', False)
                                    for simulation in case['full_master_FIFO_scenarios'])
                                for case in value['sensitivity_scenarios'])
    return {'schema_version': 1, 'utc': now.isoformat(), **value,
            'manifest': small_proof(MANIFEST), 'queue': small_proof(QUEUE), 'v4_budget': small_proof(BUDGET),
            'source_proofs': {key: {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(raw_bytes[key]).hexdigest()}
                              for key, path in [('raw_runtime', raw_path), ('master_state', state_path), ('master_queue', master_path)]},
            'raw_source_age_seconds': age, 'master_pid': state['scheduler_pid'], 'raw_master_jobs': raw['master_queue_jobs'],
            'current_master_jobs': len(master), 'baseline_already_completed': baseline,
            'main66_completed_integrity_and_running_timing': rows,
            'C_three_models_100_steps_excluding10_available': all_models,
            'C_runtime_samples': qualifying,
            'can_root_review_timing_proposal_now': all_models,
            'minimum_additional_training_stop_reserve_minutes_required_by_root': 15,
            'all_full_FIFO_scenarios_preserve_required15min_timing_reserve': meets15,
            'timing_only_recommendation': ('monitor_only_phase_already_enqueued' if value['main66_jobs_present_in_master'] else
                                           'root_may_review_insertion_but_no_authorization_implied' if meets15 else
                                           'defer_all66_insertion_under_current_timing_evidence'),
            'no_enqueue_or_GPU_launch': True, 'scores_or_predictions_parsed': False,
            'source_queue_job_definitions_changed': False, 'producer_source': small_proof(__file__),
            'limitations': ['Only this frozen main66 phase is repriced; all A and other raw work remains unchanged.',
                            'The actual frozen generic allowance is300/job. A used180; no allowance is edited.',
                            'Running phase work is max(proxy minus observed elapsed,0); beyond-proxy jobs carry unknown extra work and cannot produce a fit claim.',
                            'Existing nonphase forecast already contains its own method and C progress assumptions. The10% margin is applied once.',
                            'Completed jobs need exact definition, rc0 and full attestation/output SHA; score JSON is never parsed.',
                            'Before enqueue this is a hypothetical new phase; after enqueue only this phase remaining work replaces raw generic charges.',
                            'Final-tail work is removed only for verified same-definition completed jobs; earlier execution does not save total compute.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-runtime', type=Path, default=ROOT / 'results/runtime_budget_latest.json')
    p.add_argument('--state', type=Path, default=ROOT / 'logs/supplemental_job_state.json')
    p.add_argument('--master-queue', type=Path, default=ROOT / 'configs/supplemental_v1_queue.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new immutable timing proposal output')
    directory = args.output.with_name(args.output.stem + '_inputs')
    directory.mkdir(parents=True, exist_ok=False)
    snapshots = []
    origins = []
    for name, path in [('raw_runtime', args.raw_runtime), ('master_state', args.state), ('master_queue', args.master_queue)]:
        raw = path.read_bytes()
        target = directory / (name + '.json')
        with target.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        snapshots.append(target)
        origins.append({'name': name, 'source': str(path.resolve()), 'snapshot': small_proof(target)})
    fsync_directory(directory)
    value = reconcile(*snapshots)
    value['live_input_origins_and_immutable_snapshots'] = origins
    atomic(args.output, value)
    print(json.dumps({'mode': value['mode'], 'present': value['main66_jobs_present_in_master'],
                      'three_C_models_ready': value['C_three_models_100_steps_excluding10_available'],
                      'margins': {row['name']: row['margin_before_2350_minutes'] for row in value['sensitivity_scenarios']}}))


if __name__ == '__main__':
    main()
