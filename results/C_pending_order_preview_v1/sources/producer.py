#!/usr/bin/env python3
"""One immutable, score-blind, finite-candidate ordering preview. Never mutates a queue."""
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from reconcile_main_behavior_runtime import fifo_projection
from schedule_jobs import digest_job

PINNED = {
    'FIFO_source.py': ('scripts/reconcile_main_behavior_runtime.py', '17d9b7839aef14e8c073124489c455d67512033f6de7658029411e88047a6f17'),
    'scheduler_source.py': ('scripts/schedule_jobs.py', 'c5c264b058ff655315ef896ccb0a89b471637c6612013a51999fc6f48b2d16d9'),
    'budget_v5.json': ('results/evaluation_timing_budget_v5.json', '940a0a0651cc7dd20da16d46321d8d4cac6b4dbbf7dd3e1ba6fbdd787e0b8f7d'),
}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def write(path, value):
    with path.open('xb') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def raw_objects(blob):
    """Retain byte-identical JSON job objects for a later independent ordering proof."""
    text = blob.decode('utf-8')
    decoder = json.JSONDecoder()
    index = text.index('[') + 1
    objects = {}
    while True:
        while text[index].isspace() or text[index] == ',':
            index += 1
        if text[index] == ']':
            return objects
        value, end = decoder.raw_decode(text, index)
        if value['id'] in objects:
            raise ValueError('Duplicate job ID')
        objects[value['id']] = text[index:end].encode('utf-8')
        index = end


def produce(output):
    if output.exists():
        raise FileExistsError('New immutable namespace required')
    blobs, proofs = {}, {}
    for name, (relative, wanted) in PINNED.items():
        path = ROOT / relative
        blob = path.read_bytes()
        if sha(blob) != wanted:
            raise ValueError('Pinned source changed: ' + relative)
        blobs[name] = blob
        proofs[name] = {'path': str(path), 'sha256': wanted}
    dynamic = {
        'raw_runtime.json': ROOT / 'results/runtime_budget_latest.json',
        'master_state.json': ROOT / 'logs/supplemental_job_state.json',
        'master_queue.json': ROOT / 'configs/supplemental_v1_queue.json',
        'ready_index.json': ROOT / 'artifacts/evaluation_ready_v5/ready_index.json',
        'early_work_ROOT_decision.json': ROOT / 'docs/C410_early_NLL_admission_ROOT_decision.json',
        'master_cmdline.bin': Path('/proc/2988506/cmdline'),
        'master_proc_stat.txt': Path('/proc/2988506/stat'),
    }
    for name, path in dynamic.items():
        blobs[name] = path.read_bytes()
        proofs[name] = {'path': str(path), 'sha256': sha(blobs[name])}
    source = Path(__file__).resolve()
    blobs['producer.py'] = source.read_bytes()
    proofs['producer.py'] = {'path': str(source), 'sha256': sha(blobs['producer.py'])}
    data = {key: json.loads(value) for key, value in blobs.items() if key.endswith('.json')}
    raw, state, master, budget, authority = (data[key] for key in
        ('raw_runtime.json', 'master_state.json', 'master_queue.json', 'budget_v5.json', 'early_work_ROOT_decision.json'))
    timestamp = dt.datetime.fromisoformat(raw['updated_at_utc'])
    captured = dt.datetime.now(dt.timezone.utc)
    if not 0 <= (captured - timestamp).total_seconds() < 90:
        raise ValueError('Fresh observer required')
    argv = [value.decode() for value in blobs['master_cmdline.bin'].split(b'\0') if value]
    if (argv[argv.index('--deadline') + 1] != '2026-11-07T00:00:00Z'
            or '--launch-margin-seconds' in argv or '--stop-margin-seconds' in argv
            or state['scheduler_pid'] != 2988506 or len(master) != 564
            or raw['master_queue_jobs'] != 564 or raw['failed_or_blocked_jobs']
            or not all(raw['frozen_scientific_core_matches'].values())):
        raise ValueError('Actual scheduler configuration or source requires review')
    other = next(row for row in raw['requested_matrix'] if row['phase'] == 'other_queued_jobs')
    if other['job_ids'] or other['remaining_gpu_seconds']:
        raise ValueError('Only remaining C training is priced')
    c_rows = [row for update in raw['extended_model_runtime_updates'].values() for row in update['jobs']]
    model = {row['id']: tag for tag, update in raw['extended_model_runtime_updates'].items() for row in update['jobs']}
    c_ids = set(model)
    unfinished = {jid for jid, record in state['jobs'].items() if record['status'] != 'completed'}
    if (len(c_ids) != 63 or unfinished != {row['id'] for row in c_rows if row['scheduler_status'] != 'completed'}
            or any(state['jobs'][row['id']]['status'] != row['scheduler_status'] for row in c_rows)
            or any(row['total_seconds'] != 0 for row in c_rows if row['scheduler_status'] == 'completed')):
        raise ValueError('Observer/state completion race; recapture in a new invocation')
    lookup = {job['id']: job for job in master}
    if len(lookup) != len(master) or any(state['jobs'][jid]['definition_sha256'] != digest_job(lookup[jid]) for jid in lookup):
        raise ValueError('Frozen job definitions differ')
    if any(lookup[jid]['expected_seconds'] != 3600 for jid in c_ids):
        raise ValueError('C admission allowance changed')
    running = [jid for jid in unfinished if state['jobs'][jid]['status'] == 'running']
    if len(running) != 4 or len({state['jobs'][jid]['gpu'] for jid in running}) != 4:
        raise ValueError('Exactly four actual occupied slots required')
    pending = {tag: [job['id'] for job in master if job['id'] in c_ids and model[job['id']] == tag
                    and state['jobs'][job['id']]['status'] == 'pending'] for tag in raw['extended_model_runtime_updates']}
    ones, large = pending['pythia-1b'], pending['pythia-2.8b']
    positions = [index for index, job in enumerate(master) if job['id'] in set(ones + large)]
    if [master[index]['id'] for index in positions] != ones + large:
        raise ValueError('Expected current pending 1B then2.8B order')
    candidates = {
        'current_1b_then_2p8b': ones + large,
        'pending_2p8b_then_1b': large + ones,
        'last3_1b_after_2p8b': ones[:-3] + large + ones[-3:],
        'last4_1b_after_2p8b': ones[:-4] + large + ones[-4:],
    }
    ready = next(row for row in data['ready_index.json']['phases'] if row['id'] == 'extended_pythia_410m_100m_nll')
    if ready.get('ready'):
        raise ValueError('Actual READY requires a separate real-plan preview')
    c410 = {jid for jid, tag in model.items() if tag == 'pythia-410m'}
    initial = {'id': 'timing_only_C410_extended_NLL_initial', 'depends_on': sorted(c410)}
    phase = [initial] + [{'id': 'timing_only_C410_NLL_' + kind + '_' + jid, 'depends_on': [initial['id']]}
                       for kind in ('final', 'best') for jid in sorted(c410)]
    high = next(case for case in budget['sensitivity_scenarios'] if case['name'] == 'high')
    rates = {row['id'].rsplit('_', 1)[-1]: row['per_job_whole_slot_seconds_proxy'] for row in high['per_phase']
             if row['id'].startswith('C_pythia_410m_nll_')}
    if set(rates) != {'initial', 'final', 'best'}:
        raise ValueError('All frozen rate strata required')
    durations = {job['id']: rates['initial' if job is initial else 'final' if '_NLL_final_' in job['id'] else 'best'] for job in phase}
    nonphase = raw['remaining_slot_hours'] * 3600
    remaining = {row['id']: row['total_seconds'] for row in c_rows}
    cutoff = dt.datetime.fromisoformat('2026-11-06T22:30:00+00:00')
    stop = dt.datetime.fromisoformat('2026-11-06T23:50:00+00:00')
    tail = dt.datetime.fromisoformat('2026-11-07T02:56:32+00:00')
    old_raw_objects = raw_objects(blobs['master_queue.json'])
    results, orders, dispatch_rows = [], {}, []
    baseline_seconds = None
    for name, order in candidates.items():
        candidate = list(master)
        for position, jid in zip(positions, order):
            candidate[position] = lookup[jid]
        if ({job['id'] for job in candidate} != set(lookup)
                or any(job is not lookup[job['id']] for job in candidate)
                or any(candidate[index] is not master[index] for index in range(len(master)) if index not in positions)):
            raise AssertionError('Only the approved finite pending-order domain may move')
        for tag in ('pythia-410m', 'pythia-1b', 'pythia-2.8b'):
            if ([job['id'] for job in candidate if job['id'] in c_ids and model[job['id']] == tag]
                    != [job['id'] for job in master if job['id'] in c_ids and model[job['id']] == tag]):
                raise AssertionError('Within-model relative order changed')
        reconstructed = b'[\n' + b',\n'.join(old_raw_objects[job['id']] for job in candidate) + b'\n]\n'
        if json.loads(reconstructed) != candidate:
            raise AssertionError('Raw-object-preserving candidate proof failed')
        orders[name] = {'all564_ordered_ids': [job['id'] for job in candidate],
            'raw_object_preserving_reconstruction_sha256': sha(reconstructed),
            'reconstruction_rule': "b'[\\n' + b',\\n'.join(original_raw_JSON_object[jid] for jid in all564_ordered_ids) + b'\\n]\\n'",
            'job_object_count': len(candidate), 'all_original_raw_objects_and_definition_hashes_identical': True,
            'noncandidate_positions_identical': True, 'running_positions_identical': True,
            'C410_positions_and_order_identical': True, 'within_model_original_relative_order_preserved': True}
        variants = []
        for early in (False, True):
            forecast = fifo_projection(candidate, state, phase if early else [], durations if early else {}, raw, nonphase, True)
            if forecast['status'] != 'complete_time_simulation' or abs(forecast['unassigned_conservative_nonphase_slot_seconds']) > .01:
                raise ValueError('Every actual remaining job must be priced')
            if name == 'current_1b_then_2p8b' and not early:
                baseline_seconds = forecast['makespan_seconds_with_10pct']
            future = {row['id']: row for row in forecast['future_dispatch_timing_rows']}
            launches, ends = [], {}
            for jid in sorted(c_ids):
                record = state['jobs'][jid]
                if record['status'] == 'pending':
                    start = timestamp + dt.timedelta(seconds=future[jid]['start_seconds'])
                    basis = 'forecast_pending_dispatch'
                    ends[jid] = future[jid]['end_seconds']
                else:
                    start = dt.datetime.fromisoformat(record['started_at_utc'])
                    basis = 'actual_already_started'
                    ends[jid] = remaining[jid] * 1.1 if record['status'] == 'running' else 0.
                row = {'candidate': name, 'variant': 'event_gated_C49_high' if early else 'authorized_C_only',
                       'id': jid, 'model': model[jid], 'start_utc': start.isoformat(), 'basis': basis,
                       'headroom_to_2230_seconds': (cutoff - start).total_seconds()}
                launches.append(row)
            latest = max(launches, key=lambda row: row['start_utc'])
            finish = timestamp + dt.timedelta(seconds=forecast['makespan_seconds_with_10pct'])
            min_headroom = min(row['headroom_to_2230_seconds'] for row in launches)
            model_ends = {tag: (timestamp + dt.timedelta(seconds=max(ends[jid] for jid in c_ids if model[jid] == tag))).isoformat()
                          for tag in pending}
            variant = {'variant': 'event_gated_C49_high' if early else 'authorized_C_only',
                'last_C_training_launch': latest, 'minimum_launch_headroom_seconds': min_headroom,
                'strict_C_launch_admission_satisfied': min_headroom > 0,
                'all_training_launches_at_least_300seconds_before_cutoff': min_headroom >= 300,
                'finish_utc': finish.isoformat(), 'finish_margin_before_2350_seconds': (stop - finish).total_seconds(),
                'finish_at_least_900seconds_before_stop': (stop - finish).total_seconds() >= 900,
                'extra_wall_seconds_relative_to_current_C_only': forecast['makespan_seconds_with_10pct'] - baseline_seconds,
                'model_complete_phase_times_utc_proxy': model_ends,
                'all63_actual_or_forecast_training_starts': launches, 'FIFO': forecast}
            if early:
                gate = max(ends[jid] for jid in c410)
                if future[initial['id']]['start_seconds'] < gate - 1e-7:
                    raise AssertionError('Full C410 phase gate violated')
                if any(future[job['id']]['start_seconds'] < future[initial['id']]['end_seconds'] - 1e-7 for job in phase[1:]):
                    raise AssertionError('Matched initial dependency violated')
                envelope = finish + dt.timedelta(seconds=high['wall_seconds_with_dispatch_tail_and_handoff_proxy'])
                variant.update({'matched_extended_initial_count': 1, 'adapted_final_plus_best_forward_count': 48,
                    'full_C410_phase_gate_utc_proxy': (timestamp + dt.timedelta(seconds=gate)).isoformat(),
                    'new_initial_dispatch_utc_proxy': (timestamp + dt.timedelta(seconds=future[initial['id']]['start_seconds'])).isoformat(),
                    'zero_extra_READY_delay_assumed_not_measured': True, 'unproved_alias_credit': 0,
                    'extra_wall_seconds_relative_to_same_candidate_C_only': forecast['makespan_seconds_with_10pct'] - variants[0]['FIFO']['makespan_seconds_with_10pct'],
                    'unchanged_full_v5_tail_conservative_envelope_finish_utc': envelope.isoformat(),
                    'unchanged_full_v5_tail_margin_before_delivery_tail_seconds': (tail - envelope).total_seconds(),
                    'envelope_keeps_same49_again_and_is_not_a_cost_ledger': True,
                    'root_optional_new_early_work_both_time_gates': min_headroom >= 300 and (stop - finish).total_seconds() >= 900})
            variants.append(variant)
            dispatch_rows.extend(launches)
        results.append({'candidate': name, 'moved_pending_ids_in_order': order, 'variants': variants})
    definition_rows = [{'id': job['id'], 'definition_sha256': digest_job(job),
                        'original_raw_JSON_object_sha256': sha(old_raw_objects[job['id']])} for job in master]
    result = {'schema_version': 1, 'status': 'finite_CPU_candidate_preview_only_not_execution_authority',
        'captured_at_utc': captured.isoformat(), 'runtime_source_utc': timestamp.isoformat(), 'sources': proofs,
        'actual_master': {'pid': 2988506, 'argv': argv, 'job_count': len(master),
                         'launch_margin_seconds': 1800, 'stop_margin_seconds': 600, 'all_C_expected_seconds': 3600,
                         'C_strict_launch_cutoff_utc': cutoff.isoformat()},
        'current_four_running_ids_and_gpus': [{'id': jid, 'gpu': state['jobs'][jid]['gpu']} for jid in sorted(running)],
        'pending_by_model': {tag: len(ids) for tag, ids in pending.items()},
        'candidate_count': 4, 'scientific_C_cohort_count': 63, 'all564_definition_proofs': definition_rows,
        'candidate_order_proofs': orders, 'results': results,
        'frozen_C410_NLL_high_rate_seconds': rates, 'C49_sensitivity_factor_already_in_rates': high['whole_wall_sensitivity_factor'],
        'whole_FIFO_10percent_applied_once_to_every_remaining_duration': True,
        'scope_limits': ['Finite four specified candidate orderings only; no subset or horizon search.',
            'Current four running processes and every pending C410 position/order remain fixed.',
            'All original564 raw job objects, argv/env/expected_seconds and within-model relative orders preserved.',
            'No candidate queue written into configs or actual master; original phase queues/registries untouched.',
            'C49 is only a future full24 event-gated upperbound with new matched initial; zero extra audit/proof/READY latency is not measured.',
            'The high training timing proxy remains the same observer estimate plus10percent once; a timing forecast does not guarantee admission.',
            'No new evaluation completion credit; main66 deferred, optional2.8QA paused.',
            'Any actual reorder requires separate ROOT authority and current-state/source/definition checks.',
            'Actual early work requires real READY and a new real-manifest/current-runtime review plus all ROOT time gates.'],
        'source_master_queue_unmodified': sha(dynamic['master_queue.json'].read_bytes()) == sha(blobs['master_queue.json']),
        'score_semantics_read': False, 'new_GPU_scheduler_watcher_or_authority': False}
    if not result['source_master_queue_unmodified']:
        raise ValueError('Live queue changed during pure preview')
    output.mkdir()
    (output / 'sources').mkdir()
    for name, blob in blobs.items():
        write(output / 'sources' / name, blob)
    write(output / 'preview.json', encoded(result))
    with (output / 'summary.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['candidate', 'variant', 'last_C_launch_utc', 'launch_headroom_minutes', 'finish_utc',
                         'finish_margin_before2350_minutes', 'extra_wall_vs_current_C_only_minutes', 'early_both_gates'])
        for case in results:
            for variant in case['variants']:
                writer.writerow([case['candidate'], variant['variant'], variant['last_C_training_launch']['start_utc'],
                    variant['minimum_launch_headroom_seconds'] / 60, variant['finish_utc'],
                    variant['finish_margin_before_2350_seconds'] / 60, variant['extra_wall_seconds_relative_to_current_C_only'] / 60,
                    variant.get('root_optional_new_early_work_both_time_gates')])
    print(json.dumps({'preview': str(output / 'preview.json'), 'sha256': sha((output / 'preview.json').read_bytes()),
                     'runtime_utc': timestamp.isoformat(), 'results': [{'candidate': case['candidate'], 'variants':
                         [{key: variant[key] for key in ('variant', 'minimum_launch_headroom_seconds', 'finish_margin_before_2350_seconds',
                                                        'extra_wall_seconds_relative_to_current_C_only')} for variant in case['variants']]}
                         for case in results]}))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: producer.py NEW_RESULT_DIRECTORY')
    produce(Path(sys.argv[1]).resolve())
