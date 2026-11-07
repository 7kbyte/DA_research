#!/usr/bin/env python3
"""One-shot, score-blind B60 time proposal. Never changes or starts a queue."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from handoff_evaluation import completion_proof, job_definition
from reconcile_main_behavior_runtime import fifo_projection
from schedule_jobs import digest_job

FIXED = {
    'B_queue.json': ('artifacts/evaluation_ready_v5/supplement_B_nll/queue.json', 'd166e2f4aaee69870328952274c471a804a2aec387b9838deaf653a9fe353240'),
    'B_manifest.json': ('artifacts/evaluation_ready_v5/supplement_B_nll/manifest.json', 'ffa3304307004d4e013b7414a841c5cdc54df619dd5f0376b276491ab2c515b4'),
    'B_phase_spec.json': ('artifacts/evaluation_ready_v5/supplement_B_nll/phase_spec.json', 'e3dde341a797982967db09ccef0ae6b63541be41509350e3890ecc34b6351578'),
    'A_completion.json': ('results/evaluation_timing_budget_v4_sources/A_runtime_completion.json', 'b87e50543f9383c686248362088ea2c602ce9c2f3b0c28073ecf6c2d1a3516a8'),
    'budget_v4.json': ('results/evaluation_timing_budget_v4.json', 'a5154c67177e5d43bc77593c8927ebc06c330989197e8db0dc701a80da381328'),
    'FIFO_source.py': ('scripts/reconcile_main_behavior_runtime.py', '17d9b7839aef14e8c073124489c455d67512033f6de7658029411e88047a6f17'),
    'observer_source.py': ('scripts/monitor_runtime_budget.py', 'aaf4c774b47d00bef8794ac4151f1194ad04c6d6d6dce830ceceb337f331d0f4'),
}
DYNAMIC = {
    'raw_runtime.json': 'results/runtime_budget_latest.json',
    'master_state.json': 'logs/supplemental_job_state.json',
    'master_queue.json': 'configs/supplemental_v1_queue.json',
    'B_READY.json': 'artifacts/evaluation_ready_v5/supplement_B_nll/READY.json',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def materialize(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def run(out):
    blobs, proofs = {}, {}
    for name, (relative, wanted) in FIXED.items():
        source = ROOT / relative
        payload = source.read_bytes()
        if sha(payload) != wanted:
            raise ValueError('Frozen source changed: ' + relative)
        blobs[name] = payload
        proofs[name] = {'path': str(source), 'sha256': wanted}
    for name, relative in DYNAMIC.items():
        source = ROOT / relative
        blobs[name] = source.read_bytes()
        proofs[name] = {'path': str(source), 'sha256': sha(blobs[name])}
    parsed = {name: json.loads(raw) for name, raw in blobs.items() if name.endswith('.json')}
    raw, state, master = (parsed[name] for name in ['raw_runtime.json', 'master_state.json', 'master_queue.json'])
    phase, manifest, ready = (parsed[name] for name in ['B_queue.json', 'B_manifest.json', 'B_READY.json'])
    when = dt.datetime.fromisoformat(raw['updated_at_utc'])
    captured = dt.datetime.now(dt.timezone.utc)
    if not 0 <= (captured - when).total_seconds() < 90:
        raise ValueError('Require a current runtime observer sample')
    if (state.get('scheduler_pid') != 2988506 or len(master) != 504
            or raw['master_queue_jobs'] != 504 or raw['failed_or_blocked_jobs']
            or not all(raw['frozen_scientific_core_matches'].values())):
        raise ValueError('Master/science status requires review')
    if ready.get('status') != 'ready' or not ready.get('ready'):
        raise ValueError('B must be genuinely READY')
    for key, name in [('queue', 'B_queue.json'), ('manifest', 'B_manifest.json'), ('phase_spec', 'B_phase_spec.json')]:
        if ready[key]['sha256'] != proofs[name]['sha256']:
            raise ValueError('READY identity mismatch')
    ids = {job['id'] for job in phase}
    master_lookup = {job['id']: job for job in master}
    if len(ids) != 60 or ids & (set(master_lookup) | set(state['jobs'])):
        raise ValueError('Only the complete, not-yet-enqueued B60 is supported')
    entries = {entry['id']: entry for entry in manifest['jobs']}
    workload = manifest['workload']
    if (manifest['mode'] != 'nll' or set(entries) != ids
            or any(entry['kind'] != 'adapted' for entry in entries.values())
            or workload['total_inference_jobs'] != 60 or workload['initial_baselines'] != 0
            or workload['reuse_materialization_jobs'] != 0):
        raise ValueError('B60 must contain exactly 60 new forwards')
    groups = manifest['baseline_groups']
    if len(groups) != 1:
        raise ValueError('Require the single original matched 410M baseline')
    group = next(iter(groups.values()))
    origin = group['initial_origin']
    origin_path = Path(origin['manifest'])
    origin_raw = origin_path.read_bytes()
    if sha(origin_raw) != origin['manifest_sha256']:
        raise ValueError('Prior baseline authority changed')
    origin_manifest = json.loads(origin_raw)
    base_id = group['initial_job_id']
    base_job = master_lookup[base_id]
    base_state = state['jobs'][base_id]
    if (base_state.get('status') != 'completed' or base_state.get('returncode') != 0
            or base_state.get('missing_outputs') or 'stop_requested_at' in base_state
            or base_state['definition_sha256'] != digest_job(base_job)):
        raise ValueError('Original complete baseline missing')
    authorities = {(str((ROOT / FIXED['B_manifest.json'][0]).resolve()), proofs['B_manifest.json']['sha256']): manifest,
                   (str(origin_path.resolve()), origin['manifest_sha256']): origin_manifest}
    inputs = {}
    baseline_binding = job_definition(base_job, authorities, inputs)
    base_output_proof = completion_proof(base_job, baseline_binding)
    mapping = []
    for job in phase:
        binding = job_definition(job, authorities, inputs)
        if job.get('depends_on') != [base_id] or job['expected_seconds'] != 180:
            raise ValueError('Frozen B dependency/actual allowance differs')
        if binding['manifest']['baseline_groups'][binding['entry']['baseline_group']]['inference_policy'] != group['inference_policy']:
            raise ValueError('B inference policy differs')
        mapping.append({'id': job['id'], 'definition_sha256': digest_job(job),
                        'actual_run_id': entries[job['id']]['actual_run_id'],
                        'checkpoint_sha256': entries[job['id']]['checkpoint_sha256'],
                        'depends_on': job['depends_on'], 'expected_seconds': job['expected_seconds']})
    c_rows = [row for update in raw['extended_model_runtime_updates'].values() for row in update['jobs']]
    unfinished = {jid for jid, record in state['jobs'].items() if record.get('status') != 'completed'}
    if (unfinished != {row['id'] for row in c_rows} or len(c_rows) != 63
            or any(state['jobs'][row['id']]['status'] != row['scheduler_status'] for row in c_rows)
            or sum(record.get('status') == 'running' for record in state['jobs'].values()) != 4
            or any(state['jobs'][jid]['definition_sha256'] != digest_job(master_lookup[jid]) for jid in unfinished)):
        raise ValueError('Observer/current C statuses are not a consistent 4-running+59-pending snapshot')
    other = next(row for row in raw['requested_matrix'] if row['phase'] == 'other_queued_jobs')
    if other['job_ids'] or other['remaining_gpu_seconds'] != 0:
        raise ValueError('Unexpected generic non-training remainder')
    a = parsed['A_completion.json']
    if a['status'] != 'passed' or a['new_forward_jobs_completed'] != 108 or a['materialization_only_jobs_completed'] != 18:
        raise ValueError('A reference must be the whole completed grid')
    for row in a['jobs']:
        record = state['jobs'][row['id']]
        if (record.get('status') != 'completed' or record.get('returncode') != 0
                or record['definition_sha256'] != row['definition_sha256']
                or record['elapsed_seconds'] != row['whole_slot_seconds']):
            raise ValueError('A timing receipt differs from exact completed state')
    reference = a['whole_slot_statistics']['adapted']['median']
    nonphase = raw['remaining_slot_hours'] * 3600
    base = fifo_projection(master, state, [], {}, raw, nonphase, True)
    stop = dt.datetime.fromisoformat('2026-11-06T23:50:00+00:00')
    scenarios = []
    for name, factor in [('low', 1.), ('central', 1.25), ('high', 1.5)]:
        duration = reference * factor
        projection = fifo_projection(master, state, phase, {jid: duration for jid in ids}, raw, nonphase, True)
        finish = when + dt.timedelta(seconds=projection['makespan_seconds_with_10pct'])
        v4case = next(case for case in parsed['budget_v4.json']['sensitivity_scenarios'] if case['name'] == name)
        backlog = next(row for row in v4case['per_phase'] if row['id'] == 'B_nll')
        scenarios.append({'name': name, 'whole_slot_seconds_proxy': duration, 'reference_factor': factor,
                          'phase_slot_seconds_before_common_10pct': len(phase) * duration,
                          'FIFO_priority_insertion': projection,
                          'increment_over_C_only_minutes': (projection['makespan_seconds_with_10pct'] - base['makespan_seconds_with_10pct']) / 60,
                          'finish_utc': finish.isoformat(), 'margin_before_2350_minutes': (stop - finish).total_seconds() / 60,
                          'meets_15min_reserve_under_proxy': (stop - finish).total_seconds() >= 900,
                          'still_in_final_189_NLL_backlog_jobs': 60,
                          'currently_removed_from_final_backlog_jobs': 0,
                          'future_same_definition_completion_only_credit_at_v4_slot_seconds': backlog['sum_slot_seconds_proxy']})
    result = {'schema_version': 1, 'status': 'read_only_timing_proposal', 'captured_at_utc': captured.isoformat(),
              'runtime_source_utc': when.isoformat(), 'source_age_at_capture_seconds': (captured - when).total_seconds(),
              'source_proofs': proofs, 'hypothetical_only': True, 'scheduler_pid': state['scheduler_pid'],
              'candidate_queue': {'path': str(out / 'candidate_queue.json'), 'sha256': proofs['B_queue.json']['sha256']},
              'B_manifest': ready['manifest'], 'B_phase_spec': ready['phase_spec'],
              'job_definition_mapping': mapping, 'baseline': {'id': base_id, 'charged_seconds': 0,
                'definition_sha256': digest_job(base_job), 'opaque_output_and_attestation_proof': base_output_proof,
                'original_manifest': {'path': str(origin_path), 'sha256': origin['manifest_sha256']}},
              'observed_A108_whole_slot_seconds': a['whole_slot_statistics']['adapted'],
              'current_C_only_FIFO_without_new_phase': base, 'scenarios': scenarios,
              'A126_or_baseline_work_deducted_again': 0, 'total_compute_saving_claimed_seconds': 0,
              'B_new_forward_count': 60, 'B_new_baseline_count': 0, 'B_reuse_count': 0,
              'B_current_backlog_credit': 0, 'main66_still_unqueued': True,
              'inputs_fresh_full_SHA_verification_deferred_to_frozen_worker': True,
              'no_score_semantics_read': True, 'no_GPU_or_scheduler_or_enqueue_started': True,
              'limitations': ['Pure time scenarios are not completion guarantees.',
                'All current running children keep their slots; priority only precedes pending C.',
                'Common 10 percent overhead is charged exactly once to all remaining C and hypothetical B slots.',
                'B definition allowance remains 180/job; no frozen job or live queue is changed.',
                'B can leave final 189-NLL backlog only after all exact jobs actually complete with integrity evidence.',
                'Earlier B execution moves the same work; it does not reduce total computation.']}
    out.mkdir(exist_ok=False)
    sources = out / 'sources'
    sources.mkdir()
    for name, payload in blobs.items():
        materialize(sources / name, payload)
    materialize(out / 'candidate_queue.json', blobs['B_queue.json'])
    materialize(out / 'proposal.json', (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode())
    print(json.dumps({'path': str(out / 'proposal.json'), 'sha256': sha((out / 'proposal.json').read_bytes()),
                      'runtime_source_utc': when.isoformat(), 'scenarios': [{k: row[k] for k in ['name', 'finish_utc', 'margin_before_2350_minutes', 'increment_over_C_only_minutes']} for row in scenarios]}, ensure_ascii=False))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: producer.py NEW_OUTPUT_DIRECTORY')
    run(Path(sys.argv[1]).resolve())
