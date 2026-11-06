#!/usr/bin/env python3
"""Independent, read-only postpublication verification; writes only new evidence."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from reconcile_main_behavior_runtime import fifo_projection
from schedule_jobs import digest_job


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def objects(blob):
    text = blob.decode(); decoder = json.JSONDecoder(); pos = text.index('[') + 1
    order, chunks = [], {}
    while True:
        while text[pos].isspace() or text[pos] == ',':
            pos += 1
        if text[pos] == ']':
            return order, chunks
        job, end = decoder.raw_decode(text, pos)
        if job['id'] in chunks:
            raise ValueError('Duplicate ID')
        order.append(job['id']); chunks[job['id']] = text[pos:end].encode(); pos = end


def main(output):
    wall, cpu = time.perf_counter(), time.process_time()
    if output.exists():
        raise FileExistsError('Immutable output namespace required')
    blobs, proofs = {}, {}
    def read(name, path, wanted=None):
        path = Path(path); blob = path.read_bytes()
        if wanted is not None and sha(blob) != wanted:
            raise ValueError('Pinned input differs: ' + str(path))
        blobs[name] = blob; proofs[name] = {'path': str(path.resolve()), 'sha256': sha(blob), 'bytes': len(blob)}
        return json.loads(blob) if name.endswith('.json') else blob
    authority = read('authority.json', ROOT / 'docs/C_pending_order_ROOT_authority.json',
                     '145a24d4054cf897bd812f1a2300584c839d9d96e6ba158ecf000dc27da64878')
    execution = ROOT / 'artifacts/provenance/C_pending_order_apply_v1/execution'
    applied = read('execution_receipt.json', execution / 'execution_receipt.json',
                   'ba943ecf01e4021a0d8fc5ff4161981787e175135c3737fb105eec515bcce7c6')
    preview = read('preview.json', ROOT / 'results/C_pending_order_preview_v1/preview.json', authority['preview_sha256'])
    for name, proof in applied['proofs'].items():
        read('published_' + name, proof['path'], proof['sha256'])
    read('apply_source.py', applied['producer']['path'], applied['producer']['sha256'])
    freeze = read('formal_freeze.json', ROOT / 'results/extended_formal_freeze.json')
    model = {}
    for phase in freeze['phases']:
        registry = read(phase['model'] + '_registry.json', phase['active_registry'], phase['active_registry_sha256'])
        for row in registry:
            if row['id'] in model:
                raise ValueError('Duplicate formal C ID')
            model[row['id']] = phase['model']
        for key in ('queue', 'snapshot', 'protocol'):
            path = Path(phase[key]); path = path if path.is_absolute() else ROOT / path
            read(phase['model'] + '_' + key + '.json', path, phase[key + '_sha256'])
    assert len(model) == 63 and freeze['formal_runs'] == 63
    queue = read('live_queue.json', ROOT / 'configs/supplemental_v1_queue.json', authority['after_queue_sha256'])
    index = read('live_index.json', ROOT / 'configs/supplemental_v1_registry_index.json',
                 applied['proofs']['before_index.json']['sha256'])
    state = read('live_state.json', ROOT / 'logs/supplemental_job_state.json')
    runtime = read('runtime.json', ROOT / 'results/runtime_budget_latest.json')
    read('master_cmdline.bin', '/proc/2988506/cmdline')
    read('master_proc_stat.txt', '/proc/2988506/stat')
    observer_source = read('observer_source.py', ROOT / 'scripts/monitor_runtime_budget.py', runtime['observer_source_sha256'])
    read('FIFO_source.py', ROOT / 'scripts/reconcile_main_behavior_runtime.py',
         preview['sources']['FIFO_source.py']['sha256'])
    read('scheduler_source.py', ROOT / 'scripts/schedule_jobs.py', preview['sources']['scheduler_source.py']['sha256'])
    read('verification_source.py', Path(__file__).resolve())
    old_order, old_chunks = objects(blobs['published_before_queue.json'])
    new_order, new_chunks = objects(blobs['live_queue.json'])
    before_state = json.loads(blobs['published_before_state.json'])
    assert len(old_order) == len(new_order) == len(queue) == 564 and old_chunks == new_chunks
    assert new_order == preview['candidate_order_proofs']['last4_1b_after_2p8b']['all564_ordered_ids']
    assert blobs['live_queue.json'] == blobs['published_after_queue.json'] == blobs['published_authorized_candidate.json']
    eligible = {jid for jid in model if model[jid] in ('pythia-1b', 'pythia-2.8b')
                and before_state['jobs'][jid]['status'] == 'pending'}
    assert len(eligible) == 37 and all(state['jobs'][jid]['status'] == 'pending' for jid in eligible)
    for position, jid in enumerate(old_order):
        assert jid in eligible or new_order[position] == jid
        if before_state['jobs'][jid]['status'] in ('running', 'completed'):
            assert new_order[position] == jid
        if jid not in model or model[jid] == 'pythia-410m':
            assert new_order[position] == jid
    for tag in set(model.values()):
        assert [jid for jid in old_order if model.get(jid) == tag] == [jid for jid in new_order if model.get(jid) == tag]
    definition_proofs = {row['id']: row for row in preview['all564_definition_proofs']}
    for job in queue:
        jid = job['id']
        assert digest_job(job) == state['jobs'][jid]['definition_sha256'] == definition_proofs[jid]['definition_sha256']
        assert sha(new_chunks[jid]) == definition_proofs[jid]['original_raw_JSON_object_sha256']
    assert blobs['live_index.json'] == blobs['published_before_index.json'] == blobs['published_after_index.json']
    argv = [part.decode() for part in blobs['master_cmdline.bin'].split(b'\0') if part]
    assert sha(blobs['master_cmdline.bin']) == preview['sources']['master_cmdline.bin']['sha256']
    master_pids = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if len(args) > 1 and args[1].endswith(b'/schedule_jobs.py'):
            master_pids.append(int(path.parent.name))
    assert master_pids == [2988506] and state['scheduler_pid'] == index['scheduler_pid'] == 2988506
    actual_start_ticks = blobs['master_proc_stat.txt'].decode().rsplit(')', 1)[1].split()[19]
    preview_stat = Path(ROOT / 'results/C_pending_order_preview_v1/sources/master_proc_stat.txt').read_text()
    assert actual_start_ticks == preview_stat.rsplit(')', 1)[1].split()[19]
    assert blobs['master_proc_stat.txt'].decode().rsplit(')', 1)[1].split()[0] not in ('T', 't', 'Z')
    timestamp = dt.datetime.fromisoformat(runtime['updated_at_utc']); now = dt.datetime.now(dt.timezone.utc)
    assert timestamp > dt.datetime.fromisoformat(applied['confirmed_at_utc']) and 0 <= (now - timestamp).total_seconds() < 90
    assert runtime['status'] == 'monitoring' and not runtime['failed_or_blocked_jobs']
    assert runtime['master_queue_jobs'] == 564 and all(runtime['frozen_scientific_core_matches'].values())
    rows = [row for update in runtime['extended_model_runtime_updates'].values() for row in update['jobs']]
    assert {row['id'] for row in rows} == set(model)
    assert all(state['jobs'][row['id']]['status'] == row['scheduler_status'] for row in rows)
    assert {jid for jid, record in state['jobs'].items() if record['status'] != 'completed'} == {
        row['id'] for row in rows if row['scheduler_status'] != 'completed'}
    assert not any(record['status'] in ('failed', 'blocked') for record in state['jobs'].values())
    assert all(job['expected_seconds'] == 3600 for job in queue if job['id'] in model)
    forecast = fifo_projection(queue, state, [], {}, runtime, runtime['remaining_slot_hours'] * 3600, True)
    assert forecast['status'] == 'complete_time_simulation' and abs(forecast['unassigned_conservative_nonphase_slot_seconds']) < .01
    cutoff = dt.datetime.fromisoformat('2026-11-06T22:30:00+00:00'); stop = dt.datetime.fromisoformat('2026-11-06T23:50:00+00:00')
    future = {row['id']: row for row in forecast['future_dispatch_timing_rows']}
    starts = [{'id': jid, 'model': model[jid], 'basis': 'pending_forecast' if jid in future else 'actual_already_started',
               'start_utc': (timestamp + dt.timedelta(seconds=future[jid]['start_seconds'])).isoformat()
               if jid in future else state['jobs'][jid]['started_at_utc']} for jid in sorted(model)]
    latest = max(starts, key=lambda row: dt.datetime.fromisoformat(row['start_utc']))
    headroom = (cutoff - dt.datetime.fromisoformat(latest['start_utc'])).total_seconds()
    finish = timestamp + dt.timedelta(seconds=forecast['makespan_seconds_with_10pct'])
    assert (ROOT / 'configs/supplemental_v1_queue.json').read_bytes() == blobs['live_queue.json']
    assert (ROOT / 'configs/supplemental_v1_registry_index.json').read_bytes() == blobs['live_index.json']
    result = {'schema_version': 1, 'status': 'passed_independent_read_only_postpublication_verification', 'utc': now.isoformat(),
        'authority': proofs['authority.json'], 'execution_receipt': proofs['execution_receipt.json'], 'source_proofs': proofs,
        'all564_raw_JSON_objects_byte_identical': True, 'all564_definition_hashes_identical': True, 'index_byte_identical': True,
        'all_noneligible_completed_running_and_C410_positions_unchanged': True, 'all_within_model_relative_orders_unchanged': True,
        'exact_authorized_last4_candidate_published': True, 'all37_moved_domain_jobs_still_pending': True,
        'formal_C_ID_source': 'Three exact-SHA registries in extended_formal_freeze; no prefix matching',
        'formal_C_status_counts_by_model': {tag: {status: sum(model[jid] == tag and state['jobs'][jid]['status'] == status for jid in model)
                                               for status in ('completed', 'running', 'pending')} for tag in sorted(set(model.values()))},
        'single_master_pid': master_pids[0], 'master_argv': argv, 'master_start_ticks_unchanged': True,
        'runtime_observer': {'updated_at_utc': timestamp.isoformat(), 'healthy_after_publication': True,
            'reads_live_master_queue_every_refresh': b"queue=read(ROOT/'configs/supplemental_v1_queue.json')" in observer_source,
            'emits_order_or_queue_SHA': False, 'aggregate_slot_forecast_order_agnostic': True,
            'order_effect_independently_evaluated_below_not_claimed_from_observer_aggregate': True},
        'new_C_only_FIFO_admission_forecast': {'runtime_source_utc': timestamp.isoformat(), 'all63_start_rows': starts,
            'last_training_start': latest, 'launch_headroom_to2230_seconds': headroom, 'strict_launch_admission_satisfied': headroom > 0,
            'finish_utc': finish.isoformat(), 'finish_margin_before2350_seconds': (stop - finish).total_seconds(),
            'all_remaining_durations_receive10percent_once': True, 'FIFO': forecast,
            'early_evaluation_considered_or_authorized': False, 'forecast_not_guarantee': True},
        'limitations': ['C410 completion growth since publication is normal; all formal63 IDs bound to frozen registries.',
            'Original observer refreshes actual queue but its aggregate slot/4 forecast does not quantify FIFO order effects.',
            'This receipt authorizes no new early work; existing ROOT early-work READY/admission/finish/tail gates remain.',
            'No source/index/state/queue modified; evidence files only.'],
        'wall_seconds': time.perf_counter() - wall, 'process_cpu_seconds': time.process_time() - cpu,
        'scores_read': False, 'new_GPU_scheduler_or_watcher_started': False}
    output.mkdir(); (output / 'sources').mkdir()
    for name, blob in blobs.items():
        with (output / 'sources' / name).open('xb') as stream:
            stream.write(blob); stream.flush(); os.fsync(stream.fileno())
    target = output / 'receipt.json'
    with target.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    for folder in (output / 'sources', output):
        fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    print(json.dumps({'receipt': str(target), 'sha256': sha(target.read_bytes()), 'last_start': latest['start_utc'],
                      'launch_headroom_minutes': headroom / 60, 'finish_utc': finish.isoformat(),
                      'finish_margin_before2350_minutes': (stop - finish).total_seconds() / 60,
                      'formal_C_counts': result['formal_C_status_counts_by_model']}))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: verify.py NEW_OUTPUT_DIRECTORY')
    main(Path(sys.argv[1]).resolve())
