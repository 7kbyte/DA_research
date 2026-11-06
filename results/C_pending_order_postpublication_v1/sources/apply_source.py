#!/usr/bin/env python3
"""Apply the explicitly accepted pending-only order, preserving every job byte."""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_C_runtime_order_review import job_segments
from apply_C_runtime_order import master_pids
from schedule_jobs import digest_job, validate_queue
from stat_bound_digest_cache import atomic, fsync_directory, small_proof

PREVIEW_SHA = '24b25d5ba2418f825ee441aa08c22a2514d102a7d5008e1bd77a2ac577283502'
BEFORE_SHA = 'f9a05e3687c6d0c282180fa0f142770373120c92801087bc26b08128b6f56c42'
AFTER_SHA = 'b4a2c80b1d0ced8f47aa4bb79c41ba14594a7c446dd2ff2c8acfe93eb04edf99'
PID = 2988506

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def check(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    if sys.argv[1:] != ['--execute-root-authorized-order']:
        raise ValueError('Explicit root execution argument required')
    started = time.perf_counter(); cpu = time.process_time()
    preview_path = ROOT / 'results/C_pending_order_preview_v1/preview.json'
    check(small_proof(preview_path)['sha256'] == PREVIEW_SHA, 'Preview differs')
    preview = json.loads(preview_path.read_bytes())
    authority_path = ROOT / 'docs/C_pending_order_ROOT_authority.json'
    authority = json.loads(authority_path.read_bytes())
    check(authority['authorized'] is True and authority['candidate'] == 'last4_1b_after_2p8b', 'Wrong authority')
    check((authority['preview_sha256'], authority['before_queue_sha256'], authority['after_queue_sha256']) == (PREVIEW_SHA, BEFORE_SHA, AFTER_SHA), 'Authority digest differs')
    check(authority['producer_sha256'] == small_proof(__file__)['sha256'], 'Producer differs')
    check(authority['early_NLL_authorized'] is False, 'Order-only authority required')
    folder = Path(__file__).parent / 'execution'
    check(not folder.exists(), 'Execution evidence must use a new namespace')
    source = ROOT / 'configs/supplemental_v1_queue.json'
    index = ROOT / 'configs/supplemental_v1_registry_index.json'
    state_path = ROOT / 'logs/supplemental_job_state.json'
    with source.with_suffix('.mutation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = source.read_bytes(); index_before = index.read_bytes()
        state_raw = state_path.read_bytes(); state = json.loads(state_raw)
        check(sha(before) == BEFORE_SHA, 'Live source changed')
        old, segments = job_segments(before)
        new = preview['candidate_order_proofs']['last4_1b_after_2p8b']['all564_ordered_ids']
        check(len(old) == len(new) == len(set(new)) == 564 and set(old) == set(new), 'IDs differ')
        candidate_raw = b'[\n' + b',\n'.join(segments[n] for n in new) + b'\n]\n'
        check(sha(candidate_raw) == AFTER_SHA, 'Candidate bytes differ')
        candidate = validate_queue(json.loads(candidate_raw))
        new_order, new_segments = job_segments(candidate_raw)
        check(new_order == new and new_segments == segments, 'Raw job bytes differ')
        definitions = {r['id']: r for r in preview['all564_definition_proofs']}
        for job in candidate:
            name = job['id']; digest = digest_job(job)
            check(digest == definitions[name]['definition_sha256'] == state['jobs'][name]['definition_sha256'], 'Job definition differs: ' + name)
            check(sha(segments[name]) == definitions[name]['original_raw_JSON_object_sha256'], 'Original job bytes differ')
        eligible = {n for n in old if n.startswith(('extended_v2_pythia_1b_m32_formal_', 'extended_v2_pythia_2p8b_m8_h25m_formal_')) and state['jobs'][n]['status'] == 'pending'}
        check(len(eligible) == 37, 'Expected37 pending candidate jobs')
        for i, name in enumerate(old):
            check(name in eligible or new[i] == name, 'A fixed position changed')
        for prefix in ('extended_v2_pythia_410m_m64_formal_', 'extended_v2_pythia_1b_m32_formal_', 'extended_v2_pythia_2p8b_m8_h25m_formal_'):
            check([n for n in old if n.startswith(prefix)] == [n for n in new if n.startswith(prefix)], 'Within-model order differs')
        running = state['active_jobs']
        check(state['scheduler_pid'] == PID and master_pids() == [PID], 'Master identity differs')
        check(len(running) == 4 and all(n.startswith('extended_v2_pythia_410m_m64_formal_') for n in running), 'Four C410 workers required as lead-in')
        check(sum(n.startswith('extended_v2_pythia_410m_m64_formal_') and state['jobs'][n]['status'] == 'pending' for n in old) >= 4, 'Insufficient untouched C410 lead-in')
        check(sha(Path(f'/proc/{PID}/cmdline').read_bytes()) == preview['sources']['master_cmdline.bin']['sha256'], 'Master argv differs')
        for path, entry in json.loads((ROOT / 'artifacts/provenance/formal_deterministic_repository_freeze.json').read_bytes())['core_sources'].items():
            check(small_proof(ROOT / path)['sha256'] == entry['current'], 'Scientific core changed')
        execution = json.loads((ROOT / 'docs/supplemental_evaluation_preparation_v5_execution.json').read_bytes())
        for path, wanted in {**execution['sources'], **execution['frozen_scorer_input_sha256']}.items():
            check(small_proof(path)['sha256'] == wanted, 'Frozen evaluation source differs')
        folder.mkdir()
        for name, raw in [('before_queue.json', before), ('before_index.json', index_before), ('before_state.json', state_raw), ('authorized_candidate.json', candidate_raw)]:
            with (folder / name).open('xb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        fsync_directory(folder)
        temp = source.with_name(source.name + f'.root_pending_order_{os.getpid()}.tmp')
        with temp.open('xb') as stream:
            os.chmod(temp, source.stat().st_mode & 0o777)
            stream.write(candidate_raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, source); fsync_directory(source.parent)
        published = dt.datetime.now(dt.timezone.utc).isoformat()
    end = time.monotonic() + 40
    while True:
        after_raw = state_path.read_bytes(); after = json.loads(after_raw)
        if after['updated_at_utc'] != state['updated_at_utc']:
            break
        if time.monotonic() > end:
            raise RuntimeError('Order published; reload confirmation timed out, inspect without retrying mutation')
        time.sleep(1)
    check(source.read_bytes() == candidate_raw and index.read_bytes() == index_before, 'Queue/index changed after publication')
    check(after['scheduler_pid'] == PID and master_pids() == [PID], 'Master identity changed')
    check(all(after['jobs'][n]['status'] == 'pending' for n in eligible), 'A moved job dispatched before confirmation')
    check(all(after['jobs'][j['id']]['definition_sha256'] == digest_job(j) for j in candidate), 'Reloaded definition differs')
    check(not any(after['jobs'][n]['status'] in ('failed', 'blocked') for n in new), 'Active queue contains failure')
    for name, raw in [('after_queue.json', source.read_bytes()), ('after_index.json', index.read_bytes()), ('after_state.json', after_raw)]:
        with (folder / name).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fsync_directory(folder)
    record = {'status': 'published_reload_confirmed', 'published_at_utc': published, 'confirmed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'master_pid_before_and_after': PID,
        'authority': small_proof(authority_path), 'preview': small_proof(preview_path), 'producer': small_proof(__file__),
        'proofs': {p.name: small_proof(p) for p in folder.iterdir()}, 'all564_original_objects_and_definitions_identical': True,
        'all_noneligible_positions_unchanged': True, 'all_three_within_model_orders_unchanged': True, 'all37_candidate_jobs_still_pending_after_reload': True,
        'index_unchanged': True, 'single_master_unchanged': True, 'scientific_and_evaluation_sources_unchanged': True,
        'scheduler_or_children_paused_or_restarted': False, 'job_arguments_or_expected_seconds_changed': False,
        'new_GPU_work_or_early_NLL_authorized': False, 'scores_read': False, 'wall_seconds': time.perf_counter() - started, 'cpu_seconds': time.process_time() - cpu}
    atomic(folder / 'execution_receipt.json', record)
    print(json.dumps({'receipt': small_proof(folder / 'execution_receipt.json'), 'status': record['status'], 'after_queue_sha256': AFTER_SHA}))

if __name__ == '__main__':
    main()
