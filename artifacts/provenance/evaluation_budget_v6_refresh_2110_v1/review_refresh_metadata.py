#!/usr/bin/env python3
"""Describe a frozen producer output and actual argv admission; no scoring reads."""
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from schedule_jobs import digest_job

proofs = {}
def read(path, expected=None, binary=False):
    path = Path(path)
    raw = path.read_bytes()
    assert len(raw) < 32 << 20
    digest = hashlib.sha256(raw).hexdigest()
    assert expected is None or digest == expected
    proofs[str(path.resolve())] = {'path': str(path.resolve()), 'sha256': digest, 'bytes': len(raw)}
    return raw if binary else json.loads(raw)

def output(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')

utc = lambda text: dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
new_path = ROOT / 'results/evaluation_timing_budget_v6_refresh_2110.json'
old_path = ROOT / 'results/evaluation_timing_budget_v6_refresh_1925.json'
new = read(new_path)
old = read(old_path, '2b505e58310d7cef34d70a7fe367b4845150d921754a913319fccccbbf09e333')
read(ROOT / 'scripts/describe_remaining_evaluation_budget_v6.py',
     'd02d430869f5c6f91f71496b2b77ea5e0388386ed432bdcec22240270c706a50', True)
for value in (old, new):
    assert value['producer_source']['sha256'] == proofs[str((ROOT / 'scripts/describe_remaining_evaluation_budget_v6.py').resolve())]['sha256']
assert old['accounting'] == new['accounting']
assert new['accounting']['remaining_queue_jobs'] == 224
assert new['accounting']['other_C_unproved_alias_discounts'] == 0
comparisons = []
for a, b in zip(old['sensitivity_scenarios'], new['sensitivity_scenarios']):
    assert a['name'] == b['name']
    for key in ('per_phase', 'four_slot_dispatch', 'whole_wall_sensitivity_factor',
                'wall_seconds_with_dispatch_tail_and_handoff_proxy',
                'handoff_preflight_allowance_seconds_unmeasured'):
        assert a[key] == b[key]
    comparisons.append({'name': b['name'], 'unchanged_remaining_evaluation_seconds': b['wall_seconds_with_dispatch_tail_and_handoff_proxy'],
        'old_finish_utc': a['from_current_training_forecast']['evaluation_finish_utc'],
        'new_finish_utc': b['from_current_training_forecast']['evaluation_finish_utc'],
        'forecast_finish_change_seconds': (utc(b['from_current_training_forecast']['evaluation_finish_utc']) - utc(a['from_current_training_forecast']['evaluation_finish_utc'])).total_seconds(),
        'new_margin_before_025632_minutes': b['from_current_training_forecast']['margin_before_30min_delivery_tail_minutes']})

sources = HERE / 'sources'
queue = read(sources / 'master_queue.json', new['source_proofs']['master_queue.json']['sha256'])
state = read(sources / 'scheduler_state.json', new['source_proofs']['scheduler_state.json']['sha256'])
runtime = read(sources / 'runtime.json', new['source_proofs']['runtime.json']['sha256'])
lookup = {job['id']: job for job in queue}
assert len(queue) == len(lookup) == len(state['jobs']) == 607
assert runtime['master_status_counts'] == dict(collections.Counter(row['status'] for row in state['jobs'].values()))
for job in queue:
    record = state['jobs'][job['id']]
    assert digest_job(job) == record['definition_sha256']
    assert not any(key.startswith('stop_requested') for key in record)
    if record['status'] == 'completed':
        assert type(record['returncode']) is int and record['returncode'] == 0 and not record.get('missing_outputs')
    else:
        assert record['status'] in ('pending', 'running')
    if 'argv' in record:
        assert record['argv'] == job['argv']

scheduler_argv = read(Path('/proc') / str(state['scheduler_pid']) / 'cmdline', binary=True).rstrip(b'\0').decode().split('\0')
assert '--exit-when-idle' not in scheduler_argv
assert '--launch-margin-seconds' not in scheduler_argv and '--stop-margin-seconds' not in scheduler_argv
assert scheduler_argv[scheduler_argv.index('--deadline') + 1] == '2026-11-07T00:00:00Z'
read(ROOT / 'scripts/schedule_jobs.py', 'c5c264b058ff655315ef896ccb0a89b471637c6612013a51999fc6f48b2d16d9', True)
read(ROOT / 'scripts/handoff_evaluation.py', binary=True)
read(ROOT / 'scripts/prepare_supplemental_evaluations.py', binary=True)

base = utc(new['training_forecast']['updated_at_utc'])
pending = {key for key, record in state['jobs'].items() if record['status'] == 'pending'}
projection = new['training_forecast']['complete_C_only_FIFO']
assert projection['all_remaining_job_durations_receive_10pct_once'] is True
starts = projection['future_dispatch_timing_rows']
assert len(starts) == len(pending) and {row['id'] for row in starts} == pending
cutoff = utc('2026-11-06T23:30:00Z')  # midnight minus original1800s launch margin
admission = []
for row in starts:
    job = lookup[row['id']]
    start = base + dt.timedelta(seconds=row['start_seconds'])
    allowance = job['expected_seconds']
    strict_latest = cutoff - dt.timedelta(seconds=allowance)
    assert allowance == 3600 and strict_latest == utc('2026-11-06T22:30:00Z')
    assert start < strict_latest
    config_path = Path(job['argv'][job['argv'].index('--config') + 1])
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    read(config_path)
    admission.append({'id': row['id'], 'argv': job['argv'], 'definition_sha256': digest_job(job),
        'expected_seconds': allowance, 'forecast_start_utc': start.isoformat(),
        'strict_latest_launch_utc': strict_latest.isoformat(),
        'strict_launch_headroom_seconds_proxy': (strict_latest - start).total_seconds(),
        'admitted_under_time_forecast_only': True})

# Whole actual source snapshots, without reading any inference output or checkpoint.
ready_bytes = read(ROOT / 'artifacts/evaluation_ready_v5/ready_index.json', binary=True)
ready = json.loads(ready_bytes)
with (HERE / 'ready_index_observed.json').open('xb') as stream: stream.write(ready_bytes)
cache_bytes = read(ROOT / 'artifacts/evaluation_ready_v5/cache_progress.json', binary=True)
cache = json.loads(cache_bytes)
with (HERE / 'cache_progress_observed.json').open('xb') as stream: stream.write(cache_bytes)
cache_snapshot = cache['cache_snapshot']
snapshot_bytes = read(cache_snapshot['path'], cache_snapshot['sha256'], True)
with (HERE / 'immutable_cache_snapshot_observed.json').open('xb') as stream: stream.write(snapshot_bytes)
audit_exists = (ROOT / 'results/extended_run_audit.json').exists()

future_groups = []
for path, expected, count in [
    (ROOT / 'configs/primary_evaluation_v2_execution_matched_behavior_queue.json', 300, 66),
    (ROOT / 'artifacts/evaluation_ready_v5/supplement_A_behavior/queue.json', 600, 18),
    (ROOT / 'artifacts/evaluation_ready_v5/extended_pythia_410m_100m_behavior/queue.json', 600, 21),
]:
    jobs = read(path)
    jobs = [job for job in jobs if job['id'] not in lookup]
    assert len(jobs) == count and {job['expected_seconds'] for job in jobs} == {expected}
    future_groups.append({'source_queue': str(path), 'actual_unenqueued_jobs': len(jobs),
        'expected_seconds': expected, 'strict_latest_launch_utc': (utc('2026-11-07T02:49:42Z') - dt.timedelta(seconds=expected)).isoformat(),
        'jobs': [{'id': job['id'], 'argv': job['argv'], 'definition_sha256': digest_job(job)} for job in jobs]})

read(__file__, binary=True)
result = {'status': 'score_blind_original_producer_refresh_and_admission_description_only',
    'observed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'new_forecast': proofs[str(new_path.resolve())], 'comparison_baseline': proofs[str(old_path.resolve())],
    'unchanged_accounting': new['accounting'], 'low_central_high_comparison': comparisons,
    'old_training_finish_utc': old['training_forecast']['finish_utc'],
    'new_training_finish_utc': new['training_forecast']['finish_utc'],
    'master_state_counts': dict(collections.Counter(row['status'] for row in state['jobs'].values())),
    'all_607_definition_bindings_and_completed_strict_integer_rc0_verified': True,
    'scheduler_argv': scheduler_argv, 'pending_actual_argv_admission': admission,
    'latest_pending_forecast_start_utc': max((row['forecast_start_utc'] for row in admission), default=None),
    'minimum_strict_2230_launch_headroom_seconds_proxy': min((row['strict_launch_headroom_seconds_proxy'] for row in admission), default=None),
    'future_actual_queues': future_groups,
    'future_not_READY_NLL_and_behavior': {'NLL_expected_seconds_from_unchanged_builder': 180,
        'behavior_expected_seconds_from_unchanged_builder': 600,
        'not_actual_final_queues_until_READY': True, 'no_future_alias_discount': True},
    'protected_deadlines': {'hard_deadline_utc': '2026-11-07T03:26:32Z',
        'protected_delivery_tail_begins_utc': '2026-11-07T02:56:32Z',
        'successor_stop_margin_seconds': 1910, 'successor_launch_margin_seconds': 2210,
        'successor_TERM_window_begins_utc': '2026-11-07T02:54:42Z',
        'strict_start_plus_expected_seconds_before_utc': '2026-11-07T02:49:42Z',
        'strict_latest_start_by_expected_seconds': {'180': '2026-11-07T02:46:42Z',
            '300': '2026-11-07T02:44:42Z', '600': '2026-11-07T02:39:42Z'}},
    'current_handoff_not_ready': {'ready_phase_count': ready['ready_phase_count'],
        'all_requested_phases_ready': ready['all_requested_phases_ready'],
        'all63_audit_file_exists': audit_exists, 'captured_immutable_cache_snapshot': cache_snapshot,
        'captured_cache_is_not_final_handoff_cache': True,
        'required_later': ['all607_completed_rc0_no_stop', 'all63audit_passed', 'all9READY',
            'execution_time_latest_immutable_cache_and_current_stats', 'active_jobs_empty',
            'ROOT_terminate_old_idle_master_then_confirm_all_PIDs_PGroups_exit_and_lock_release']},
    'interpretation': ['Training10pct is applied exactly once by original FIFO producer.',
        'All low/central/high remaining workloads, rates, sensitivity factors and dispatch tails equal19:25.',
        'Evaluation includes existing whole-slot/scorer/inputverification cost once; nested timers are not added.',
        '300s handoff preflight is an unmeasured allowance, not full audit/statistics/publication guarantee.',
        'Forward forecast is LPT proxy, not actual final successor FIFO/admission, success, or launch authorization.',
        'Actual old master is idle-persistent; forecast completion does not terminate it.',
        'No score/prediction parsing, GPU, source/queue/state/margin mutation, or payload read by this description.'],
    'source_proofs': list(proofs.values())}
output('comparison_and_admission.json', result)
with (HERE / 'pending_admission.csv').open('x', newline='') as stream:
    fields = ['id', 'expected_seconds', 'forecast_start_utc', 'strict_latest_launch_utc', 'strict_launch_headroom_seconds_proxy', 'definition_sha256', 'argv']
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
    for row in admission:
        writer.writerow({key: json.dumps(row[key]) if key == 'argv' else row[key] for key in fields})
print(json.dumps({'comparison_and_admission_sha256': hashlib.sha256((HERE/'comparison_and_admission.json').read_bytes()).hexdigest(),
    'latest_pending_forecast_start_utc': result['latest_pending_forecast_start_utc'],
    'minimum_2230_headroom_seconds': result['minimum_strict_2230_launch_headroom_seconds_proxy'],
    'ready_phase_count': ready['ready_phase_count'], 'all63_audit_exists': audit_exists}))
