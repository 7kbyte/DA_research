import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import reconcile_main_behavior_runtime as module
from handoff_evaluation import paths_for_output

NOW = dt.datetime(2026, 11, 6, 11, tzinfo=dt.timezone.utc)


def fixture():
    jobs = [{'id': 'b' + str(i), 'expected_seconds': 300} for i in range(66)]
    rows = [{'id': job['id'], 'status': 'not_enqueued', 'elapsed_seconds': 0.} for job in jobs]
    raw = {'requested_matrix': [{'phase': 'other_queued_jobs', 'job_ids': ['A_unrelated'],
                               'remaining_gpu_seconds': 180}], 'remaining_slot_hours': 10000 / 3600}
    budget = {'sensitivity_scenarios': []}
    for name, rate in [('low', 100.), ('central', 125.), ('high', 150.)]:
        budget['sensitivity_scenarios'].append({'name': name, 'per_phase': [{'id': 'main_behavior_adapted',
            'remaining_forward_jobs': 66, 'materialization_jobs': 0, 'per_job_whole_slot_seconds_proxy': rate}]})
    return jobs, rows, raw, {'whole_slot_seconds': 100.}, budget


def test_hypothetical_keeps_A_and_does_not_charge_completed_baseline():
    jobs, rows, raw, baseline, budget = fixture()
    result = module.calculate(raw, jobs, rows, baseline, budget, NOW)
    assert result['mode'] == 'hypothetical_complete_main66_insertion'
    assert result['raw_generic_main66_slot_seconds_removed_once'] == 0
    assert result['non_main66_remaining_slot_seconds_kept_from_raw'] == pytest.approx(10000)
    assert result['raw_other_non_main66_job_ids_kept'] == ['A_unrelated']
    assert result['A126_generic_or_completed_cost_removed_by_this_tool'] == 0
    assert result['baseline_charged_to_phase_seconds'] == 0
    assert result['sensitivity_scenarios'][0]['remaining_main66_slot_seconds_proxy'] == 6600


def test_exact300_charge_and_running_elapsed_subtracted_once_capped_nonnegative():
    jobs, rows, raw, baseline, budget = fixture()
    for row in rows: row['status'] = 'pending'
    rows[0].update(status='running', elapsed_seconds=40.)
    raw['requested_matrix'][0].update(job_ids=['A_unrelated', *[job['id'] for job in jobs]], remaining_gpu_seconds=66 * 300 + 180)
    raw['remaining_slot_hours'] = (10000 + 66 * 300) / 3600
    result = module.calculate(raw, jobs, rows, baseline, budget, NOW)
    assert result['mode'] == 'exact_remaining_main66_reconciliation'
    assert result['raw_generic_main66_slot_seconds_removed_once'] == 19800
    assert result['non_main66_remaining_slot_seconds_kept_from_raw'] == pytest.approx(10000)
    case = result['sensitivity_scenarios'][0]
    assert case['remaining_main66_slot_seconds_proxy'] == 6560
    assert case['running_elapsed_seconds_credited_once_capped_per_job'] == 40
    rows[0]['elapsed_seconds'] = 200.
    case = module.calculate(raw, jobs, rows, baseline, budget, NOW)['sensitivity_scenarios'][0]
    assert case['remaining_main66_slot_seconds_proxy'] == 6500
    assert case['running_beyond_proxy_ids_unknown_extra_remaining'] == ['b0']
    assert not case['fits_before_2350_under_timing_proxy']


def test_only_verified_same_definition_completed_work_leaves_final_tail_budget():
    jobs, rows, raw, baseline, budget = fixture()
    for row in rows: row['status'] = 'pending'
    rows[0].update(status='completed', elapsed_seconds=95.)
    raw['requested_matrix'][0].update(job_ids=[job['id'] for job in jobs], remaining_gpu_seconds=66 * 300)
    raw['remaining_slot_hours'] = (10000 + 66 * 300) / 3600
    result = module.calculate(raw, jobs, rows, baseline, budget, NOW)
    case = result['sensitivity_scenarios'][0]
    assert result['main66_completed_actual_slot_seconds_once'] == 95
    assert case['remaining_main66_slot_seconds_proxy'] == 6500
    assert case['completed_same_definition_future_slot_seconds_removed_at_v4_proxy'] == 100
    assert case['main66_still_in_final_evaluation_backlog_slot_seconds_at_v4_proxy'] == 6500
    assert result['same_work_moved_earlier_total_compute_saved_seconds'] == 0


def test_full_FIFO_actual_running_slots_and_long_job_tail():
    running = [{'id': 'c' + str(i), 'expected_seconds': 999} for i in range(4)]
    pending = [{'id': 'c4'}, {'id': 'c5'}]
    phase = [{'id': 'b', 'depends_on': ['baseline']}]
    state = {'jobs': {'baseline': {'status': 'completed', 'returncode': 0}}}
    state['jobs'].update({job['id']: {'status': 'running', 'gpu': str(i)} for i, job in enumerate(running)})
    state['jobs'].update({job['id']: {'status': 'pending'} for job in pending})
    costs = [100., 200., 300., 400., 500., 600.]
    raw = {'extended_model_runtime_updates': {'model': {'jobs': [{'id': 'c' + str(i), 'total_seconds': cost} for i, cost in enumerate(costs)]}}}
    result = module.fifo_projection(running + pending, state, phase, {'b': 50.}, raw, sum(costs), True)
    assert result['status'] == 'complete_time_simulation'
    first = result['future_dispatch_timing_rows'][0]
    assert first['id'] == 'b' and first['start_seconds'] == pytest.approx(110.)
    assert result['longrun_FIFO_imbalance_seconds_above_ideal'] > 0
    assert result['makespan_seconds_with_10pct'] > (sum(costs) + 50) * 1.1 / 4


def test_C_only_excludes_zero_time_behavior_placeholders_without_changing_makespan():
    master = [{'id': 'c0'}, {'id': 'c1'}]
    state = {'jobs': {'baseline': {'status': 'completed', 'returncode': 0},
                      'c0': {'status': 'running', 'gpu': '0'}, 'c1': {'status': 'pending'}}}
    raw = {'extended_model_runtime_updates': {'model': {'jobs': [
        {'id': 'c0', 'total_seconds': 200.}, {'id': 'c1', 'total_seconds': 100.}]}}}
    placeholders = [{'id': 'behavior' + str(index), 'depends_on': ['baseline']} for index in range(66)]
    old = module.fifo_projection(master, state, placeholders,
                                 {job['id']: 0. for job in placeholders}, raw, 300., True)
    current = module.fifo_projection(master, state, [], {}, raw, 300., True)
    assert old['makespan_seconds_with_10pct'] == current['makespan_seconds_with_10pct']
    assert len([row for row in old['future_dispatch_timing_rows'] if row['id'].startswith('behavior')]) == 66
    assert not [row for row in current['future_dispatch_timing_rows'] if row['id'].startswith('behavior')]
    assert all(row['id'].startswith('c') for row in current['future_dispatch_timing_rows'])


def frozen_phase(tmp_path):
    group = {'tasks': {'toy': {}}}
    entries, jobs = [], []
    manifest_sha = 'f' * 64
    manifest_path = tmp_path / 'manifest.json'
    for index in range(67):
        jid = 'initial' if index == 0 else 'adapted' + str(index)
        output = tmp_path / (jid + '.json')
        entry = {'id': jid, 'kind': 'initial' if index == 0 else 'adapted', 'baseline_group': 'g',
                 'output': str(output), 'argv': [sys.executable, str(module.ROOT / 'scripts/evaluate_tasks.py')], 'input_sha256': {}}
        if index: entry.update(depends_on=['initial'], checkpoint_sha256='a' * 64)
        entries.append(entry)
        job = {'id': jid, 'argv': [sys.executable, str(module.ROOT / 'scripts/run_frozen_evaluation.py'),
                '--manifest', str(manifest_path), '--manifest-sha256', manifest_sha, '--job-id', jid, '--device', 'cuda:0'],
               'expected_seconds': 300, 'env': {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'},
               'outputs': sorted(paths_for_output(output, 'behavior', group))}
        if index: job['depends_on'] = ['initial']
        jobs.append(job)
    common = {str(module.ROOT / ('scripts/' + name)): hashlib.sha256((module.ROOT / ('scripts/' + name)).read_bytes()).hexdigest()
              for name in ['run_frozen_evaluation.py', 'evaluate_tasks.py']}
    manifest = {'mode': 'behavior', 'jobs': entries, 'records': [], 'baseline_groups': {'g': group}, 'common_input_sha256': common}
    def complete(index):
        entry = entries[index]
        output = Path(entry['output'])
        paths = [output, output.with_name(output.stem + '_toy.jsonl')]
        for path in paths: path.write_bytes(b'SCORES_NOT_JSON_DO_NOT_PARSE')
        receipt = {'manifest_sha256': manifest_sha, 'job_id': entry['id'], 'verified_frozen_inputs': True,
                   'checkpoint_sha256': entry.get('checkpoint_sha256'), 'inference_alias': False,
                   'actual_argv': entry['argv'], 'output_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
        output.with_name(output.stem + '_execution.json').write_text(json.dumps(receipt))
        return {'status': 'completed', 'returncode': 0, 'definition_sha256': module.digest_job(jobs[index]), 'elapsed_seconds': 100.}
    state = {'jobs': {'initial': complete(0)}}
    return jobs, manifest, manifest_path, manifest_sha, state, complete


def test_baseline_full_SHA_and_exact66_definitions_without_score_parse(tmp_path):
    jobs, manifest, path, sha, state, complete = frozen_phase(tmp_path)
    phase, rows, baseline = module.inspect_phase(jobs, manifest, path, sha, [jobs[0]], state, NOW)
    assert len(phase) == 66 and baseline['whole_slot_seconds'] == 100
    assert all(row['status'] == 'not_enqueued' for row in rows)
    Path(manifest['jobs'][0]['output']).write_bytes(b'CHANGED')
    with pytest.raises(ValueError, match='changed after execution'):
        module.inspect_phase(jobs, manifest, path, sha, [jobs[0]], state, NOW)


def test_completed_adapted_missing_outputs_and_changed_definition_rejected(tmp_path):
    jobs, manifest, path, sha, state, complete = frozen_phase(tmp_path)
    for job in jobs[1:]: state['jobs'][job['id']] = {'status': 'pending', 'definition_sha256': module.digest_job(job)}
    state['jobs'][jobs[1]['id']] = complete(1)
    phase, rows, _ = module.inspect_phase(jobs, manifest, path, sha, jobs, state, NOW)
    assert rows[0]['status'] == 'completed' and rows[0]['attestation_sha256']
    Path(manifest['jobs'][1]['output']).unlink()
    with pytest.raises(ValueError, match='Missing completed output'):
        module.inspect_phase(jobs, manifest, path, sha, jobs, state, NOW)
    complete(1)
    changed = copy.deepcopy(jobs)
    changed[1]['expected_seconds'] = 180
    with pytest.raises(ValueError, match='definition differs'):
        module.inspect_phase(jobs, manifest, path, sha, changed, state, NOW)
