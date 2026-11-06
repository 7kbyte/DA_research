import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import describe_remaining_evaluation_budget_v5 as module


def phase_fixture(tmp_path):
    manifest_sha = 'f' * 64
    manifest = {'mode': 'nll', 'baseline_groups': {'g': {'tasks': {}}}}
    queue, bindings, records = [], {}, {}
    for index in range(2):
        identity = 'job' + str(index)
        output = tmp_path / (identity + '.json')
        output.write_bytes(b'OPAQUE_NON_JSON_SCORE_BYTES')
        execution = output.with_name(output.stem + '_execution.json')
        entry = {'id': identity, 'kind': 'adapted', 'actual_run_id': 'run' + str(index),
                 'checkpoint_sha256': 'c' * 64, 'baseline_group': 'g', 'output': str(output),
                 'argv': ['frozen_scorer', identity]}
        execution.write_text(json.dumps({'manifest_sha256': manifest_sha, 'job_id': identity,
            'verified_frozen_inputs': True, 'checkpoint_sha256': entry['checkpoint_sha256'],
            'inference_alias': False, 'actual_argv': entry['argv'],
            'output_sha256': {str(output): hashlib.sha256(output.read_bytes()).hexdigest()},
            'inference_subprocess_wall_seconds': 25., 'verification_wall_seconds': 3.}))
        job = {'id': identity, 'outputs': [str(output), str(execution)]}
        queue.append(job)
        bindings[identity] = {'manifest': manifest, 'entry': entry, 'manifest_sha256': manifest_sha, 'records': []}
        records[identity] = {'status': 'completed', 'returncode': 0,
            'definition_sha256': module.digest_job(job), 'elapsed_seconds': 40.,
            'started_at_utc': '2026-11-06T11:30:00+00:00', 'ended_at_utc': '2026-11-06T11:30:40+00:00'}
    return queue, bindings, {'jobs': records}


def test_partial_phase_does_not_read_any_attestation_or_output(tmp_path, monkeypatch):
    queue, bindings, state = phase_fixture(tmp_path)
    state['jobs'][queue[1]['id']]['status'] = 'running'
    def forbidden(*args):
        raise AssertionError('No output verification before full phase gate')
    monkeypatch.setattr(module, 'completion_proof', forbidden)
    with pytest.raises(module.WaitForGate, match='not fully complete'):
        module.complete_phase(queue, bindings, queue, state)


def test_complete_phase_hashes_opaque_scores_and_counts_whole_slot_once(tmp_path):
    queue, bindings, state = phase_fixture(tmp_path)
    rows = module.complete_phase(queue, bindings, queue, state)
    assert len(rows) == 2
    assert sum(row['whole_slot_seconds'] for row in rows) == 80.
    assert sum(row['scorer_subprocess_wall_seconds_nested'] for row in rows) == 50.
    assert all(row['execution_attestation_sha256'] for row in rows)


def test_missing_changed_failed_or_definition_mismatched_completion_rejected(tmp_path):
    queue, bindings, state = phase_fixture(tmp_path)
    bad = copy.deepcopy(state)
    bad['jobs'][queue[0]['id']]['returncode'] = 1
    with pytest.raises(ValueError, match='not successful'):
        module.complete_phase(queue, bindings, queue, bad)
    changed = copy.deepcopy(queue)
    changed[0]['expected_seconds'] = 999
    with pytest.raises(ValueError, match='definition differs'):
        module.complete_phase(queue, bindings, changed, state)
    output = Path(queue[0]['outputs'][0])
    output.write_bytes(b'CHANGED_OPAQUE_BYTES')
    with pytest.raises(ValueError, match='changed after execution'):
        module.complete_phase(queue, bindings, queue, state)
    output.unlink()
    with pytest.raises(ValueError, match='Missing completed output'):
        module.complete_phase(queue, bindings, queue, state)


def future_fixture():
    primary = {'jobs': [{'id': 'main' + str(index), 'kind': 'adapted'} for index in range(66)]}
    A = {'jobs': [{'id': 'A' + str(index), 'kind': 'adapted'} for index in range(15)] +
         [{'id': 'alias' + str(index), 'kind': 'reuse', 'inference_origin': {'job_id': 'main' + str(index)}} for index in range(3)]}
    phases = []
    def add(identity, count, rate, alias=False):
        phases.append({'id': identity, 'remaining_forward_jobs': 0 if alias else count,
            'materialization_jobs': count if alias else 0, 'per_job_whole_slot_seconds_proxy': rate,
            'sum_slot_seconds_proxy': count * rate, 'model': None})
    add('main_behavior_adapted', 66, 150.)
    add('B_nll', 60, 40.)
    for name, count in [('pythia_410m', 24), ('pythia_1b', 24), ('pythia_2p8b', 15)]:
        add('C_' + name + '_nll_final', count, 40.)
        add('C_' + name + '_nll_best', count, 30.)
        add('C_' + name + '_nll_initial', 1, 25.)
        add('C_' + name + '_behavior_final', count, 150.)
    add('A_selected_behavior', 15, 150.)
    add('A_selected_behavior_materialization', 3, 20., True)
    cases = []
    for name, factor in [('low', 1.), ('central', 1.25), ('high', 1.5)]:
        rows = copy.deepcopy(phases)
        for row in rows:
            row['per_job_whole_slot_seconds_proxy'] *= factor
            row['sum_slot_seconds_proxy'] *= factor
        cases.append({'name': name, 'whole_wall_sensitivity_factor': factor,
            'A_observed_410m_final_reference_seconds_before_factor': 40.,
            'future_410m_NLL_final_whole_slot_seconds_proxy': 40. * factor,
            'per_phase': rows, 'handoff_preflight_allowance_seconds_unmeasured': 300.})
    completion = {'whole_slot_statistics': {'adapted': {'min': 10., 'median': 20., 'max': 30.}},
                  'actual_whole_slot_seconds_sum': 2400.}
    return {'sensitivity_scenarios': cases}, completion, primary, A


def test_only_B60_removed_and_fast_samples_do_not_lower_factors_or_rates():
    old, completion, primary, A = future_fixture()
    cases = module.future_cases(old, completion, primary, A)
    for prior, case in zip(old['sensitivity_scenarios'], cases):
        assert case['whole_wall_sensitivity_factor'] == prior['whole_wall_sensitivity_factor']
        assert case['remaining_forward_jobs'] == 273
        assert case['remaining_materialization_jobs'] == 3
        assert case['four_slot_dispatch']['job_count'] == 276
        assert case['future_410m_NLL_final_whole_slot_seconds_proxy'] == prior['future_410m_NLL_final_whole_slot_seconds_proxy']
        assert case['C_future_D_B4_equality_discount_spent'] is False
        assert case['earlier_A_and_baseline_future_work_removed_again'] == 0
        assert 'B_nll' not in {row['id'] for row in case['per_phase']}


def test_slower_B_can_raise_only_410m_final_nll_transfer():
    old, completion, primary, A = future_fixture()
    completion['whole_slot_statistics']['adapted'] = {'min': 45., 'median': 50., 'max': 60.}
    cases = module.future_cases(old, completion, primary, A)
    for prior, case in zip(old['sensitivity_scenarios'], cases):
        before = {row['id']: row for row in prior['per_phase']}
        for row in case['per_phase']:
            if row['id'] == 'C_pythia_410m_nll_final':
                assert row['per_job_whole_slot_seconds_proxy'] > before[row['id']]['per_job_whole_slot_seconds_proxy']
            else:
                assert row == before[row['id']]
