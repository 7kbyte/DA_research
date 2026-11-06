import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/watch_extended_phase_equivalence.py'
spec = importlib.util.spec_from_file_location('extended_phase_equivalence', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def fixture_cohort(tmp_path):
    rows, jobs, state = [], [], {'jobs': {}}
    for seed in [11, 22, 33]:
        for group in sorted(module.GROUPS):
            name = f'{group}_{seed}'
            output = tmp_path / name
            summary = write_json(output / 'summary.json', {})
            row = {'id': name, 'group': group, 'seed': seed, 'output_dir': str(output),
                   'summary_path': str(summary), 'config_path': str(output / 'config.json')}
            job = {'id': name, 'argv': ['python', 'run_train.py', '--config', row['config_path']],
                   'outputs': [str(summary)]}
            state['jobs'][name] = {'status': 'completed', 'returncode': 0,
                                  'definition_sha256': module.digest_job(job)}
            rows.append(row)
            jobs.append(job)
    return rows, jobs, state


def fixture_producer(tmp_path, monkeypatch, unequal_best=False):
    rows, jobs, state = fixture_cohort(tmp_path)
    for row in rows:
        output = Path(row['output_dir'])
        config = {'run_label': row['id'], 'model_name_or_path': 'tiny-model', 'model_revision': 'revision1'}
        write_json(Path(row['config_path']), config)
        (output / 'events.jsonl').write_text('{"event":"finish"}\n')
        best_path, final_path = output / 'best.pt', output / 'final.pt'
        trajectory = [
            {'step': 1, 'target_prediction_tokens': 10, 'replay_prediction_tokens': 0,
             'metrics': {'target': {'nll': 1.0}}},
            {'step': 2, 'target_prediction_tokens': 20, 'replay_prediction_tokens': 1,
             'metrics': {'target': {'nll': 2.0}}}]
        best = {'step': 1, 'target_prediction_tokens': 10, 'nll': 1.0, 'checkpoint_path': str(best_path)}
        counters = {'step': 2, 'target_prediction_tokens': 20, 'replay_prediction_tokens': 1}
        summary = {'config': config, 'counters': counters, 'checkpoint_path': str(final_path),
                   'best_target_validation': best, 'validation_trajectory': trajectory}
        write_json(Path(row['summary_path']), summary)
        if row['group'] not in {'B4', 'D'}:
            continue
        weights = {'first': torch.arange(4.).reshape(2, 2), 'last': torch.zeros(1025)}
        torch.save({'model_state': weights, 'config': config, 'counters': counters,
                    'best_target_validation': best, 'validation_trajectory': trajectory}, final_path)
        if unequal_best and row['group'] == 'D' and row['seed'] == 11:
            # Change the final element of a second tensor, never a sampled prefix.
            weights['last'][-1] = torch.nextafter(torch.tensor(0.), torch.tensor(1.))
        torch.save({'model_state': weights, 'config': config,
                    'checkpoint_kind': 'model_only_best_target_validation',
                    'counters': {k: trajectory[0][k] for k in counters}, 'selection': best}, best_path)
    registry = write_json(tmp_path / 'registry.json', rows)
    snapshot = write_json(tmp_path / 'snapshot.json', {'source': 'synthetic'})
    phase = {'phase': 'synthetic_410m_formal', 'model': 'pythia-410m',
             'active_registry': str(registry), 'snapshot': str(snapshot)}
    raw = {'status': 'passed', 'completed_runs': 24, 'expected_runs': 24, 'violations': []}
    monkeypatch.setattr(module, 'audit_phase', lambda *args, **kwargs: copy.deepcopy(raw))
    gate = module.complete_cohort(rows, state, jobs, jobs)
    return phase, state, gate, rows, raw


def test_cohort_gate_requires_all24_successful_exits_and_exact_definitions(tmp_path):
    rows, jobs, state = fixture_cohort(tmp_path)
    assert module.complete_cohort(rows, state, jobs, jobs)['ready']
    state['jobs'][rows[-1]['id']]['returncode'] = 1
    assert not module.complete_cohort(rows, state, jobs, jobs)['ready']
    state['jobs'][rows[-1]['id']]['returncode'] = 0
    changed = copy.deepcopy(jobs)
    changed[-1]['argv'].append('--changed')
    with pytest.raises(ValueError, match='Master job differs'):
        module.complete_cohort(rows, state, jobs, changed)
    with pytest.raises(ValueError, match='all eight groups'):
        module.complete_cohort(rows[:-1], state, jobs, jobs)


def test_complete_state_is_not_enough_if_completed_summary_is_missing(tmp_path):
    rows, jobs, state = fixture_cohort(tmp_path)
    Path(rows[0]['summary_path']).unlink()
    with pytest.raises(FileNotFoundError, match='Completed cohort output missing'):
        module.complete_cohort(rows, state, jobs, jobs)


def test_publishes_actual_final_and_best_bytes_with_false_equality_preserved(tmp_path, monkeypatch):
    phase, state, gate, rows, _ = fixture_producer(tmp_path, monkeypatch, unequal_best=True)
    result = module.publish_phase(phase, state, gate, tmp_path / 'proofs', [])
    assert result['status'] == 'passed'
    assert result['comparisons_computed'] == 6
    assert result['all_pairs_bitwise_equal'] is False
    assert result['non_equivalent_comparison_ids'] == ['synthetic_410m_formal_D_B4_s11_best']
    final = module.read(result['proofs']['final']['path'])
    best = module.read(result['proofs']['best']['path'])
    assert len(final['comparisons']) == len(best['comparisons']) == 3
    assert all(item['bitwise_model_state_equal'] for item in final['comparisons'])
    assert not best['comparisons'][0]['bitwise_model_state_equal']
    left, right = final['comparisons'][0]['left'], final['comparisons'][0]['right']
    assert left['checkpoint_sha256'] != right['checkpoint_sha256']
    assert left['model_state_fingerprint'] == right['model_state_fingerprint']
    assert left['checkpoint_exposure']['target_prediction_tokens_at_checkpoint'] == 20
    assert best['comparisons'][0]['left']['actual_selection']['step'] == 1
    assert best['comparisons'][0]['left']['checkpoint_exposure']['target_prediction_tokens_at_checkpoint'] == 10
    module.existing_completion(tmp_path / 'proofs' / phase['phase'] / 'completion.json', phase)
    # An endpoint mutation invalidates the evidence even if all prior metadata is valid.
    Path(rows[0]['summary_path']).write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        module.existing_completion(tmp_path / 'proofs' / phase['phase'] / 'completion.json', phase)


def test_checkpoint_selection_is_bound_to_saved_exposure_not_only_summary(tmp_path, monkeypatch):
    _, _, _, rows, _ = fixture_producer(tmp_path, monkeypatch)
    row = next(row for row in rows if row['group'] == 'D')
    summary = module.read(row['summary_path'])
    path = Path(summary['best_target_validation']['checkpoint_path'])
    saved = torch.load(path, map_location='cpu', weights_only=False)
    saved['counters']['target_prediction_tokens'] += 1
    torch.save(saved, path)
    with pytest.raises(ValueError, match='actual selected endpoint'):
        module.checkpoint_side(row, summary, 'best')


def test_source_audit_failure_publishes_no_ready_completion(tmp_path, monkeypatch):
    phase, state, gate, _, raw = fixture_producer(tmp_path, monkeypatch)
    raw['violations'] = [{'run_id': 'B0_11', 'check': 'same_seed_target_stream'}]
    raw['status'] = 'failed'
    with pytest.raises(ValueError, match='source/state audit failed'):
        module.publish_phase(phase, state, gate, tmp_path / 'proofs', [])
    directory = tmp_path / 'proofs' / phase['phase']
    assert module.read(directory / 'audit.json')['status'] == 'failed'
    assert not (directory / 'completion.json').exists()


def test_previously_published_completion_rechecks_checkpoint_bytes(tmp_path, monkeypatch):
    phase, state, gate, _, _ = fixture_producer(tmp_path, monkeypatch)
    result = module.publish_phase(phase, state, gate, tmp_path / 'proofs', [])
    final = module.read(result['proofs']['final']['path'])
    path = Path(final['comparisons'][0]['left']['checkpoint'])
    path.write_bytes(path.read_bytes() + b'changed serialization')
    with pytest.raises(ValueError, match='changed'):
        module.existing_completion(tmp_path / 'proofs' / phase['phase'] / 'completion.json', phase)


def test_numeric_difference_is_reported_separately_from_source_validity(tmp_path, monkeypatch):
    phase, state, gate, _, raw = fixture_producer(tmp_path, monkeypatch)
    raw['violations'] = [{'run_id': 'D_11', 'check': 'D_B4_exact_numerical_trajectories'}]
    raw['status'] = 'failed'
    result = module.publish_phase(phase, state, gate, tmp_path / 'proofs', [])
    audit = module.read(result['audit']['path'])
    assert audit['status'] == 'passed'
    assert audit['numerical_null_control_status'] == 'different'
    assert audit['phase_audit']['status'] == 'failed'

