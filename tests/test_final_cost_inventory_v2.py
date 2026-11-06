"""Small receipt fixtures only; no GPU, score parsing or checkpoint reads."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import inventory_study_costs_final_v2 as module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')
    return path


def proof(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(tmp_path):
    root = tmp_path / 'exports'
    exporter = tmp_path / 'exporter.py'
    exporter.write_text('# explicit synthetic exporter authority\n')
    phases, models = [], []
    for model in ('pythia-410m', 'pythia-1b', 'pythia-2.8b'):
        identity = model + '_B4_s11'
        definition = {'id': identity, 'argv': ['frozen_training', identity]}
        queue = write(tmp_path / model / 'queue.json', [definition])
        snapshot = write(tmp_path / model / 'snapshot.json', {'queue_path': str(queue), 'queue_sha256': proof(queue)['sha256']})
        checkpoint = tmp_path / model / 'DO_NOT_READ_CHECKPOINT.pt'
        checkpoint.write_bytes(b'NOT_A_CHECKPOINT_PAYLOAD')
        registry = write(tmp_path / model / 'registry.json', [{'id': identity, 'checkpoint_path': str(checkpoint)}])
        phase = {'model': model, 'phase': 'phase_' + model, 'runs': 1,
            'active_registry': str(registry), 'active_registry_sha256': proof(registry)['sha256'],
            'snapshot': str(snapshot), 'snapshot_sha256': proof(snapshot)['sha256']}
        phases.append(phase)
        models.append({'model': model, 'registry': proof(registry), 'publication_receipt': None})
        if model == 'pythia-410m':
            record = {'status': 'verified_draft_not_deliverable', 'run_id': identity, 'model_tag': model,
                'phase_name': phase['phase'], 'data_family': 'extended_v1', 'role': 'extended',
                'input_snapshot_sha256': phase['snapshot_sha256'], 'source_checkpoint': str(checkpoint),
                'definition': definition, 'definition_sha256': module.digest_job(definition),
                'completed_scheduler_record': {'status': 'completed', 'returncode': 0,
                    'definition_sha256': module.digest_job(definition)},
                'all_nonweight_metadata_roundtrip_verified': True, 'additional_encoding_calls': 1,
                'wall_seconds': 10., 'process_cpu_seconds': 9.,
                'encode': {'wall_seconds': 6.}, 'decode': {'wall_seconds': 2.},
                'bytewise_verification': {'all_tensor_bytes_equal': True, 'wall_seconds': 1.},
                'encode_process_cpu_seconds': 5., 'decode_process_cpu_seconds': 2., 'verification_process_cpu_seconds': 1.}
            write(root / model / 'records' / (identity + '.json'), record)
    freeze = write(tmp_path / 'freeze.json', {'phases': phases})
    write(root / 'progress.json', {'status': 'watching_validation_metadata_only', 'elapsed_wall_seconds': 500., 'process_cpu_seconds': 20.})
    return {'root': str(root), 'freeze': proof(freeze), 'exporter_source': proof(exporter), 'models': models}


def test_exact_registry_enumeration_counts_only_actual_draft_not_daemon_or_children(tmp_path, monkeypatch):
    source = fixture(tmp_path)
    original = Path.read_bytes
    def guarded(path):
        if path.suffix in {'.pt', '.zip', '.tar'}:
            raise AssertionError('Model/archive bytes must not be read by a cost collector')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    reader = module.CostReceiptReader()
    operations, coverage = module.C_operations(source, reader)
    for item in operations:
        reader.register(item['source']['path'])
    rows, _ = module.v1.operation_rows({'operations': operations}, reader)
    assert [row['actual_first_created_records_observed'] for row in coverage] == [1, 0, 0]
    assert not any(row['assembly_complete'] for row in coverage)
    assert all(row['publication_source_status'] == 'not_authorized_source_unassigned' for row in coverage)
    assert sum(row['recorded_wall_seconds'] for row in rows if row['included_in_disjoint_recorded_wall_sum']) == 10.
    assert len(reader.exact_cost_bytes) >= 10


def test_later_assembly_is_disjoint_from_draft_but_never_implies_publication(tmp_path):
    source = fixture(tmp_path)
    write(Path(source['root']) / 'pythia-410m' / 'completion.json', {'status': 'passed', 'model': 'pythia-410m',
        'phase': 'phase_pythia-410m', 'logical_final_endpoints': 1, 'fullphase_source_gate_passed': True,
        'timer_scope': 'assembly only; prior draft operations excluded', 'wall_seconds': 5., 'process_cpu_seconds': 4.})
    reader = module.CostReceiptReader()
    operations, coverage = module.C_operations(source, reader)
    for item in operations:
        reader.register(item['source']['path'])
    rows, _ = module.v1.operation_rows({'operations': operations}, reader)
    assert coverage[0]['assembly_complete'] is True
    assert coverage[0]['publication_completed_assumed'] is False
    assert sum(row['recorded_wall_seconds'] for row in rows if row['included_in_disjoint_recorded_wall_sum']) == 15.


def test_duplicate_or_unregistered_C_source_and_incomplete_draft_rejected(tmp_path):
    source = fixture(tmp_path)
    duplicate = copy.deepcopy(source)
    duplicate['models'].append(copy.deepcopy(duplicate['models'][0]))
    with pytest.raises(ValueError, match='Duplicate model'):
        module.C_operations(duplicate, module.CostReceiptReader())
    record_dir = Path(source['root']) / 'pythia-410m' / 'records'
    extra = write(record_dir / 'UNREGISTERED.json', {'wall_seconds': 999})
    with pytest.raises(ValueError, match='Unregistered JSON'):
        module.C_operations(source, module.CostReceiptReader())
    extra.unlink()
    path = next(record_dir.glob('*.json'))
    value = json.loads(path.read_text())
    value['completed_scheduler_record']['returncode'] = 1
    write(path, value)
    with pytest.raises(ValueError, match='exact completed run binding'):
        module.C_operations(source, module.CostReceiptReader())


def test_missing_or_partial_completion_cannot_claim_completed_model(tmp_path):
    source = fixture(tmp_path)
    write(Path(source['root']) / 'pythia-1b' / 'completion.json', {'status': 'passed', 'model': 'pythia-1b',
        'phase': 'phase_pythia-1b', 'logical_final_endpoints': 1, 'fullphase_source_gate_passed': True,
        'timer_scope': 'assembly only', 'wall_seconds': 5.})
    with pytest.raises(ValueError, match='complete, separately timed'):
        module.C_operations(source, module.CostReceiptReader())


def test_exact_cost_copy_keeps_observed_bytes_and_rejects_big_payload_registration(tmp_path):
    path = write(tmp_path / 'receipt.json', {'wall_seconds': 1.})
    reader = module.CostReceiptReader()
    reader.register(path)
    reader.read(path)
    original = reader.exact_cost_bytes[str(path.resolve())]
    write(path, {'wall_seconds': 99.})
    assert json.loads(original)['wall_seconds'] == 1.
    assert hashlib.sha256(original).hexdigest() == reader.sources[str(path.resolve())]['sha256']
    with pytest.raises(ValueError, match='small JSON'):
        reader.register(tmp_path / 'model.pt')


def test_unknown_B_process_guard_lower_bound_and_watch_attempts_not_wait_are_additive(tmp_path):
    parent = write(tmp_path / 'B.json', {'complete': True, 'started': '2026-11-06T11:00:00Z', 'ended': '2026-11-06T11:20:00Z'})
    guard = write(tmp_path / 'guard.json', {'ready': True, 'wall_seconds': 1.3})
    watch = write(tmp_path / 'watch.json', {'status': 'completed', 'wait': 720., 'attempts': .9, 'successful': .26})
    operations = [module.op('B_unknown', {'path': str(parent)}, category='cpu_analysis', status_pointer='/complete',
        accepted_statuses=[True], wall_pointer=None, process_cpu_pointer=None, add_to_recorded_wall_totals=False),
        module.op('B_guard', {'path': str(guard)}, category='cpu_analysis', status_pointer='/ready',
            accepted_statuses=[True], process_cpu_pointer=None, parent_id='B_unknown'),
        module.op('watch_calendar', {'path': str(watch)}, category='cpu_analysis', accepted_statuses=['completed'],
            wall_pointer='/wait', process_cpu_pointer=None, timing_semantics='wall_including_waiting', add_to_recorded_wall_totals=False),
        module.op('watch_attempts', {'path': str(watch)}, category='cpu_analysis', accepted_statuses=['completed'],
            wall_pointer='/attempts', process_cpu_pointer=None, parent_id='watch_calendar'),
        module.op('watch_success_child', {'path': str(watch)}, category='cpu_analysis', accepted_statuses=['completed'],
            wall_pointer='/successful', process_cpu_pointer=None, parent_id='watch_attempts',
            timing_semantics='nested_measured_child', add_to_recorded_wall_totals=False)]
    rows, _ = module.v1.operation_rows({'operations': operations}, module.CostReceiptReader())
    assert rows[0]['recorded_wall_seconds'] is None
    assert sum(row['recorded_wall_seconds'] for row in rows if row['included_in_disjoint_recorded_wall_sum']) == pytest.approx(2.2)
