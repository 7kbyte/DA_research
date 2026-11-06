"""Synthetic metadata and CPU time arithmetic only; no real READY or scores."""
import copy
import datetime as dt
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import preflight_C410_ready_nll as module


def test_ROOT_exact_reserve_boundaries_and_one_microsecond_late():
    cutoff = module.utc('2026-11-06T22:30:00Z')
    stop = module.utc('2026-11-06T23:50:00Z')
    rows = [{'id': 'future_C', 'start_utc': '2026-11-06T22:25:00Z'},
            {'id': 'already_completed_C', 'start_utc': '2026-11-06T23:59:00Z'}]
    finish = module.utc('2026-11-06T23:35:00Z')
    assert module.timing_gates(rows, finish, cutoff, stop, {'future_C'})['both_ROOT_time_gates']
    changed = copy.deepcopy(rows)
    changed[0]['start_utc'] = '2026-11-06T22:25:00.000001Z'
    assert not module.timing_gates(changed, finish, cutoff, stop, {'future_C'})['both_ROOT_time_gates']
    assert not module.timing_gates(rows, finish + dt.timedelta(microseconds=1), cutoff, stop, {'future_C'})['both_ROOT_time_gates']
    changed[0]['start_utc'] = '2026-11-06T22:30:00Z'
    assert not module.timing_gates(changed, finish, cutoff, stop, {'future_C'})['both_ROOT_time_gates']


def test_actual_canonical_scope_prices_mixed_endpoints_once_without_second_factor():
    queue = [{'id': 'new_initial'}, {'id': 'merged'}, {'id': 'other_best'}]
    manifest = {'jobs': [{'id': 'new_initial', 'kind': 'initial'},
                         {'id': 'merged', 'kind': 'adapted'}, {'id': 'other_best', 'kind': 'adapted'}],
                'records': [{'kind': 'final', 'inference_job_id': 'merged'},
                            {'kind': 'best', 'inference_job_id': 'merged'},
                            {'kind': 'best', 'inference_job_id': 'other_best'}]}
    rates = {'initial': 15., 'final': 30., 'best': 25.}  # Already sensitivity-priced.
    assert module.price_jobs(queue, manifest, rates) == {'new_initial': 15., 'merged': 30., 'other_best': 25.}
    manifest['jobs'][1]['kind'] = 'reuse'
    with pytest.raises(ValueError, match='ROOT review'):
        module.price_jobs(queue, manifest, rates)


def test_FIFO_preserves_four_occupied_slots_pending_order_and_new_initial_dependency():
    master = [{'id': 'running' + str(i)} for i in range(4)] + [{'id': 'Cnext'}, {'id': 'Clast'}]
    state = {'jobs': {job['id']: {'status': 'running', 'gpu': str(i)} for i, job in enumerate(master[:4])}}
    state['jobs'].update({job['id']: {'status': 'pending'} for job in master[4:]})
    raw = {'extended_model_runtime_updates': {'toy': {'jobs': [{'id': job['id'], 'total_seconds': 10.} for job in master]}}}
    phase = [{'id': 'new_initial'}, {'id': 'adapted', 'depends_on': ['new_initial']}]
    before = copy.deepcopy((master, state, phase, raw))
    base = module.fifo_projection(master, state, [], {}, raw, 60., True)
    actual = module.fifo_projection(master, state, phase, {'new_initial': 2., 'adapted': 3.}, raw, 60., True)
    rows = {row['id']: row for row in actual['future_dispatch_timing_rows']}
    assert rows['new_initial']['start_seconds'] == pytest.approx(11.)
    assert rows['adapted']['start_seconds'] == pytest.approx(13.2)
    assert [row['id'] for row in actual['future_dispatch_timing_rows'] if row['id'].startswith('C')] == ['Cnext', 'Clast']
    assert not any(row['id'].startswith('running') for row in actual['future_dispatch_timing_rows'])
    assert base['makespan_seconds_with_10pct'] == pytest.approx(22.)
    assert actual['makespan_seconds_with_10pct'] == pytest.approx(22.)
    assert (master, state, phase, raw) == before


def test_actual_occupied_long_run_delays_ready_initial_naturally():
    master = [{'id': 'C' + str(i)} for i in range(4)]
    state = {'jobs': {job['id']: {'status': 'running', 'gpu': str(i)} for i, job in enumerate(master)}}
    raw = {'extended_model_runtime_updates': {'toy': {'jobs': [{'id': job['id'], 'total_seconds': 1000.} for job in master]}}}
    result = module.fifo_projection(master, state, [{'id': 'new_initial'}], {'new_initial': 20.}, raw, 4000., True)
    assert result['future_dispatch_timing_rows'][0]['start_seconds'] == pytest.approx(1100.)
    assert result['makespan_seconds_with_10pct'] == pytest.approx(1122.)


def test_input_SHA_and_changed_snapshot_fail_closed(tmp_path):
    path = tmp_path / 'state.json'
    path.write_text(json.dumps({'status': 'pending'}))
    source = module.Inputs()
    source.read('state.json', path, module.sha(path.read_bytes()))
    source.unchanged()
    path.write_text(json.dumps({'status': 'running'}))
    with pytest.raises(ValueError, match='changed during preflight'):
        source.unchanged()
    with pytest.raises(ValueError, match='SHA differs'):
        module.Inputs().read('state.json', path, '0' * 64)


def test_metadata_replaced_by_identical_bytes_still_rejected(tmp_path):
    path = tmp_path / 'state.json'
    path.write_text('{}')
    source = module.Inputs()
    source.read('state.json', path)
    replacement = tmp_path / 'new.json'
    replacement.write_text('{}')
    replacement.replace(path)
    with pytest.raises(ValueError, match='changed during preflight'):
        source.unchanged()


def test_source_symlink_retarget_rejected_even_when_old_canonical_is_unchanged(tmp_path):
    old = tmp_path / 'old.json'; new = tmp_path / 'new.json'; alias = tmp_path / 'alias.json'
    old.write_text('{}'); new.write_text('{}'); alias.symlink_to(old)
    source = module.Inputs()
    source.read('source.json', alias)
    alias.unlink(); alias.symlink_to(new)
    with pytest.raises(ValueError, match='changed during preflight'):
        source.unchanged()


def test_nonmetadata_file_rejected_before_reading_payload(tmp_path):
    path = tmp_path / 'large_file'
    with path.open('wb') as stream:
        stream.truncate((32 << 20) + 1)
    with pytest.raises(ValueError, match='Only small metadata'):
        module.Inputs().read('bad', path, binary=True)


def test_existing_output_rejected_before_any_real_input_read(tmp_path):
    from types import SimpleNamespace
    with pytest.raises(FileExistsError, match='immutable'):
        module.run(SimpleNamespace(output=tmp_path))
