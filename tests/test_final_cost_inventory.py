"""Temporary study fixtures: never import Torch or read a scientific score."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import inventory_study_costs_final as module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')
    return path


def proof(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def operation(id, path, **extra):
    return {'id': id, 'category': 'lossless_export', 'source': {'path': str(path)}, 'status_pointer': '/status',
            'accepted_statuses': ['passed'], 'wall_pointer': '/wall', 'timing_semantics': 'single_measured_stage',
            'add_to_recorded_wall_totals': True, 'scope': 'Explicit synthetic wall', **extra}


def test_cumulative_packages_and_nested_guard_not_counted_as_disjoint_stages(tmp_path):
    parent = write(tmp_path / 'export.json', {'status': 'passed', 'wall': 30, 'packages': [{'wall': 10}, {'wall': 30}]})
    analysis = write(tmp_path / 'analysis.json', {'status': 'passed', 'wall': 5, 'guard': 1})
    items = [operation('export', parent), operation('analysis', analysis, category='cpu_analysis')]
    for index in range(2):
        items.append(operation('pkg' + str(index), parent, parent_id='export', wall_pointer=f'/packages/{index}/wall',
                               timing_semantics='cumulative_parent_sample', add_to_recorded_wall_totals=False))
    items.append(operation('guard', analysis, category='cpu_analysis', parent_id='analysis', wall_pointer='/guard',
                           timing_semantics='nested_measured_child', add_to_recorded_wall_totals=False))
    rows, _ = module.operation_rows({'operations': items}, module.legacy.Reader())
    assert sum(row['recorded_wall_seconds'] for row in rows if row['included_in_disjoint_recorded_wall_sum']) == 35
    assert rows[-1]['recorded_wall_seconds'] == 1
    bad = copy.deepcopy(items)
    bad[-1]['add_to_recorded_wall_totals'] = True
    with pytest.raises(ValueError, match='parent and child'): module.operation_rows({'operations': bad}, module.legacy.Reader())
    bad = [operation('wait', parent, timing_semantics='wall_including_waiting')]
    with pytest.raises(ValueError, match='waiting'): module.operation_rows({'operations': bad}, module.legacy.Reader())


def test_recovery_timer_is_additive_but_calendar_and_grower_wait_are_context(tmp_path):
    path = write(tmp_path / 'publication.json', {'status': 'passed', 'recovery': 128.689,
                 'started': '2026-11-06T06:42:00Z', 'ended': '2026-11-06T06:49:00Z', 'wall': 456})
    items = [operation('calendar', path, wall_pointer=None, timing_semantics='calendar_envelope',
                       add_to_recorded_wall_totals=False, calendar_start_pointer='/started', calendar_end_pointer='/ended'),
             operation('recovery', path, wall_pointer='/recovery', parent_id='calendar'),
             operation('grower', path, timing_semantics='wall_including_waiting', add_to_recorded_wall_totals=False)]
    rows, _ = module.operation_rows({'operations': items}, module.legacy.Reader())
    assert rows[0]['calendar_window_seconds'] == 420 and rows[0]['recorded_wall_seconds'] is None
    assert sum(row['recorded_wall_seconds'] for row in rows if row['included_in_disjoint_recorded_wall_sum']) == 128.689
    assert not any(row['pure_CPU_or_GPU_work_claim'] for row in rows)


def test_prehash_full_read_receipts_dedup_hits_and_never_read_opaque_input(tmp_path):
    stat = {'dev': 1, 'ino': 2, 'size': 123, 'mtime_ns': 4, 'ctime_ns': 5}
    receipt = {'method': 'sha256_entire_open_fd_to_eof_v1', 'complete_to_eof': True,
               'canonical_path': str(tmp_path / 'DO_NOT_READ_CHECKPOINT'), 'sha256': 'b' * 64,
               'stat_before': stat, 'stat_after': stat, 'fd_before': stat, 'fd_after': stat,
               'read_bytes': 123, 'full_read_seconds': 2., 'reader_source': {'sha256': 'reader'}}
    raw = (json.dumps(receipt) + '\n').encode()
    sha = hashlib.sha256(raw).hexdigest()
    sources = []
    for name in ['first', 'copy']:
        folder = tmp_path / name
        (folder / 'receipts').mkdir(parents=True)
        (folder / 'receipts' / (sha + '.json')).write_bytes(raw)
        progress = write(folder / 'progress.json', {'counters_this_process': {'full_reads': 1, 'cache_hits': 999}})
        sources.append({'directory': str(folder), 'reader_sha256': 'reader', 'progress': str(progress)})
    rows, progress = module.opaque_hash_receipts({'opaque_digest_caches': sources}, module.legacy.Reader())
    assert len(rows) == 1 and rows[0]['historical_full_read_seconds'] == 2
    assert rows[0]['current_inventory_input_payload_reads'] == 0
    assert len(progress) == 2 and all(not row['full_read_or_cache_hit_counts_added_as_wall'] for row in progress)
    damaged = copy.deepcopy(sources)
    damaged[0]['reader_sha256'] = 'other'
    with pytest.raises(ValueError, match='stable complete-read'): module.opaque_hash_receipts({'opaque_digest_caches': damaged}, module.legacy.Reader())


def test_state_union_keeps_identical_completed_baseline_once_and_rejects_redefinition(tmp_path):
    done = {'status': 'completed', 'returncode': 0, 'elapsed_seconds': 7, 'definition_sha256': 'same'}
    first = write(tmp_path / 'first.json', {'jobs': {'baseline': done}})
    second = write(tmp_path / 'second.json', {'jobs': {'baseline': dict(done)}})
    merged, _ = module.merge_states([{'path': str(first)}, {'path': str(second)}], module.legacy.Reader())
    assert len(merged['jobs']) == 1 and merged['jobs']['baseline']['elapsed_seconds'] == 7
    write(second, {'jobs': {'baseline': {**done, 'definition_sha256': 'changed'}}})
    with pytest.raises(ValueError, match='definition'): module.merge_states([{'path': str(first)}, {'path': str(second)}], module.legacy.Reader())
    write(second, {'jobs': {'baseline': {**done, 'elapsed_seconds': 8}}})
    with pytest.raises(ValueError, match='attempt'): module.merge_states([{'path': str(first)}, {'path': str(second)}], module.legacy.Reader())


def test_only_SHA_bound_ready_manifests_enter_and_duplicate_manifest_is_once(tmp_path):
    catalog = write(tmp_path / 'catalog.json', {'plans': [{'id': 'A'}, {'id': 'C'}]})
    manifest = write(tmp_path / 'manifest.json', {'schema_version': 2, 'mode': 'nll', 'jobs': [], 'records': []})
    queue = write(tmp_path / 'queue.json', [])
    base = write(tmp_path / 'base.json', {'evaluation_manifests': [{**proof(manifest), 'category': 'supplement'}]})
    index = write(tmp_path / 'index.json', {'catalog': proof(catalog), 'phases': [
        {'id': 'A', 'status': 'ready', 'ready': True, 'manifest': proof(manifest), 'queue': proof(queue)},
        {'id': 'C', 'status': 'waiting_training', 'ready': False}]})
    spec = {'legacy_source_spec': proof(base), 'evaluation_ready_indexes': [{'path': str(index), 'catalog': proof(catalog)}]}
    effective, readiness = module.effective_sources(spec, module.legacy.Reader(), tmp_path / 'state.json')
    assert len(effective['evaluation_manifests']) == 1 and len(readiness) == 2
    assert readiness[-1]['ready'] is False
    manifest.write_text(manifest.read_text() + ' ')
    with pytest.raises(ValueError, match='SHA mismatch'): module.effective_sources(spec, module.legacy.Reader(), tmp_path / 'state.json')


def test_missing_operations_and_unstarted_cohort_remain_unknown(tmp_path):
    rows, _ = module.operation_rows({'operations': [operation('future', tmp_path / 'missing.json')]}, module.legacy.Reader())
    assert rows[0]['recorded_wall_seconds'] is None and not rows[0]['included_in_disjoint_recorded_wall_sum']
    registry = write(tmp_path / 'registry.json', [{'id': 'futureD', 'group': 'D', 'output_dir': str(tmp_path / 'not_started')}])
    spec = {'training_cohorts': [{'id': 'C', 'registry': proof(registry), 'expected_logical_rows': 1,
                                 'expected_physical_runs': 1, 'expected_training_aliases': 0}]}
    row = module.cohort_rows(spec, module.legacy.Reader(), {'physical_runs': []})[0]
    assert row['D_group_independent_training_definitions'] == 1 and row['D_group_observed_physical_runs'] == 0
    assert row['known_parent_wall_sum_seconds'] is None and row['incomplete_or_unobserved_physical_runs'] == 1


def test_full_collection_counts_both_D_B4_training_and_no_alias_forward_or_score_parse(tmp_path):
    runs = tmp_path / 'runs'
    registry = []
    for group in ['D', 'B4']:
        run = runs / group
        counts = {'setup_seconds': 1, 'training_seconds': 10, 'validation_seconds': 2, 'checkpoint_seconds': 3,
                  'training_teacher_forward_seconds': 4, 'validation_teacher_forward_seconds': 1,
                  'validation_kl_arithmetic_seconds': .2, 'target_prediction_tokens': 100, 'replay_prediction_tokens': 5}
        write(run / 'config.json', {'output_dir': str(run), 'target_prediction_tokens': 100})
        write(run / 'summary.json', {'config': {'output_dir': str(run)}, 'status': 'completed', 'counters': counts})
        registry.append({'id': group, 'group': group, 'output_dir': str(run)})
    registry.append({'id': 'A_alias', 'alias_of': 'B4', 'execution_kind': 'alias', 'counted_as_new_training': False})
    reg = write(tmp_path / 'registry.json', registry)
    output = tmp_path / 'score.json'
    output.write_bytes(b'PRIVATE_SCORES_INVALID_JSON')
    job = {'id': 'initial', 'kind': 'initial', 'output': str(output), 'baseline_group': 'g'}
    manifest = write(tmp_path / 'manifest.json', {'schema_version': 2, 'mode': 'nll', 'jobs': [job], 'records': [],
                     'baseline_groups': {'g': {'initial_job_id': 'initial', 'initial_output': str(output)}}})
    write(tmp_path / 'score_execution.json', {'job_id': 'initial', 'manifest_sha256': proof(manifest)['sha256'],
          'verified_frozen_inputs': True, 'checkpoint_sha256': None, 'verification_wall_seconds': 1.,
          'inference_subprocess_wall_seconds': 5., 'inference_alias': False,
          'output_sha256': {str(output): proof(output)['sha256']}})
    state = write(tmp_path / 'state.json', {'jobs': {'initial': {'status': 'completed', 'returncode': 0,
                  'elapsed_seconds': 10, 'definition_sha256': 'frozen'}}})
    base = write(tmp_path / 'base.json', {'registries': [{'path': str(reg), 'category': 'primary'}],
                 'evaluation_manifests': [{**proof(manifest), 'category': 'primary', 'scheduler_state': str(state), 'verify_output_hashes': True}]})
    spec = {'legacy_source_spec': proof(base), 'runs_root': str(runs), 'evaluation_scheduler_states': [{'path': str(state)}],
            'training_cohorts': [{'id': 'all', 'registry': proof(reg), 'expected_logical_rows': 3,
                'expected_physical_runs': 2, 'expected_training_aliases': 1}]}
    reader = module.legacy.Reader()
    result = module.collect(spec, tmp_path / 'result', reader)
    cohort = result['training_cohort_coverage'][0]
    assert cohort['D_group_independent_training_definitions'] == cohort['B4_group_independent_training_definitions'] == 1
    assert cohort['D_group_observed_physical_runs'] == cohort['B4_group_observed_physical_runs'] == 1
    assert cohort['known_parent_wall_sum_seconds'] == 32  # teacher children excluded
    assert cohort['training_aliases_not_new_runs'] == 1
    assert len(result['legacy_method_and_preparation_ledger']['physical_runs']) == 2
    assert len(result['legacy_method_and_preparation_ledger']['evaluation_canonical_inferences']) == 1
    assert reader.sources[str(output.resolve())]['read_purpose'] == 'SHA_only_no_score_deserialization'
    assert not result['accounting_contract']['whole_study_completely_instrumented']
    assert b'PRIVATE_SCORES' not in json.dumps(result).encode()
    with pytest.raises(FileExistsError): module.collect(spec, tmp_path / 'result')
