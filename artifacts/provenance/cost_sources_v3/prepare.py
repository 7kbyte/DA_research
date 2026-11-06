#!/usr/bin/env python3
"""Prepare explicit cost source revision3 without altering either collector."""
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def proof(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def main():
    old = ROOT / 'configs/cost_inventory_final_sources_v2.json'
    assert proof(old)['sha256'] == '3c0286f39fb19f386bbb09a68143c80cd4fe45677efaf360309b032712b14ce7'
    assert proof(ROOT / 'scripts/inventory_study_costs_final_v2.py')['sha256'] == '864effb89f20dbfce3ba6ce51db8eb8351b087c2ac92d255274fe6267753f6ae'
    spec = copy.deepcopy(json.loads(old.read_text()))
    spec['source_spec_revision'] = 3
    spec['created_at_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    spec['preserved_v2_source_spec'] = proof(old)
    spec['unchanged_v2_collector'] = proof(ROOT / 'scripts/inventory_study_costs_final_v2.py')
    spec['accounting_contract'].update({'v1_v2_collectors_and_old_specs_unchanged': True,
        'source_revision3_uses_unchanged_schema2_collector_contract': True,
        'copied_sources_and_nested_verification_are_not_new_operations': True,
        'pending_order_preview_unmeasured_parent_remains_unknown': True,
        'bank_diagnostics_history_regions_counted_once_current_geometry_not_added_again': True,
        'no_future_C_publication_unsealing_or_GPU_QA_imputed': True})
    new_ids = []
    def add(identity, path, wall, cpu=None, *, category='cpu_analysis', status=None, accepted=None,
            semantics='independent_measured_region', additive=True, scope=''):
        source = proof(ROOT / path if not Path(path).is_absolute() else path)
        item = {'id': identity, 'category': category, 'source': source,
            'status_pointer': status, 'accepted_statuses': accepted or [], 'wall_pointer': wall,
            'process_cpu_pointer': cpu, 'timing_semantics': semantics,
            'add_to_recorded_wall_totals': additive, 'parent_id': None,
            'canonical_id': 'explicit_timer:' + source['path'] + '#' + str(wall), 'scope': scope}
        spec['additional_operations'].append(item); new_ids.append(identity)
    add('snapshot1205_actual_allocation', 'docs/snapshot_1205_storage_reservation.json', '/wall_seconds', '/process_cpu_seconds',
        category='storage_wait_and_reservation', status='/status', accepted=['actual_capacity_reserved'],
        semantics='single_measured_stage', scope='Actual initial capacity allocation for1205 snapshot; independent of publication. Bytes are capacity, not time.')
    add('snapshot1205_durable_publication', 'docs/snapshot_1205_durable_publication.json', '/wall_seconds',
        category='storage_publication', status='/status', accepted=['passed'], semantics='single_measured_stage',
        scope='Actual1205 archive publication invocation only. Source archive creation and publication CPU time unmeasured; no archive payload read by this inventory.')
    for version in ('v1', 'v2'):
        add('B_test_review_' + version + '_independent_region', 'results/supplement_B_test_review_' + version + '/review_receipt.json',
            '/review_wall_seconds', '/review_process_CPU_seconds', status='/review_complete', accepted=[True],
            scope='One distinct authorized review process measured region: source/numerical review and plots through final source recheck. Excludes interactive writing/receipt serialization. Never the original B analysis parent or inference cost; no score semantics consumed here.')
    add('B_doc_weight_decomposition_independent_verification',
        'results/supplement_B_test_review_v1/target_doc_weight_decomposition_verification.json',
        '/wall_seconds', '/process_CPU_seconds', status='/verified', accepted=[True],
        scope='Separate measured doc-weight scalar verification region after review; not copied table generation, bootstrap, or new model forward.')
    history_path = ROOT / 'results/extended_bank_diagnostics/timing_history.json'
    history = json.loads(history_path.read_text())
    assert len(history['prior_runs']) == 3
    assert abs(sum(row['wall_seconds'] for row in history['prior_runs']) + history['current_run']['wall_seconds'] - history['analysis_region_wall_sum_seconds']) < 1e-12
    assert abs(sum(row['process_cpu_seconds'] for row in history['prior_runs']) + history['current_run']['process_cpu_seconds'] - history['analysis_region_cpu_sum_seconds']) < 1e-12
    for index in range(3):
        add('C_actual_bank_diagnostics_development_region_' + str(index + 1), str(history_path),
            '/prior_runs/' + str(index) + '/wall_seconds', '/prior_runs/' + str(index) + '/process_cpu_seconds',
            scope='Distinct development invocation measured input-analysis region only. Imports/tests/output/docs excluded; historical source revisions may not be archived. Never imputed whole-process CPU/wall.')
    add('C_actual_bank_diagnostics_current_region', str(history_path), '/current_run/wall_seconds', '/current_run/process_cpu_seconds',
        scope='Current measured input-analysis region counted from timing_history only; identical timer in bank_geometry.json is not another invocation. Imports/tests/output/docs excluded.')
    add('C_pending_order_apply_with_reload_wait_context',
        'artifacts/provenance/C_pending_order_apply_v1/execution/execution_receipt.json', '/wall_seconds', '/cpu_seconds',
        status='/status', accepted=['published_reload_confirmed'], semantics='wall_including_waiting', additive=False,
        scope='Actual order publication invocation plus6second reload wait. Wall nonadditive context, measured process CPU retained per row; no active-write wall split inferred. Source copies are not new executions.')
    add('C_pending_order_postpublication_verification_region', 'results/C_pending_order_postpublication_v1/receipt.json',
        '/wall_seconds', '/process_cpu_seconds', status='/status', accepted=['passed_independent_read_only_postpublication_verification'],
        scope='Independent postpublication measured source/object/process/FIFO verification region only; excludes imports and later evidence serialization/copies. Parent preparation/writing time unknown.')
    add('final_handoff_gap_read_only_region', 'results/final_handoff_gap_v1/receipt.json', '/wall_seconds', '/process_cpu_seconds',
        status='/status', accepted=['read_only_final_handoff_gap_review_not_authority'],
        scope='Independent read-only source/allowance/deadline coarse-bound review region; excludes imports/evidence serialization/copies and interactive code/doc work. No prepare/start/final FIFO.')
    telemetry_path = ROOT / 'results/resources_1303_intermediate/collection_receipt.json'
    telemetry = json.loads(telemetry_path.read_text()); assert telemetry['returncode'] == 0
    normalized = {'schema_version': 1, 'status': 'explicit_normalization_only_original_unchanged',
        'original_source': proof(telemetry_path), 'wall_seconds': telemetry['wall_seconds'],
        'child_user_cpu_seconds': telemetry['child_user_cpu_seconds'], 'child_sys_cpu_seconds': telemetry['child_sys_cpu_seconds'],
        'child_process_cpu_seconds': telemetry['child_user_cpu_seconds'] + telemetry['child_sys_cpu_seconds'],
        'normalization': 'child_process_cpu_seconds = original child_user_cpu_seconds + original child_sys_cpu_seconds; wall copied verbatim',
        'scope': 'Only the actual1303 telemetry-collection child process. Not root interpreter CPU, training, inference, score analysis, or later visual review.',
        'new_operation_claimed': False, 'original_fields_or_timer_semantics_modified': False}
    normalized_path = HERE / 'telemetry_1303_normalized.json'; write(normalized_path, normalized)
    add('telemetry1303_collection_child', str(normalized_path), '/wall_seconds', '/child_process_cpu_seconds',
        status='/status', accepted=['explicit_normalization_only_original_unchanged'],
        scope='Measured1303 collection child process: wall copied from original and child CPU explicitly normalized as user+sys. Normalization creates no second operation.')
    metadata = [old, telemetry_path, ROOT / 'results/extended_bank_diagnostics/bank_geometry.json',
        ROOT / 'results/C_pending_order_preview_v1/preview.json', ROOT / 'results/C_pending_order_preview_v1/verification_receipt.json',
        ROOT / 'results/final_handoff_gap_v1/verification_receipt.json']
    spec['metadata_sources'].extend(proof(path) for path in metadata)
    spec['unmeasured_additional_gaps'].extend([
        {'scope': 'Pending-order preview and its independent verification have no structured operation timer. Forecast start/end_seconds are simulated durations, not measured CPU work. Do not infer from UTC or tool wall.'},
        {'scope': 'Final handoff gap independent verification receipt has no timer; source-copy/receipt/doc/interactive authoring work outside measured regions remains unknown.'},
        {'scope': 'Snapshot1205 original archive construction whole-process time and publication CPU not measured by the newly registered receipts.'},
        {'scope': 'B original authorized CPU analysis parent remains unknown; two distinct later review regions and doc-weight verification do not fill that missing parent.'},
        {'scope': 'C actual-bank diagnostics imports/tests/output/document/rank-readback outside four measured regions unknown. First historical source revision not archived.'}])
    spec['revision3_additional_operation_ids'] = new_ids
    spec['revision3_preparation_source'] = proof(__file__)
    ids = [row['id'] for row in spec['additional_operations']]
    assert len(ids) == len(set(ids))
    assert all(source['publication_receipt'] is None for source in spec['C_exports']['models'])
    target = ROOT / 'configs/cost_inventory_final_sources_v3.json'; write(target, spec)
    print(json.dumps({'source_spec': proof(target), 'new_measured_or_context_operations': len(new_ids),
                      'normalized_telemetry_child_CPU_seconds': normalized['child_process_cpu_seconds']}))


if __name__ == '__main__':
    main()
