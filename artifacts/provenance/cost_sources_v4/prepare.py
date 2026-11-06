#!/usr/bin/env python3
"""Cost source revision4; normalize only explicit measured timer metadata."""
import copy
import datetime as dt
import hashlib
import json
import math
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
    old = ROOT / 'configs/cost_inventory_final_sources_v3.json'
    assert proof(old)['sha256'] == '4fdc24ea2c5995e76b032d29803de1e038114d667b9dd7ef7753b220abf7ea2a'
    assert proof(ROOT / 'scripts/inventory_study_costs_final_v2.py')['sha256'] == '864effb89f20dbfce3ba6ce51db8eb8351b087c2ac92d255274fe6267753f6ae'
    spec = copy.deepcopy(json.loads(old.read_text()))
    spec.update({'source_spec_revision': 4, 'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                 'preserved_v3_source_spec': proof(old), 'revision4_preparation_source': proof(__file__)})
    spec['accounting_contract'].update({'v1_v2_v3_sources_and_all_collectors_unchanged': True,
        'inclusive_parent_or_child_CPU_normalized_from_explicit_user_sys_only': True,
        'whole_parent_stage_timers_not_added_to_nested_read_decode_fingerprint_or_bootstrap_timers': True,
        'original_best_identity_only_is_not_GPU_QA_or_a_new_training_seed': True,
        'primary_behavior_static_IO_review_is_not_an_actual_analysis_or_measured_zero_IO_event': True})
    ids, origins, normalization = [], [], []
    def add(identity, path, wall, cpu=None, category='cpu_analysis', *, scope, original_status='passed'):
        item = {'id': identity, 'category': category, 'source': proof(path), 'status_pointer': '/status',
            'accepted_statuses': [original_status], 'wall_pointer': wall, 'process_cpu_pointer': cpu,
            'timing_semantics': 'independent_measured_complete_operation', 'add_to_recorded_wall_totals': True,
            'parent_id': None, 'canonical_id': 'explicit_timer:' + str(Path(path).resolve()) + '#' + wall, 'scope': scope}
        spec['additional_operations'].append(item); ids.append(identity)
    def normalized(identity, relative, wall_key, cpu_keys, category, scope):
        source = ROOT / relative; value = json.loads(source.read_text())
        assert value.get('returncode', 0) == 0
        if 'status' in value:
            assert value['status'] in ('passed', 'passed_independent_small_evidence_binding_check')
        wall = value[wall_key]; components = {key: value[key] for key in cpu_keys}
        assert math.isfinite(wall) and wall >= 0
        assert all(math.isfinite(number) and number >= 0 for number in components.values())
        result = {'schema_version': 1, 'status': 'normalized_actual_measured_operation',
            'original_source': proof(source), 'wall_seconds': wall, 'original_wall_pointer': '/' + wall_key,
            'CPU_components': components, 'CPU_component_pointers': ['/' + key for key in cpu_keys],
            'recorded_inclusive_CPU_seconds': sum(components.values()),
            'normalization': 'Wall copied verbatim; CPU is sum of only the explicitly listed disjoint user/sys components.',
            'scope': scope, 'nested_subtimers_added': False, 'new_operation_created_by_normalization': False,
            'original_source_fields_or_semantics_changed': False}
        target = HERE / (identity + '_normalized.json'); write(target, result)
        add(identity, target, '/wall_seconds', '/recorded_inclusive_CPU_seconds', category,
            scope=scope, original_status='normalized_actual_measured_operation')
        origins.append(proof(source)); normalization.append(proof(target))
    normalized('durable_weight_restore_example_actual', 'docs/已交付权重恢复示例_实际恢复收据.json',
        'parent_wall_seconds', ['parent_self_user_seconds', 'parent_self_sys_seconds', 'children_user_seconds', 'children_sys_seconds'],
        'archive_qa', 'Actual CPU durable-example parent operation: extraction, two archived decodes, literal doc metadata example, fresh restored-state fingerprints and new-artifact inventory. Inclusive CPU=self+children once. Stages nested. Excludes initial driver/directory preparation and later doc/evidence copying. No GPU, original optimizerCP or wholeTAR hash.')
    normalized('optional_C2p8b_original_best_identity_actual',
        'results/reproduction_qa/optional_C2p8b_original_best_identity_v1/whole_process_timing.json',
        'whole_child_process_wall_seconds', ['whole_child_user_cpu_seconds', 'whole_child_sys_cpu_seconds'], 'archive_qa',
        'Actual complete original-best identity-only child process spawn-through-exit. Includes imports/load/full-state fingerprint/fresh fileSHA/proof serialization; worker parent_timing/read_operations are nested and not added. This is not reproduction training, numerical new seed or score analysis. Historical prehash references are not new reads.')
    normalized('optional_C2p8b_identity_independent_binding_check_actual',
        'results/reproduction_qa/optional_C2p8b_original_best_identity_v1/independent_binding_check.json',
        'wall_seconds', ['user_cpu_seconds', 'sys_cpu_seconds'], 'archive_qa',
        'Separate later small-evidence binding check, explicitly measured outside the identity child operation. No second checkpoint-payload pass; does not include its own entire interactive preparation.')
    normalized('snapshot1325_whole_package_child', 'artifacts/provenance/snapshot_1325_build/whole_process_timing.json',
        'package_child_wall_seconds', ['package_child_user_seconds', 'package_child_sys_seconds'], 'lossless_export',
        'Full repository code/data snapshot package subprocess including compression and all member verification. Excludes later shared publication. This category is archive creation, not model-weight encoding or GPU training. Nested compression/member timers are not additional operations.')
    allocation = ROOT / 'docs/snapshot_1325_storage_reservation.json'
    value = json.loads(allocation.read_text()); assert value['status'] == 'actual_capacity_reserved'
    add('snapshot1325_actual_allocation', allocation, '/wall_seconds', '/process_cpu_seconds', 'storage_wait_and_reservation',
        scope='Actual1325 allocation, independent of package creation and publication; capacity bytes are not time.', original_status='actual_capacity_reserved')
    publication = ROOT / 'docs/snapshot_1325_durable_publication.json'
    value = json.loads(publication.read_text()); assert value['status'] == 'passed'
    add('snapshot1325_durable_publication', publication, '/wall_seconds', None, 'storage_publication',
        scope='Actual1325 durable publication invocation only, separate from package/allocation; publication CPU not measured. NoTAR payload read in this collector.')
    for identity, relative, scope in [
        ('A_gradient_clipping_whole_analysis_child', 'results/A_gradient_clipping_posttest_v1/whole_process_timing.json',
         'Complete post-test descriptive training-log child: imports, exact source checks/snapshot, full log parse/statistics and figures. COMPLETE nested timers already included.18 oldmain references do not add training. Tests/docs/manual interpretation outside child unknown.'),
        ('A_gradient_clipping_independent_numeric_child', 'results/A_gradient_clipping_posttest_v1/independent_numeric_review_whole_process_timing.json',
         'Separate complete post-analysis CSV arithmetic QA child; not nested in the earlier wholeanalysis. No new inference or statistics selection.'),
        ('A_gradient_clipping_presentation_v2_child', 'results/A_gradient_clipping_posttest_v1/presentation_v2/whole_process_timing.json',
         'Separate complete plotting-only child including input/oldoutputSHA checks and PNG/PDF/SVG serialization. Old document backup/latercaption edits/manual review outside child unknown; no new statistics or GPU.'),
    ]:
        normalized(identity, relative, 'whole_child_process_wall_seconds', ['whole_child_user_cpu_seconds', 'whole_child_sys_cpu_seconds'], 'cpu_analysis', scope)
    spec['metadata_sources'].extend([proof(old), *origins])
    spec['revision4_additional_operation_ids'] = ids
    spec['revision4_normalized_sources'] = normalization
    spec['unmeasured_additional_gaps'].extend([
        {'scope': 'Actual durable restore-example initial driver/directory preparation and later document/evidence copying not in parent timer remain unknown.'},
        {'scope': 'A gradient14tests, Chinese document/caption edits and manual plot interpretation outside child timers are unmeasured.'},
        {'scope': 'Snapshot1325 publication CPU unmeasured; package child CPU is not publication CPU.'},
        {'scope': 'Primarybehavior static source audit was not a primaryanalysis execution; no actual0-byte or0second analysis cost is imputed.'}])
    assert len(ids) == 9
    all_ids = [item['id'] for item in spec['additional_operations']]
    assert len(all_ids) == len(set(all_ids))
    assert all(item['publication_receipt'] is None for item in spec['C_exports']['models'])
    target = ROOT / 'configs/cost_inventory_final_sources_v4.json'; write(target, spec)
    print(json.dumps({'source_spec': proof(target), 'new_operation_ids': ids, 'normalized_sources': len(normalization)}))


if __name__ == '__main__':
    main()
