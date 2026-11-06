#!/usr/bin/env python3
"""Revision5 adds six explicit measured CPU regions; no source discovery."""
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


def pointer(value, path):
    for part in path.strip('/').split('/'):
        value = value[part]
    return value


def main():
    old = ROOT / 'configs/cost_inventory_final_sources_v4.json'
    assert proof(old)['sha256'] == 'a1b56243519aa3a649859b88351156e8ce9b8006d139114635c55fdbef85c959'
    assert proof(ROOT / 'scripts/inventory_study_costs_final_v2.py')['sha256'] == '864effb89f20dbfce3ba6ce51db8eb8351b087c2ac92d255274fe6267753f6ae'
    spec = copy.deepcopy(json.loads(old.read_text()))
    spec.update({'source_spec_revision': 5, 'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                 'preserved_v4_source_spec': proof(old), 'revision5_preparation_source': proof(__file__)})
    spec['accounting_contract'].update({'v1_v2_v3_v4_sources_and_all_collectors_unchanged': True,
        'revision5_exactly_six_independent_measured_operations_or_regions': True,
        'post_stdlib_preparation_body_region_is_not_a_complete_parent_process_timer': True,
        'new_GPU_reproduction_training_not_imputed_from_CPU_preparation': True})
    ids, origins, normalizations = [], [], []

    def add(identity, source, wall, cpu, category, scope, status, *, original=None, semantics='independent_measured_complete_operation'):
        original = original or (source, wall)
        spec['additional_operations'].append({'id': identity, 'category': category, 'source': proof(source),
            'status_pointer': '/status', 'accepted_statuses': [status], 'wall_pointer': wall, 'process_cpu_pointer': cpu,
            'timing_semantics': semantics, 'add_to_recorded_wall_totals': True, 'parent_id': None,
            'canonical_id': 'explicit_timer:' + str(Path(original[0]).resolve()) + '#' + original[1], 'scope': scope})
        ids.append(identity)

    def normalize(identity, relative, wall, cpu, category, scope, *, accepted_status=None,
                  semantics='independent_measured_complete_operation'):
        source = ROOT / relative; value = json.loads(source.read_text())
        assert value.get('returncode', 0) == 0
        if accepted_status is not None:
            assert value['status'] == accepted_status
        duration = pointer(value, wall); components = {key: pointer(value, key) for key in cpu}
        assert math.isfinite(duration) and duration >= 0
        assert all(math.isfinite(number) and number >= 0 for number in components.values())
        result = {'schema_version': 1, 'status': 'normalized_actual_measured_operation', 'original_source': proof(source),
            'wall_seconds': duration, 'original_wall_pointer': wall, 'CPU_components': components,
            'CPU_component_pointers': cpu, 'recorded_inclusive_CPU_seconds': sum(components.values()),
            'normalization': 'Copy the explicit measured wall; sum only listed disjoint user/sys components. No timestamp inference.',
            'scope': scope, 'timing_semantics': semantics, 'nested_subtimers_added': False,
            'new_operation_created_by_normalization': False, 'original_source_fields_or_semantics_changed': False}
        target = HERE / (identity + '_normalized.json'); write(target, result)
        add(identity, target, '/wall_seconds', '/recorded_inclusive_CPU_seconds', category, scope,
            'normalized_actual_measured_operation', original=(source, wall), semantics=semantics)
        origins.append(proof(source)); normalizations.append(proof(target))

    normalize('snapshot1415_whole_package_child', 'artifacts/provenance/snapshot_1415_build/whole_process_timing.json',
        '/whole_child_wall_seconds', ['/whole_child_user_cpu_seconds', '/whole_child_sys_cpu_seconds'], 'lossless_export',
        'Complete actual1415 package child spawn-through-exit including compression/member verification. Earlier0732/1205/1325 package operations remain separate historical work, not repeated. Nested member/compression timers not added; publication outside child.')
    allocation = ROOT / 'docs/snapshot_1415_storage_reservation.json'
    value = json.loads(allocation.read_text()); assert value['status'] == 'actual_capacity_reserved'
    add('snapshot1415_actual_allocation', allocation, '/wall_seconds', '/process_cpu_seconds', 'storage_wait_and_reservation',
        'Actual1415 capacity allocation only, independent of package and publication. No time inferred from bytes or file timestamps.', 'actual_capacity_reserved')
    publication = ROOT / 'docs/snapshot_1415_durable_publication.json'
    value = json.loads(publication.read_text()); assert value['status'] == 'passed'
    add('snapshot1415_durable_publication', publication, '/wall_seconds', None, 'storage_publication',
        'Actual1415 durable publication invocation only. Its CPU remains unknown; nested durability events are not new operations. Collector reads this receipt, not TAR payload.', 'passed')
    normalize('optional_C1b_original_best_identity_actual',
        'results/reproduction_qa/optional_C1b_original_best_identity_v1/whole_process_timing.json',
        '/whole_child_process_wall_seconds', ['/whole_child_user_cpu_seconds', '/whole_child_sys_cpu_seconds'], 'archive_qa',
        'Complete fixed original1B-best identity child spawn-through-exit, including imports, originalCP loading/full-state fingerprint/fresh fileSHA and proof serialization. Worker parent/read_operations are nested. CPU identity work is not GPU reproduction, a new scientific seed, or score analysis.',
        accepted_status='passed_child_process')
    normalize('optional_three_model_1415_selected_assessment_tool_extraction_actual',
        'results/reproduction_qa/optional_three_model_B4_preparation_v1/assessment_tool_extraction_1415.json',
        '/wall_seconds', ['/user_cpu_seconds', '/sys_cpu_seconds'], 'archive_qa',
        'Independent actual selected-tool extraction stage: streams1415 archive and writes only archived verifier plus PACKAGE_MANIFEST. Does not recompute wholeTAR SHA or repeat package QA cost. Does not relocate base or launch GPU reproduction.',
        accepted_status='passed_selected_archived_tool_and_manifest_CPU_extraction_only')
    normalize('optional_three_model_B4_preparation_body_region_actual',
        'docs/optional_three_model_B4_archive_QA_CPU_preparation_receipt_v1.json',
        '/preparation_stage_timing/wall_seconds', ['/preparation_stage_timing/user_cpu_seconds', '/preparation_stage_timing/sys_cpu_seconds'], 'archive_qa',
        'Independent builder body region after stdlib imports through fresh0732 required payload checks and template construction only. Excludes earlier engineering, imports, separate C1B identity child and1415 selected extraction. Not a complete parent-process timer or executed GPU QA.',
        accepted_status='passed_CPU_preparation_only_FALSE_templates_not_execution_ready',
        semantics='independent_measured_body_region_after_stdlib_imports')
    spec['metadata_sources'].extend([proof(old), *origins])
    spec['revision5_additional_operation_ids'] = ids
    spec['revision5_normalized_sources'] = normalizations
    spec['unmeasured_additional_gaps'].extend([
        {'scope': 'New QA preparation engineering, source review, QC/whole CPU tests and any work outside six explicit timers remain unknown; body-region time is not whole-parent time.'},
        {'scope': 'Snapshot1415 publication CPU is unmeasured; old archive/package SHA work is not newly charged by references.'},
        {'scope': 'Optional three-model GPU reproduction has not executed and is not counted as completed training or a new scientific seed.'}])
    assert len(ids) == len(set(ids)) == 6
    all_ids = [item['id'] for item in spec['additional_operations']]
    assert len(all_ids) == len(set(all_ids))
    assert spec['C_exports'] == json.loads(old.read_text())['C_exports']
    target = ROOT / 'configs/cost_inventory_final_sources_v5.json'; write(target, spec)
    print(json.dumps({'source_spec': proof(target), 'new_operation_ids': ids, 'normalized_sources': len(normalizations)}))


if __name__ == '__main__':
    main()
