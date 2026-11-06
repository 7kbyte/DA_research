#!/usr/bin/env python3
"""Explicit cost-only source revision6; unchanged collectors and old sources."""
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
    if path.stat().st_size > 32 << 20:
        raise ValueError('Explicit metadata unexpectedly exceeds32MiB')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def read(path):
    if Path(path).stat().st_size > 32 << 20:
        raise ValueError('Only explicit small metadata is read')
    return json.loads(Path(path).read_text())


def pointer(value, path):
    if path is None:
        return None
    for part in path.strip('/').split('/'):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def main():
    old = ROOT / 'configs/cost_inventory_final_sources_v5.json'
    assert proof(old)['sha256'] == '3d5aba267d9d92756b512d6feae468ab1deb199030379d887608e9e6a385d53a'
    collector = ROOT / 'scripts/inventory_study_costs_final_v2.py'
    assert proof(collector)['sha256'] == '864effb89f20dbfce3ba6ce51db8eb8351b087c2ac92d255274fe6267753f6ae'
    candidates_path = ROOT / 'results/cost_sources_v6_candidates_v1/candidate_sources.json'
    assert proof(candidates_path)['sha256'] == 'a0a506ae0fcced8e22975ec9d7a97e6b8d4fd34c5d5c74badcd0af0023e8f1eb'
    candidates = read(candidates_path)
    spec = copy.deepcopy(read(old))
    old_ops = copy.deepcopy(spec['additional_operations'])
    spec.update(source_spec_revision=6, created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                preserved_v5_source_spec=proof(old), revision6_preparation_source=proof(__file__))
    spec['accounting_contract'].update({
        'v1_through_v5_sources_and_all_collectors_unchanged': True,
        'revision6_C410_actual43_inferences_collected_only_by_existing_manifest_inventory': True,
        'revision6_whole_children_and_nested_regions_explicitly_separated': True,
        'revision6_extra_A_B_whole_batches_not_assumed_completed': True,
        'revision6_no_new_score_semantics_or_GB_checkpoint_payload_reads': True,
    })
    ids, origins, normalized, measured, contexts = [], [], [], [], []
    seen = {item.get('canonical_id', item['id']) for item in old_ops}

    def add(identity, source, wall, cpu, category, scope, semantics, *, additive=True,
            parent=None, original=None, accepted=(), status_pointer='/status', **extra):
        canonical = 'explicit_timer:' + str(Path(original[0]).resolve()) + '#' + str(original[1]) if original else identity
        if canonical in seen:
            raise ValueError('A newly registered operation duplicates an old timer: ' + canonical)
        seen.add(canonical)
        item = {'id': identity, 'category': category, 'source': proof(source),
                'status_pointer': status_pointer, 'accepted_statuses': list(accepted),
                'wall_pointer': wall, 'process_cpu_pointer': cpu,
                'timing_semantics': semantics, 'add_to_recorded_wall_totals': additive,
                'parent_id': parent, 'canonical_id': canonical, 'scope': scope, **extra}
        spec['additional_operations'].append(item)
        ids.append(identity)
        return item

    def normalize(identity, source, wall, cpu, user, system, category, scope,
                  *, semantics='independent_measured_complete_operation', returncode=None):
        source = Path(source)
        value = read(source)
        if returncode is not None:
            if type(value.get('returncode')) is not int or value['returncode'] != returncode:
                raise ValueError('Unexpected measured child outcome: ' + identity)
        duration = pointer(value, wall)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise ValueError('Measured nonnegative finite wall required')
        components = {}
        if cpu:
            components[cpu] = pointer(value, cpu)
        else:
            for component in (user, system):
                if component:
                    components[component] = pointer(value, component)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in components.values()):
            raise ValueError('Measured nonnegative finite CPU components required')
        inclusive_cpu = sum(components.values()) if components else None
        result = {'schema_version': 1, 'status': 'normalized_actual_measured_operation',
                  'original_source': proof(source), 'original_wall_pointer': wall,
                  'wall_seconds': duration, 'CPU_component_pointers': list(components), 'CPU_components': components,
                  'recorded_inclusive_CPU_seconds': inclusive_cpu, 'original_returncode': value.get('returncode'),
                  'scope': scope, 'timing_semantics': semantics,
                  'normalization': 'Copy measured wall and sum only explicitly listed disjoint CPU components; never infer from UTC/bytes.',
                  'nested_subtimers_added': False, 'new_operation_created_by_normalization': False,
                  'original_source_fields_or_semantics_changed': False}
        target = HERE / (identity + '_normalized.json')
        write(target, result)
        add(identity, target, '/wall_seconds', '/recorded_inclusive_CPU_seconds', category, scope,
            semantics, accepted=['normalized_actual_measured_operation'], original=(source, wall))
        origins.append(proof(source)); normalized.append(proof(target))
        measured.append({'id': identity, 'original_source': proof(source), 'wall_pointer': wall,
                         'measured_wall_seconds': duration, 'known_inclusive_CPU_seconds': inclusive_cpu,
                         'scope': scope, 'timing_semantics': semantics})

    # Existing candidate bytes are frozen; this is explicit enumeration, not
    # recursive timer discovery. C assembly remains the old dynamic row.
    for item in candidates['candidates']:
        source = item['source']
        if proof(source['path']) != {k: source[k] for k in ('path', 'sha256')}:
            raise ValueError('Frozen candidate metadata changed: ' + item['id'])
        classification = item['classification']
        if classification == 'new_explicit_registration':
            semantics = 'independent_measured_complete_operation'
            if item['id'] == 'C410_NLL_priority_append_region':
                semantics = 'independent_measured_append_region_not_whole_parent'
            elif item['id'].startswith('extra_export_preflight_') or item['id'].endswith('_region') or item['id'].endswith('_body'):
                semantics = 'independent_measured_body_region_after_imports'
            normalize(item['id'], source['path'], item['wall_pointer'], item.get('cpu_pointer'),
                      item.get('user_pointer'), item.get('sys_pointer'), item['category'], item['scope'],
                      semantics=semantics)
        elif classification == 'existing_dynamic_collector_now_completed':
            assert item['id'] == 'C_assembly:pythia-410m'
            contexts.append({'id': item['id'], 'handling': 'Existing unchanged C_operations dynamic row only; no new explicit operation.', 'source': source})
        elif classification == 'nested_reference_never_add':
            add(item['id'], source['path'], item['wall_pointer'], None, item['category'], item['scope'],
                'nested_measured_child', additive=False, parent=item['parent_id'],
                original=(source['path'], item['wall_pointer']), accepted=['passed'])
        elif classification == 'nonadditive_calendar_context':
            add(item['id'], source['path'], None, None, 'storage_wait_and_reservation', item['scope'],
                'calendar_envelope', additive=False,
                calendar_start_pointer='/started_at_utc', calendar_end_pointer='/finished_at_utc')
        elif classification == 'nonadditive_wall_including_waits':
            add(item['id'], source['path'], item['wall_pointer'], None, item['category'], item['scope'],
                'wall_including_waiting', additive=False, parent=item['parent_id'],
                original=(source['path'], item['wall_pointer']))
        else:
            raise ValueError('Unknown explicit candidate classification')

    # Four separate actually measured pytest children, not four aliases to the
    # same regression. Initial manual48-test parent was not measured.
    explicit = [
        ('guard_v3_author_initial95_synthetic_tests_whole_child',
         'artifacts/provenance/supplemental_guard_v3_repair_v1/synthetic_tests_whole_child_timing.json',
         '/whole_child_process_wall_seconds', '/whole_child_user_cpu_seconds', '/whole_child_sys_cpu_seconds', 0,
         'Initial independent repair95-test child. Superseded source remains snapshotted; actual historical CPU work still counted once.'),
        ('guard_v3_author_final97_synthetic_tests_whole_child',
         'artifacts/provenance/supplemental_guard_v3_repair_v2/synthetic_tests_whole_child_timing.json',
         '/whole_child_process_wall_seconds', '/whole_child_user_cpu_seconds', '/whole_child_sys_cpu_seconds', 0,
         'Separate final plan-bound97-test child; not nested in the initial author test child.'),
        ('guard_v3_independent_candidate48_tests_whole_child',
         'results/supplemental_guard_v3_independent_review_v1/candidate_1/receipt.json',
         '/whole_child_wall_seconds', '/whole_child_user_cpu_seconds', '/whole_child_sys_cpu_seconds', 0,
         'Separate reviewer child48tests before plan-binding amendment; source snapshot retained.'),
        ('guard_v3_independent_final104_tests_whole_child',
         'results/supplemental_guard_v3_independent_review_v1/final_stable/test_receipt.json',
         '/whole_child_wall_seconds', '/whole_child_user_cpu_seconds', '/whole_child_sys_cpu_seconds', 0,
         'Separate reviewer final104-test child; source/AST review engineering outside child remains unknown.'),
        ('CPU_full793_test_suite_whole_child', 'logs/tests_after_supplemental_guard_v3_full.json',
         '/whole_child_wall_seconds', '/user_cpu_seconds', '/sys_cpu_seconds', 0,
         'ROOT complete793-test child spawn-through-exit; overlaps test content but is a genuinely separate execution, not nested in previous pytest children.'),
        ('C410_aux_v2_failed_check_whole_child',
         'artifacts/provenance/C410_nll_analysis_actual_v1/check_whole_child_timing.json',
         '/wall_seconds', '/user_cpu_seconds', '/sys_cpu_seconds', 1,
         'Real failedv2 score-blind check child before any score parse/authority/journal; charged as actual failed work, never recast as successful analysis.'),
        ('C410_aux_v3_actual_check_whole_child',
         'artifacts/provenance/C410_nll_analysis_actual_v3/check_whole_child_timing.json',
         '/wall_seconds', '/user_cpu_seconds', '/sys_cpu_seconds', 0,
         'Real successfulv3 score-blind check child; guard/source/output hashing below is a nested child and is not added again.'),
        ('C410_aux_v3_actual_analyze_whole_child',
         'artifacts/provenance/C410_nll_analysis_actual_v3/analyze_whole_child_timing.json',
         '/wall_seconds', '/user_cpu_seconds', '/sys_cpu_seconds', 0,
         'ROOT-authorized complete C410 NLL analysis child, including its own repeated gate, collection/bootstrap/write; no other model scores parsed by this cost collector.'),
        ('evaluation_budget_v6_actual_completion_whole_child',
         'artifacts/provenance/evaluation_budget_v6_execution_v1/whole_child_timing.json',
         '/wall_seconds', '/user_cpu_seconds', '/sys_cpu_seconds', 0,
         'Actual-completion score-blind evaluation budgetv6 child. Internal budget collector/verification timers are nested, not new operations.'),
    ]
    for identity, source, wall, user, system, rc, scope in explicit:
        normalize(identity, ROOT / source, wall, None, user, system, 'cpu_analysis', scope, returncode=rc)

    for identity, source, parent in [
        ('C410_aux_v3_actual_check_nested_guard', 'artifacts/provenance/C410_nll_analysis_actual_v3/check_readiness.json', 'C410_aux_v3_actual_check_whole_child'),
        ('C410_aux_v3_actual_analyze_nested_guard', 'results/supplemental_analysis_v3/extended_pythia_410m_100m_nll/whole_phase_readiness.json', 'C410_aux_v3_actual_analyze_whole_child'),
    ]:
        value = read(ROOT / source)
        assert value.get('ready') is True and value.get('score_json_parsed') is False
        assert value['counts']['closure_queue_jobs'] == 43 and value['counts']['logical_endpoints'] == 48
        assert value['verification_work']['counters']['checkpoint_fresh_bytes'] == 0
        add(identity, ROOT / source, '/score_blind_verification_wall_seconds', None, 'cpu_analysis',
            'Nested guard inside the corresponding whole child only. Historical cached checkpoint bytes are not current reads or additional timed work.',
            'nested_measured_child', additive=False, parent=parent, status_pointer='/ready', accepted=[True],
            original=(ROOT / source, '/score_blind_verification_wall_seconds'))

    # Successful real analysis is checked using COMPLETE metadata only. Do not
    # open endpoints/statistics/bootstrap or scored evaluation output semantics.
    complete_path = ROOT / 'results/supplemental_analysis_v3/extended_pythia_410m_100m_nll/COMPLETE.json'
    complete = read(complete_path)
    assert complete['complete'] is True and complete['GPU_inference_started'] is False
    assert complete['selection_recomputed'] is False
    assert complete['plan']['sha256'] == '61f44eba006214462680f656fcb6cc3a90aae550ae506f84f2ec27446f2292fb'

    # Keep dynamic C publication disabled, because it would add the inner
    # publisher wall beside the measured whole child registered above.
    assert all(row['publication_receipt'] is None for row in spec['C_exports']['models'])
    spec['resolved_previously_planned_operations'] = [
        {'id': 'C_publication:pythia-410m', 'registered_as': 'C410_final_publication_whole_child',
         'nested_receipt': proof(ROOT / 'docs/C410_final_weights_durable_publication.json'),
         'inner_dynamic_publication_enabled': False}]
    spec['planned_operations_without_source'] = [row for row in spec['planned_operations_without_source'] if row['id'] != 'C_publication:pythia-410m']
    spec['unmeasured_additional_gaps'].extend([
        {'scope': 'C410 phase audit/equivalence completion succeeded but no whole-parent wall/CPU was measured. Historical opaque prehash receipts do not reconstruct this parent.'},
        {'scope': 'Original optional five-package planning/verification and publication-runbook preparation parents have no structured full timer; proxies/timestamps are not producer CPU costs.'},
        {'scope': 'Earlier90-test invocation and first manual48-test invocation lack structured whole-child timers; pytest displayed seconds are not whole-parent timers.'},
        {'scope': 'Initial ROOT C410 v3 expected_outputs KeyError happened before a child spawn; ROOT_precheck_v1_failure.json explicitly states whole_parent_timing unknown.'},
        {'scope': 'Repair engineering, AST/source review, documentation, first-read review/authority construction and source copying outside measured pytest/check/analyze child timers remain unknown.'},
        {'scope': 'A/B additional-export whole batches are not yet completed/registered here. Their standalone first samples are actual measured independent children; a future batch reusing those samples must not repeat their encoding costs.'},
        {'scope': 'Other C model analysis/publication, future optional C-best exports and GPU reproduction are not manufactured from preparation or capacity reservations.'},
    ])
    spec['revision6_additional_operation_ids'] = ids
    spec['revision6_normalized_sources'] = normalized
    spec['revision6_candidate_source'] = proof(candidates_path)
    spec['revision6_C410_assembly_handling'] = contexts
    spec['metadata_sources'].extend([proof(old), proof(candidates_path), *origins,
        proof(complete_path), proof(ROOT / 'artifacts/provenance/C410_nll_analysis_actual_v3/ROOT_precheck_v1_failure.json')])
    assert spec['additional_operations'][:len(old_ops)] == old_ops
    all_ids = [item['id'] for item in spec['additional_operations']]
    assert len(all_ids) == len(set(all_ids))
    assert spec['C_exports'] == read(old)['C_exports']
    target = ROOT / 'configs/cost_inventory_final_sources_v6.json'
    write(target, spec)
    write(HERE / 'explicit_new_registration_table.json', measured)
    write(HERE / 'preparation_receipt.json', {
        'schema_version': 1, 'status': 'prepared_explicit_cost_sources_not_whole_study_complete',
        'source_spec': proof(target), 'preserved_v5_source': proof(old),
        'unchanged_collector': proof(collector), 'producer': proof(__file__),
        'additional_operation_rows': len(ids), 'additional_disjoint_measured_rows': len(measured),
        'explicit_disjoint_recorded_wall_sum_seconds': sum(row['measured_wall_seconds'] for row in measured),
        'explicit_known_inclusive_CPU_sum_seconds': sum(row['known_inclusive_CPU_seconds'] or 0 for row in measured),
        'table': proof(HERE / 'explicit_new_registration_table.json'),
        'C410_real_NLL_new_forward_jobs': 43,
        'C410_manual_additional_inference_cost_rows': 0,
        'existing_C_assembly_dynamic_row_added_again': False,
        'C410_nested_publication_added_to_whole_child': False,
        'GB_checkpoint_or_archive_payload_reads': 0,
        'score_or_prediction_semantics_read': False,
        'whole_study_complete': False,
    })
    print(json.dumps({'source_spec': proof(target), 'new_disjoint_operations': len(measured), 'new_total_rows': len(ids)}))


if __name__ == '__main__':
    main()
