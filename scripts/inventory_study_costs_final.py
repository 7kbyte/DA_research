#!/usr/bin/env python3
"""Score-blind final-collection entry; keeps the legacy method ledger unchanged."""
import argparse
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import inventory_study_costs as legacy
from stat_bound_digest_cache import atomic, small_proof

OPERATION_CATEGORIES = {'cpu_analysis', 'opaque_prehash', 'lossless_export',
                        'storage_publication', 'archive_qa', 'storage_wait_and_reservation'}


def read_bound(reader, proof, optional=False):
    path = str(Path(proof['path']).resolve())
    value = reader.read(path)
    if value is None:
        if optional:
            return None
        raise ValueError('Required source missing or invalid: ' + path)
    if proof.get('sha256') and reader.sources[path]['sha256'] != proof['sha256']:
        raise ValueError('Source SHA mismatch: ' + path)
    return value


def scalar(value, pointer):
    matches = legacy.pointer_values(value, pointer) if value is not None and pointer else []
    if len(matches) > 1:
        raise ValueError('A cost metric must select exactly one value')
    return matches[0][1] if matches else None


def merge_states(sources, reader):
    merged = {'jobs': {}}
    bindings = []
    for source in sources:
        state = read_bound(reader, source, source.get('optional', False))
        path = str(Path(source['path']).resolve())
        bindings.append({'path': path, **reader.sources[path]})
        if state is None:
            continue
        for jid, row in state.get('jobs', {}).items():
            previous = merged['jobs'].get(jid)
            if previous is None:
                merged['jobs'][jid] = row
                continue
            if not previous.get('definition_sha256') or previous['definition_sha256'] != row.get('definition_sha256'):
                raise ValueError('Conflicting scheduler job definition: ' + jid)
            completed = lambda entry: entry.get('status') == 'completed' and entry.get('returncode') == 0
            if completed(previous) and completed(row):
                if any(previous.get(key) != row.get(key) for key in ('elapsed_seconds', 'started_at_utc', 'ended_at_utc')):
                    raise ValueError('Two different completed attempts require explicit attempt accounting: ' + jid)
            elif completed(row):
                merged['jobs'][jid] = row
            elif not completed(previous):
                # Preserve the last explicit state's active status, never invent success.
                merged['jobs'][jid] = row
    return merged, bindings


def effective_sources(spec, reader, merged_state_path):
    base = copy.deepcopy(read_bound(reader, spec['legacy_source_spec']))
    plans = list(base.get('evaluation_manifests', []))
    readiness = []
    for source in spec.get('evaluation_ready_indexes', []):
        catalog = read_bound(reader, source['catalog'])
        index = read_bound(reader, {'path': source['path']}, source.get('optional', False))
        if index is None:
            readiness.append({'path': source['path'], 'status': 'missing_ready_index'})
            continue
        if index.get('catalog', {}).get('sha256') != source['catalog']['sha256']:
            raise ValueError('Ready index is not bound to the explicit catalog')
        expected_ids = {plan['id'] for plan in catalog['plans']}
        phases = index.get('phases', [])
        if len({phase['id'] for phase in phases}) != len(phases) or {phase['id'] for phase in phases} != expected_ids:
            raise ValueError('Ready index phase identities differ from catalog')
        for phase in phases:
            row = {'id': phase['id'], 'status': phase['status'], 'ready': phase.get('ready', False)}
            readiness.append(row)
            if not phase.get('ready'):
                continue
            manifest = read_bound(reader, phase['manifest'])
            queue = read_bound(reader, phase['queue'])
            row.update(manifest=phase['manifest'], queue=phase['queue'], logical_jobs=len(manifest['jobs']))
            if len(queue) != len(manifest['jobs']):
                raise ValueError('Ready queue/manifest job counts differ')
            plans.append({**phase['manifest'], 'category': 'supplement', 'verify_output_hashes': True})
    unique = {}
    for item in plans:
        manifest = read_bound(reader, item)
        sha = item['sha256']
        if sha in unique and unique[sha]['category'] != item['category']:
            raise ValueError('One manifest has conflicting accounting categories')
        unique[sha] = {**item, 'scheduler_state': str(merged_state_path.resolve())}
    base['evaluation_manifests'] = list(unique.values())
    # Old collector and its postprocessing interpretation remain an explicit
    # reference sub-ledger. Operational totals below never consume that subtotal.
    return base, readiness


def operation_rows(spec, reader):
    rows = []
    by_id = {}
    duplicates = []
    for item in spec.get('operations', []):
        if item['category'] not in OPERATION_CATEGORIES:
            raise ValueError('Unsupported operational category')
        source = item['source']
        receipt = read_bound(reader, source, optional=True)
        path = str(Path(source['path']).resolve())
        status = scalar(receipt, item.get('status_pointer'))
        accepted = receipt is not None and (not item.get('accepted_statuses') or status in item['accepted_statuses'])
        wall = legacy.number(scalar(receipt, item.get('wall_pointer')))
        cpu = legacy.number(scalar(receipt, item.get('process_cpu_pointer')))
        start = scalar(receipt, item.get('calendar_start_pointer'))
        end = scalar(receipt, item.get('calendar_end_pointer'))
        window = None
        if start and end:
            window = (dt.datetime.fromisoformat(end.replace('Z', '+00:00')) -
                      dt.datetime.fromisoformat(start.replace('Z', '+00:00'))).total_seconds()
            if window < 0:
                raise ValueError('Negative calendar envelope')
        additive = bool(accepted and wall is not None and item.get('add_to_recorded_wall_totals', False))
        if item['timing_semantics'] in {'calendar_envelope', 'wall_including_waiting', 'cumulative_parent_sample'} and additive:
            raise ValueError('Context, waiting, or cumulative child cannot be counted as disjoint work')
        row = {'id': item['id'], 'category': item['category'], 'parent_id': item.get('parent_id'),
               'source': path, 'source_sha256': reader.sources[path].get('sha256'),
               'source_wall_pointer': item.get('wall_pointer'), 'receipt_status': status,
               'accepted_receipt': accepted, 'recorded_wall_seconds': wall if accepted else None,
               'recorded_process_cpu_seconds': cpu if accepted else None,
               'calendar_window_seconds': window, 'timing_semantics': item['timing_semantics'],
               'included_in_disjoint_recorded_wall_sum': additive,
               'pure_CPU_or_GPU_work_claim': False, 'included_in_method_comparison_costs': False,
               'unrecorded_work_is_zero': False, 'scope': item['scope']}
        identity = item.get('canonical_id', item['id'])
        if identity in by_id:
            other = by_id[identity]
            if any(other[key] != row[key] for key in ('category', 'recorded_wall_seconds', 'recorded_process_cpu_seconds', 'timing_semantics')):
                raise ValueError('Conflicting operational receipt duplicates')
            duplicates.append({'kept_id': other['id'], 'duplicate_id': row['id'], 'canonical_id': identity})
            continue
        by_id[identity] = row
        rows.append(row)
    ids = {row['id']: row for row in rows}
    for row in rows:
        parent = row['parent_id']
        seen = {row['id']}
        while parent:
            if parent in seen or parent not in ids:
                raise ValueError('Missing or cyclic operational parent')
            seen.add(parent)
            ancestor = ids[parent]
            if row['included_in_disjoint_recorded_wall_sum'] and ancestor['included_in_disjoint_recorded_wall_sum']:
                raise ValueError('An included parent and child would double-count wall')
            parent = ancestor['parent_id']
    return rows, duplicates


def opaque_hash_receipts(spec, reader):
    rows = []
    seen = set()
    progress = []
    for source in spec.get('opaque_digest_caches', []):
        for path in sorted(Path(source['directory']).glob('receipts/*.json')):
            receipt = read_bound(reader, {'path': str(path)})
            sha = reader.sources[str(path.resolve())]['sha256']
            if sha != path.stem:
                raise ValueError('Opaque full-read receipt filename does not bind its bytes')
            if sha in seen:
                continue
            seen.add(sha)
            stats = [receipt.get(key) for key in ('stat_before', 'stat_after', 'fd_before', 'fd_after')]
            seconds = legacy.number(receipt.get('full_read_seconds'))
            if (receipt.get('method') != 'sha256_entire_open_fd_to_eof_v1'
                    or not receipt.get('complete_to_eof') or not stats[0]
                    or any(value != stats[0] for value in stats)
                    or receipt.get('read_bytes') != stats[0]['size'] or seconds is None
                    or receipt.get('reader_source', {}).get('sha256') != source['reader_sha256']):
                raise ValueError('Invalid stable complete-read cost evidence')
            rows.append({'receipt': str(path.resolve()), 'receipt_sha256': sha,
                         'canonical_input': receipt['canonical_path'], 'input_sha256': receipt['sha256'],
                         'historical_full_read_seconds': seconds, 'historical_full_read_bytes': receipt['read_bytes'],
                         'source_authority': receipt.get('source_authority'),
                         'current_inventory_input_payload_reads': 0,
                         'historical_read_is_current_cache_hit': False})
        value = read_bound(reader, {'path': source['progress']}, optional=True)
        progress.append({'source': source['progress'], 'snapshot': value,
                         'counters_are_process_cumulative_not_disjoint_stages': True,
                         'full_read_or_cache_hit_counts_added_as_wall': False})
    return rows, progress


def cohort_rows(spec, reader, result):
    physical = {row['canonical_run_dir']: row for row in result['physical_runs']}
    rows = []
    for cohort in spec['training_cohorts']:
        registry = read_bound(reader, cohort['registry'])
        aliases = [row for row in registry if legacy.alias(row)]
        actual = [row for row in registry if not legacy.alias(row)]
        paths = {str(Path(row['output_dir']).resolve()) for row in actual}
        if (len(registry) != cohort['expected_logical_rows'] or len(actual) != cohort['expected_physical_runs']
                or len(aliases) != cohort['expected_training_aliases'] or len(paths) != len(actual)):
            raise ValueError('Explicit formal cohort size or training alias contract differs')
        observed = [physical[path] for path in paths if path in physical]
        known_parent = [row['known_parent_component_sum_seconds'] for row in observed
                        if row['known_parent_component_sum_seconds'] is not None]
        group_paths = {group: {str(Path(row['output_dir']).resolve()) for row in actual if row.get('group') == group}
                       for group in ('D', 'B4')}
        rows.append({'id': cohort['id'], 'source_registry': cohort['registry'],
                     'expected_physical_runs': len(actual), 'logical_rows': len(registry),
                     'training_aliases_not_new_runs': len(aliases), 'observed_run_directories': len(observed),
                     'summary_completed_runs': sum(row['status'] == 'completed' for row in observed),
                     'incomplete_or_unobserved_physical_runs': len(actual) - sum(row['status'] == 'completed' for row in observed),
                     'D_group_independent_training_definitions': len(group_paths['D']),
                     'B4_group_independent_training_definitions': len(group_paths['B4']),
                     'D_group_observed_physical_runs': sum(path in physical for path in group_paths['D']),
                     'B4_group_observed_physical_runs': sum(path in physical for path in group_paths['B4']),
                     'known_parent_wall_sum_seconds': sum(known_parent) if known_parent else None,
                     'parent_wall_missing_or_unobserved_runs': len(actual) - len(known_parent),
                     'teacher_children_added_to_parent_sum': False,
                     'committed_path_partial_not_all_attempts': True})
    return rows


def collect(spec, output, reader=None):
    reader = reader or legacy.Reader()
    for source in spec.get('legacy_collector_dependencies', []):
        path = str(Path(source['path']).resolve())
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != source['sha256']:
            raise ValueError('Legacy collector dependency changed: ' + path)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh snapshot output; never overwrite previous cost evidence')
    output.mkdir(parents=True)
    states, bindings = merge_states(spec['evaluation_scheduler_states'], reader)
    state_path = output / 'effective_evaluation_scheduler_state.json'
    atomic(state_path, states)
    effective, readiness = effective_sources(spec, reader, state_path)
    atomic(output / 'effective_legacy_source_spec.json', effective)
    result = legacy.inventory(effective, spec['runs_root'], reader)
    operations, duplicates = operation_rows(spec, reader)
    hashes, progress = opaque_hash_receipts(spec, reader)
    if hashes:
        operations.append({'id': 'opaque_stable_full_file_reads', 'category': 'opaque_prehash', 'parent_id': None,
                           'recorded_wall_seconds': sum(row['historical_full_read_seconds'] for row in hashes),
                           'recorded_process_cpu_seconds': None, 'calendar_window_seconds': None,
                           'timing_semantics': 'deduplicated_historical_complete_read_receipts',
                           'included_in_disjoint_recorded_wall_sum': True, 'accepted_receipt': True,
                           'pure_CPU_or_GPU_work_claim': False, 'included_in_method_comparison_costs': False,
                           'unrecorded_work_is_zero': False,
                           'scope': 'Unique receipt SHA, not input SHA: genuine repeated full reads are counted, references/hits are not.'})
    totals = []
    for category in sorted(OPERATION_CATEGORIES):
        selected = [row for row in operations if row['category'] == category]
        included = [row for row in selected if row['included_in_disjoint_recorded_wall_sum']]
        totals.append({'category': category, 'records': len(selected), 'included_disjoint_recorded_wall_rows': len(included),
                       'known_disjoint_recorded_wall_seconds': sum(row['recorded_wall_seconds'] for row in included) if included else None,
                       'missing_or_unaccepted_recorded_wall_rows': sum(not row['accepted_receipt'] or row['recorded_wall_seconds'] is None for row in selected),
                       'nonadditive_context_or_child_rows': len(selected) - len(included),
                       'sum_is_study_elapsed_or_pure_CPU_work': False, 'coverage_complete': False})
    cohorts = cohort_rows(spec, reader, result)
    return {'schema_version': 1, 'status': 'observed_snapshot_with_explicit_gaps',
            'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'legacy_method_and_preparation_ledger': result,
            'operational_costs': operations, 'operational_category_totals': totals,
            'operational_deduplication': duplicates, 'historical_opaque_hash_receipts': hashes,
            'opaque_cache_progress_context_nonadditive': progress,
            'training_cohort_coverage': cohorts, 'v5_evaluation_readiness': readiness,
            'unmeasured_activity_gaps': spec.get('unmeasured_activity_gaps', []),
            'scheduler_source_bindings': bindings, 'source_snapshots': reader.sources,
            'accounting_contract': {
                'whole_study_completely_instrumented': False, 'unmeasured_or_rolled_back_work_imputed_zero': False,
                'scores_or_prediction_JSON_semantics_parsed': False, 'checkpoint_or_tensor_payloads_read': False,
                'teacher_timers_nested_not_additive': True,
                'training_aliases_are_not_new_runs': True, 'D_B4_independent_training_both_counted': True,
                'inference_identity': 'Explicit canonical manifest SHA and inference job ID; proven reuse emits zero new forward.',
                'old_inventory_and_v3_source_spec_modified': False,
                'legacy_postprocessing_subledger_added_to_operational_totals': False,
                'parallel_wall_sum_is_study_elapsed_or_GPU_seconds': False,
                'readiness_is_execution_completion': False,
                'loss_bounds_are_not_added_as_exact_cost': True,
                'historical_cache_receipt_hits_are_current_full_payload_reads': False,
                'snapshot_is_atomic_across_live_study': False,
            }}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    begin = time.perf_counter()
    cpu_begin = time.process_time()
    reader = legacy.Reader()
    spec = read_bound(reader, {'path': str(args.sources)})
    result = collect(spec, args.output, reader)
    for key in ('operational_costs', 'operational_category_totals', 'historical_opaque_hash_receipts', 'training_cohort_coverage', 'v5_evaluation_readiness'):
        legacy.csv_write(args.output / (key + '.csv'), result[key])
    base = result['legacy_method_and_preparation_ledger']
    for key in ('physical_runs', 'registry_entries', 'preparation_costs', 'loss_bounds', 'category_totals', 'evaluation_manifest_status', 'evaluation_executions', 'evaluation_canonical_inferences', 'evaluation_references', 'evaluation_category_totals'):
        legacy.csv_write(args.output / (key + '.csv'), base[key])
    result['collector_execution'] = {'source': small_proof(__file__), 'source_spec': small_proof(args.sources),
                                     'wall_seconds': time.perf_counter() - begin,
                                     'process_cpu_seconds': time.process_time() - cpu_begin,
                                     'self_timer_not_recursively_added_to_ledger': True}
    atomic(args.output / 'inventory.json', result)
    print(json.dumps({'output': str(args.output), 'cohort_physical_runs_planned': sum(row['expected_physical_runs'] for row in result['training_cohort_coverage']),
                      'evaluation_attested_forward_jobs': sum(row['recorded_additional_model_inference'] for row in base['evaluation_executions']),
                      'operational_records': len(result['operational_costs']), 'whole_study_complete': False, 'scores_parsed': False}))


if __name__ == '__main__':
    main()
