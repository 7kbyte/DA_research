#!/usr/bin/env python3
"""Independent cost arithmetic/source review; never invokes the collector."""
import csv
import datetime as dt
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NEW = ROOT / 'results/cost_inventory_final_v8_cpu_collection_20261106_attempt1'
OLD = ROOT / 'results/cost_inventory_final_v7_cpu_collection_20261106_attempt2'
evidence = {}; cache = {}; opened = []
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = str(args[0])
        assert not path.endswith(('.pt', '.safetensors', '.zip', '.tar', '.tar.gz', '.bin', '.npy', '.npz')), path
sys.addaudithook(audit)

def read(path, expected=None, parse=True):
    path = Path(path).resolve(); key = str(path)
    assert path.suffix in ('.json', '.csv', '.py', '.md', '.log')
    assert path.stat().st_size < 32 << 20
    if key not in cache:
        before = path.stat(); raw = path.read_bytes(); after = path.stat()
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        assert stamp(before) == stamp(after)
        cache[key] = raw; opened.append(key)
    raw = cache[key]; digest = hashlib.sha256(raw).hexdigest()
    assert expected is None or digest == expected, key
    evidence[key] = {'path': key, 'sha256': digest, 'bytes': len(raw)}
    return json.loads(raw) if parse else raw

def pointer(obj, text):
    for key in text.split('/')[1:]:
        key = key.replace('~1', '/').replace('~0', '~')
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj

def close(a, b): assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-7), (a, b)
def summed(rows):
    return sum((Decimal(str(r['recorded_wall_seconds'])) for r in rows if r['included_in_disjoint_recorded_wall_sum']), Decimal(0))

spec = read(ROOT/'configs/cost_inventory_final_sources_v8.json', '572cb3bf1e53bac59f0d5f5a60341e1c8a78dd1b592c1cf6b5689dfef673ec03')
previous = read(ROOT/'configs/cost_inventory_final_sources_v7.json', '0cab816da531e857ded88a004624d9a946a1bd3dae75f295ed68e1e4f40b1f0e')
read(ROOT/'configs/cost_inventory_final_sources_v6.json', 'e4c99db9b1803681ad8706bb90480ede0be584975324040df979c1c9966b17e0')
read(ROOT/'scripts/inventory_study_costs_final_v2.py', '864effb89f20dbfce3ba6ce51db8eb8351b087c2ac92d255274fe6267753f6ae', False)
read(ROOT/'artifacts/provenance/cost_sources_v7/run_collection_with_IO_audit.py', 'a3513d44a64adbab37feafcb1f7d0e15a31a61defac7572cd5482621775bede9', False)
completion = read(ROOT/'artifacts/provenance/cost_sources_v8/COMPLETION.json', 'ae5a44bec6b46e0de18b8a62ea02be02378b0201d4d8b57d3a1352574057750a')
manifest = read(ROOT/'artifacts/provenance/cost_sources_v8/source_manifest.json', '110088540df7a40406d61a2b7bfcb86d5a1223eaec2d86f9128580ec7ccbfbb4')
for row in manifest['files']:
    read(row['path'], row['sha256'], False)
receipt = read(NEW/'collection_receipt.json', '28c7f5042cf71005eba88ad66a7312d87a955ad521be2981c337aa06604aa2e9')
assert receipt['whole_study_complete'] is False and receipt['score_semantics_read'] is False
assert receipt['GB_model_or_archive_payloads_rehashed'] is False
new = read(NEW/'inventory.json', receipt['inventory']['sha256']); old = read(OLD/'inventory.json')
assert read(NEW/'source_spec_v2.json') == spec
assert spec['additional_operations'][:len(previous['additional_operations'])] == previous['additional_operations']
assert spec['metadata_sources'][:len(previous['metadata_sources'])] == previous['metadata_sources']
assert spec['C_exports'] == previous['C_exports']
base = read(spec['v1_source_spec']['path'], spec['v1_source_spec']['sha256'])
legacy = read(base['legacy_source_spec']['path'], base['legacy_source_spec']['sha256'])
assert all(row['verify_output_hashes'] is True for row in legacy['evaluation_manifests'])

copies = new['cost_only_v2_extension']['exact_cost_source_copies']
assert copies == receipt['exact_cost_source_copies'] and len(copies) == 432
assert len({row['original_path'] for row in copies}) == len({row['snapshot_path'] for row in copies}) == 432
for row in copies:
    raw = read(row['snapshot_path'], row['sha256'], False)
    assert len(raw) == row['bytes'] and Path(row['snapshot_path']).parent == NEW/'exact_cost_sources'
assert sum(row['bytes'] for row in copies) == 24620818
copy_index = {row['original_path']: row for row in copies}
def captured(path, expected=None):
    row = copy_index[str(Path(path).resolve())]
    assert expected is None or row['sha256'] == expected
    return read(row['snapshot_path'], row['sha256'])

rows = {row['id']: row for row in new['operational_costs']}
prior = {row['id']: row for row in old['operational_costs']}
assert len(rows) == len(new['operational_costs']) == 376 and len(prior) == len(old['operational_costs'])
parent_ids = spec['revision8_additional_parent_operation_ids']; region_ids = spec['revision8_separate_region_operation_ids']
assert len(parent_ids) == 46 and len(region_ids) == 14 and not (set(parent_ids) & set(region_ids))
assert set(rows) >= set(prior)
old_timer_paths = set(); old_timer_shas = set()
for operation in previous['additional_operations']:
    obj = captured(operation['source']['path'], operation['source']['sha256'])
    origin = obj.get('original_source', operation['source'])
    ptr = obj.get('original_wall_pointer', operation['wall_pointer'])
    old_timer_paths.add((str(Path(origin['path']).resolve()), ptr)); old_timer_shas.add((origin['sha256'], ptr))
seen_paths = set(); seen_shas = set(); normalized_checks = []
for operation in spec['additional_operations'][len(previous['additional_operations']):]:
    row = rows[operation['id']]
    obj = read(operation['source']['path'], operation['source']['sha256'])
    origin = obj['original_source']; original = captured(origin['path'], origin['sha256'])
    ptr = obj['original_wall_pointer']; path_key = (str(Path(origin['path']).resolve()), ptr); sha_key = (origin['sha256'], ptr)
    assert path_key not in old_timer_paths | seen_paths and sha_key not in old_timer_shas | seen_shas
    seen_paths.add(path_key); seen_shas.add(sha_key)
    assert obj['wall_seconds'] == pointer(original, ptr) == row['recorded_wall_seconds']
    cpu_values = [pointer(original, p) for p in obj['CPU_component_pointers']]
    assert obj['recorded_inclusive_CPU_seconds'] == (sum(cpu_values) if cpu_values else None) == row['recorded_process_cpu_seconds']
    assert row['included_in_disjoint_recorded_wall_sum'] is (operation['id'] in parent_ids)
    assert obj['nested_subtimers_added'] is False and obj['new_operation_created_by_normalization'] is False
    normalized_checks.append({'id': operation['id'], 'source': origin, 'wall_pointer': ptr,
        'wall_seconds': row['recorded_wall_seconds'], 'original_returncode': obj['original_returncode'],
        'additive_parent': row['included_in_disjoint_recorded_wall_sum']})
assert len(normalized_checks) == 60
parents = [rows[key] for key in parent_ids]; regions = [rows[key] for key in region_ids]
static = summed(parents); close(static, 12678.339845579118)
region_sum = sum(Decimal(str(row['recorded_wall_seconds'])) for row in regions); close(region_sum, 207.72160545364022)
failures = [row for row in normalized_checks if row['additive_parent'] and row['original_returncode'] != 0]
assert len(failures) == 7 and all(type(row['original_returncode']) is int for row in failures)
added = [row for key, row in rows.items() if key not in prior and key not in set(parent_ids + region_ids) and row['included_in_disjoint_recorded_wall_sum']]
assert len(added) == 7 and all(row['id'].startswith('C_draft:') for row in added)
assert sum(':pythia-1b:' in row['id'] for row in added) == 6 and sum(':pythia-2.8b:' in row['id'] for row in added) == 1
new_drafts = summed(added); close(new_drafts, 2190.3036724217236)
changed = []
for key in prior:
    a, b = prior[key], rows[key]
    if a['included_in_disjoint_recorded_wall_sum'] != b['included_in_disjoint_recorded_wall_sum'] or a['recorded_wall_seconds'] != b['recorded_wall_seconds']:
        changed.append(key)
assert changed == ['opaque_stable_full_file_reads']
opaque_delta = Decimal(str(rows[changed[0]]['recorded_wall_seconds'])) - Decimal(str(prior[changed[0]]['recorded_wall_seconds']))
close(opaque_delta, 153.74574811384082)
old_total = summed(prior.values()); new_total = summed(rows.values())
close(old_total, 24419.945426069); close(new_total, 39442.334692184)
close(new_total - old_total, static + new_drafts + opaque_delta)
for value in (old, new):
    opaque = value['historical_opaque_hash_receipts']
    assert len({row['receipt_sha256'] for row in opaque}) == len(opaque)
    close(sum(row['historical_full_read_seconds'] for row in opaque), next(row['recorded_wall_seconds'] for row in value['operational_costs'] if row['id'] == 'opaque_stable_full_file_reads'))
    assert all(row['current_inventory_input_payload_reads'] == 0 and row['historical_read_is_current_cache_hit'] is False for row in opaque)

io_audit = read(NEW/'IO_audit.json', completion['IO_audit']['sha256'])
assert io_audit['unique_score_outputs_stream_hashed'] == 332
assert io_audit['cumulative_score_stream_SHA_bytes'] == 456416316
assert sum(row['file_bytes'] * row['full_stream_hash_open_count'] for row in io_audit['score_output_stream_hash_reads']) == 456416316
assert len({row['path'] for row in io_audit['score_output_stream_hash_reads']}) == 332
assert io_audit['score_output_semantic_deserialization'] is False and io_audit['score_SHA_verification_policy_changed'] is False
assert io_audit['weight_base_TAR_ZIP_tensor_payload_bytes_read'] == 0 and io_audit['forbidden_weight_base_archive_tensor_payload_open_attempts'] == []
score_paths = {row['path'] for row in io_audit['score_output_stream_hash_reads']}
assert not score_paths & set(opened)
def eval_counts(value):
    return {row['category']: (row['attested_new_forward_jobs'], row['attested_reuse_jobs']) for row in value['legacy_method_and_preparation_ledger']['evaluation_category_totals']}
assert eval_counts(old) == eval_counts(new) == {'diagnostic': (17, 0), 'primary': (68, 0), 'supplement': (211, 18)}
assert new['cost_only_v2_extension']['B_NLL_additional_manual_inference_cost_rows'] == 0
assert all(row['pure_CPU_or_GPU_work_claim'] is False and row['unrecorded_work_is_zero'] is False for row in rows.values())

pub = rows['v8_C410_best_publication_internal_region']
assert pub['recorded_wall_seconds'] == 195.2169531546533 and pub['recorded_process_cpu_seconds'] is None
assert pub['included_in_disjoint_recorded_wall_sum'] is False
failure = captured(ROOT/'artifacts/provenance/C410_best_publication_actual_v1/outer_timing_receipt_failure.json')
assert failure['outer_script_returncode_observed'] == 1
assert all(failure[key] is None for key in ['publisher_returncode_not_persisted', 'exact_publisher_spawn_to_exit_wall_seconds_not_persisted', 'publisher_whole_user_CPU_seconds_not_persisted', 'publisher_whole_system_CPU_seconds_not_persisted'])
assert sum(row['recorded_wall_seconds'] == 1953.1042039617896 and row['included_in_disjoint_recorded_wall_sum'] for row in rows.values()) == 1
assert rows['v8_B_native_actual_restore']['recorded_wall_seconds'] == 36.18444097042084
assert rows['B_nll_CPU_analysis_unknown']['recorded_wall_seconds'] is None
assert not any('C410_best_publication_actual_v1/batch_whole_child_timing.json' in str(row) for row in spec['additional_operations'])
report_ids = [key for key in parent_ids if key.startswith(('v8_report_delivery_section7_render_', 'v8_report_QA_delivery_section7_'))]
assert len(report_ids) == 5; close(sum(rows[key]['recorded_wall_seconds'] for key in report_ids), 42.84236093610525)
correction = captured(ROOT/'artifacts/provenance/report_delivery_section7_update_v2/correction_receipt.json')
assert correction['status'] == 'historical_scope_correction'
assert correction['all_17_pages_extracted_text_identical'] is True and correction['failed_edit_parent_wall_seconds'] is None
assert rows['v8_snapshot1918_bounded_review_attempt1']['recorded_wall_seconds'] == 21.441286388784647
assert rows['v8_snapshot1918_bounded_review_attempt2']['recorded_wall_seconds'] == 32.06563150137663
assert rows['v8_cost_v6_successful_collection']['recorded_wall_seconds'] == 6.833468526601791
assert rows['v8_cost_v7_successful_collection']['recorded_wall_seconds'] == 8.851966518908739
timer = read(completion['collection_whole_child_timing']['path'], completion['collection_whole_child_timing']['sha256'])
prep = read(completion['source_preparation_whole_child_timing']['path'], completion['source_preparation_whole_child_timing']['sha256'])
assert timer['returncode'] == 0 and timer['whole_child_wall_seconds'] == 8.667578175663948
assert not any(row.get('source') in (timer, prep) for row in rows.values())
assert not any('cost_sources_v8/collection_whole_child_timing.json' in str(row) or 'cost_sources_v8/preparation_whole_child_timing.json' in str(row) for row in spec['additional_operations'])
assert completion['current_collection_self_registered'] is False and completion['whole_study_complete'] is False
coverage = {row['id']: row['summary_completed_runs'] for row in new['training_cohort_coverage']}
assert coverage == {'primary36': 36, 'A108_additional_plus18_training_aliases': 108, 'B60': 60, 'C410m24': 24, 'C1b24': 20, 'C2p8b15_h25m': 2}
exports = {row['model']: row['actual_first_created_records_observed'] for row in new['cost_only_v2_extension']['C_export_record_coverage']}
assert exports == {'pythia-410m': 24, 'pythia-1b': 18, 'pythia-2.8b': 2}

csv_rows = list(csv.DictReader(io.StringIO(read(NEW/'operational_costs.csv', parse=False).decode())))
assert {row['id'] for row in csv_rows} == set(rows)
for row in csv_rows:
    value = rows[row['id']]
    assert row['included_in_disjoint_recorded_wall_sum'] == str(value['included_in_disjoint_recorded_wall_sum'])
    assert row['recorded_wall_seconds'] == (str(value['recorded_wall_seconds']) if value['recorded_wall_seconds'] is not None else '')
read(__file__, parse=False)
result = {'status': 'passed_independent_cost_v8_small_metadata_and_arithmetic_review',
    'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'unmodified_original_collector_and_IO_audit': True,
    'static_parent_count': 46, 'failed_parent_count': 7, 'static_parent_wall_seconds': float(static),
    'nonadditive_region_count': 14, 'nonadditive_region_wall_seconds': float(region_sum),
    'dynamic_new_drafts': [{'id': row['id'], 'wall_seconds': row['recorded_wall_seconds']} for row in added],
    'dynamic_new_draft_wall_seconds': float(new_drafts), 'historical_opaque_wall_growth_seconds': float(opaque_delta),
    'old_disjoint_operation_wall_seconds': float(old_total), 'new_disjoint_operation_wall_seconds': float(new_total),
    'disjoint_operation_delta_seconds': float(new_total-old_total), 'all60_normalized_original_pointers_verified': normalized_checks,
    'original_path_and_SHA_plus_pointer_dedup_against_v7_passed': True,
    'exact_metadata_copy_count': 432, 'exact_metadata_copy_bytes_fresh_hashed': 24620818,
    'inference_counts': eval_counts(new), 'collector_opaque_score_bytes': 456416316,
    'independent_score_payload_reads_or_hashes': 0, 'independent_large_payload_bytes_read': 0,
    'C410best_exact_publisher_whole_wall_CPU_child_rc': None,
    'C410best_publisher_only_nonadditive_region_seconds': 195.2169531546533,
    'B_native_external_parent_only_seconds': 36.18444097042084,
    'current_v8_collection_wall_seconds_excluded_from_v8': timer['whole_child_wall_seconds'],
    'training_completed_counts': coverage, 'actual_C_final_draft_counts': exports,
    'whole_study_complete': False, 'manual_engineering_complete_parent_seconds': None,
    'source_proofs': list(evidence.values()), 'unique_explicit_small_files_read': len(opened),
    'unique_small_serialized_bytes_read': sum(len(raw) for raw in cache.values())}
with (HERE/'REVIEW.json').open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
print(json.dumps({key: result[key] for key in ['status', 'disjoint_operation_delta_seconds', 'new_disjoint_operation_wall_seconds', 'unique_explicit_small_files_read', 'unique_small_serialized_bytes_read']}))
