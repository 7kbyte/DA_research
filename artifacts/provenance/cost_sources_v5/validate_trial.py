#!/usr/bin/env python3
"""Targeted revision5 normalization/nonoverlap and live-snapshot delta checks."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/cost_inventory_final_v5_cpu_trial_20261106'
load = lambda path: json.loads(Path(path).read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pointer(value, path):
    for part in path.strip('/').split('/'):
        value = value[part]
    return value


old = load(ROOT / 'results/cost_inventory_final_v4_cpu_trial_20261106/inventory.json')
new = load(OUT / 'inventory.json')
spec = load(ROOT / 'configs/cost_inventory_final_sources_v5.json')
old_spec = load(ROOT / 'configs/cost_inventory_final_sources_v4.json')
unchanged = {
    'scripts/inventory_study_costs_final.py': 'c0c9d4605da8809e48aa24bdbeca664c86cdf613fe4aef4a0fc35614d17f8b95',
    'scripts/inventory_study_costs_final_v2.py': '864effb89f20dbfce3ba6ce51db8eb8351b087c2ac92d255274fe6267753f6ae',
    'configs/cost_inventory_final_sources_v1.json': '3b9986f853a656b7dd16cc4adc8a8b5d7194bf6f8db478ed4785ffb84167847c',
    'configs/cost_inventory_final_sources_v2.json': '3c0286f39fb19f386bbb09a68143c80cd4fe45677efaf360309b032712b14ce7',
    'configs/cost_inventory_final_sources_v3.json': '4fdc24ea2c5995e76b032d29803de1e038114d667b9dd7ef7753b220abf7ea2a',
    'configs/cost_inventory_final_sources_v4.json': 'a1b56243519aa3a649859b88351156e8ce9b8006d139114635c55fdbef85c959',
    'results/cost_inventory_final_v2_cpu_trial_20261106/inventory.json': '009c70654649936a78045234129a739ed52e95923162b6f122771019785b904f',
    'results/cost_inventory_final_v3_cpu_trial_20261106/inventory.json': '2af4855e2990e8dc5df4bfd2b491a21a17e3fb1d9231934ba299a1db6fd40ed0',
    'results/cost_inventory_final_v4_cpu_trial_20261106/inventory.json': 'e9984cde3102a4b5a22c6f7f4b045077205932fba39d683afe4b40927bf7ec39',
    'results/cost_inventory_final_v4_cpu_trial_20261106/collection_receipt.json': '0150d047db01a172189b5fd6a0f06a700d91c82069e0bf11b177b5861c18de83',
    'results/cost_inventory_final_v4_cpu_trial_20261106/delta_from_v3_and_validation.json': 'c5bd7fe03a3e290802a9d951de57df03a9fe32844f6b9cfd442c9e9c5aff0ff5',
}
for path, expected in unchanged.items():
    assert sha(ROOT / path) == expected, path
assert spec['additional_operations'][:len(old_spec['additional_operations'])] == old_spec['additional_operations']
assert spec['C_exports'] == old_spec['C_exports']
assert all(item['publication_receipt'] is None for item in spec['C_exports']['models'])
before = {row['id']: row for row in old['operational_costs']}
after = {row['id']: row for row in new['operational_costs']}
ids = spec['revision5_additional_operation_ids']
rows = [after[identity] for identity in ids]
assert len(ids) == len(set(ids)) == 6
assert all(row['accepted_receipt'] and row['included_in_disjoint_recorded_wall_sum'] for row in rows)
assert len(spec['revision5_normalized_sources']) == 4
for normalized in spec['revision5_normalized_sources']:
    assert sha(normalized['path']) == normalized['sha256']
    value = load(normalized['path']); original = value['original_source']
    assert sha(original['path']) == original['sha256']
    actual = load(original['path'])
    assert value['wall_seconds'] == pointer(actual, value['original_wall_pointer'])
    components = [pointer(actual, path) for path in value['CPU_component_pointers']]
    assert math.isclose(sum(components), value['recorded_inclusive_CPU_seconds'], abs_tol=1e-12)
    assert value['nested_subtimers_added'] is False and value['new_operation_created_by_normalization'] is False
body = next(row for row in spec['additional_operations'] if row['id'] == 'optional_three_model_B4_preparation_body_region_actual')
assert body['timing_semantics'] == 'independent_measured_body_region_after_stdlib_imports'
assert after['snapshot1415_durable_publication']['recorded_process_cpu_seconds'] is None
assert after['B_nll_CPU_analysis_unknown']['recorded_wall_seconds'] is None
assert new['cost_only_v2_extension']['B_NLL_additional_manual_inference_cost_rows'] == 0
assert len(new['legacy_method_and_preparation_ledger']['evaluation_canonical_inferences']) == len(old['legacy_method_and_preparation_ledger']['evaluation_canonical_inferences'])
assert all(sha(item['snapshot_path']) == item['sha256'] for item in new['cost_only_v2_extension']['exact_cost_source_copies'])
assert new['accounting_contract']['scores_or_prediction_JSON_semantics_parsed'] is False
assert new['accounting_contract']['checkpoint_or_tensor_payloads_read'] is False


def charged(row):
    return row['recorded_wall_seconds'] if row and row['included_in_disjoint_recorded_wall_sum'] else 0.


registered_wall = sum(charged(row) for row in rows)
known_cpu = sum(row['recorded_process_cpu_seconds'] for row in rows if row['recorded_process_cpu_seconds'] is not None)
natural_C = []
unexpected = []
for identity in sorted(set(before) | set(after)):
    difference = charged(after.get(identity)) - charged(before.get(identity))
    if identity in ids or identity == 'opaque_stable_full_file_reads' or abs(difference) < 1e-10:
        continue
    detail = {'id': identity, 'before_charged_wall_seconds': charged(before.get(identity)),
              'after_charged_wall_seconds': charged(after.get(identity)), 'wall_delta_seconds': difference}
    if identity.startswith(('C_draft:', 'C_assembly:')):
        natural_C.append(detail)
    else:
        unexpected.append(detail)
assert not unexpected, unexpected
old_hashes = {row['receipt_sha256'] for row in old['historical_opaque_hash_receipts']}
fresh_opaque = [row for row in new['historical_opaque_hash_receipts'] if row['receipt_sha256'] not in old_hashes]
assert all(row['current_inventory_input_payload_reads'] == 0 for row in fresh_opaque)
opaque_delta = charged(after['opaque_stable_full_file_reads']) - charged(before['opaque_stable_full_file_reads'])
assert math.isclose(opaque_delta, sum(row['historical_full_read_seconds'] for row in fresh_opaque), abs_tol=1e-8)
dynamic_C_wall = sum(row['wall_delta_seconds'] for row in natural_C)
total_delta = sum(charged(row) for row in after.values()) - sum(charged(row) for row in before.values())
assert math.isclose(total_delta, registered_wall + dynamic_C_wall + opaque_delta, abs_tol=1e-8)
coverage = new['cost_only_v2_extension']['C_export_record_coverage']
assert sum(row['expected_draft_records'] for row in coverage) == 63
assert all(row['publication_source_status'] == 'not_authorized_source_unassigned' for row in coverage)
delta = {'schema_version': 1, 'status': 'passed_targeted_revision5_and_snapshot_delta_checks',
    'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'inventory': {'path': str(OUT / 'inventory.json'), 'sha256': sha(OUT / 'inventory.json')},
    'source_spec': {'path': str(ROOT / 'configs/cost_inventory_final_sources_v5.json'), 'sha256': sha(ROOT / 'configs/cost_inventory_final_sources_v5.json')},
    'preserved_old_source_and_trial_SHA': unchanged,
    'revision5_new_explicit_operations_or_regions': [{'id': row['id'], 'wall_seconds': row['recorded_wall_seconds'], 'known_CPU_seconds': row['recorded_process_cpu_seconds']} for row in rows],
    'revision5_new_measured_wall_sum_seconds': registered_wall,
    'revision5_known_CPU_sum_seconds_publication_CPU_unknown': known_cpu,
    'dynamic_live_C_operation_growth_separate_from_revision5_registration': natural_C,
    'dynamic_C_measured_wall_delta_seconds': dynamic_C_wall,
    'new_historical_opaque_receipts_count': len(fresh_opaque), 'historical_opaque_wall_delta_seconds': opaque_delta,
    'new_opaque_input_paths': sorted({row['canonical_input'] for row in fresh_opaque}),
    'current_collector_checkpoint_payload_reads': 0,
    'total_operational_recorded_wall_snapshot_delta_seconds': total_delta,
    'delta_is_not_elapsed_study_time_or_revision5_only_or_whole_study_CPU': True,
    'C_export_coverage': [{key: value for key, value in row.items() if key != 'records'} for row in coverage],
    'C_daemon_changed_wall_is_nonadditive_waiting_context': True,
    'body_region_not_a_complete_parent_timer': True, 'nested_read_or_old_archive_SHA_or_package_operations_added_again': False,
    'B_original_parent_remains_unknown': True, 'unknown_QC_engineering_or_CPU_tests_imputed': False,
    'future_GPU_QA_counted_as_completed_training': False, 'score_semantics_read': False, 'new_GPU_launched': False,
    'validator': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}}
target = OUT / 'delta_from_v4_and_validation.json'
with target.open('x') as stream:
    json.dump(delta, stream, ensure_ascii=False, indent=2); stream.write('\n')
print(json.dumps({'receipt': str(target), 'sha256': sha(target), 'new_registration_wall': registered_wall,
    'new_registration_known_CPU': known_cpu, 'dynamic_C_wall': dynamic_C_wall,
    'historical_opaque_wall_delta': opaque_delta, 'total_operational_snapshot_wall_delta': total_delta,
    'new_C_records': len(natural_C), 'new_opaque_receipts': len(fresh_opaque)}))
