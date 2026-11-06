#!/usr/bin/env python3
"""Source revision4 targeted normalization, nonoverlap and snapshot-delta checks."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/cost_inventory_final_v4_cpu_trial_20261106'
load = lambda path: json.loads(Path(path).read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
old = load(ROOT / 'results/cost_inventory_final_v3_cpu_trial_20261106/inventory.json')
new = load(OUT / 'inventory.json')
spec = load(ROOT / 'configs/cost_inventory_final_sources_v4.json')
old_spec = load(ROOT / 'configs/cost_inventory_final_sources_v3.json')
unchanged = {
    'scripts/inventory_study_costs_final.py': 'c0c9d4605da8809e48aa24bdbeca664c86cdf613fe4aef4a0fc35614d17f8b95',
    'scripts/inventory_study_costs_final_v2.py': '864effb89f20dbfce3ba6ce51db8eb8351b087c2ac92d255274fe6267753f6ae',
    'configs/cost_inventory_final_sources_v1.json': '3b9986f853a656b7dd16cc4adc8a8b5d7194bf6f8db478ed4785ffb84167847c',
    'configs/cost_inventory_final_sources_v2.json': '3c0286f39fb19f386bbb09a68143c80cd4fe45677efaf360309b032712b14ce7',
    'configs/cost_inventory_final_sources_v3.json': '4fdc24ea2c5995e76b032d29803de1e038114d667b9dd7ef7753b220abf7ea2a',
    'results/cost_inventory_final_v3_cpu_trial_20261106/inventory.json': '2af4855e2990e8dc5df4bfd2b491a21a17e3fb1d9231934ba299a1db6fd40ed0',
    'results/cost_inventory_final_v2_cpu_trial_20261106/inventory.json': '009c70654649936a78045234129a739ed52e95923162b6f122771019785b904f',
}
for path, wanted in unchanged.items():
    assert sha(ROOT / path) == wanted, path
assert spec['additional_operations'][:len(old_spec['additional_operations'])] == old_spec['additional_operations']
assert spec['C_exports'] == old_spec['C_exports']
before = {row['id']: row for row in old['operational_costs']}
after = {row['id']: row for row in new['operational_costs']}
ids = spec['revision4_additional_operation_ids']
rows = [after[identity] for identity in ids]
assert len(ids) == len(set(ids)) == 9 and all(row['accepted_receipt'] and row['included_in_disjoint_recorded_wall_sum'] for row in rows)
for normalized in spec['revision4_normalized_sources']:
    assert sha(normalized['path']) == normalized['sha256']
    value = load(normalized['path']); source = value['original_source']
    assert sha(source['path']) == source['sha256']
    actual = load(source['path'])
    assert value['wall_seconds'] == actual[value['original_wall_pointer'][1:]]
    components = [actual[pointer[1:]] for pointer in value['CPU_component_pointers']]
    assert math.isclose(sum(components), value['recorded_inclusive_CPU_seconds'], abs_tol=1e-12)
    assert value['nested_subtimers_added'] is False and value['new_operation_created_by_normalization'] is False
assert after['snapshot1325_durable_publication']['recorded_process_cpu_seconds'] is None
assert after['B_nll_CPU_analysis_unknown']['recorded_wall_seconds'] is None
assert len(new['legacy_method_and_preparation_ledger']['evaluation_canonical_inferences']) == len(old['legacy_method_and_preparation_ledger']['evaluation_canonical_inferences'])
assert new['cost_only_v2_extension']['B_NLL_additional_manual_inference_cost_rows'] == 0
dynamic = [row for identity, row in after.items() if identity not in before and identity not in ids and row['included_in_disjoint_recorded_wall_sum']]
assert len(dynamic) == 3 and all(row['id'].startswith('C_draft:pythia-410m:') for row in dynamic)
old_hashes = {row['receipt_sha256'] for row in old['historical_opaque_hash_receipts']}
fresh_receipts = [row for row in new['historical_opaque_hash_receipts'] if row['receipt_sha256'] not in old_hashes]
assert fresh_receipts and all('extended_v2_pythia_410m_m64_formal_' in row['canonical_input'] for row in fresh_receipts)
assert all(row['current_inventory_input_payload_reads'] == 0 for row in fresh_receipts)
opaque_delta = after['opaque_stable_full_file_reads']['recorded_wall_seconds'] - before['opaque_stable_full_file_reads']['recorded_wall_seconds']
assert math.isclose(opaque_delta, sum(row['historical_full_read_seconds'] for row in fresh_receipts), abs_tol=1e-9)
coverage = new['cost_only_v2_extension']['C_export_record_coverage']
assert sum(row['expected_draft_records'] for row in coverage) == 63
assert all(not row['assembly_complete'] and row['publication_source_status'] == 'not_authorized_source_unassigned' for row in coverage)
for copy in new['cost_only_v2_extension']['exact_cost_source_copies']:
    assert sha(copy['snapshot_path']) == copy['sha256']
assert new['accounting_contract']['scores_or_prediction_JSON_semantics_parsed'] is False
assert new['accounting_contract']['checkpoint_or_tensor_payloads_read'] is False
new_wall = sum(row['recorded_wall_seconds'] for row in rows)
new_cpu_known = sum(row['recorded_process_cpu_seconds'] for row in rows if row['recorded_process_cpu_seconds'] is not None)
total_delta = sum(row['recorded_wall_seconds'] for row in after.values() if row['included_in_disjoint_recorded_wall_sum']) - sum(row['recorded_wall_seconds'] for row in before.values() if row['included_in_disjoint_recorded_wall_sum'])
dynamic_wall = sum(row['recorded_wall_seconds'] for row in dynamic)
assert math.isclose(total_delta, new_wall + dynamic_wall + opaque_delta, abs_tol=1e-8)
delta = {'schema_version': 1, 'status': 'passed_targeted_revision4_and_snapshot_delta_checks',
    'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'inventory': {'path': str(OUT / 'inventory.json'), 'sha256': sha(OUT / 'inventory.json')},
    'source_spec': {'path': str(ROOT / 'configs/cost_inventory_final_sources_v4.json'), 'sha256': sha(ROOT / 'configs/cost_inventory_final_sources_v4.json')},
    'preserved_old_source_and_trial_SHA': unchanged,
    'revision4_new_explicit_operations': [{'id': row['id'], 'wall_seconds': row['recorded_wall_seconds'], 'known_CPU_seconds': row['recorded_process_cpu_seconds']} for row in rows],
    'revision4_new_measured_wall_sum_seconds': new_wall,
    'revision4_known_CPU_sum_seconds_publication_CPU_not_measured': new_cpu_known,
    'requested_eight_operations_excluding_extra_independent_binding_check_wall_seconds': new_wall - after['optional_C2p8b_identity_independent_binding_check_actual']['recorded_wall_seconds'],
    'dynamic_live_C_draft_additions_separate_from_revision4_registration': [{'id': row['id'], 'wall_seconds': row['recorded_wall_seconds']} for row in dynamic],
    'dynamic_C_draft_measured_wall_sum_seconds': dynamic_wall,
    'new_historical_opaque_receipts_count': len(fresh_receipts), 'historical_opaque_wall_delta_seconds': opaque_delta,
    'new_opaque_receipts_are_C410_only_and_not_optional2p8_identity_nested_reads': True,
    'current_collector_checkpoint_payload_reads': 0,
    'total_operational_recorded_wall_snapshot_delta_seconds': total_delta,
    'delta_is_not_elapsed_study_time_or_revision4_only_or_pure_CPU_work': True,
    'C_daemon_changed_wall_is_nonadditive_waiting_context': True,
    'B_original_parent_remains_unknown': True, 'nested_children_not_added': True,
    'C_assembly_or_publication_imputed': False, 'score_semantics_read': False, 'new_GPU_or_training_launched': False,
    'validator': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}}
target = OUT / 'delta_from_v3_and_validation.json'
with target.open('x') as stream:
    json.dump(delta, stream, ensure_ascii=False, indent=2); stream.write('\n')
print(json.dumps({'receipt': str(target), 'sha256': sha(target), 'new_registration_wall': new_wall,
                  'dynamic_C_wall': dynamic_wall, 'historical_opaque_wall_delta': opaque_delta,
                  'total_operational_snapshot_wall_delta': total_delta}))
