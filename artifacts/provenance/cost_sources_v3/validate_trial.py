#!/usr/bin/env python3
"""Targeted revision3 source/ledger validation; no scoring or payload reads."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/cost_inventory_final_v3_cpu_trial_20261106'
load = lambda path: json.loads(Path(path).read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
spec = load(ROOT / 'configs/cost_inventory_final_sources_v3.json')
ledger = load(OUT / 'inventory.json')
expected_unchanged = {
    'scripts/inventory_study_costs_final.py': 'c0c9d4605da8809e48aa24bdbeca664c86cdf613fe4aef4a0fc35614d17f8b95',
    'scripts/inventory_study_costs_final_v2.py': '864effb89f20dbfce3ba6ce51db8eb8351b087c2ac92d255274fe6267753f6ae',
    'configs/cost_inventory_final_sources_v1.json': '3b9986f853a656b7dd16cc4adc8a8b5d7194bf6f8db478ed4785ffb84167847c',
    'configs/cost_inventory_final_sources_v2.json': '3c0286f39fb19f386bbb09a68143c80cd4fe45677efaf360309b032712b14ce7',
    'results/cost_inventory_final_v2_cpu_trial_20261106/inventory.json': '009c70654649936a78045234129a739ed52e95923162b6f122771019785b904f',
}
for path, wanted in expected_unchanged.items():
    assert sha(ROOT / path) == wanted, path
rows = {row['id']: row for row in ledger['operational_costs']}
new = [rows[identity] for identity in spec['revision3_additional_operation_ids']]
assert len(new) == 13 and all(row['accepted_receipt'] for row in new)
assert sum(row['included_in_disjoint_recorded_wall_sum'] for row in new) == 12
assert not rows['C_pending_order_apply_with_reload_wait_context']['included_in_disjoint_recorded_wall_sum']
assert rows['C_pending_order_apply_with_reload_wait_context']['timing_semantics'] == 'wall_including_waiting'
assert rows['B_nll_CPU_analysis_unknown']['recorded_wall_seconds'] is None
assert not rows['B_budget_successful_collection_child']['included_in_disjoint_recorded_wall_sum']
diagnostics = [row for row in new if row['id'].startswith('C_actual_bank_diagnostics_')]
assert len(diagnostics) == 4
history = load(ROOT / 'results/extended_bank_diagnostics/timing_history.json')
assert math.isclose(sum(row['recorded_wall_seconds'] for row in diagnostics), history['analysis_region_wall_sum_seconds'], abs_tol=1e-12)
assert math.isclose(sum(row['recorded_process_cpu_seconds'] for row in diagnostics), history['analysis_region_cpu_sum_seconds'], abs_tol=1e-12)
assert not any(row.get('source', '').endswith('/extended_bank_diagnostics/bank_geometry.json') for row in ledger['operational_costs'])
original_telemetry = load(ROOT / 'results/resources_1303_intermediate/collection_receipt.json')
assert math.isclose(rows['telemetry1303_collection_child']['recorded_process_cpu_seconds'],
                    original_telemetry['child_user_cpu_seconds'] + original_telemetry['child_sys_cpu_seconds'], abs_tol=1e-12)
assert rows['telemetry1303_collection_child']['recorded_wall_seconds'] == original_telemetry['wall_seconds']
assert ledger['cost_only_v2_extension']['B_NLL_additional_manual_inference_cost_rows'] == 0
coverage = ledger['cost_only_v2_extension']['C_export_record_coverage']
assert sum(row['expected_draft_records'] for row in coverage) == 63
assert all(row['publication_source_status'] == 'not_authorized_source_unassigned' and not row['publication_completed_assumed'] for row in coverage)
assert not any(row['assembly_complete'] for row in coverage)
for copy in ledger['cost_only_v2_extension']['exact_cost_source_copies']:
    assert sha(copy['snapshot_path']) == copy['sha256']
assert ledger['accounting_contract']['scores_or_prediction_JSON_semantics_parsed'] is False
assert ledger['accounting_contract']['checkpoint_or_tensor_payloads_read'] is False
assert ledger['accounting_contract']['whole_study_completely_instrumented'] is False
receipt = {'schema_version': 1, 'status': 'passed_targeted_source_revision3_validation',
    'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'inventory': {'path': str(OUT / 'inventory.json'), 'sha256': sha(OUT / 'inventory.json')},
    'source_spec': {'path': str(ROOT / 'configs/cost_inventory_final_sources_v3.json'), 'sha256': sha(ROOT / 'configs/cost_inventory_final_sources_v3.json')},
    'preserved_sources_and_old_trial_SHA': expected_unchanged, 'new_operations': 13, 'new_additive_measured_regions_or_stages': 12,
    'new_disjoint_recorded_wall_seconds': sum(row['recorded_wall_seconds'] for row in new if row['included_in_disjoint_recorded_wall_sum']),
    'new_wait_context_not_added': True, 'diagnostics_four_regions_counted_once': True, 'B_parent_still_unknown': True,
    'source_copy_SHA_verified': True, 'C_dynamic_records_observed': sum(row['actual_first_created_records_observed'] for row in coverage),
    'C_publication_or_assembly_imputed': False,
    'validator': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
    'no_score_semantics_or_GB_payload_reads': True, 'new_GPU_or_watcher': False}
target = OUT / 'targeted_validation_receipt.json'
with target.open('x') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2); stream.write('\n')
print(json.dumps({'receipt': str(target), 'sha256': sha(target), 'new_disjoint_measured_wall_seconds': receipt['new_disjoint_recorded_wall_seconds']}))
