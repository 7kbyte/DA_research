"""Directed metadata-only accounting checks; no scores or model payload reads."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results/cost_inventory_final_v6_cpu_trial_20261106'


def read(path):
    return json.loads(Path(path).read_text())


def proof(path):
    p = Path(path).resolve()
    if p.suffix == '.pt':
        raise ValueError('Checkpoint payloads forbidden')
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())


def main():
    started, cpu_started = time.perf_counter(), time.process_time()
    checks = []
    def check(condition, text):
        if not condition:
            raise AssertionError(text)
        checks.append(text)
    old_spec = read(ROOT / 'configs/cost_inventory_final_sources_v5.json')
    spec = read(ROOT / 'configs/cost_inventory_final_sources_v6.json')
    old = read(ROOT / 'results/cost_inventory_final_v5_cpu_trial_20261106/inventory.json')
    current = read(OUTPUT / 'inventory.json')
    preparation = read(HERE / 'preparation_receipt.json')
    table = read(HERE / 'explicit_new_registration_table.json')
    candidates = read(ROOT / 'results/cost_sources_v6_candidates_v1/candidate_sources.json')
    for source in candidates['old_source_specs_preserved']:
        check(proof(source['path'])['sha256'] == source['sha256'], 'Unchanged old source ' + source['path'])
    check(proof(ROOT / 'scripts/inventory_study_costs_final_v2.py')['sha256'] == '864effb89f20dbfce3ba6ce51db8eb8351b087c2ac92d255274fe6267753f6ae', 'Unchanged final v2 collector')
    check(spec['additional_operations'][:len(old_spec['additional_operations'])] == old_spec['additional_operations'], 'All pre-existing explicit operation definitions unchanged')
    check(spec['C_exports'] == old_spec['C_exports'], 'Dynamic C exporter and publication-null definitions unchanged')
    check(spec['v1_source_spec'] == old_spec['v1_source_spec'], 'Original v1 source contract retained')
    rows = {row['id']: row for row in current['operational_costs']}
    spec_rows = {row['id']: row for row in spec['additional_operations']}
    check(len(rows) == len(current['operational_costs']), 'No duplicate operation IDs')
    explicit_csv = []
    for item in table:
        row = rows[item['id']]
        check(row['included_in_disjoint_recorded_wall_sum'] is True, item['id'] + ' included once')
        check(row['recorded_wall_seconds'] == item['measured_wall_seconds'], item['id'] + ' measured wall exact')
        check(row['recorded_process_cpu_seconds'] == item['known_inclusive_CPU_seconds'], item['id'] + ' measured CPU/unknown exact')
        normalized_source = read(row['source'])
        check(normalized_source['original_source'] == item['original_source'], item['id'] + ' original source binding exact')
        check(proof(item['original_source']['path']) == item['original_source'], item['id'] + ' current original bytes match')
        explicit_csv.append({'id': item['id'], 'category': row['category'], 'wall_seconds': row['recorded_wall_seconds'],
                             'known_inclusive_CPU_seconds': row['recorded_process_cpu_seconds'],
                             'original_source': item['original_source']['path'], 'source_sha256': item['original_source']['sha256'],
                             'original_wall_pointer': item['wall_pointer'], 'timing_semantics': item['timing_semantics'], 'scope': item['scope']})
    check(len(table) == 25, 'Exactly25 newly registered disjoint measured operations/regions')
    check(math.isclose(sum(item['measured_wall_seconds'] for item in table), preparation['explicit_disjoint_recorded_wall_sum_seconds'], abs_tol=1e-10), 'Explicit new wall subtotal arithmetic exact')
    context_ids = set(spec['revision6_additional_operation_ids']) - {item['id'] for item in table}
    for identity in context_ids:
        check(rows[identity]['included_in_disjoint_recorded_wall_sum'] is False, identity + ' nested/wait/calendar is nonadditive')
    check(len(context_ids) == 9, 'Exactly9 additional nested/context rows')
    check('C_assembly:pythia-410m' not in spec['revision6_additional_operation_ids'], 'C410 assembly not duplicated as a new explicit operation')
    check(rows['C_assembly:pythia-410m']['included_in_disjoint_recorded_wall_sum'], 'Existing dynamic C410 assembly now observed')
    check('C_publication:pythia-410m' not in rows, 'Inner dynamic C410 publisher not also registered')
    check(rows['C410_final_publication_internal_child']['parent_id'] == 'C410_final_publication_whole_child', 'Publisher inner wall is nested under real whole child')
    base, old_base = current['legacy_method_and_preparation_ledger'], old['legacy_method_and_preparation_ledger']
    old_forwards = sum(row['recorded_additional_model_inference'] for row in old_base['evaluation_executions'])
    new_forwards = sum(row['recorded_additional_model_inference'] for row in base['evaluation_executions'])
    c410 = [row for row in base['evaluation_executions'] if row['manifest_sha256'] == '48f8b2235c303211e4f79a92c16c7fb1c8d3757ce5d66a97a555fba7dfcf8f32']
    check(len(c410) == 43 and sum(row['recorded_additional_model_inference'] for row in c410) == 43, 'Actual C410 physical43 forwards, not49/48')
    check(all(row['scheduler_status'] == 'completed' and type(row['scheduler_returncode']) is int and row['scheduler_returncode'] == 0 and row['output_integrity_verified_by_inventory'] for row in c410), 'All43 actual completed rc0/output-SHA verified')
    check(new_forwards - old_forwards == 43 and (old_forwards, new_forwards) == (253, 296), 'Only43 new forwards added: no repeated A/B/baselines/aliases')
    check(preparation['C410_manual_additional_inference_cost_rows'] == 0, 'No manual second C410 inference ledger')
    cohort = {row['id']: row for row in current['training_cohort_coverage']}
    for identity, runs in [('primary36', 36), ('A108_additional_plus18_training_aliases', 108), ('B60', 60), ('C410m24', 24)]:
        check(cohort[identity]['summary_completed_runs'] == runs, identity + ' complete physical coverage')
    check(cohort['A108_additional_plus18_training_aliases']['training_aliases_not_new_runs'] == 18, 'A18 training aliases are not new training')
    coverage = current['cost_only_v2_extension']['C_export_record_coverage']
    check([(r['model'], r['actual_first_created_records_observed'], r['assembly_complete']) for r in coverage] == [('pythia-410m', 24, True), ('pythia-1b', 5, False), ('pythia-2.8b', 1, False)], 'Exact observed C draft/assembly coverage, other packages not completed')
    copies = current['cost_only_v2_extension']['exact_cost_source_copies']
    for item in copies:
        check(proof(item['snapshot_path'])['sha256'] == item['sha256'], 'Exact captured cost source ' + item['sha256'])
    old_totals = {r['category']: r['known_disjoint_recorded_wall_seconds'] for r in old['operational_category_totals']}
    category_delta = []
    for row in current['operational_category_totals']:
        category = row['category']
        explicit = sum(r['wall_seconds'] for r in explicit_csv if r['category'] == category)
        delta = row['known_disjoint_recorded_wall_seconds'] - old_totals[category]
        category_delta.append({'category': category, 'v5_recorded_wall_seconds': old_totals[category],
                               'v6_recorded_wall_seconds': row['known_disjoint_recorded_wall_seconds'],
                               'change_seconds': delta, 'new_explicit_registration_seconds': explicit,
                               'existing_dynamic_C_or_opaque_change_seconds': delta - explicit,
                               'not_whole_study_elapsed_or_pure_CPU': True})
    csv_write(HERE / 'new_disjoint_operations.csv', explicit_csv)
    csv_write(HERE / 'category_delta_vs_v5.csv', category_delta)
    csv_write(HERE / 'new_nonadditive_context_rows.csv', [rows[k] for k in sorted(context_ids)])
    # This quality-check process observes only frozen ledger/receipt metadata.
    # Its region timer is not recursively added to the just-created ledger.
    write(HERE / 'directed_verification_receipt.json', {
        'schema_version': 1, 'status': 'passed_score_blind_accounting_QC',
        'source_spec': proof(ROOT / 'configs/cost_inventory_final_sources_v6.json'),
        'inventory': proof(OUTPUT / 'inventory.json'), 'collection_receipt': proof(OUTPUT / 'collection_receipt.json'),
        'checks_passed': len(checks), 'checks': checks,
        'new_explicit_disjoint_measured_rows': len(table), 'new_explicit_wall_sum_seconds': sum(r['wall_seconds'] for r in explicit_csv),
        'new_explicit_known_inclusive_CPU_sum_seconds': sum(r['known_inclusive_CPU_seconds'] or 0 for r in explicit_csv),
        'old_attested_forwards': old_forwards, 'current_attested_forwards': new_forwards,
        'C410_actual_forward_jobs': len(c410),
        'C410_scorer_subprocess_wall_sum_seconds': sum(r.get('scorer_subprocess_wall_seconds') or 0 for r in c410),
        'C410_known_disjoint_inference_and_verification_wall_sum_seconds': sum(r.get('known_disjoint_recorded_wall_seconds') or 0 for r in c410),
        'C410_scheduler_slot_sum_seconds_nonadditive': sum(r.get('scheduler_elapsed_seconds_nonadditive') or 0 for r in c410),
        'historical_opaque_receipt_counts': {'v5': len(old['historical_opaque_hash_receipts']), 'v6': len(current['historical_opaque_hash_receipts'])},
        'C_observed_records': [{k: v for k, v in r.items() if k != 'records'} for r in coverage],
        'exact_cost_source_snapshot_count': len(copies), 'category_delta_vs_v5': category_delta,
        'new_disjoint_operations_CSV': proof(HERE / 'new_disjoint_operations.csv'),
        'category_delta_CSV': proof(HERE / 'category_delta_vs_v5.csv'),
        'new_nonadditive_context_rows_CSV': proof(HERE / 'new_nonadditive_context_rows.csv'),
        'producer': proof(__file__),
        'verification_region_wall_seconds': time.perf_counter() - started,
        'verification_region_process_cpu_seconds': time.process_time() - cpu_started,
        'verification_timer_scope': 'After imports: read explicit existing cost ledgers/small receipts, arithmetic/SHA checks and write CSVs; excludes engineering/imports/receipt serialization. Not a complete parent timer.',
        'score_semantics_read': False, 'GB_model_or_archive_payloads_read': 0,
        'new_GPU_or_scheduler': False, 'whole_study_complete': False,
    })
    print(json.dumps({'checks_passed': len(checks), 'receipt': proof(HERE / 'directed_verification_receipt.json')}))


if __name__ == '__main__':
    main()
