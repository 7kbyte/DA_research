"""Cost-only arithmetic/source QC for the explicit v8 observed snapshot."""
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'results/cost_inventory_final_v7_cpu_collection_20261106_attempt2'
NEW = ROOT / 'results/cost_inventory_final_v8_cpu_collection_20261106_attempt1'


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def included(rows):
    return sum(r['recorded_wall_seconds'] for r in rows if r['included_in_disjoint_recorded_wall_sum'])


def cpu_included(rows):
    return sum(r['recorded_process_cpu_seconds'] or 0 for r in rows if r['included_in_disjoint_recorded_wall_sum'])


def main():
    start = time.monotonic()
    cpu = resource.getrusage(resource.RUSAGE_SELF)
    previous = load(ROOT/'configs/cost_inventory_final_sources_v7.json')
    spec = load(ROOT/'configs/cost_inventory_final_sources_v8.json')
    assert sha(ROOT/'configs/cost_inventory_final_sources_v7.json') == '0cab816da531e857ded88a004624d9a946a1bd3dae75f295ed68e1e4f40b1f0e'
    assert sha(ROOT/'configs/cost_inventory_final_sources_v6.json') == 'e4c99db9b1803681ad8706bb90480ede0be584975324040df979c1c9966b17e0'
    assert sha(ROOT/'scripts/inventory_study_costs_final_v2.py') == '864effb89f20dbfce3ba6ce51db8eb8351b087c2ac92d255274fe6267753f6ae'
    assert sha(ROOT/'artifacts/provenance/cost_sources_v7/run_collection_with_IO_audit.py') == 'a3513d44a64adbab37feafcb1f7d0e15a31a61defac7572cd5482621775bede9'
    assert spec['additional_operations'][:len(previous['additional_operations'])] == previous['additional_operations']
    assert spec['metadata_sources'][:len(previous['metadata_sources'])] == previous['metadata_sources']
    assert spec['C_exports'] == previous['C_exports'] and spec['v1_source_spec'] == previous['v1_source_spec']
    legacy = load(load(Path(spec['v1_source_spec']['path']))['legacy_source_spec']['path'])
    assert all(x['verify_output_hashes'] is True for x in legacy['evaluation_manifests'])
    old, new = load(OLD/'inventory.json'), load(NEW/'inventory.json')
    delta = load(HERE/'registration_delta.json')
    rows = {x['id']:x for x in new['operational_costs']}
    prior = {x['id']:x for x in old['operational_costs']}
    assert len(rows) == len(new['operational_costs']) and len(prior) == len(old['operational_costs'])
    parent_ids = spec['revision8_additional_parent_operation_ids']
    region_ids = spec['revision8_separate_region_operation_ids']
    assert len(parent_ids) == 46 and len(region_ids) == 14
    parents, regions = [rows[k] for k in parent_ids], [rows[k] for k in region_ids]
    assert all(r['accepted_receipt'] and r['included_in_disjoint_recorded_wall_sum'] for r in parents)
    assert all(r['accepted_receipt'] and not r['included_in_disjoint_recorded_wall_sum'] for r in regions)
    static = included(parents)
    static_cpu = cpu_included(parents)
    assert math.isclose(static,delta['parent_wall_increment_seconds'],abs_tol=1e-8)
    assert math.isclose(static_cpu,delta['parent_known_CPU_increment_seconds'],abs_tol=1e-8)
    assert math.isclose(sum(r['recorded_wall_seconds'] for r in regions),delta['partial_region_wall_separate_seconds'],abs_tol=1e-8)
    new_ids = set(parent_ids) | set(region_ids)
    added = [r for key,r in rows.items() if key not in prior and key not in new_ids and r['included_in_disjoint_recorded_wall_sum']]
    changed, missing = [], []
    for key,left in prior.items():
        if key not in rows:
            missing.append(key)
            continue
        right = rows[key]
        b = left['recorded_wall_seconds'] if left['included_in_disjoint_recorded_wall_sum'] else 0
        a = right['recorded_wall_seconds'] if right['included_in_disjoint_recorded_wall_sum'] else 0
        cb = (left['recorded_process_cpu_seconds'] or 0) if left['included_in_disjoint_recorded_wall_sum'] else 0
        ca = (right['recorded_process_cpu_seconds'] or 0) if right['included_in_disjoint_recorded_wall_sum'] else 0
        if b != a or cb != ca:
            changed.append({'id':key,'old_wall_seconds':b,'new_wall_seconds':a,'wall_delta_seconds':a-b,'known_CPU_delta_seconds':ca-cb})
    assert not missing, 'previous measured cost disappeared'
    dynamic = included(added) + sum(r['wall_delta_seconds'] for r in changed)
    dynamic_cpu = cpu_included(added) + sum(r['known_CPU_delta_seconds'] for r in changed)
    actual_delta = included(new['operational_costs']) - included(old['operational_costs'])
    actual_cpu_delta = cpu_included(new['operational_costs']) - cpu_included(old['operational_costs'])
    assert math.isclose(actual_delta,static+dynamic,abs_tol=1e-7)
    assert math.isclose(actual_cpu_delta,static_cpu+dynamic_cpu,abs_tol=1e-7)
    copies = new['cost_only_v2_extension']['exact_cost_source_copies']
    for source in copies:
        assert sha(source['snapshot_path']) == source['sha256']
        assert Path(source['snapshot_path']).stat().st_size == source['bytes']
    io = load(NEW/'IO_audit.json')
    assert io['status'] == 'passed_cost_only_IO_audit' and io['weight_base_TAR_ZIP_tensor_payload_bytes_read'] == 0
    assert not io['forbidden_weight_base_archive_tensor_payload_open_attempts']
    assert io['score_output_semantic_deserialization'] is False and io['score_SHA_verification_policy_changed'] is False
    assert sum(r['actual_full_stream_hash_bytes'] for r in io['score_output_stream_hash_reads']) == io['cumulative_score_stream_SHA_bytes']
    timing = load(HERE/'collection_whole_child_timing.json')
    assert type(timing['returncode']) is int and timing['returncode'] == 0
    assert 'stop_requested' not in timing and 'stop_requested_at' not in timing
    assert timing['source_spec_sha256'] == sha(ROOT/'configs/cost_inventory_final_sources_v8.json')
    assert rows['v8_C410_best_publication_internal_region']['recorded_wall_seconds'] == 195.2169531546533
    assert rows['v8_C410_best_publication_internal_region']['recorded_process_cpu_seconds'] is None
    assert not rows['v8_C410_best_publication_internal_region']['included_in_disjoint_recorded_wall_sum']
    assert rows['v8_B_native_actual_restore']['recorded_wall_seconds'] == 36.18444097042084
    assert not any('C410_best_publication_actual_v1/batch_whole_child_timing.json' in str(r) for r in spec['additional_operations'])
    def eval_counts(value):
        return [{k:r[k] for k in ('category','attested_new_forward_jobs','attested_reuse_jobs')} for r in value['legacy_method_and_preparation_ledger']['evaluation_category_totals']]
    training = new['training_cohort_coverage']
    for key,count in [('primary36',36),('A108_additional_plus18_training_aliases',108),('B60',60),('C410m24',24)]:
        cohort = next(x for x in training if x['id'] == key)
        assert cohort['summary_completed_runs'] == count
    result = {'status':'passed_v8_cost_source_and_arithmetic_QC','source_spec_sha256':sha(ROOT/'configs/cost_inventory_final_sources_v8.json'),
        'inventory_sha256':sha(NEW/'inventory.json'),'v6_v7_and_collectors_unchanged':True,'static_new_parent_operations':len(parents),
        'static_parent_wall_increment_seconds':static,'static_parent_known_CPU_increment_seconds':static_cpu,
        'partial_regions_nonadditive':len(regions),'partial_region_wall_separate_seconds':sum(r['recorded_wall_seconds'] for r in regions),
        'dynamic_added_parent_operations':[{k:r[k] for k in ('id','category','recorded_wall_seconds','recorded_process_cpu_seconds')} for r in added],
        'dynamic_changed_parent_operations':changed,'dynamic_wall_delta_seconds':dynamic,'dynamic_known_CPU_delta_seconds':dynamic_cpu,
        'old_operational_disjoint_wall_seconds':included(old['operational_costs']),'new_operational_disjoint_wall_seconds':included(new['operational_costs']),
        'total_operational_wall_delta_seconds':actual_delta,'total_known_CPU_delta_seconds':actual_cpu_delta,
        'exact_metadata_copy_bindings':len(copies),'unique_exact_metadata_copy_files':len({x['snapshot_path'] for x in copies}),
        'exact_metadata_copy_unique_bytes':sum(Path(x).stat().st_size for x in {r['snapshot_path'] for r in copies}),
        'training_cohort_coverage':training,'C_observed_export_coverage':new['cost_only_v2_extension']['C_export_record_coverage'],
        'evaluation_count_comparison':{'v7':eval_counts(old),'v8':eval_counts(new)},
        'B_NLL_extra_manual_forward_charge':new['cost_only_v2_extension']['B_NLL_additional_manual_inference_cost_rows'],
        'score_semantics_parsed':False,'weight_base_TAR_ZIP_tensor_bytes_read':0,
        'unique_score_outputs_stream_hashed':io['unique_score_outputs_stream_hashed'],
        'score_SHA_stream_bytes_actual_reads':io['cumulative_score_stream_SHA_bytes'],
        'current_successful_collection_parent_wall_seconds_not_in_ledger':timing['whole_child_wall_seconds'],
        'current_collection_source_preparation_and_QC_not_recursively_charged':True,'whole_study_complete':False,
        'manual_engineering_whole_seconds':None,
        'QC_region':{'wall_seconds':time.monotonic()-start,'self_user_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime-cpu.ru_utime,
            'self_system_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_stime-cpu.ru_stime,
            'scope':'This QC main-entry small cost-ledger/source arithmetic and metadata-copy hash region only; final serialization and manual reading excluded. Not included in v8.'}}
    with (HERE/'QC.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    fields = ['id','category','wall_seconds','known_CPU_seconds','original_returncode','additive_parent','source']
    with (NEW/'registered_v8_increment.csv').open('x',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in delta['parents'] + delta['regions_separate_nonadditive']:
            writer.writerow({k:json.dumps(r[k],ensure_ascii=False) if k=='source' else r[k] for k in fields})
    print(json.dumps({k:result[k] for k in ('status','static_new_parent_operations','static_parent_wall_increment_seconds','partial_regions_nonadditive','dynamic_wall_delta_seconds','total_operational_wall_delta_seconds','score_SHA_stream_bytes_actual_reads')},ensure_ascii=False))


if __name__ == '__main__':
    main()
