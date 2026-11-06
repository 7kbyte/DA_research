"""Targeted ledger arithmetic/provenance checks; reads only cost metadata."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OLD=ROOT/'results/cost_inventory_final_v6_cpu_trial_20261106'
NEW=ROOT/'results/cost_inventory_final_v7_cpu_collection_20261106_attempt2'


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def included(rows):return sum(r['recorded_wall_seconds'] for r in rows if r['included_in_disjoint_recorded_wall_sum'])


def main():
    previous=load(ROOT/'configs/cost_inventory_final_sources_v6.json');spec=load(ROOT/'configs/cost_inventory_final_sources_v7.json')
    assert sha(ROOT/'configs/cost_inventory_final_sources_v6.json')=='e4c99db9b1803681ad8706bb90480ede0be584975324040df979c1c9966b17e0'
    assert sha(ROOT/'scripts/inventory_study_costs_final_v2.py')=='864effb89f20dbfce3ba6ce51db8eb8351b087c2ac92d255274fe6267753f6ae'
    assert spec['additional_operations'][:len(previous['additional_operations'])]==previous['additional_operations']
    assert spec['metadata_sources'][:len(previous['metadata_sources'])]==previous['metadata_sources']
    assert spec['v1_source_spec']==previous['v1_source_spec'] and spec['C_exports']==previous['C_exports']
    old,new=load(OLD/'inventory.json'),load(NEW/'inventory.json');registration=load(HERE/'registration_delta.json')
    assert len(spec['revision7_additional_parent_operation_ids'])==34
    assert len(spec['revision7_separate_region_operation_ids'])==10
    rows={r['id']:r for r in new['operational_costs']};oldrows={r['id']:r for r in old['operational_costs']}
    parents=[rows[k] for k in spec['revision7_additional_parent_operation_ids']]
    regions=[rows[k] for k in spec['revision7_separate_region_operation_ids']]
    assert all(r['accepted_receipt'] and r['included_in_disjoint_recorded_wall_sum'] for r in parents)
    assert all(r['accepted_receipt'] and not r['included_in_disjoint_recorded_wall_sum'] for r in regions)
    static=included(parents);assert math.isclose(static,registration['parent_wall_increment_seconds'],abs_tol=1e-8)
    added=[r for k,r in rows.items() if k not in oldrows and k not in spec['revision7_additional_parent_operation_ids'] and r['included_in_disjoint_recorded_wall_sum']]
    changed=[]
    for key,left in oldrows.items():
        right=rows[key]
        before=left['recorded_wall_seconds'] if left['included_in_disjoint_recorded_wall_sum'] else 0
        after=right['recorded_wall_seconds'] if right['included_in_disjoint_recorded_wall_sum'] else 0
        if before!=after:changed.append(dict(id=key,old_wall_seconds=before,new_wall_seconds=after,delta_seconds=after-before))
    dynamic=included(added)+sum(r['delta_seconds'] for r in changed)
    assert math.isclose(included(new['operational_costs'])-included(old['operational_costs']),static+dynamic,abs_tol=1e-7)
    copies=new['cost_only_v2_extension']['exact_cost_source_copies']
    for source in copies:
        assert sha(source['snapshot_path'])==source['sha256'] and Path(source['snapshot_path']).stat().st_size==source['bytes']
    def training(v):return {r['id']:r for r in v['training_cohort_coverage']}
    training_new=training(new)
    for key,count in [('primary36',36),('A108_additional_plus18_training_aliases',108),('B60',60),('C410m24',24)]:
        assert training_new[key]['summary_completed_runs']==count
    assert training_new['A108_additional_plus18_training_aliases']['training_aliases_not_new_runs']==18
    assert training_new['primary36']['D_group_observed_physical_runs']==training_new['primary36']['D_group_independent_training_definitions']
    assert new['cost_only_v2_extension']['B_NLL_additional_manual_inference_cost_rows']==0
    def eval_counts(v):return [(r['category'],r['attested_new_forward_jobs'],r['attested_reuse_jobs']) for r in v['legacy_method_and_preparation_ledger']['evaluation_category_totals']]
    assert eval_counts(new)==eval_counts(old)
    io=load(NEW/'IO_audit.json');assert io['weight_base_TAR_ZIP_tensor_payload_bytes_read']==0 and not io['forbidden_weight_base_archive_tensor_payload_open_attempts']
    assert io['score_output_semantic_deserialization'] is False
    result=dict(status='passed_targeted_v7_cost_source_and_ledger_QC',source_spec_sha256=sha(ROOT/'configs/cost_inventory_final_sources_v7.json'),
        inventory_sha256=sha(NEW/'inventory.json'),old_v6_spec_and_collectors_unchanged=True,
        static_new_parent_operations=34,static_registered_parent_wall_increment_seconds=static,
        static_registered_parent_known_CPU_increment_seconds=sum(r['recorded_process_cpu_seconds'] or 0 for r in parents),
        partial_regions_nonadditive=10,partial_region_wall_separate_seconds=sum(r['recorded_wall_seconds'] for r in regions),
        dynamic_added_parent_operations=[dict(id=r['id'],wall_seconds=r['recorded_wall_seconds']) for r in added],
        dynamic_changed_parent_operations=changed,dynamic_C_or_opaque_wall_delta_seconds=dynamic,
        old_operational_disjoint_wall_seconds=included(old['operational_costs']),new_operational_disjoint_wall_seconds=included(new['operational_costs']),
        total_operational_delta_seconds=static+dynamic,exact_metadata_copy_bindings=len(copies),
        unique_exact_metadata_copy_files=len({r['snapshot_path'] for r in copies}),
        exact_metadata_copy_unique_bytes=sum(Path(p).stat().st_size for p in {r['snapshot_path'] for r in copies}),
        evaluation_forward_reuse_counts_unchanged=eval_counts(new),
        B_NLL_extra_manual_forward_charge=0,whole_study_complete=False,
        score_stream_hash_bytes_current_successful_collection=io['cumulative_score_stream_SHA_bytes'],
        failed_attempt1_score_hash_bytes='unknown; no IO receipt persisted before category failure',
        score_semantics_parsed=False,weight_payload_bytes_read=0,
        current_successful_collection_timer_not_recursively_charged=True)
    with (HERE/'QC.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    fields=['id','category','wall_seconds','known_CPU_seconds','additive_parent','source']
    with (NEW/'registered_v7_increment.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in registration['parents']+registration['regions_separate_nonadditive']:
            writer.writerow({k:json.dumps(row[k],ensure_ascii=False) if k=='source' else row[k] for k in fields})
    print(json.dumps(result))


if __name__=='__main__':main()
