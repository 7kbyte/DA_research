"""Independent cost-only arithmetic and receipt review; no collector execution."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import time

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
V6=ROOT/'results/cost_inventory_final_v6_cpu_trial_20261106'
V7=ROOT/'results/cost_inventory_final_v7_cpu_collection_20261106_attempt2'
D=ROOT/'artifacts/provenance/cost_sources_v7'
START=time.perf_counter();CPU=resource.getrusage(resource.RUSAGE_SELF);SOURCES={};RAW={};CHECKS=[]
def read(path,expected=None):
    path=Path(path).resolve()
    assert path.suffix in ('.json','.csv','.py','.md','.toml','.log') and path.stat().st_size<16<<20,path
    assert not any(part in path.parts for part in ('future_evaluation_v1','future_evaluation_v2')),path
    if str(path) not in RAW:
        a=path.stat();raw=path.read_bytes();b=path.stat()
        sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        assert sig(a)==sig(b),path
        RAW[str(path)]=raw;SOURCES[str(path)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'five_stat':sig(b)}
    if expected:assert SOURCES[str(path)]['sha256']==expected,path
    return RAW[str(path)]
def load(path,expected=None):return json.loads(read(path,expected))
def proof(p):return load(p['path'],p['sha256'])
def rows(path):return list(csv.DictReader(io.StringIO(read(path).decode())))
def pointer(obj,path):
    for key in path.strip('/').split('/'):obj=obj[int(key)] if isinstance(obj,list) else obj[key]
    return obj
def equal(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-8),(a,b)
def check(label):CHECKS.append(label)

complete=load(D/'COMPLETION.json');qc=proof(complete['QC']);delta=load(D/'registration_delta.json')
spec=proof(complete['source_spec']);old=proof(spec['preserved_v6_source_spec'])
manifest=proof(complete['source_manifest'])
for mapping in ('source_specs_v1_through_v7','unchanged_collectors','scientific_10_frozen_SHA'):
    for p,h in manifest[mapping].items():read(ROOT/p,h)
for p in manifest['files'].values():read(p['path'],p['sha256'])
assert spec['additional_operations'][:len(old['additional_operations'])]==old['additional_operations']
assert spec['metadata_sources'][:len(old['metadata_sources'])]==old['metadata_sources']
assert spec['C_exports']==old['C_exports']
check('v1-v6 specs, four collectors and ten scientific sources retain frozen SHA;82 old operations and metadata prefix/C config preserved')
oldops={r['id']:r for r in old['additional_operations']};ops={r['id']:r for r in spec['additional_operations']}
newids=set(ops)-set(oldops);parents=spec['revision7_additional_parent_operation_ids'];regions=spec['revision7_separate_region_operation_ids']
assert len(oldops)==82 and len(ops)==126 and len(parents)==34 and len(regions)==10
assert newids==set(parents)|set(regions) and not set(parents)&set(regions)
ledger={r['id']:r for r in rows(V7/'operational_costs.csv')};oldledger={r['id']:r for r in rows(V6/'operational_costs.csv')}
registered={r['id']:r for r in rows(V7/'registered_v7_increment.csv')};assert set(registered)==newids
for key in newids:
    op=ops[key];norm=proof(op['source']);original=proof(norm['original_source'])
    equal(pointer(original,norm['original_wall_pointer']),norm['wall_seconds'])
    equal(norm['wall_seconds'],ledger[key]['recorded_wall_seconds']);equal(norm['wall_seconds'],registered[key]['wall_seconds'])
    assert norm['CPU_components']=={p:pointer(original,p) for p in norm['CPU_component_pointers']}
    if norm['recorded_inclusive_CPU_seconds'] is not None:
        equal(sum(norm['CPU_components'].values()),norm['recorded_inclusive_CPU_seconds'])
        equal(norm['recorded_inclusive_CPU_seconds'],ledger[key]['recorded_process_cpu_seconds'])
    assert norm['nested_subtimers_added'] is False and norm['new_operation_created_by_normalization'] is False
    assert op['add_to_recorded_wall_totals']==(key in parents)
    assert ledger[key]['included_in_disjoint_recorded_wall_sum']==str(key in parents)
parent_wall=sum(float(ledger[k]['recorded_wall_seconds']) for k in parents)
parent_cpu=sum(float(ledger[k]['recorded_process_cpu_seconds'] or 0) for k in parents)
region_wall=sum(float(ledger[k]['recorded_wall_seconds']) for k in regions)
equal(parent_wall,6844.962586797774);equal(parent_cpu,6504.615013000002);equal(region_wall,.31587640196084976)
equal(sum(float(ledger[k]['recorded_wall_seconds']) for k in parents[:18]),351.465067)
equal(sum(float(ledger[k]['recorded_wall_seconds']) for k in parents[18:33]),6485.179102)
check('All34 parent and10 excluded region normalized wall/CPU pointers equal fresh original receipts, registered CSV and ledger; no normalized nested subtimer added')
def included(mapping):return {k:r for k,r in mapping.items() if r['included_in_disjoint_recorded_wall_sum']=='True'}
a,b=included(oldledger),included(ledger)
added=set(b)-set(a);dynamic=added-set(parents)
assert len(dynamic)==7 and all(k.startswith('C_draft:pythia-1b:') for k in dynamic)
changed={k for k in a.keys()&b.keys() if a[k]['recorded_wall_seconds']!=b[k]['recorded_wall_seconds']}
assert changed=={'opaque_stable_full_file_reads'}
newC=sum(float(b[k]['recorded_wall_seconds']) for k in dynamic);equal(newC,1781.1728460863233)
opaque=float(b['opaque_stable_full_file_reads']['recorded_wall_seconds'])-float(a['opaque_stable_full_file_reads']['recorded_wall_seconds'])
equal(opaque,139.0627657622099)
old_total=sum(float(r['recorded_wall_seconds']) for r in a.values());new_total=sum(float(r['recorded_wall_seconds']) for r in b.values())
equal(new_total-old_total,parent_wall+newC+opaque);equal(new_total-old_total,8765.198198646307)
equal(new_total,24419.9454260692)
assert len(set(ledger)-set(oldledger))==51
for f in ('operational_category_totals.csv',):equal(sum(float(r['known_disjoint_recorded_wall_seconds'] or 0) for r in rows(V7/f)),new_total)
check('v6-v7 ledger reconciles exactly34 parents+7 new C1B drafts+opaque receipt growth;10 extra region rows remain nonadditive')
receipt=proof(complete['collection_receipt']);previous=load(V6/'collection_receipt.json')
copies={v['original_path']:v for v in receipt['exact_cost_source_copies']}
coverage=load(V7/'C_export_record_coverage.json')
assert {v['model']:v['actual_first_created_records_observed'] for v in coverage}=={'pythia-410m':24,'pythia-1b':12,'pythia-2.8b':1}
for key in dynamic:
    row=b[key];source=copies[row['source']]
    draft=load(source['snapshot_path'],source['sha256']);equal(draft['wall_seconds'],row['recorded_wall_seconds'])
    assert draft['run_id']==key.split(':',2)[2] and draft['status']=='verified_draft_not_deliverable'
inv=proof(complete['inventory']);oldinv=proof(previous['inventory'])
opaque_rows={r['receipt_sha256']:r for r in inv['historical_opaque_hash_receipts']}
old_opaque={r['receipt_sha256']:r for r in oldinv['historical_opaque_hash_receipts']}
assert len(opaque_rows)==1128 and len(old_opaque)==1104 and old_opaque.items()<=opaque_rows.items()
fresh_receipts=set(opaque_rows)-set(old_opaque);assert len(fresh_receipts)==24
for key in fresh_receipts:
    row=opaque_rows[key];r=load(row['receipt'],key)
    assert r['method']=='sha256_entire_open_fd_to_eof_v1' and r['complete_to_eof'] is True
    assert r['stat_before']==r['stat_after']==r['fd_before']==r['fd_after']
    assert r['read_bytes']==r['stat_before']['size'] and row['current_inventory_input_payload_reads']==0
    equal(r['full_read_seconds'],row['historical_full_read_seconds'])
equal(sum(opaque_rows[k]['historical_full_read_seconds'] for k in fresh_receipts),opaque)
check('7 C1B draft costs verified against exact captured JSON, not future publication;24 new opaque full-read receipts unique by receipt SHA, no current stat-hit cost imputed')
for name in ('evaluation_category_totals.csv',):
    assert rows(V7/name)==rows(V6/name)
evals=rows(V7/'evaluation_category_totals.csv')
assert sum(int(r['attested_new_forward_jobs']) for r in evals)==296 and sum(int(r['attested_reuse_jobs']) for r in evals)==18
audit=proof(complete['IO_audit']);reads=audit['score_output_stream_hash_reads']
assert len(reads)==len({r['path'] for r in reads})==332
assert all(r['actual_full_stream_hash_bytes']==r['file_bytes']*r['full_stream_hash_open_count'] for r in reads)
assert sum(r['actual_full_stream_hash_bytes'] for r in reads)==456416316
assert sum(r['file_bytes'] for r in reads)==434689335
assert audit['score_output_semantic_deserialization'] is False and audit['weight_base_TAR_ZIP_tensor_payload_bytes_read']==0
assert audit['forbidden_weight_base_archive_tensor_payload_open_attempts']==[]
check('Forward296/reuse18 and evaluation fees unchanged; successful collection SHA-only456416316 bytes includes repeat opens, not semantic unsealing or physical-disk I/O; weight/archive reads0')
failed=load(ROOT/'artifacts/provenance/cost_sources_v7_failed_attempt1/FAILURE_RECEIPT.json')
ft=load(ROOT/'artifacts/provenance/cost_sources_v7_failed_attempt1/collection_whole_child_timing.json',failed['whole_child_timing_sha256'])
assert ft['returncode']==1 and isinstance(failed['score_output_stream_SHA_bytes_for_failed_attempt'],str)
equal(ft['whole_child_wall_seconds'],ledger['cost_v7_initial_failed_collection_whole_child']['recorded_wall_seconds'])
assert not (ROOT/failed['original_output']/'inventory.json').exists()
unknown=ledger['B_nll_CPU_analysis_unknown'];assert unknown['recorded_wall_seconds']=='' and unknown['included_in_disjoint_recorded_wall_sum']=='False'
assert complete['whole_study_complete'] is False
assert not any('behavioral_presentation' in json.dumps(r) for r in spec['additional_operations'])
check('Failed attempt1 rc1 retained and charged once; unsuccessful IO bytes unknown; B full CPU analysis unknown; late presentation v1/v2 not retroactively charged')
for name in ('native_restore_author_28_development_tests','native_restore_author_initial29_tests','native_restore_author_timeout6_targeted_tests','native_restore_ROOT_post_repair_original29_tests','CPU_full840_test_suite_whole_child'):
    norm=proof(ops[name]['source']);timer=proof(norm['original_source']);assert timer['returncode']==0
    if name.startswith('native_'):assert any('artifacts/provenance/' in v for v in timer['argv'])
    else:assert timer['argv']==['python','-m','pytest','-q']
assert 'testpaths = ["tests"]' in read(ROOT/'pyproject.toml').decode()
for item in spec['revision7_nested_source_exclusions']:
    proof(item['source']);assert not any(r.get('source',{}).get('path')==item['source']['path'] for r in spec['additional_operations'] if r['id'] in newids)
check('840 test discovery excludes provenance native29/timeout6; separate earlier targeted calls remain actual costs; B assembly/publisher and duplicate probe references excluded')
for p,s in SOURCES.items():
    st=Path(p).stat();assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==tuple(s['five_stat'])
after=resource.getrusage(resource.RUSAGE_SELF)
result={'status':'passed_independent_cost_only_review','checks':CHECKS,'sources':SOURCES,
 'arithmetic':{'new_parent_count':34,'excluded_regions':10,'new_parent_wall_seconds':parent_wall,'new_parent_known_CPU_seconds':parent_cpu,
 'new_C1B_draft_wall_seconds':newC,'opaque_growth_seconds':opaque,'new_opaque_receipts':24,'operational_total_delta_seconds':new_total-old_total,
 'operational_recorded_wall_seconds':new_total},
 'actual_score_files_or_weight_archive_payloads_opened':False,'collector_or_tests_executed':False,'v7_originals_modified':False,
 'scope_limit':'Cost metadata only (including finite v6/v7 ledger JSON); current live C/cache may grow. This is neither whole-study completion nor elapsed/pure GPU time.',
 'review_process_region_wall_seconds':time.perf_counter()-START,'review_process_region_user_seconds':after.ru_utime-CPU.ru_utime,
 'review_process_region_system_seconds':after.ru_stime-CPU.ru_stime,'timing_scope':'In-process audit region, excludes interpreter/import/authoring and receipt serialization; not a full engineering parent timer',
 'fresh_cost_metadata_bytes':sum(r['bytes'] for r in SOURCES.values())}
with (HERE/'REVIEW.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
print(json.dumps({'status':result['status'],'checks':len(CHECKS),'metadata_bytes':result['fresh_cost_metadata_bytes'],'arithmetic':result['arithmetic']},indent=2))
