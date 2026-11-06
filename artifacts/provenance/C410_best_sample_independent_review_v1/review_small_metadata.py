"""Read only small proofs and payload stat; never open CP/ZIP/TAR/base weights."""
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import resource
import shutil
import time

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
DELIVERY=Path('/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4')
SAMPLE=ROOT/'artifacts/provenance/additional_endpoint_actual_v4/C410_best_sample'
START=time.perf_counter();CPU=resource.getrusage(resource.RUSAGE_SELF);SOURCES={};RAW={}
def sig(p):
    s=Path(p).stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def raw(p,expected=None):
    p=Path(p).resolve();assert p.suffix in ('.json','.py','.csv','.log') and p.stat().st_size<16<<20,p
    if str(p) not in RAW:
        before=sig(p);b=p.read_bytes();assert before==sig(p)
        RAW[str(p)]=b;SOURCES[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'five_stat':before}
    if expected:assert SOURCES[str(p)]['sha256']==expected,p
    return RAW[str(p)]
def load(p,h=None):return json.loads(raw(p,h))
def proof(p):return load(p['path'],p['sha256'])
authority_path=ROOT/'artifacts/provenance/additional_endpoint_actual_v4/ROOT_C410_best_sample_authority.json'
auth=load(authority_path,'f4e2f428167089cab0ce423b49dbf6691b35abb348b1a7a19b815dd7f0e0c154')
record_path=DELIVERY/'C410_best/records/extended_v2_pythia_410m_m64_formal_B0_s11_w0__best.json'
r=load(record_path);timer=load(SAMPLE/'whole_child_timing.json');launch=load(SAMPLE/'child_launch.json')
spec=load(launch['spec_path'],launch['spec_sha256']);scope=proof(auth['scope']);raw(auth['exporter']['path'],auth['exporter']['sha256']);raw(auth['wrapper']['path'],auth['wrapper']['sha256'])
assert spec['argv']==launch['argv'] and spec['authority']['sha256']==SOURCES[str(authority_path.resolve())]['sha256']
assert spec['env']['CUDA_VISIBLE_DEVICES']=='' and all(spec['env'][k]=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
assert auth['max_new_weights']==1 and auth['package']==r['package']=='C410_best' and auth['slot']==spec['slot']=='slot1'
assert r['status']=='verified_additional_endpoint_not_yet_assembled' and r['kind']=='best' and r['group']=='B0' and r['training_seed']==11
assert r['run_id']==auth['first_sample_run_id'] and r['source_checkpoint_sha256']==auth['first_sample_checkpoint_sha256']==r['frozen_checkpoint_identity_sha256']
assert r['source_checkpoint_sha256']=='dd25db838ede6d137cd630818e5ecb3ad8f9100ef47eba86740a48f04a260bf4'
assert r['exporter_sha256']==auth['exporter']['sha256'] and r['prepared_scope_sha256']==auth['scope']['sha256']
phase=scope['phases']['C410_best']['context'];manifest=proof(auth['manifest']);ready=proof(auth['READY']);completion=proof(auth['phase_completion']);audit=proof(auth['phase_audit']);pairs=proof(auth['best_model_equivalence'])
assert completion['status']=='passed' and completion['audit']==auth['phase_audit'] and completion['proofs']['best']==auth['best_model_equivalence']
assert audit['status']=='passed' and audit['expected_runs']==audit['completed_runs']==24 and audit['source_violations']==[]
assert pairs['source_audit']==completion['audit'] and pairs['source_registry']==completion['source_registry'] and pairs['source_snapshot']==completion['input_snapshot']
for p in (completion['source_registry'],completion['input_snapshot']):proof(p)
assert len(phase['rows'])==len(phase['logical_run_ids'])==24 if 'logical_run_ids' in phase else len(phase['rows'])==24
best={b['run_id']:b for b in manifest['records'] if b['kind']=='best'}
assert len(best)==24 and set(best)==set(auth['selected_completed_records'])=={row['id'] for row in phase['rows']}
for key,state in auth['selected_completed_records'].items():
    assert state==audit['completed_scheduler_records'][key]
    assert state['status']=='completed' and type(state['returncode']) is int and state['returncode']==0 and state['missing_outputs']==[]
    assert not any('stop_requested' in k for k in state)
bound=best[r['run_id']]
assert bound['checkpoint_sha256']==r['source_checkpoint_sha256'] and bound['checkpoint']==r['source_checkpoint']
assert bound['step']==r['actual_exposure']['checkpoint_step']==1224 and bound['target_prediction_tokens_at_checkpoint']==r['actual_exposure']['target_prediction_tokens']==40029696
assert r['actual_exposure']['replay_prediction_tokens']==0 and r['actual_exposure']['full_training_target_prediction_tokens']==100000000
assert r['source_stat']==r['source_fd_five_stat_before']==r['source_fd_five_stat_after']==sig(r['source_checkpoint'])
assert sig(r['archive_path'])==r['archive_stat'] and not Path(r['source_checkpoint']).is_symlink()
raw(bound['config_path'],r['config_sha256']);raw(bound['summary_path'],r['summary_sha256'])  # summary SHA only, no metric parsing
assert bound['input_sha256'][bound['config_path']]==r['config_sha256'] and bound['input_sha256'][bound['summary_path']]==r['summary_sha256']
assert r['definition']==phase['definitions'][r['run_id']] and r['definition_sha256']==r['completed_scheduler_record']['definition_sha256']
assert r['completed_scheduler_record']==auth['selected_completed_records'][r['run_id']]
fp=r['model_fingerprint'];assert fp['state_elements']==405334016
assert all(r[k]['model_fingerprint']==fp for k in ('encode','decode','bytewise_verification'))
assert r['decode']['tensor_raw_hashes_verified']==r['bytewise_verification']['tensor_count']==292
assert r['bytewise_verification']['status']=='passed' and r['bytewise_verification']['all_tensor_bytes_equal'] is True
assert r['all_nonweight_metadata_roundtrip_verified'] is True and r['encode']['lossless_weight_bits'] is True and r['encode']['model_only'] is True
assert r['decode']['model_only_evaluation_format'] is True and r['additional_encoding_calls']==1
assert r['encode']['source_checkpoint_sha256']==r['bytewise_verification']['source_checkpoint_sha256']==r['source_checkpoint_sha256']
assert r['archive_sha256']==r['encode']['archive_sha256']==r['bytewise_verification']['archive_sha256']
assert r['encode']['base_sha256']==phase['base_sha']=='3e09001631bb18ec8a6f4522f8e6ab26a53099e69228f9822adc91bfbe27511b'
base=next(x for x in scope['phases']['C410_best']['base_files'] if x['path']==phase['base'])
assert base['sha256']==phase['base_sha']
aliases=[]
assert pairs['status']=='passed' and pairs['endpoint']=='best' and len(pairs['comparisons'])==3
for pair in pairs['comparisons']:
    assert pair['seed'] in (11,22,33) and pair['bitwise_model_state_equal'] is True and pair['endpoint']=='best'
    assert pair['left']['model_state_fingerprint']==pair['right']['model_state_fingerprint']
    for side in (pair['left'],pair['right']):
        b=best[side['run_id']];assert side['kind']=='best' and side['checkpoint_sha256']==b['checkpoint_sha256'] and side['checkpoint']==b['checkpoint']
        assert side['checkpoint_exposure']['step']==b['step']
        for key in ('target_prediction_tokens_at_checkpoint','replay_prediction_tokens_at_checkpoint'):assert side['checkpoint_exposure'][key]==b[key]
        raw(side['config_path'],side['config_sha256']);raw(side['summary_path'],side['summary_sha256'])
        st=sig(side['checkpoint']);assert st['bytes']==b['checkpoint_bytes'] and st['mtime_ns']==b['checkpoint_mtime_ns']
    aliases.append({'seed':pair['seed'],'left':pair['left']['run_id'],'right':pair['right']['run_id'],'proof_source_SHA_bound':True,'eligible_not_yet_materialized':True})
assert {x['seed'] for x in aliases}=={11,22,33}
assert type(timer['returncode']) is int and timer['returncode']==0 and timer['wall_seconds']>=r['wall_seconds']
assert sum(r[k]['wall_seconds'] for k in ('encode','decode','bytewise_verification'))<=r['wall_seconds']
assert timer['child_peak_rss_bytes']==r['peak_rss_bytes_process_cumulative']==3686518784 and timer['GPU_used'] is False and timer['publication_performed'] is False
def events(text):return {k:int(v) for k,v in (line.split() for line in text.splitlines())}
before=events(r['cgroup_memory_before']['memory.events']);after=events(r['cgroup_memory_after']['memory.events'])
assert all(before[k]==after[k]==0 for k in ('oom','oom_kill','oom_group_kill'))
current={k:(Path('/sys/fs/cgroup')/k).read_text().strip() for k in ('memory.current','memory.max','memory.events','memory.stat')}
stats=events(current['memory.stat']);clean=max(0,stats['inactive_file']-stats['file_dirty']-stats['file_writeback']-stats['unevictable'])
available=max(0,int(current['memory.max'])-int(current['memory.current']))+clean
projection_time=dt.datetime.now(dt.timezone.utc);encode=20*120;prepub=encode+360+540;total=prepub+900+600
assert prepub==3300 and total==4800
pub=projection_time+dt.timedelta(seconds=prepub);finish=projection_time+dt.timedelta(seconds=total)
assert pub<dt.datetime(2026,11,7,2,tzinfo=dt.timezone.utc) and finish<=dt.datetime(2026,11,7,2,56,32,tzinfo=dt.timezone.utc)
costs=list(csv.DictReader(io.StringIO(raw(ROOT/'results/cost_inventory_final_v7_cpu_collection_20261106_attempt2/operational_costs.csv').decode())))
v6timer=ROOT/'artifacts/provenance/cost_sources_v6/trial_whole_child_timing.json';v7timer=ROOT/'artifacts/provenance/cost_sources_v7/collection_whole_child_timing.json'
unregistered=[]
for p in (v6timer,v7timer):
    t=load(p);assert not any(row['source']==str(p) for row in costs)
    assert not any(row['recorded_wall_seconds'] and abs(float(row['recorded_wall_seconds'])-t['whole_child_wall_seconds'])<1e-10 for row in costs)
    unregistered.append({'path':str(p),'sha256':SOURCES[str(p.resolve())]['sha256'],'wall_seconds':t['whole_child_wall_seconds'],'status':'next_version_candidate_not_v7_charge'})
for path,s in SOURCES.items():assert sig(path)==s['five_stat']
resource_cpu=resource.getrusage(resource.RUSAGE_SELF)
out={'status':'passed_small_evidence_and_resource_review_not_batch_authority','sources':SOURCES,'record':{'path':str(record_path),'sha256':SOURCES[str(record_path)]['sha256']},
 'full_state_receipt_consistency':{'elements':fp['state_elements'],'tensor_count':292,'fingerprint':fp['sha256_model_state'],'fresh_independent_weight_bytes_read':0,'scope':'Prior actual exporter encode/decode/all-uint8 proof + source/code/stat bindings; not a second payload roundtrip'},
 'source_cohort_completed':24,'source_bound_best_alias_eligible':aliases,'currently_observed_sample_records':len(list((DELIVERY/'C410_best/records').glob('*.json'))),
 'remaining_plan':{'remaining_nonalias_encoding_targets':20,'per_target_seconds':120,'three_alias_source_verification_allowance_seconds':360,'assembly_seconds':540,'combined_assembly_allowance_seconds':900,'publication_seconds':900,'finish_margin_seconds':600,
 'seconds_to_projected_publication_start':prepub,'seconds_including_publication_and_margin':total,'observation_utc':projection_time.isoformat(),'projected_publication_start_utc':pub.isoformat(),'projected_protected_finish_utc':finish.isoformat(),'latest_launch_publication_constraint':'strictly before2026-11-07T01:05:00Z','latest_launch_finish_constraint':'at or before2026-11-07T01:36:32Z','binding_constraint':'strict publication-start boundary; future fresh ROOT admission required'},
 'sample_cost':{'whole_child_wall_seconds':timer['wall_seconds'],'whole_child_user_seconds':timer['user_cpu_seconds'],'whole_child_system_seconds':timer['sys_cpu_seconds'],'endpoint_parent_seconds':r['wall_seconds'],'codec_decode_verify_children_seconds':{k:r[k]['wall_seconds'] for k in ('encode','decode','bytewise_verification')},'nested_timers_not_additive':True},
 'resource':{'sample_peak_rss_bytes':timer['child_peak_rss_bytes'],'sample_before_events':before,'sample_after_events':after,'max_event_delta':after['max']-before['max'],'current_cgroup':current,'clean_inactive_credit_bytes':clean,'conservative_available_bytes_observed':available,'scratch_free_bytes':shutil.disk_usage('/tmp/grad-research-20261106').free,'not_a_reservation_or_guarantee':True},
 'unregistered_collector_parent_candidates':unregistered,'batch_or_publication_authorized_by_this_review':False,'existing_sources_modified':False,'CP_ZIP_TAR_base_payload_bytes_read':0,'scoring_or_GPU_invoked':False,
 'audit_region_wall_seconds':time.perf_counter()-START,'audit_region_user_seconds':resource_cpu.ru_utime-CPU.ru_utime,'audit_region_system_seconds':resource_cpu.ru_stime-CPU.ru_stime,'timing_scope':'In-process small metadata/stat review, not complete authoring or interpreter child parent',
 'payload_open_policy':'This reviewer only read json/csv/py/log; source summary files were SHA-only, no validation/test score parsing; CP/ZIP/base operations stat only'}
with (HERE/'REVIEW.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'status':out['status'],'record_sha256':out['record']['sha256'],'projection':out['remaining_plan'],'conservative_available_bytes':available,'collector_timers_unregistered':unregistered},indent=2))
