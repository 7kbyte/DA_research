"""Independent actual-B recovery receipt review: small metadata and stat only."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import os
import stat
import sys

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
PARENT=ROOT/'artifacts/provenance/B_native_restore_actual_parent_v1'
INNER=ROOT/'artifacts/provenance/B_native_restore_actual_v1'
SOURCES={};OPEN_DENIALS=[]
QA=Path('/tmp/grad-research-20261106/reproduction_qa/B_native_B4_tr11_gen11_draw1001_k20_final_v1')
def audit(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes)):return
    p=Path(args[0]).resolve()
    if p.suffix in {'.tar','.zip','.safetensors','.pt','.bin'} or p.is_relative_to(QA/'base'):
        OPEN_DENIALS.append(str(p));raise RuntimeError('Large/base/model payload open forbidden: '+str(p))
sys.addaudithook(audit)
def small_bytes(path):
    p=Path(path).resolve();assert p.stat().st_size<20*1024*1024
    value=p.read_bytes();SOURCES[str(p)]={'sha256':hashlib.sha256(value).hexdigest(),'bytes':len(value)};return value
def read(path):return json.loads(small_bytes(path))
def proof(path):
    small_bytes(path);return {'path':str(Path(path).resolve()),**SOURCES[str(Path(path).resolve())]}
def bound(p):
    value=small_bytes(p['path']);assert SOURCES[str(Path(p['path']).resolve())]['sha256']==p['sha256'];return json.loads(value)
def identical(a,b):return Path(a['path']).resolve()==Path(b['path']).resolve() and a['sha256']==b['sha256']
def fstat(path):
    s=Path(path).stat();return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'uid':s.st_uid,'gid':s.st_gid,'mode':oct(stat.S_IMODE(s.st_mode)),'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def write(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

root_proof=proof(PARENT/'ROOT_acceptance.json');assert root_proof['sha256']=='0194effe57624ce7ffac21396c57383cce5cbc067e93451738f755325eb25071'
root_accept=read(root_proof['path']);whole=bound(root_accept['whole_process_timing']);inner=bound(root_accept['internal_timing']);authority=bound(root_accept['execution_authority']);plan=bound(root_accept['prepared_plan'])
assert root_accept['status']=='ROOT_accepted_one_fixed_B_native_durable_package_actual_CPU_restore'
assert root_accept['execution_authority']['sha256']=='93c7456bbcfd01cc0d86000e459a9a37604c03944281c398372ffae6aa0ee009'
assert root_accept['prepared_plan']['sha256']=='61c19ee07d69936da329846fcb8aa8bd67ea8d8fe1ff8cd6b3084c032c3fa3c7'
assert authority['authorized'] is True and authority['authorized_by']=='root' and authority['not_ROOT_authority'] is False
assert identical(authority['plan'],root_accept['prepared_plan']) and authority['runner']==plan['runner'] and authority['checker']==plan['checker']
for p in (authority['runner'],authority['checker']):assert proof(p['path'])['sha256']==p['sha256']
assert authority['runner']['sha256']=='6d7245e78e225c3879eaf931b85e7be073b4ba9777d7263b7e0e1c39b1630a02'
prep=bound(authority['independent_preparation_review']);assert authority['independent_preparation_review']['sha256']=='294c812ba4661f325121cf3655ba1c098cc7cdb11d691fcfd0ed4c32488236e8'
assert identical(prep['plan'],authority['plan']) and identical(prep['runner'],authority['runner'])
evidence={k:bound(p) for k,p in plan['evidence'].items()}
pub=evidence['publication'];assert pub['status']=='passed' and pub['destination']==plan['archive'] and pub['destination_sha256']==plan['historical_full_TAR_sha256']==pub['source_sha256']
assert evidence['completion']['tar_sha256']==pub['destination_sha256'] and evidence['completion']['manifest_sha256']==plan['evidence']['manifest']['sha256']
assert evidence['completion']['file_fsync_completed'] is True and evidence['completion']['parent_directory_fsync_completed'] is True
assert evidence['manifest']['logical_endpoints']==evidence['manifest']['unique_weights']==60 and evidence['manifest']['optimizer_included'] is False
assert evidence['ROOT_acceptance']['publication_receipt']==pub and evidence['ROOT_acceptance']['unique_encoded_weights']==60

small=root_accept['small_QA_evidence'];assert len(small)==9
for p in small.values():assert proof(p['path'])['sha256']==p['sha256']
receipt=bound(small['actual_restore_receipt.json']);fp=bound(small['independent_fingerprint.json']);decode=bound(small['restored/B4_seed11_final.pt.decode.json'])
assert bound(small['bound_plan.json'])==plan and receipt['plan']==authority['plan'] and receipt['authority']['sha256']==root_accept['execution_authority']['sha256']
assert json.loads(small_bytes(small['logs/decode.stdout.log']['path']))==decode
assert json.loads(small_bytes(small['logs/independent_fingerprint.stdout.log']['path']))==fp
assert small_bytes(small['logs/decode.stderr.log']['path'])==small_bytes(small['logs/independent_fingerprint.stderr.log']['path'])==b''
selected=plan['selected'];run='bankvar_B4_tr11_gen11_draw1001_k20_b0p05_w0p05'
assert plan['run_id']==root_accept['run_id']==receipt['run_id']==fp['run_id']==run and receipt['status']==fp['status']=='passed'
assert (selected['group'],selected['training_seed'],selected['generation_seed'],selected['selection_seed'],selected['bank_chunks'],selected['kind'])==('B4',11,11,1001,20,'final')
assert selected['actual_exposure']=={'checkpoint_step':306,'target_prediction_tokens':10000000,'replay_prediction_tokens':500000,'full_training_target_prediction_tokens':10000000,'full_training_replay_prediction_tokens':500000}
assert selected['ratio']==selected['weight']==.05 and selected['unique_bank_prediction_tokens']==10220
matches=[r for r in evidence['manifest']['logical_runs'] if r['run_id']==run];assert len(matches)==1 and matches[0]==selected
expected=plan['members'];actual=receipt['extraction']['files'];assert len(expected)==len(actual)==12 and len({x['member'] for x in actual})==12
member_rows=[]
for row in actual:
    assert row['sha256']==expected[row['member']]['sha256'] and row['bytes']==expected[row['member']]['bytes']
    assert (QA/row['member']).stat().st_size==row['bytes']
    member_rows.append({'member':row['member'],'bytes':row['bytes'],'planned_sha256':expected[row['member']]['sha256'],'actual_extraction_reported_fresh_sha256':row['sha256'],'review_fresh_payload_read':False})
assert len([x for x in actual if x['member'].startswith('base/')])==6
assert plan['native_base']=='base/pythia-410m-final/model.safetensors' and receipt['extraction']['selected_member_payload_bytes']==sum(x['bytes'] for x in actual)==plan['selected_member_bytes']==2058150693
assert receipt['extraction']['source_TAR_bytes_returned_to_Python']==root_accept['source_TAR_bytes_returned_to_Python']==2058355626
expected_fp=selected['model_fingerprint'];assert expected_fp==fp['fingerprint']==decode['model_fingerprint']==selected['encode']['model_fingerprint']==selected['decode']['model_fingerprint']==selected['bytewise_verification']['model_fingerprint']==root_accept['whole_model_fingerprint']
assert expected_fp=={'sha256_model_state':'05724eda110e91eb76c8f6d45b54386b374e486662207a266166a0fde0ffb790','state_elements':405334016}
assert fp['tensor_count']==decode['tensor_raw_hashes_verified']==selected['bytewise_verification']['tensor_count']==plan['tensor_count']==root_accept['tensor_count']==292
assert fp['independent_tensor_payload_bytes_hashed']==plan['model_raw_bytes']==expected_fp['state_elements']*4==root_accept['independent_full_tensor_bytes_hashed']==1621336064
assert fp['fresh_new_restored_file_sha256']==decode['restored_file_sha256']==root_accept['new_restored_file_SHA256']
assert fp['fresh_new_restored_serialized_bytes_hashed']==decode['restored_bytes']==root_accept['fresh_serialized_restored_bytes_hashed']==1621450850
assert fp['source_checkpoint_sha256']==selected['source_checkpoint_sha256']==selected['frozen_checkpoint_identity_sha256']
assert fp['original_checkpoint_read'] is fp['original_CP_fresh_uint8_comparison'] is fp['Adam_resume_verified'] is fp['GPU_used'] is False
assert fp['archive_only_native_base'] is True and receipt['original_checkpoint_or_live_base_opened'] is receipt['whole_source_TAR_rehash'] is receipt['new_encoding'] is receipt['GPU_used'] is False
assert decode['status']=='decoded_and_hash_verified' and decode['gpu_used'] is False
assert root_accept['ROOT_payload_bytes_rehashed']==0

for value in (whole,inner):assert type(value['returncode']) is int and value['returncode']==0 and value['GPU_used'] is False
assert whole==root_accept['whole_process_cost'] and whole['argv'][1]=='-I'
assert inner['timed_out'] is False and type(inner['direct_child_returncode']) is int and inner['direct_child_returncode']==0
assert inner['process_group_cleanup']['whole_group_absence_confirmed'] is True and inner['process_group_cleanup']['final_observation']['members']==[]
pgid=inner['process_group_cleanup']['pgid']
try:os.killpg(pgid,0);pg_exists=True
except ProcessLookupError:pg_exists=False
assert not pg_exists
lock=Path(plan['sequential_restore_lock']);s=lock.stat();lock_key=f'{os.major(s.st_dev):02x}:{os.minor(s.st_dev):02x}:{s.st_ino}'
locks=[line for line in Path('/proc/locks').read_text().splitlines() if lock_key in line];assert not locks
tar_stat=fstat(plan['archive']);restored_stat=fstat(QA/decode['restored'])
for k,v in plan['archive_stat'].items():assert tar_stat[k]==v
for k,v in root_accept['published_B_TAR_current_stat'].items():assert tar_stat[k]==v
for k,v in root_accept['new_restored_file_current_stat'].items():assert restored_stat[k]==v
assert tar_stat['bytes']==pub['bytes']==69578864640
events={k:int(v) for k,v in [line.split() for line in Path('/sys/fs/cgroup/memory.events').read_text().splitlines()]}
before=receipt['resource_admission']['memory_events'];delta={k:events[k]-before[k] for k in before}
assert all(before[k]==events[k]==0 for k in ('oom','oom_kill','oom_group_kill'))
assert not OPEN_DENIALS

with (BASE/'selected_member_hash_binding.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(member_rows[0]));w.writeheader();w.writerows(member_rows)
copies=BASE/'small_source_copies';inventory=[]
for path,value in sorted(SOURCES.items()):
    p=Path(path);assert p.is_relative_to(ROOT);dst=copies/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==value['sha256'];dst.write_bytes(data);inventory.append({'path':path,**value,'copy':str(dst.relative_to(ROOT))})
write(BASE/'source_manifest.json',{'schema_version':1,'sources':inventory})
write(BASE/'REVIEW.json',{'status':'passed_independent_metadata_review_of_actual_fixed_B_restore','observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ROOT_acceptance':root_proof,'authority':root_accept['execution_authority'],'plan':root_accept['prepared_plan'],'source_manifest':proof(BASE/'source_manifest.json'),'selected_member_count':12,'all_reported_fresh_member_SHA_and_size_match_plan':True,'native_base_files':6,'fixed_sample':{k:selected[k] for k in ('run_id','group','training_seed','generation_seed','selection_seed','bank_chunks','kind','actual_exposure')},'model_fingerprint':expected_fp,'tensor_count':292,'new_serialized_model_sha256':fp['fresh_new_restored_file_sha256'],'full_parent_wall_seconds':whole['wall_seconds'],'full_parent_user_seconds':whole['user_cpu_seconds'],'full_parent_sys_seconds':whole['sys_cpu_seconds'],'full_parent_peak_RSS_bytes':whole['child_peak_rss_bytes'],'all_inner_PG_body_stages_nested_not_additive':True,'PG_current_exists':pg_exists,'PG_cleanup_receipt':inner['process_group_cleanup'],'shared_lock_current_proc_locks_matches':locks,'lock_observation_is_not_a_lease':True,'TAR_current_stat':tar_stat,'new_restored_model_current_stat':restored_stat,'memory_events_current':events,'memory_event_delta_since_body_admission':delta,'memory_pressure_cannot_be_attributed_to_QA':True,'independent_reviewer_payload_bytes_read':0,'independent_reviewer_payload_hashes_recomputed':0,'independent_reviewer_did_not_run_decode_or_tensor_fingerprint':True,'review_checks_existing_fresh_recovery_evidence_not_fresh_original_CP_uint8':True,'manual_review_parent_wall_seconds':None,'manual_review_parent_timing_status':'unknown; automated review child timer is separate','claim':'One preselected actual B final model recovered using durable native archive. No all60 restore, original CP recompare, Adam resume, GPU training or scientific score inference.'})
print(json.dumps({'status':'passed_metadata_only','selected_members':12,'tensor_count':292,'source_files':len(inventory),'review_payload_read_bytes':0,'PG_absent':not pg_exists,'lock_no_record':not locks,'current_memory_events':events}))
