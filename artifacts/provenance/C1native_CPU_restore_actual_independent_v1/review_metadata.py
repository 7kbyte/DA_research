#!/usr/bin/env python3
"""Review only already-produced small recovery evidence and current stats."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT=Path('/tmp/grad-research-20261106/worktree')
UTIL=ROOT/'artifacts/provenance/C2native_CPU_restore_bound_plan_independent_v1/review_metadata.py'
UTIL_SHA='e064ad995cfec5e888e4a1db4b1db651e4d46424357b008300d595747b29f30b'
assert hashlib.sha256(UTIL.read_bytes()).hexdigest()==UTIL_SHA
spec=importlib.util.spec_from_file_location('metadata_only_utilities',UTIL)
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
raw,load,proof,need=H.raw,H.load,H.proof,H.demand
IN=ROOT/'artifacts/provenance/C_native_final_CPU_restore_actual_v1/1b'
QA=Path('/tmp/grad-research-20261106/reproduction_qa/native_Cfinal_1b_B4s11_final_v1')
OUT=Path(__file__).resolve().parent

def main():
    raw(UTIL,UTIL_SHA)
    p=load(IN/'bound_plan.json','820774f9b6bf9a2d54db73460a126cbcbf62e2a5be4cf6bb9d064340b52dd4fe')
    authority=load(IN/'ROOT_execution_authority.json','12cfd458053fd02e0cd3f9e5a6a9bec1ed32ff431b7c178e3e2242b9e4a41c19')
    need(authority['authorized'] is True and authority['authorized_by']=='ROOT' and authority['sequential_only'] is True and
         authority['scope']=='one_fixed_native_Cfinal_CPU_recovery' and authority['model']==p['model'] and authority['run_id']==p['run_id'],'Exact authority mismatch')
    need(authority['plan']==dict(path=str(IN/'bound_plan.json'),sha256='820774f9b6bf9a2d54db73460a126cbcbf62e2a5be4cf6bb9d064340b52dd4fe'),'Authority plan mismatch')
    for k in ('runner','checker'):
        need(authority[k]==p[k],'Runtime source mismatch');raw(p[k]['path'],p[k]['sha256'])
    raw(authority['whole_wrapper']['path'],authority['whole_wrapper']['sha256'])
    pre=proof(authority['ROOT_resource_gate']);proof(authority['independent_metadata_review'])
    for path,v in pre['source_checks'].items():raw(path,v['sha256'])
    ev=IN/'restore_whole_child_evidence'
    launch=load(ev/'launch.json','f3e1c1e332eed1bc8be391ee97973023bdd99d253ed69d8a8f820963cab0d791')
    whole=load(ev/'whole_child_timing.json','3b3878cd1321b4c6d45f29c819ef06578088f6d500214f1be845af81dc0d72c9')
    need(launch['argv']==['/usr/bin/python','-I',p['runner']['path'],'execute','--plan',authority['plan']['path'],
          '--plan-sha256',authority['plan']['sha256'],'--authority',str(IN/'ROOT_execution_authority.json'),
          '--authority-sha256','12cfd458053fd02e0cd3f9e5a6a9bec1ed32ff431b7c178e3e2242b9e4a41c19'],'Actual isolated invocation mismatch')
    need(launch['PYTHONPATH_absent'] is True and launch['GPU_hidden'] is True and
         launch['wrapper_source_sha256']==authority['whole_wrapper']['sha256'],'Actual launch source/isolation mismatch')
    need(all(type(whole[k]) is int and whole[k]==0 for k in ('returncode','direct_child_returncode')) and whole['timed_out'] is False,'Restore not strict rc0')
    cleanup=whole['process_group_cleanup'];pgid=cleanup['pgid']
    need(pgid==3460499 and cleanup['whole_group_absence_confirmed'] is True and cleanup['status']=='group_absent_no_cleanup_needed' and
         cleanup['signals']==[] and cleanup['final_observation']['killpg_0_exists'] is False and cleanup['final_observation']['members']==[],'Actual group cleanup not confirmed')
    current_group=[]
    for f in Path('/proc').glob('[0-9]*/stat'):
        try:
            b=f.read_text();z=b[b.rfind(')')+2:].split()
            if int(z[2])==pgid:current_group.append(dict(pid=int(f.parent.name),state=z[0]))
        except (OSError,ValueError,IndexError):pass
    need(not current_group,'Prior recovery process group exists now')
    try:os.killpg(pgid,0);exists=True
    except ProcessLookupError:exists=False
    need(not exists,'Prior recovery PGID remains')
    result=load(QA/'actual_restore_receipt.json','6b97a1d0706e5c7db7f6573417de6e0a4d644d61278996816dfc342dd5ed162d')
    independent=load(QA/'independent_fingerprint.json','3f20daf2c11170176fb298d0a6075f7a37e972223c3462b8efbda84a77ba0eee')
    need(result['status']=='passed' and result['model']==p['model'] and result['run_id']==p['run_id'] and
         result['plan']==authority['plan'] and result['authority']==dict(path=str(IN/'ROOT_execution_authority.json'),sha256='12cfd458053fd02e0cd3f9e5a6a9bec1ed32ff431b7c178e3e2242b9e4a41c19'),'Recovery source closure differs')
    need(result['independent_check']==independent and independent['status']=='passed' and independent['fingerprint']==p['selected']['model_fingerprint'] and
         independent['tensor_count']==196 and independent['independent_tensor_payload_bytes_hashed']==4047126528 and
         independent['source_checkpoint_sha256']==p['selected']['source_checkpoint_sha256'],'Complete recovered FP differs')
    decode=load(QA/'restored/B4_seed11_final.pt.decode.json')
    need(decode==json.loads(raw(QA/'logs/decode.stdout.log')) and decode['status']=='decoded_and_hash_verified' and
         decode['model_fingerprint']==independent['fingerprint'] and decode['tensor_raw_hashes_verified']==196 and
         decode['restored_file_sha256']==independent['fresh_new_restored_file_sha256']=='b9480d726b7d126c194ca8f5e23c921951a40abf696e883e3161f38c0992233d' and
         decode['restored_bytes']==independent['fresh_new_restored_serialized_bytes_hashed']==4047222722,'Decoder/checker new output SHA differs')
    need(json.loads(raw(QA/'logs/independent_fingerprint.stdout.log'))==independent,'Checker stdout/result mismatch')
    need([s['name'] for s in result['stages']]==['decode','independent_fingerprint'] and
         all(type(s['returncode']) is int and s['returncode']==0 for s in result['stages']),'Actual children not strict int0')
    need(result['stages'][0]['argv']==['/usr/bin/python','-I','tools/scripts/archive_model_weights.py','--threads','2','decode','--archive',p['selected']['archive_member'],
          '--base',p['native_base'],'--output','restored/B4_seed11_final.pt'] and
         result['stages'][1]['argv']==['/usr/bin/python','-I','check_restored.py'],'Decode/checker archived-only argv differs')
    need(all(result[k] is False for k in ('network_calls_requested','GPU_used','original_checkpoint_or_livebase_opened','whole_source_TAR_rehash')) and
         independent['original_checkpoint_read'] is False and independent['original_CP_fresh_uint8_comparison'] is False and
         independent['Adam_resume_verified'] is False and independent['archive_only_native_base'] is True,'Recovery scope overstated')
    need(p['selected']['actual_exposure']==dict(checkpoint_step=3058,target_prediction_tokens=100000000,replay_prediction_tokens=5000000,
         full_training_target_prediction_tokens=100000000,full_training_replay_prediction_tokens=5000000),'Fixed final exposure differs')
    raw(QA/'bound_plan.json',authority['plan']['sha256']);raw(QA/'check_restored.py',p['checker']['sha256'])
    files=result['extraction']['files'];need(len(files)==12 and {f['member'] for f in files}==set(p['members']),'Not exact12 selected members')
    member_stats={}
    for f in files:
        planned=p['members'][f['member']]
        need(f['sha256']==planned['sha256'] and (planned['bytes'] is None or f['bytes']==planned['bytes']),'Actual extraction member SHA/size differs')
        st=H.sig(QA/f['member']);need(st['bytes']==f['bytes'],'Current extracted file size changed');member_stats[f['member']]=st
    need(sum(f['bytes'] for f in files)==result['extraction']['selected_member_payload_bytes']==5175105157 and
         result['extraction']['source_TAR_bytes_returned_to_Python']==5175337756 and result['extraction']['full_TAR_SHA_recomputed'] is False,'Actual extraction read scope differs')
    tarstat=H.sig(p['archive']);need(tarstat==p['archive_stat'],'Durable TAR current stat differs')
    restoredstat=H.sig(QA/'restored/B4_seed11_final.pt');need(restoredstat['bytes']==4047222722,'Current new restored size differs')
    for name in ('tools/scripts/archive_model_weights.py','tools/src/replay_study/weight_archive.py','tools/src/replay_study/__init__.py'):
        raw(QA/name,p['members'][name]['sha256'])
    mem=result['resource_admission'];policy=p['resource_policy']
    clean=max(0,min(mem['inactive_file_bytes'],mem['file_bytes']-mem['shmem_bytes'])-mem['dirty_bytes']-mem['writeback_bytes']-mem['unevictable_bytes'])
    need(clean==mem['conservative_clean_inactive_credit_bytes'] and max(0,mem['memory_max_bytes']-mem['memory_current_bytes'])+clean==mem['conservative_available_bytes'] and
         mem['conservative_available_bytes']>=policy['required_reclaim_aware_memory_bytes'] and mem['disk_free_bytes']>=policy['required_scratch_bytes'],'Actual resource admission mismatch')
    lock=QA.parent/'.native_Cfinal_restore.lock';st=lock.stat();needle=f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}'
    locks=[l for l in Path('/proc/locks').read_text().splitlines() if needle in l]
    events={k:int(v) for k,v in (l.split() for l in Path('/sys/fs/cgroup/memory.events').read_text().splitlines())}
    copies=OUT/'small_actual_evidence';copies.mkdir()
    for path in [QA/'actual_restore_receipt.json',QA/'independent_fingerprint.json',QA/'restored/B4_seed11_final.pt.decode.json',
                 ev/'whole_child_timing.json',ev/'launch.json',IN/'ROOT_execution_authority.json']:
        b=raw(path);target=copies/path.name
        with target.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    receipt=dict(status='accepted_independent_metadata_review_of_one_actual_native_CPU_restore',at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        model=p['model'],fixed_run_id=p['run_id'],actual_exposure=p['selected']['actual_exposure'],native_files=6,selected_members=12,
        source_CP_SHA_provenance=p['selected']['source_checkpoint_sha256'],full_model_fingerprint=independent['fingerprint'],tensor_count=196,
        current_durable_TAR_stat=tarstat,current_new_restored_stat=restoredstat,extracted_member_current_stats=member_stats,
        actual_fresh_checks_reported=dict(selected_member_payload_SHA_bytes=5175105157,source_TAR_bytes_returned_to_Python=5175337756,
            independent_tensor_bytes=4047126528,new_serialized_SHA=independent['fresh_new_restored_file_sha256'],new_serialized_SHA_bytes=4047222722),
        actual_whole_child_timing=whole,timer_accounting='84.33324014395475s whole is inclusive; extraction/body/decode/checker/cleanup all nested and never added',
        current_prior_PG_absent=True,shared_restore_lock_current_observation=dict(path=str(lock),inode=st.st_ino,matching_proc_locks=locks,
            observation_not_a_lease=True,prior_runner_process_group_fully_gone=True),
        cgroup_memory_events_before=mem['memory_events'],cgroup_memory_events_current=events,
        cgroup_pressure_not_attributable_to_this_recovery=True,
        scope='One fixed B4 seed11 final model-only CPU recovery, not all24 endpoints, optimizer resume, or GPU retraining',
        limitations=['Historical wholeTAR SHA supported by publisher; this review does not rehash TAR/member/model payloads',
                     'No fresh original checkpoint uint8 comparison; source SHA is bound provenance',
                     'No Adam/optimizer resume and no new scientific seed or performance inference'],
        this_review_large_payload_bytes_read=0,this_review_score_output_bytes_read=0,this_review_restore_or_bind_or_hint_or_C2_operation_executed=False,
        ROOT_acceptance_created=False,tests_run=False,metadata_sources=H.reads,
        source=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    H.persist(OUT/'receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],receipt_sha256=hashlib.sha256((OUT/'receipt.json').read_bytes()).hexdigest(),current_lock_records=len(locks)),indent=2))

if __name__=='__main__':main()
