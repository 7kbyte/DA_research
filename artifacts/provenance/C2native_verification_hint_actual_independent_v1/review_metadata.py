#!/usr/bin/env python3
"""Read only small, completed hint evidence and current target stats."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT=Path('/tmp/grad-research-20261106/worktree')
IN=ROOT/'artifacts/provenance/C2native_verification_hint_actual_v1'
PREP=ROOT/'artifacts/provenance/C2native_verification_hint_preparation_v1'
OUT=Path(__file__).resolve().parent
source=PREP/'hint_once.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='33120a9e18a64e5084512bc2536d85f5e8d20524591592c57da1c290d49b0501'
spec=importlib.util.spec_from_file_location('inert_fixed_hint_metadata_functions',source)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
reads={}

def read(path,sha=None):
    path=Path(path);assert path.suffix in {'.json','.py'} and path.stat().st_size<16<<20
    before=M.signature(path.stat());b=path.read_bytes();h=hashlib.sha256(b).hexdigest()
    assert M.signature(path.stat())==before and (sha is None or h==sha)
    reads[str(path)]=dict(sha256=h,bytes=len(b));return json.loads(b) if path.suffix=='.json' else b

def save(path,o):
    with path.open('x') as f:json.dump(o,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
    r=read(IN/'receipt.json','48e28f764f69cddb76c57d830e2e5ac5e500a8ca97573b9d012637c0502d0f54')
    w=read(IN/'whole_timing/whole_child_timing.json','bfb3b9f8182c01e8bfd241638f45ec7a9b8d06eb461b1c2f3a6f9addbec8dde4')
    p=read(r['plan']['path'],r['plan']['sha256']);read(r['source']['path'],r['source']['sha256'])
    auth=read(IN/'ROOT_authority.json','0f87004ced5a82c07f73997544661fe972648a1ae51e35831fa11208041553ce')
    launch=read(IN/'ROOT_launch.json')
    assert auth['authorized'] is True and auth['authorized_by']=='ROOT' and auth['scope']==r['scope']==M.SCOPE
    assert auth['plan']==r['plan'] and auth['source']==r['source']
    assert type(w['returncode']) is int and w['returncode']==0
    assert w['argv']==['python',str(source),'--execute','--plan',r['plan']['path'],'--plan-sha256',r['plan']['sha256'],
      '--authority',str(IN/'ROOT_authority.json'),'--authority-sha256','0f87004ced5a82c07f73997544661fe972648a1ae51e35831fa11208041553ce',
      '--receipt',str(IN/'receipt.json')]
    timer_source=ROOT/'artifacts/provenance/final_evaluation_handoff_actual_v1/time_child.py'
    assert launch['argv']==['python',str(timer_source),str(IN/'whole_timing')]+w['argv']
    read(timer_source)
    assert r['status']=='passed_bounded_completed_derived_verification_cache_hint' and r['operation_executed'] is True
    assert r['advice_attempts']==r['completed_advice_count']==4
    assert len(r['items'])==4 and r['total_advised_range_bytes']==p['total_advised_range_bytes']==44404036408<=48<<30
    assert r['maximum_range_bytes']==48<<30 and r['required_memory_bytes']==p['required_memory_bytes']==48698310656
    for q in p['source_proofs']:read(q['path'],q['sha256'])
    independent=[]
    for x,planned in zip(r['items'],p['items']):
        assert all(x[k]==v for k,v in planned.items()) and x['after']==x['before']==M.target_stat(Path(x['path']))
        rec=read(x['record']['path'],x['record']['sha256']);copy=read(x['record_copy']['path'],x['record_copy']['sha256']);assert rec==copy
        d,v,e=rec['decode'],rec['bytewise_verification'],rec['encode']
        assert rec['status']=='verified_additional_endpoint_not_yet_assembled' and rec['package']=='C2p8B_best' and rec['additional_encoding_calls']==1
        assert d['status']=='decoded_and_hash_verified' and v['status']=='passed' and v['all_tensor_bytes_equal'] is True
        assert d['model_fingerprint']==v['model_fingerprint']==e['model_fingerprint']==rec['model_fingerprint']==x['model_fingerprint']
        assert d['tensor_raw_hashes_verified']==v['tensor_count']==388 and d['restored_file_sha256']==x['historical_restored_file_sha256']
        assert e['source_checkpoint_sha256']==v['source_checkpoint_sha256']==rec['source_checkpoint_sha256']==x['source_CP_sha256']
        assert e['archive_sha256']==v['archive_sha256']==rec['archive_sha256']
        assert d['restored']==x['path'] and d['restored_bytes']==x['before']['bytes'] and x['readonly_fd_fsync_region_seconds']>=0 and x['advice_region_seconds']>=0
        independent.append(dict(path=x['path'],record_sha256=x['record']['sha256'],current_full_stat=x['after'],model_fingerprint=x['model_fingerprint']))
    for key in ('references_before_open','references_with_only_own_readonly_FDs'):
        assert r[key]==dict(other_FD_references=[],mappings=[],unreadable_process_metadata=[])
    assert r['per_file_shared_lock_is_advisory_not_producer_lease'] is True and r['producer_global_locks_not_acquired_or_modified'] is True
    for key in ('memory_at_entry','memory_immediately_before_open','memory_after_locks','memory_after'):
        m=r[key];s=m['stat'];credit=max(0,min(s['inactive_file'],s['file']-s['shmem'])-s['file_dirty']-s['file_writeback']-s['unevictable'])
        assert credit==m['clean_inactive_credit'] and max(0,m['maximum']-m['current'])+credit==m['conservative_available_bytes']
        if key!='memory_after':assert m['conservative_available_bytes']<r['required_memory_bytes']
    assert r['memory_at_entry']['conservative_available_bytes']==15515938816 and r['memory_after']['conservative_available_bytes']==54662733824
    assert r['payload_bytes_read']==0 and r['source_bytes_deleted_or_modified'] is False and r['global_drop_caches'] is False and r['original_CP_base_ZIP_TAR_opened'] is False
    assert r['advice_is_not_guaranteed_reclaim'] is True and r['exact_reclaimed_bytes'] is None
    failure=read(ROOT/'artifacts/provenance/C_native_final_CPU_restore_actual_v1/2p8b/ROOT_preexecution_resource_gate_attempt2_failure.json',
      'ee5b75111866668f9293b4f9bba8e113bb1cd710b177d967c317032fd0a86482')
    assert failure['status']=='original_fresh_resource_gate_failed_before_authority_or_launch' and failure['exact_failed_gate_available_bytes'] is None
    assert failure['TRUE_created'] is False and failure['restore_launched'] is False and failure['failed_inline_parent_wall_seconds'] is None
    assert failure['later_observation_not_exact_failed_gate']['conservative_available_bytes']==23809355776
    copies=OUT/'small_actual_evidence';copies.mkdir()
    for src,name in [(IN/'receipt.json','actual_hint_receipt.json'),(IN/'whole_timing/whole_child_timing.json','actual_hint_whole_child_timing.json'),
      (IN/'ROOT_authority.json','actual_hint_ROOT_authority.json'),(ROOT/'artifacts/provenance/C_native_final_CPU_restore_actual_v1/2p8b/ROOT_preexecution_resource_gate_attempt2_failure.json','post_hint_fresh_gate_failure.json')]:
        with (copies/name).open('xb') as f:f.write(src.read_bytes());f.flush();os.fsync(f.fileno())
    result=dict(status='accepted_independent_metadata_review_of_actual_bounded_hint',at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
      actual_receipt=dict(path=str(IN/'receipt.json'),sha256='48e28f764f69cddb76c57d830e2e5ac5e500a8ca97573b9d012637c0502d0f54'),
      actual_parent_timing=w,four_completed_derived_outputs=independent,actual_total_advised_range_bytes=44404036408,
      actual_before_available_bytes=15515938816,actual_after_available_bytes=54662733824,exact_reclaimed_bytes=None,
      all_three_pre_hint_fresh_observations_were_below_unchanged_required_bytes=True,
      scope='Four exact completed derived restored verification outputs only; readonly fsync+DONTNEED, metadata stat unchanged',
      limitations=['Advice is not guaranteed reclaim or memory reservation','Shared locks and no-FD/maps scans are point-in-time advisory checks, not writer leases',
                  'Whole3.770098101347685s includes body3.72962012514472s and per-file regions; never add nested timers',
                  'Post-hint C2 fresh gate failed; exact failed available and failed parent timer unknown. Later23809355776B is not the failed value'],
      post_hint_failed_resource_gate=failure,model_restore_authority_or_launch_implied=False,
      this_review_payload_bytes_read=0,this_review_hint_or_cache_or_restore_operation=False,tests_run=False,
      read_small_metadata_sources=reads,source=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    save(OUT/'receipt.json',result)
    print(json.dumps(dict(status=result['status'],receipt_sha256=hashlib.sha256((OUT/'receipt.json').read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':main()
