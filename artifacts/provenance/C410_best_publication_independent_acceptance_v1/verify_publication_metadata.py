#!/usr/bin/env python3
"""Bounded small metadata + stat-only independent publication verification."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
ACTUAL=ROOT/'artifacts/provenance/C410_best_publication_actual_v1'
SHA='0e05b8b3d7f8088220b6a1cc9b8daf9d44b30eb758014d8732390361a22424a0'
NBYTES=26490163200
evidence={};cache={};bytes_read=0
def small(path,expected=None,parse=True):
    global bytes_read
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    s=path.lstat();assert stat.S_ISREG(s.st_mode) and not path.is_symlink()
    assert path.suffix in ('.json','.jsonl','.py','.md','.log') and s.st_size<32<<20
    key=str(path.resolve())
    if key not in cache:cache[key]=path.read_bytes();bytes_read+=len(cache[key])
    raw=cache[key];digest=hashlib.sha256(raw).hexdigest();after=path.lstat()
    sig=lambda t:(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns)
    assert sig(s)==sig(after) and (expected is None or digest==expected)
    evidence[key]=dict(path=key,sha256=digest,bytes=len(raw))
    return json.loads(raw) if parse else raw
def stamp(path):
    s=Path(path).lstat();assert stat.S_ISREG(s.st_mode) and not Path(path).is_symlink()
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def public_stat(path):
    s=Path(path).lstat();assert stat.S_ISREG(s.st_mode) and not Path(path).is_symlink()
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,allocated_bytes=s.st_blocks*512,
        uid=s.st_uid,gid=s.st_gid,mode=oct(stat.S_IMODE(s.st_mode)),links=s.st_nlink,
        mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)

rp=ROOT/'docs/C410_best_weights_durable_publication.json'
receipt=small(rp,'139e54cfcbae7763a1b651abd4b54c2646dbc1af99dff26787cce0b28107091a')
authority=small(ROOT/'docs/C410_best_weights_ROOT_publication_authority.json','01c2119a873a26984b4e75cee930368275d7e49ae9f874f82e1c273d99968b75')
accept=small(ACTUAL/'ROOT_acceptance.json','ce7c7905d5f7a424eb3832bd83080f17b906b5b45fddd685170338ef3623e3a8')
assert accept['publication_receipt']==receipt
for key in ('publication','metadata_review','root_authority','completion','manifest','publication_stdout',
            'publication_stderr','publication_progress','outer_timing_receipt_failure'):
    ref=accept[key];small(ref['path'],ref['sha256'],parse=key not in ('publication_stdout','publication_stderr'))
assert receipt['status']=='passed' and receipt['bytes']==NBYTES
assert receipt['source_sha256']==receipt['destination_sha256']==SHA
assert all(receipt[k] is True for k in ('original_inode_preserved','permissions_and_ownership_preserved',
    'source_retained','file_fsync_after_copy','prefix_sha_verified_before_truncate','file_fsync_after_truncate',
    'atomic_destination_commit_noreplace','parent_directory_fsync'))
assert receipt['overwrite'] is False and receipt['gpu_used'] is False
assert receipt['publication_method']=='atomic_hard_link_noreplace_fsync_then_unlink'
assert [e['event'] for e in receipt['durability_events']]==[
    'source_sha_and_stable_stat_verified','full_prefix_write_fsync_completed',
    'entire_valid_prefix_sha_verified_before_truncate','truncate_to_verified_length_and_file_fsync_completed',
    'parent_directory_fsync_completed']
for event in receipt['durability_events']:
    if 'sha256' in event:assert event['sha256']==SHA
    if 'bytes' in event:assert event['bytes']==NBYTES
progress=small(ROOT/'artifacts/provenance/C410_best_weights_publication_progress.json')
assert progress['status']=='passed' and progress['stage']=='complete' and progress['published'] is True
for k in ('source','destination','wall_seconds','started_at_utc'):assert progress[k]==receipt[k]
assert progress['expected_sha256']==SHA and progress['bytes_written']==progress['expected_bytes']==NBYTES
assert json.loads(small(ACTUAL/'stdout.log',parse=False))==receipt
assert small(ACTUAL/'stderr.log',parse=False)==b''
small(ROOT/'scripts/publish_reserved_archive.py','9c194743383a16f98313f767fe2e722c309ec57c1aafc65d9b99d08e5407022e',parse=False)
assert receipt['publisher_source_sha256']=='9c194743383a16f98313f767fe2e722c309ec57c1aafc65d9b99d08e5407022e'
assert small(ACTUAL/'launch.json')['argv']==authority['argv']
assert authority['authorized_by']=='root' and authority['allow_publication'] is True
assert authority['source']==receipt['source'] and authority['destination']==receipt['destination']
assert authority['expected_sha256']==SHA and authority['expected_bytes']==NBYTES
for k in ('metadata_review','reviewer','completion','manifest','whole_batch_timing','publisher','current_master_state'):
    ref=authority[k];small(ref['path'],ref['sha256'],parse=k not in ('reviewer','publisher'))
meta=small(authority['metadata_review']['path']);manifest=small(authority['manifest']['path']);completion=small(authority['completion']['path'])
assert meta['logical_endpoints']==completion['logical_endpoints']==manifest['logical_endpoints']==24
assert meta['unique_weights']==completion['unique_weights']==manifest['unique_weights']==21
assert meta['alias_count']==completion['alias_count']==manifest['aliases']==3
assert completion['verified_tar_members']==len(meta['derived_safe_TAR_members'])==155
assert completion['tar_sha256']==meta['expected_sha256']==SHA and completion['tar_bytes']==NBYTES
assert manifest['package']=='C410_best' and manifest['kind']=='best' and manifest['optimizer_included'] is False
assert len(manifest['base_dependency']['files'])==6
checker=Path(authority['reviewer']['path']);spec=importlib.util.spec_from_file_location('indie_C410_best_inventory',checker)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert module.member_inventory(manifest)==meta['derived_safe_TAR_members']

fresh_text=0;sidecar_stats=0;historical=[]
for path,record in meta['metadata_sources'].items():
    p=Path(path)
    if p.resolve()==(ROOT/'logs/supplemental_job_state.json').resolve():historical.append(path);continue
    if p.name.endswith('.source_metadata.pt'):
        assert stamp(p)==record['stat'];sidecar_stats+=1;continue
    small(p,record['sha256'],parse=False);fresh_text+=1
assert len(historical)==1 and fresh_text==135 and sidecar_stats==24
for path,expected in meta['payload_stats_only'].items():assert stamp(path)==expected
snapshot=small(authority['current_master_state']['path']);rows={r['run_id']:r for r in manifest['logical_runs']}
selected=authority['selected_24_completed_scheduler_records_rechecked_exactly']
assert set(rows)==set(selected) and len(rows)==24
for rid,row in rows.items():
    state=selected[rid];assert state==snapshot['jobs'][rid]==row['completed_scheduler_record']
    assert state['status']=='completed' and type(state['returncode']) is int and state['returncode']==0
    assert not any(k.startswith('stop_requested') for k in state) and not state.get('missing_outputs')
    assert state['argv']==row['definition']['argv'] and state['definition_sha256']==row['definition_sha256']
    assert hashlib.sha256(json.dumps(row['definition'],sort_keys=True).encode()).hexdigest()==row['definition_sha256']
destination=public_stat(receipt['destination']);source=public_stat(receipt['source'])
for k,v in receipt['destination_stat'].items():assert destination[k]==v
for k,v in receipt['source_stable_stat'].items():assert source[k]==v
assert destination['device']==106 and destination['inode']==467366543 and destination['uid']==destination['gid']==11471
assert destination['mode']=='0o600' and destination['links']==1 and destination['bytes']==destination['allocated_bytes']==NBYTES
assert not os.path.lexists(receipt['original_reservation'])
assert receipt['original_reservation_stat']['inode']==destination['inode']
adjacent_path=Path(receipt['destination']+'.receipt.json')
adjacent=small(adjacent_path,'afcde1091d51b1320ddf8421f28425af22bd85247b196fca1bf72aac9e878f44')
assert adjacent['ROOT_acceptance']==accept and adjacent['publication']==receipt
assert adjacent['archive_bytes']==NBYTES and adjacent['archive_sha256']==SHA
assert public_stat(adjacent_path)['uid']==public_stat(adjacent_path)['gid']==11471
assert public_stat(adjacent_path)['mode']=='0o600'
failure=small(ACTUAL/'outer_timing_receipt_failure.json')
assert failure['outer_script_returncode_observed']==1 and failure['publication_rerun'] is False
br=failure['retained_misplaced_batch_timing'];batch=small(br['path'],br['sha256'])
small(failure['original_batch_timing']['path'],failure['original_batch_timing']['sha256'])
assert br['sha256']==failure['original_batch_timing']['sha256']=='c24ba9a50917cf88478bdccd71c79feb2f0541fe4a2572afa155b68c99f74d5a'
assert batch['publication_performed'] is False and batch['wall_seconds']==1953.1042039617896
assert not (ACTUAL/'whole_child_timing.json').exists()
for k in ('publisher_returncode_not_persisted','exact_publisher_spawn_to_exit_wall_seconds_not_persisted',
          'publisher_whole_user_CPU_seconds_not_persisted','publisher_whole_system_CPU_seconds_not_persisted'):
    assert failure[k] is None
assert receipt['wall_seconds']==195.2169531546533
small(__file__,parse=False);doc=ROOT/'docs/C410best持久发布_独立元数据验收.md';small(doc,parse=False)
result=dict(status='accepted_durable_C410_best_metadata_stat_with_outer_timing_failure_retained',
    checked_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),publication=evidence[str(rp.resolve())],
    ROOT_acceptance=evidence[str((ACTUAL/'ROOT_acceptance.json').resolve())],
    adjacent_receipt=evidence[str(adjacent_path.resolve())],destination=receipt['destination'],
    historical_publisher_verified_TAR_sha256=SHA,bytes=NBYTES,current_destination_stat=destination,current_source_stat=source,
    counts=dict(logical_best_endpoints=24,unique_encoded_weights=21,proved_aliases=3,TAR_members=155,native_base_files=6),
    independently_rechecked=dict(fresh_text_source_SHAs=fresh_text,sidecar_current_stat_only=sidecar_stats,
        stat_observations=len(meta['payload_stats_only']),historical_live_state_not_rehashed=historical,
        ROOT_prepublication_snapshot_exact_completed_records=24,receipt_stdout_progress_and_durable_stat=True),
    accounting=dict(publisher_internal_region_seconds=195.2169531546533,
        region_excludes_receipt_progress_FDclose_stdout=True,exact_external_whole_wall_seconds=None,
        publisher_whole_user_CPU_seconds=None,publisher_whole_sys_CPU_seconds=None,publisher_child_returncode=None,
        outer_wrapper_observed_returncode=1,retained_batch_seconds_not_publication=1953.1042039617896,
        batch_copy_is_zero_new_operation=True),
    boundaries=dict(actual_independent_restore_from_durable_best=False,optimizer_included=False,
        TAR_header_CP_ZIP_base_sidecar_or_score_payload_bytes_read=0,publication_rerun=False,live_source_changes=False),
    document=evidence[str(doc.resolve())],source_evidence=list(evidence.values()),small_serialized_bytes_read=bytes_read)
with (HERE/'REVIEW.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
review_sha=hashlib.sha256((HERE/'REVIEW.json').read_bytes()).hexdigest()
complete=dict(status=result['status'],review=dict(path=str(HERE/'REVIEW.json'),sha256=review_sha),
    publication=result['publication'],ROOT_acceptance=result['ROOT_acceptance'],adjacent_receipt=result['adjacent_receipt'],
    destination=receipt['destination'],bytes=NBYTES,historical_publisher_verified_TAR_sha256=SHA,
    exact_external_publication_whole_wall_CPU_and_child_rc_unknown=True,independent_large_payload_bytes_read=0)
with (HERE/'COMPLETE.json').open('x') as f:json.dump(complete,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(COMPLETE_sha256=hashlib.sha256((HERE/'COMPLETE.json').read_bytes()).hexdigest(),REVIEW_sha256=review_sha,
    fresh_text_source_SHAs=fresh_text,sidecar_stat_only=sidecar_stats,stat_observations=len(meta['payload_stats_only']),
    small_serialized_bytes_read=bytes_read,large_payload_bytes_read=0)))
