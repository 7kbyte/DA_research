"""Independent small-metadata review; never imports or runs advice code."""
from pathlib import Path
import datetime as dt,hashlib,json,os,stat
R=Path('/tmp/grad-research-20261106/worktree');O=Path(__file__).resolve().parent;reads={}
def sig(p):
 s=Path(p).stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mode=oct(stat.S_IMODE(s.st_mode)),uid=s.st_uid,gid=s.st_gid,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def read(p,h=None):
 p=Path(p);assert p.suffix in {'.json','.py'} and p.stat().st_size<16<<20
 before=sig(p);b=p.read_bytes();s=hashlib.sha256(b).hexdigest();assert sig(p)==before and (h is None or h==s)
 reads[str(p)]=dict(sha256=s,bytes=len(b));return json.loads(b) if p.suffix=='.json' else b

def memory(m):
 s=m['stat'];clean=max(0,min(s['inactive_file'],s['file']-s['shmem'])-s['file_dirty']-s['file_writeback']-s['unevictable'])
 assert clean==m['clean_inactive_credit'] and max(0,m['maximum']-m['current'])+clean==m['conservative_available_bytes']
def persist(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 d=R/'artifacts/provenance/C1best_completed_verification_hint_for_C2restore_actual_v1'
 r=read(d/'receipt.json','6a15ba42b01e3bbdf870c80b87a3ac429f41f59e2df87179265b78a0f7741527')
 w=read(d/'whole_timing/whole_child_timing.json','680a9f49e05a1a4f96e9c1521e8af8da9e287cbce154069cf0ed42eb2bf082f2')
 a=read(d/'ROOT_authority.json','e80f10b7b1c6440c411a8a9957f40cf9aa8465661de39f5d6a6e85da740f1164');p=read(r['plan']['path'],r['plan']['sha256']);launch=read(d/'ROOT_launch.json')
 source=read(r['source']['path'],r['source']['sha256']).decode();old=read(a['original_helper']['path'],a['original_helper']['sha256']).decode()
 for x,y in a['literal_guard_changes'].items():assert old.count(x)==1;old=old.replace(x,y)
 for x,y in [('fixed four-file','fixed eight-file'),('completed frozen C2 best','completed frozen C1 best'),('Only four fixed','Only eight fixed'),('total<=48<<30','total<=32<<30'),('total/48GiB','total/32GiB'),('maximum_range_bytes=48<<30','maximum_range_bytes=32<<30')]:
  assert old.count(x)==1;old=old.replace(x,y)
 assert old==source
 assert a['authorized'] is True and a['authorized_by']=='ROOT' and a['scope']==r['scope'] and a['source']==r['source'] and a['plan']==r['plan']
 assert type(w['returncode']) is int and w['returncode']==0 and launch['argv'][3:]==w['argv']
 assert w['argv']==['python',r['source']['path'],'--execute','--plan',r['plan']['path'],'--plan-sha256',r['plan']['sha256'],'--authority',str(d/'ROOT_authority.json'),'--authority-sha256','e80f10b7b1c6440c411a8a9957f40cf9aa8465661de39f5d6a6e85da740f1164','--receipt',str(d/'receipt.json')]
 assert r['status']=='passed_bounded_completed_derived_verification_cache_hint' and r['operation_executed'] is True
 assert r['advice_attempts']==r['completed_advice_count']==len(r['items'])==len(p['items'])==8
 assert r['total_advised_range_bytes']==p['total_advised_range_bytes']==32377645928<=r['maximum_range_bytes']==32<<30
 assert r['required_memory_bytes']==p['required_memory_bytes']==48698310656
 for v in p['source_proofs']:read(v['path'],v['sha256'])
 completed=[]
 for x,y in zip(r['items'],p['items']):
  assert all(x[k]==v for k,v in y.items()) and x['before']==x['after']==sig(x['path'])
  rec=read(x['record']['path'],x['record']['sha256']);de,v,en=rec['decode'],rec['bytewise_verification'],rec['encode']
  assert rec['package']=='C1B_best' and rec['status']=='verified_additional_endpoint_not_yet_assembled' and rec['kind']=='best' and rec['additional_encoding_calls']==1
  assert rec['prepared_scope_sha256']==p['prepared_scope_sha256'] and rec['exporter_sha256']==p['exporter_sha256']
  assert de['status']=='decoded_and_hash_verified' and v['status']=='passed' and v['all_tensor_bytes_equal'] is True and de['tensor_raw_hashes_verified']==v['tensor_count']==196
  assert de['model_fingerprint']==v['model_fingerprint']==en['model_fingerprint']==rec['model_fingerprint']==x['model_fingerprint']
  assert de['restored_file_sha256']==x['historical_restored_file_sha256'] and de['restored']==x['path'] and de['restored_bytes']==x['before']['bytes']
  assert v['source_checkpoint_sha256']==en['source_checkpoint_sha256']==rec['source_checkpoint_sha256'] and v['archive_sha256']==en['archive_sha256']==rec['archive_sha256']
  assert rec['all_nonweight_metadata_roundtrip_verified'] is True and rec['GPU_used'] is False
  for key in ('source_checkpoint','archive_path','metadata_path'):
   z=sig(rec[key]);assert (z['device'],z['inode'])!=(x['before']['device'],x['before']['inode'])
  completed.append(dict(path=x['path'],record=x['record'],current_full_stat=x['after'],model_fingerprint=x['model_fingerprint'],historical_restored_file_sha256=x['historical_restored_file_sha256']))
 assert len({(x['after']['device'],x['after']['inode']) for x in r['items']})==8
 for key in ('references_before_open','references_with_only_own_readonly_FDs'):assert r[key]==dict(other_FD_references=[],mappings=[],unreadable_process_metadata=[])
 assert r['per_file_shared_lock_is_advisory_not_producer_lease'] and r['producer_global_locks_not_acquired_or_modified']
 for key in ('memory_at_entry','memory_immediately_before_open','memory_after_locks','memory_after'):
  memory(r[key])
  if key!='memory_after':assert r[key]['conservative_available_bytes']<48698310656
 assert r['memory_at_entry']['conservative_available_bytes']==29640605696 and r['memory_after']['conservative_available_bytes']==59898904576
 assert r['payload_bytes_read']==0 and r['source_bytes_deleted_or_modified'] is False and r['global_drop_caches'] is False and r['original_CP_base_ZIP_TAR_opened'] is False and r['exact_reclaimed_bytes'] is None
 td=R/'artifacts/provenance/C2native_completed_scratch_hint_actual_v1'
 t=read(td/'receipt.json','53d2cc3d65c9a4b0728692dad59da4437bc603ec2c7d0c07307b41b995d0d06f')
 tw=read(td/'whole_timing/whole_child_timing.json','ef8671118bea0eba14037589a24b9b0a52181617e70d71fdc6c94d80c54d8805')
 ts=read(td/'ROOT_hint_once.py',t['actual_source_sha256']).decode();read(t['helper_source']['path'],t['helper_source']['sha256'])
 proofs=[read(v['path'],v['sha256']) for v in t['source_proofs']];pub,done,accepted=proofs
 assert pub['source']==done['tar_path']==t['source'] and pub['status']==done['status']=='passed' and accepted['status']=='accepted'
 assert pub['source_retained'] and done['file_fsync_completed'] and done['parent_directory_fsync_completed']
 assert pub['source_sha256']==pub['destination_sha256']==done['tar_sha256']=='7ca63d1fd4aab0ae491df8577ca2f16483d1e92445e70feaff46a342c837d313'
 assert t['source_before']==t['source_after']==sig(t['source'])
 assert all(t['source_before'][k]==v for k,v in pub['source_stable_stat'].items() if k!='allocated_bytes')
 assert t['status']=='passed_one_64GiB_completed_published_scratch_TAR_hint' and t['operation_executed'] and t['offset']==0 and t['maximum_length_bytes']==64<<30
 assert type(tw['returncode']) is int and tw['returncode']==0 and tw['argv']==['python','artifacts/provenance/C2native_completed_scratch_hint_actual_v1/ROOT_hint_once.py']
 for k in ('references_before_open','references_after_open'):assert t[k]==dict(other_FD_references=[],mappings=[],unreadable_process_metadata=[])
 for k in ('memory_before','memory_before_advice','memory_after'):memory(t[k])
 assert t['memory_before']['conservative_available_bytes']<48698310656 and t['memory_before_advice']['conservative_available_bytes']<48698310656
 assert t['payload_bytes_read']==0 and t['bytes_modified_or_deleted'] is False and t['global_drop_caches'] is False and t['original_CP_base_ZIP_or_durable_TAR_opened'] is False and t['exact_reclaimed_bytes'] is None
 # Source inspection additionally confirms sole target open O_RDONLY|O_NOFOLLOW, shared flock, same-FD fsync/advice; no payload reader.
 assert 'os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW' in ts and 'fcntl.LOCK_SH|fcntl.LOCK_NB' in ts and 'os.posix_fadvise(fd,0,64<<30,os.POSIX_FADV_DONTNEED)' in ts
 copies=O/'small_actual_evidence';copies.mkdir()
 for prefix,paths in [('C1best8',[d/'receipt.json',d/'ROOT_authority.json',d/'bound_plan.json',d/'whole_timing/whole_child_timing.json']),('C2scratch',[td/'receipt.json',td/'whole_timing/whole_child_timing.json'])]:
  for q in paths:
   with (copies/(prefix+'_'+q.name)).open('xb') as f:f.write(q.read_bytes())
 result=dict(status='accepted_independent_small_metadata_review_of_two_actual_prehints',utc=dt.datetime.now(dt.timezone.utc).isoformat(),
  C1best_eight_derived=dict(total_advised_bytes=32377645928,completed=completed,whole_child_timing=w,body_seconds_nested=r['operation_body_wall_seconds'],available_before=29640605696,available_after=59898904576,exact_helper_scope_literal_adaptation_verified=True),
  C2_completed_published_scratch_TAR=dict(advised_offset=0,advised_length=64<<30,whole_child_timing=tw,body_seconds_nested=t['operation_region_seconds'],current_full_stat=t['source_after'],available_before=t['memory_before']['conservative_available_bytes'],available_after=t['memory_after']['conservative_available_bytes']),
  exact_reclaimed_bytes=None,limits=['Advice is not guaranteed reclaim or memory reservation','Shared flock and no-FD/maps observations are advisory point-in-time guards, not producer leases','Whole2.491835881s and1.143689170s are independent; each contains all own body/FD/fsync/advice regions','Historical full-file SHA/FP evidence bound by completed records; no payload rehash here','Successful cache hints do not themselves authorize or prove model recovery'],
  this_review_payload_bytes_read=0,this_review_cache_hint_restore_or_bind_executed=False,tests_run=False,metadata_sources=reads,source=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 persist(O/'receipt.json',result);print(json.dumps({'status':result['status'],'sha256':hashlib.sha256((O/'receipt.json').read_bytes()).hexdigest()}))
if __name__=='__main__':main()
