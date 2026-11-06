from pathlib import Path
import os,stat,json,hashlib,time,datetime as dt,fcntl,errno
root=Path.cwd();out=Path(__file__).parent
base=Path('/tmp/grad-research-20261106/delivery')
records=[base/'additional_formal_endpoints_v4/C2p8B_best/records'/name for name in ['extended_v2_pythia_2p8b_m8_h25m_formal_B1_s11_w0p05__best.json','extended_v2_pythia_2p8b_m8_h25m_formal_B2_s11_w0p05__best.json']]+[base/'extended_C_final_v1/pythia-2.8b/records/extended_v2_pythia_2p8b_m8_h25m_formal_M_s33_w0p05.json']
producer_sources={str(root/'scripts/export_additional_endpoints.py'):'c93e0c1d35fe39b3d71155a126db841b76648ba6b3afd58d475be4eb9c9da853',str(root/'scripts/export_C_final_weights.py'):'bc057d04298b91a8189dae58fb6189c19b69faf70c9c83298d7854bf1cc38390'}
for p,h in producer_sources.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
def signature(s):return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mode=oct(stat.S_IMODE(s.st_mode)),uid=s.st_uid,gid=s.st_gid,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def memory():
 c=Path('/sys/fs/cgroup');s={k:int(v) for k,v in (x.split() for x in (c/'memory.stat').read_text().splitlines())};mx=int((c/'memory.max').read_text());cur=int((c/'memory.current').read_text());credit=max(0,min(s['inactive_file'],s['file']-s['shmem'])-s['file_dirty']-s['file_writeback']-s['unevictable']);return dict(observed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),memory_max=mx,memory_current=cur,**{k:s[k] for k in ['anon','file','shmem','active_file','inactive_file','file_dirty','file_writeback','unevictable']},conservative_clean_inactive_credit=credit,conservative_available=max(0,mx-cur)+credit,memory_events={k:int(v) for k,v in (x.split() for x in (c/'memory.events').read_text().splitlines())})
items=[]
for rp in records:
 raw=rp.read_bytes();r=json.loads(raw);de=r['decode'];v=r['bytewise_verification'];assert r['status'] in ['verified_draft_not_deliverable','verified_additional_endpoint_not_yet_assembled'] and r['additional_encoding_calls']==1
 assert de['status']=='decoded_and_hash_verified' and de['model_only_evaluation_format'] is True and de['tensor_raw_hashes_verified']==388
 assert v['status']=='passed' and v['all_tensor_bytes_equal'] is True and v['model_fingerprint']==de['model_fingerprint']==r['model_fingerprint']==r['encode']['model_fingerprint']
 p=Path(de['restored']);assert p.resolve()==p and not p.is_symlink() and not p.parent.is_symlink() and p.parent==rp.parent.parent/'verification' and p.name.endswith('.restored.pt')
 st=p.stat();before=signature(st);assert stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_uid==0 and st.st_size==de['restored_bytes'] and st.st_mtime_ns<=rp.stat().st_mtime_ns
 for original in [r['source_checkpoint'],r['archive_path'],r['metadata_path']]:
  other=Path(original).stat();assert (other.st_dev,other.st_ino)!=(st.st_dev,st.st_ino)
 copied=out/('record_'+rp.name);copied.write_bytes(raw)
 items.append(dict(record_source=str(rp),record_sha256=hashlib.sha256(raw).hexdigest(),record_copy=str(copied),path=str(p),before=before,historical_restored_file_sha256=de['restored_file_sha256'],model_fingerprint=de['model_fingerprint']))
assert len({(x['before']['device'],x['before']['inode']) for x in items})==3
total=sum(x['before']['bytes'] for x in items);assert total==33303037786 and total<=48<<30
identities={(x['before']['device'],x['before']['inode']) for x in items};map_ids={(os.major(d),os.minor(d),i) for d,i in identities};open_refs=[];map_refs=[];unreadable=[]
for process in Path('/proc').iterdir():
 if not process.name.isdigit():continue
 try:
  for fd in (process/'fd').iterdir():
   try:s=fd.stat()
   except OSError:continue
   if (s.st_dev,s.st_ino) in identities:open_refs.append(dict(pid=int(process.name),fd=fd.name,device=s.st_dev,inode=s.st_ino))
 except OSError as e:
  if e.errno not in [errno.ENOENT,errno.ESRCH]:unreadable.append(dict(pid=int(process.name),kind='fd',errno=e.errno))
 try:
  for line in (process/'maps').read_text().splitlines():
   x=line.split(maxsplit=5);major,minor=(int(a,16) for a in x[3].split(':'))
   if (major,minor,int(x[4])) in map_ids:map_refs.append(dict(pid=int(process.name),mapping_fields=x[:5]))
 except OSError as e:
  if e.errno not in [errno.ENOENT,errno.ESRCH]:unreadable.append(dict(pid=int(process.name),kind='maps',errno=e.errno))
assert not open_refs and not map_refs and not unreadable,(open_refs,map_refs,unreadable)
pre=memory();assert pre['memory_events']['oom']==0 and pre['memory_events']['oom_kill']==0
for x in items:
 p=Path(x['path']);assert signature(p.stat())==x['before'];fd=os.open(p,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
 try:
  fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB);assert signature(os.fstat(fd))==x['before'];t=time.perf_counter();os.fsync(fd);x['readonly_fd_fsync_region_seconds']=time.perf_counter()-t;t=time.perf_counter();os.posix_fadvise(fd,0,x['before']['bytes'],os.POSIX_FADV_DONTNEED);x['advice_region_seconds']=time.perf_counter()-t;x['after']=signature(os.fstat(fd));assert x['after']==x['before']==signature(p.stat())
 finally:os.close(fd)
 assert hashlib.sha256(Path(x['record_source']).read_bytes()).hexdigest()==x['record_sha256']
post=memory();r=dict(status='passed_bounded_completed_derived_verification_file_cache_hint',at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),items=items,producer_sources=producer_sources,total_advised_range_bytes=total,maximum_range_bytes=48<<30,no_existing_open_descriptors_observed=open_refs,no_existing_mappings_observed=map_refs,unreadable_process_metadata=unreadable,readonly_FD_fsync_before_advice=True,per_file_shared_flock_is_only_advisory_not_a_producer_lease=True,original_daemon_global_lock_not_acquired_or_changed=True,completed_producer_resume_does_not_reopen_restored_outputs=True,operation='os.posix_fadvise POSIX_FADV_DONTNEED offset0 exact completed output length',cgroup_before=pre,cgroup_after=post,model_payload_read_bytes=0,original_training_CP_base_ZIP_TAR_touched=False,source_bytes_deleted_or_modified=False,global_drop_caches=False,reclaimed_bytes_measured=None,source_identity_caveat='Historical complete decode/bytewise proof and current stat/size bind the selected derived paths; no current full payload SHA re-read. Hint changes cache only. Advisory locks cannot prevent unrelated writers.',scope_note='Three fixed derived CPU verification outputs only; all original checkpoints and files retained. Concurrent cgroup changes prevent attributing exact reclaimed bytes. No batch authority.')
with (out/'receipt.json').open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'status':r['status'],'range_bytes':total,'before_available_GiB':pre['conservative_available']/(1<<30),'after_available_GiB':post['conservative_available']/(1<<30),'all_target_stats_unchanged':True,'payload_read_bytes':0}))
