from pathlib import Path
import os,stat,json,hashlib,time,datetime as dt,fcntl
root=Path.cwd();out=Path(__file__).parent
publication=root/'docs/A_remaining_final_weights_durable_publication.json'
completion=Path('/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/A_remaining_final/completion.json')
def proof(p):
 raw=Path(p).read_bytes();return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest())
assert proof(publication)['sha256']=='449fcd5a88991473f4fbd042b633382718450c975b4e4c2811c66a542e7720cb'
assert proof(completion)['sha256']=='0b79a1b870644304a2d671b1f8cc1ed12c8dfb24cd3c86bf52a53baf18a2bb6b'
pub=json.loads(publication.read_text());done=json.loads(completion.read_text());source=Path(pub['source'])
assert str(source)=='/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/A_remaining_final/A_remaining_final_weights_lossless_v1.tar'
assert pub['status']=='passed' and pub['source_retained'] is True and done['file_fsync_completed'] is True and done['parent_directory_fsync_completed'] is True
assert done['tar_path']==str(source) and done['tar_sha256']==pub['source_sha256']==pub['destination_sha256']=='9e818d15cb826f0948f7092188e684382fea0d03605deac765e1ac19e5ea08b1'
def signature(s):return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,allocated_bytes=s.st_blocks*512,mode=oct(stat.S_IMODE(s.st_mode)),uid=s.st_uid,gid=s.st_gid,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
assert not source.is_symlink();before=signature(source.stat());assert before==pub['source_stable_stat'] and (before['device'],before['inode'])==(139,4436072948)
open_refs=[]
for process in Path('/proc').iterdir():
 if not process.name.isdigit():continue
 try:
  for fd in (process/'fd').iterdir():
   try:s=fd.stat()
   except (OSError,PermissionError):continue
   if (s.st_dev,s.st_ino)==(before['device'],before['inode']):open_refs.append(dict(pid=int(process.name),fd=fd.name))
 except (OSError,PermissionError):continue
assert not open_refs,open_refs
def memory():
 c=Path('/sys/fs/cgroup');s={k:int(v) for k,v in (x.split() for x in (c/'memory.stat').read_text().splitlines())};mx=int((c/'memory.max').read_text());cur=int((c/'memory.current').read_text());credit=max(0,min(s['inactive_file'],s['file']-s['shmem'])-s['file_dirty']-s['file_writeback']-s['unevictable']);events={k:int(v) for k,v in (x.split() for x in (c/'memory.events').read_text().splitlines())};return dict(observed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),memory_max=mx,memory_current=cur,**{k:s[k] for k in ['anon','file','shmem','active_file','inactive_file','file_dirty','file_writeback','unevictable']},conservative_clean_inactive_credit=credit,conservative_available=max(0,mx-cur)+credit,memory_events=events)
pre=memory();assert pre['memory_events']['oom']==0 and pre['memory_events']['oom_kill']==0
flags=os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW;fd=os.open(source,flags)
try:
 fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB);assert signature(os.fstat(fd))==before==signature(source.stat());started=time.perf_counter();os.posix_fadvise(fd,0,64<<30,os.POSIX_FADV_DONTNEED);region=time.perf_counter()-started;after=signature(os.fstat(fd));assert after==before==signature(source.stat())
finally:os.close(fd)
post=memory();r=dict(status='single_bounded_completed_owned_scratch_TAR_cache_hint_returned',at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source=str(source),publication=proof(publication),completion=proof(completion),source_before=before,source_after=after,no_existing_open_descriptors_observed=open_refs,source_historical_fsync_and_publication_verified=True,fd_mode='O_RDONLY|O_CLOEXEC|O_NOFOLLOW + shared nonblocking flock',operation='os.posix_fadvise',advice='POSIX_FADV_DONTNEED',offset=0,length_bytes=64<<30,operation_region_wall_seconds=region,cgroup_before=pre,cgroup_after=post,source_payload_read_bytes=0,source_bytes_deleted_or_modified=False,durable_destination_touched=False,CP_base_ZIP_train_files_touched=False,global_drop_caches=False,reclaimed_bytes_measured=None,scope_note='Hint only; observed cgroup changes include concurrent work and are not proof that exactly64GiB was reclaimed. No batch authority; ROOT must recheck fresh resource gate.')
with (out/'receipt.json').open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'status':r['status'],'hint_region_seconds':region,'before_available_GiB':pre['conservative_available']/(1<<30),'after_available_GiB':post['conservative_available']/(1<<30),'source_stat_unchanged':True,'payload_read_bytes':0}))
