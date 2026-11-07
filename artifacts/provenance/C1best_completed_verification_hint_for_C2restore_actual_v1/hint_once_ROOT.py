#!/usr/bin/env python3
"""Prepared fixed eight-file cache advice; inert without exact ROOT authority."""
import argparse
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import time

REQUIRED=48698310656
SCOPE='one_bounded_eight_C1best_derived_verification_cache_hint_for_C2_native_restore'
BASE=Path('/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/C1B_best')

def require(ok,why):
    if not ok:raise ValueError(why)

def signature(s):
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mode=oct(stat.S_IMODE(s.st_mode)),uid=s.st_uid,
        gid=s.st_gid,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)

def target_stat(p):
    require(p.resolve()==p and not p.is_symlink() and not p.parent.is_symlink(),'Noncanonical/symlink target')
    s=p.lstat();require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_uid==0,'Derived output owner/type/link differs')
    return signature(s)

def small(proof):
    p=Path(proof['path']);require(p.suffix in {'.json','.py'} and p.stat().st_size<=16<<20,'Only explicit small proof/source reads')
    before=signature(p.stat());b=p.read_bytes()
    require(signature(p.stat())==before and hashlib.sha256(b).hexdigest()==proof['sha256'],'Small metadata/source changed')
    return b

def memory():
    p=Path('/sys/fs/cgroup');s={k:int(v) for k,v in (x.split() for x in (p/'memory.stat').read_text().splitlines())}
    maximum=int((p/'memory.max').read_text());current=int((p/'memory.current').read_text())
    clean=max(0,min(s['inactive_file'],s['file']-s['shmem'])-s['file_dirty']-s['file_writeback']-s['unevictable'])
    return dict(at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),maximum=maximum,current=current,stat=s,
        clean_inactive_credit=clean,conservative_available_bytes=max(0,maximum-current)+clean,
        events={k:int(v) for k,v in (x.split() for x in (p/'memory.events').read_text().splitlines())})

def references(items,allowed_self_fds=()):
    ids={(x['before']['device'],x['before']['inode']) for x in items}
    mids={(os.major(d),os.minor(d),i) for d,i in ids};fds=[];maps=[];unreadable=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        pid=int(process.name)
        try:
            for f in (process/'fd').iterdir():
                try:s=f.stat()
                except OSError as e:
                    if e.errno not in (errno.ENOENT,errno.ESRCH):unreadable.append(dict(pid=pid,kind='fdstat',errno=e.errno))
                    continue
                if (s.st_dev,s.st_ino) in ids and not(pid==os.getpid() and int(f.name) in allowed_self_fds):
                    fds.append(dict(pid=pid,fd=f.name,device=s.st_dev,inode=s.st_ino))
        except OSError as e:
            if e.errno not in (errno.ENOENT,errno.ESRCH):unreadable.append(dict(pid=pid,kind='fds',errno=e.errno))
        try:
            for line in (process/'maps').read_text().splitlines():
                fields=line.split(maxsplit=5);ma,mi=(int(z,16) for z in fields[3].split(':'))
                if (ma,mi,int(fields[4])) in mids:maps.append(dict(pid=pid,mapping_fields=fields[:5]))
        except OSError as e:
            if e.errno not in (errno.ENOENT,errno.ESRCH):unreadable.append(dict(pid=pid,kind='maps',errno=e.errno))
    require(not fds and not maps and not unreadable,'Active FD/maps or unreadable process metadata: '+repr((fds,maps,unreadable)))
    return dict(other_FD_references=fds,mappings=maps,unreadable_process_metadata=unreadable)

def validate_items(plan):
    require(len(plan['items'])==8 and plan['required_memory_bytes']==REQUIRED,'Wrong fixed scope')
    for p in plan['source_proofs']:small(p)
    items=[]
    for x in plan['items']:
        r=json.loads(small(x['record']));de=r['decode'];v=r['bytewise_verification'];fp=r['model_fingerprint']
        require(r['package']=='C1B_best' and r['status']=='verified_additional_endpoint_not_yet_assembled' and
                r['additional_encoding_calls']==1 and r['exporter_sha256']==plan['exporter_sha256'] and
                r['prepared_scope_sha256']==plan['prepared_scope_sha256'],'Source record is not completed frozen C1 best')
        require(de['status']=='decoded_and_hash_verified' and de['model_only_evaluation_format'] is True and
                de['tensor_raw_hashes_verified']==v['tensor_count']==196 and v['status']=='passed' and v['all_tensor_bytes_equal'] is True and
                de['model_fingerprint']==v['model_fingerprint']==r['encode']['model_fingerprint']==fp==x['model_fingerprint'],'Full roundtrip evidence mismatch')
        require(de['restored_file_sha256']==x['historical_restored_file_sha256'] and de['restored_bytes']==x['before']['bytes'] and
                v['source_checkpoint_sha256']==r['encode']['source_checkpoint_sha256']==r['source_checkpoint_sha256'] and
                v['archive_sha256']==r['encode']['archive_sha256']==r['archive_sha256'],'Source CP/ZIP/restored SHA provenance differs')
        path=Path(de['restored'])
        require(path==Path(x['path']) and path.parent==BASE/'verification' and path.name.endswith('.restored.pt') and
                Path(x['record']['path']).parent==BASE/'records','Only eight fixed derived verification paths accepted')
        require(target_stat(path)==x['before'] and path.stat().st_mtime_ns<=Path(x['record']['path']).stat().st_mtime_ns,'Current derived identity changed')
        for name in ('source_checkpoint','archive_path','metadata_path'):
            st=Path(r[name]).stat();require((st.st_dev,st.st_ino)!=(x['before']['device'],x['before']['inode']),'Target aliases an original payload')
        items.append(dict(x))
    require(len({(x['before']['device'],x['before']['inode']) for x in items})==8,'Repeated target inode')
    total=sum(x['before']['bytes'] for x in items)
    require(total==plan['total_advised_range_bytes'] and total<=32<<30,'Range exceeds fixed total/32GiB')
    return items

def write_new(path,value):
    path=Path(path);require(not path.exists() and path.parent.is_dir(),'Receipt must be new in an existing directory')
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temp.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    try:os.link(temp,path)
    finally:temp.unlink()
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true',required=True)
    for n in ('plan','plan-sha256','authority','authority-sha256','receipt'):ap.add_argument('--'+n,required=True)
    args=ap.parse_args();start=time.monotonic();pc=time.process_time()
    pp=dict(path=str(Path(args.plan).resolve()),sha256=args.plan_sha256)
    plan=json.loads(small(pp));authority=json.loads(small(dict(path=args.authority,sha256=args.authority_sha256)))
    source=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    require(authority.get('authorized') is True and authority.get('authorized_by')=='ROOT' and authority.get('scope')==SCOPE and
            authority.get('plan')==pp and authority.get('source')==source,'Exact ROOT authority required; preparation is inert')
    require(plan['status']=='prepared_not_execution_authority' and plan['required_memory_bytes']==REQUIRED,'Bad prepared plan')
    receipt=Path(args.receipt)
    require(receipt.parent.is_dir() and not receipt.exists() and not receipt.is_symlink(),'New receipt path required before any hint')
    result=dict(scope=SCOPE,plan=pp,source=source,required_memory_bytes=REQUIRED,operation_executed=False,advice_attempts=0,completed_advice_count=0,
        payload_bytes_read=0,source_bytes_deleted_or_modified=False,global_drop_caches=False,
        original_CP_base_ZIP_TAR_opened=False,advice_is_not_guaranteed_reclaim=True,exact_reclaimed_bytes=None,
        timer_scope='Operation body only, including guards and advice; ROOT must separately measure full child spawn-to-exit')
    opened=[]
    try:
        # Critically conditional: sufficient memory succeeds by SKIPPING, never asserts failure then falls through.
        result['memory_at_entry']=memory()
        if result['memory_at_entry']['conservative_available_bytes']>=REQUIRED:
            result['status']='skipped_memory_already_sufficient_no_target_FD_opened'
        else:
            items=validate_items(plan);result['items']=items;result['references_before_open']=references(items)
            result['memory_immediately_before_open']=memory()
            if result['memory_immediately_before_open']['conservative_available_bytes']>=REQUIRED:
                result['status']='skipped_memory_recovered_during_metadata_checks_no_target_FD_opened'
            else:
                require(result['memory_immediately_before_open']['events']['oom']==0 and result['memory_immediately_before_open']['events']['oom_kill']==0,'OOM event gate')
                for x in items:
                    p=Path(x['path']);require(target_stat(p)==x['before'],'Target changed before open')
                    fd=os.open(p,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW);opened.append((fd,x))
                    fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB)
                    require(signature(os.fstat(fd))==x['before'],'FD/path stat differs')
                result['references_with_only_own_readonly_FDs']=references(items,{fd for fd,x in opened})
                result['memory_after_locks']=memory()
                if result['memory_after_locks']['conservative_available_bytes']>=REQUIRED:
                    result['status']='skipped_memory_recovered_after_readonly_locks_no_hint'
                else:
                    for fd,x in opened:
                        require(signature(os.fstat(fd))==target_stat(Path(x['path']))==x['before'],'Same FD/path changed')
                        t=time.monotonic();os.fsync(fd);x['readonly_fd_fsync_region_seconds']=time.monotonic()-t
                        result['operation_executed']=True;result['advice_attempts']+=1
                        t=time.monotonic();os.posix_fadvise(fd,0,x['before']['bytes'],os.POSIX_FADV_DONTNEED)
                        result['completed_advice_count']+=1
                        x['advice_region_seconds']=time.monotonic()-t;x['after']=signature(os.fstat(fd))
                        require(x['after']==target_stat(Path(x['path']))==x['before'],'Bytes identity/stat changed across advice')
                        small(x['record'])
                    result.update(status='passed_bounded_completed_derived_verification_cache_hint',operation_executed=True,
                        total_advised_range_bytes=plan['total_advised_range_bytes'],maximum_range_bytes=32<<30,
                        per_file_shared_lock_is_advisory_not_producer_lease=True,producer_global_locks_not_acquired_or_modified=True)
    except Exception as e:
        result.update(status='failed_guards_or_operation',error=repr(e));raise
    finally:
        for fd,x in reversed(opened):os.close(fd)
        result['memory_after']=memory();result['operation_body_wall_seconds']=time.monotonic()-start
        result['operation_process_cpu_seconds']=time.process_time()-pc
        write_new(args.receipt,result)
    print(json.dumps(dict(status=result['status'],operation_executed=result['operation_executed'],
        available_after=result['memory_after']['conservative_available_bytes'])))

if __name__=='__main__':main()
