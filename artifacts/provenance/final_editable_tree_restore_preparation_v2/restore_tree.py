#!/usr/bin/env python3
"""ROOT-bound candidate tree restore; atomic marker, never a directory rename."""
import argparse, fcntl, gzip, hashlib, json, os, re, stat, sys, tarfile, time, traceback
from pathlib import Path, PurePosixPath

TARGET_UID = TARGET_GID = 11471
BLOCK = 1 << 20
PUBLICATION_MODE = 'verified_tree_with_atomic_completion_marker'

def require(ok, message):
    if not ok: raise ValueError(message)

def signature(s):
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)

def details(s):
    return {**signature(s),'allocated_bytes':s.st_blocks*512,'uid':s.st_uid,'gid':s.st_gid,'mode':oct(stat.S_IMODE(s.st_mode))}

def digest(value): return hashlib.sha256(value).hexdigest()

def absolute(path):
    p=Path(path); require(p.is_absolute() and '..' not in p.parts,'explicit absolute path required')
    return Path(os.path.abspath(p))

def safe_relative(name):
    p=PurePosixPath(name)
    require(bool(p.parts) and not p.is_absolute() and '..' not in p.parts and '\\' not in name
            and '\x00' not in name and str(p)==name,'unsafe/noncanonical member path')
    return p

def real_directory(path):
    p=absolute(path)
    for item in [Path('/'),*reversed([p,*p.parents][:-1])]:
        require(stat.S_ISDIR(item.lstat().st_mode),'parent chain must contain real directories only')
    return p

def absent(path):
    try: Path(path).lstat()
    except FileNotFoundError: return
    raise FileExistsError(str(path))

def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try: os.fsync(fd)
    finally: os.close(fd)

def write_json(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    fsync_dir(Path(path).parent)

def chown_file_sync(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try: os.fchmod(fd,0o600);os.fchown(fd,TARGET_UID,TARGET_GID);os.fsync(fd)
    finally: os.close(fd)

def bound_json(source):
    path=absolute(source['path']);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    with os.fdopen(fd,'rb') as f:
        before=os.fstat(f.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size<=32<<20,'bounded regular JSON required')
        raw=f.read()
        require(signature(before)==signature(os.fstat(f.fileno()))==signature(path.lstat()),'JSON changed during read')
    require(digest(raw)==source['sha256'],'bound JSON SHA mismatch')
    return json.loads(raw)

def link_marker_noreplace(source,destination):
    os.link(source,destination,follow_symlinks=False)

def probe_marker_commit(namespace):
    a,b=namespace/'.commit_probe_a',namespace/'.commit_probe_b'
    with a.open('xb') as stream:
        stream.write(b'private marker capability probe\n');stream.flush();os.fsync(stream.fileno())
    link_marker_noreplace(a,b)
    require(a.stat().st_ino==b.stat().st_ino and a.stat().st_dev==b.stat().st_dev,'probe did not create same-inode hardlink')
    try: link_marker_noreplace(a,b)
    except FileExistsError: pass
    else: raise ValueError('filesystem overwrote probe target')
    a.unlink();b.unlink();fsync_dir(namespace)

def authorize(plan_path,authority_path):
    raw=absolute(plan_path).read_bytes();authority_raw=absolute(authority_path).read_bytes()
    plan=json.loads(raw);auth=json.loads(authority_raw)
    require(auth.get('authorized') is True and auth.get('authorized_by')=='root'
            and auth.get('action')=='restore_final_editable_tree','future explicit ROOT execution authority required')
    require(auth.get('plan_sha256')==digest(raw) and auth.get('tool_sha256')==digest(Path(__file__).read_bytes()),'authority plan/tool SHA mismatch')
    require(plan.get('publication_mode')==auth.get('publication_mode')==PUBLICATION_MODE,'explicit marker publication mode required')
    namespace=absolute(plan['namespace']);parent=real_directory(namespace.parent);absent(namespace)
    archive,reserve=absolute(plan['archive_path']),absolute(plan['reserve_path'])
    require(archive.parent==reserve.parent==parent,'archive/reserve/namespace must share explicit durable parent')
    require(auth.get('namespace')==str(namespace) and auth.get('destination')==str(namespace/'research'),'authority destination mismatch')
    for key in ('archive_path','archive_sha256','reserve_path','reserve_stat'):
        require(auth.get(key)==plan[key],'authority does not explicitly bind '+key)
    require(plan.get('owner_uid')==auth.get('owner_uid')==TARGET_UID and plan.get('owner_gid')==auth.get('owner_gid')==TARGET_GID,'explicit expected owner required')
    protected=absolute(plan['protected_workspace'])
    for a,b in [(namespace,protected),(namespace.resolve(),protected.resolve())]:
        require(a!=b and b not in a.parents and a not in b.parents,'namespace overlaps protected workspace')
    for field in ('publication','ROOT_publication_acceptance','adjacent_receipt','reserve_receipt'):
        require(auth.get(field+'_sha256')==plan[field]['sha256'],'authority receipt SHA mismatch')
    pub,accepted,adjacent,reservation=[bound_json(plan[k]) for k in ('publication','ROOT_publication_acceptance','adjacent_receipt','reserve_receipt')]
    sha=plan['archive_sha256'];require(re.fullmatch('[0-9a-f]{64}',sha) is not None,'archive SHA required')
    require(pub.get('status')=='passed' and pub['destination']==str(archive) and pub['destination_sha256']==pub['source_sha256']==sha,'publisher identity mismatch')
    require(plan['ROOT_acceptance_status']==accepted.get('status')=='accepted_durable_final_snapshot'
            and accepted['archive']==adjacent['archive']==str(archive)
            and accepted['sha256']==adjacent['sha256']==sha,'ROOT/adjacent identity mismatch')
    require(accepted['publication']['sha256']==adjacent['publication_receipt_sha256']==plan['publication']['sha256'],'publication proof crossbinding mismatch')
    require(adjacent.get('verified_all_payloads') is True and type(adjacent['files']) is int
            and type(plan['expected_payload_files']) is int and adjacent['files']==plan['expected_payload_files'],'prior complete verification required')
    for flag in ('file_fsync_after_copy','prefix_sha_verified_before_truncate','file_fsync_after_truncate','atomic_destination_commit_noreplace','parent_directory_fsync'):
        require(pub.get(flag) is True,'publisher durable event missing')
    current=archive.lstat()
    require(stat.S_ISREG(current.st_mode) and signature(current)==plan['archive_stat5'],'archive source identity changed')
    require(signature(current)=={k:pub['destination_stat'][k] for k in signature(current)},'publisher stat binding differs')
    require(current.st_size==pub['bytes']==adjacent['bytes']==accepted['bytes'],'archive byte size mismatch')
    require(reservation.get('status')=='allocated_for_final_editable_tree_only' and reservation['path']==str(reserve),'wrong reserve receipt/purpose')
    expected=plan['reserve_stat']
    for key in ('device','inode','bytes','allocated_bytes','uid','gid','mode'):
        require(expected[key]==reservation[key],'plan changes original reserve identity')
    rs=reserve.lstat()
    require(stat.S_ISREG(rs.st_mode) and details(rs)==expected,'reserve identity changed')
    require(rs.st_uid==TARGET_UID and rs.st_gid==TARGET_GID and stat.S_IMODE(rs.st_mode)==0o600 and rs.st_nlink==1,'wrong private reserve owner/mode/link count')
    require(rs.st_dev==parent.stat().st_dev and (rs.st_dev,rs.st_ino)!=(current.st_dev,current.st_ino),'wrong reserve filesystem/inode')
    require(type(plan['max_total_regular_bytes']) is int and 0<plan['max_total_regular_bytes']<=rs.st_blocks*512,'explicit bounded payload ceiling required')
    plan['_admitted_plan_sha256']=digest(raw);plan['_admitted_authority_sha256']=digest(authority_raw)
    plan['_admitted_tool_sha256']=auth['tool_sha256']
    return plan,namespace,archive,reserve

class HashingReader:
    def __init__(self,stream): self.stream,self.sha,self.bytes=stream,hashlib.sha256(),0
    def read(self,n=-1):
        b=self.stream.read(n);self.sha.update(b);self.bytes+=len(b);return b

def create_member(root_fd,relative,mode):
    parts=safe_relative(relative).parts;directory=os.dup(root_fd)
    try:
        for name in parts[:-1]:
            try: os.mkdir(name,0o700,dir_fd=directory)
            except FileExistsError: pass
            child=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=directory)
            os.close(directory);directory=child
        fd=os.open(parts[-1],os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600,dir_fd=directory)
        os.fchmod(fd,0o600|(mode&0o100));return fd
    finally: os.close(directory)

def extract_and_check(source_fd,archive,expected_stat,staging,plan):
    actual,manifest,manifest_raw,total={},None,None,0
    root_fd=os.open(staging,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        with os.fdopen(os.dup(source_fd),'rb',buffering=0) as raw:
            counted=HashingReader(raw)
            with gzip.GzipFile(fileobj=counted,mode='rb') as compressed:
                with tarfile.open(fileobj=compressed,mode='r|',bufsize=BLOCK,ignore_zeros=True) as tar:
                    for member in tar:
                        require(manifest is None,'PACKAGE_MANIFEST must be last')
                        pp=safe_relative(member.name)
                        require(len(pp.parts)>=2 and pp.parts[0]=='research' and member.isfile(),'one research/ top-level and regular files only')
                        name='/'.join(pp.parts[1:]);require(name not in actual,'duplicate tar member')
                        require(len(actual)<plan['expected_payload_files']+1,'more archive members than authorized')
                        total+=member.size;require(member.size>=0 and total<=plan['max_total_regular_bytes'],'regular payload ceiling exceeded')
                        is_manifest=name=='PACKAGE_MANIFEST.json'
                        if is_manifest: require(member.size<=32<<20,'manifest too large')
                        fd=create_member(root_fd,name,member.mode);hashed,copied,pieces=hashlib.sha256(),0,[]
                        with os.fdopen(fd,'wb') as target:
                            payload=tar.extractfile(member)
                            while True:
                                b=payload.read(BLOCK)
                                if not b: break
                                copied+=len(b);hashed.update(b);target.write(b)
                                if is_manifest: pieces.append(b)
                            require(copied==member.size,'truncated member')
                            target.flush();os.fchown(target.fileno(),TARGET_UID,TARGET_GID);os.fsync(target.fileno())
                        actual[name]={'bytes':copied,'sha256':hashed.hexdigest()}
                        if is_manifest:
                            manifest_raw=b''.join(pieces);manifest=json.loads(manifest_raw)
                while compressed.read(BLOCK): pass
            while counted.read(BLOCK): pass
            require(signature(os.fstat(source_fd))==expected_stat==signature(archive.lstat()),'archive changed during extraction')
            require(counted.bytes==expected_stat['bytes'] and counted.sha.hexdigest()==plan['archive_sha256'],'full compressed extraction-stream SHA mismatch')
    finally: os.close(root_fd)
    require(manifest is not None and not manifest.get('omitted_symlinks'),'missing manifest/omitted symlinks')
    expected={}
    for row in manifest['files']:
        name=str(safe_relative(row['path']));require(name!='PACKAGE_MANIFEST.json' and name not in expected,'duplicate/recursive manifest record')
        require(type(row['bytes']) is int and row['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',row['sha256']),'invalid manifest identity')
        expected[name]={'bytes':row['bytes'],'sha256':row['sha256']}
    payload={n:r for n,r in actual.items() if n!='PACKAGE_MANIFEST.json'}
    require(payload==expected and len(payload)==plan['expected_payload_files'],'manifest names/lengths/SHA mismatch')
    for directory,_,_ in os.walk(staging,topdown=False,followlinks=False):
        os.chmod(directory,0o700);os.chown(directory,TARGET_UID,TARGET_GID);fsync_dir(directory)
    return {'verified_payload_files':len(payload),'payload_regular_bytes':sum(r['bytes'] for r in payload.values()),
            'all_regular_bytes_with_manifest':total,'PACKAGE_MANIFEST_sha256':digest(manifest_raw),
            'compressed_stream_bytes_read':counted.bytes,'compressed_stream_sha256':counted.sha.hexdigest(),
            'all_member_SHA_verified':True,'verified_members':actual,
            'source_stat5_after':signature(archive.lstat())}

def restore(plan_path,authority_path):
    start=time.monotonic();plan,namespace,archive,reserve=authorize(plan_path,authority_path)
    namespace.mkdir(mode=0o700);fsync_dir(namespace.parent)
    evidence=namespace/'restore_evidence';evidence.mkdir(mode=0o700);fsync_dir(namespace)
    candidate=namespace/'research'
    completion,temp=evidence/'completion.json',evidence/'.completion.json.prepared'
    source_fd=reserve_fd=lock_fd=None
    state={'status':'running_candidate_uncommitted','namespace':str(namespace),'candidate':str(candidate),
           'publication_mode':PUBLICATION_MODE,'atomic_directory_publication':False,
           'reserve_consumed':False,'candidate_created':False,'completion_marker_present':False,
           'completion_marker_durability':'not_published','workspace_symlink_modified':False,
           'compressed_stream_bytes_read':None}
    try:
        lock_fd=os.open(namespace/'.restore.lock',os.O_CREAT|os.O_EXCL|os.O_RDWR|os.O_NOFOLLOW,0o600)
        fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source_fd=os.open(archive,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC);fcntl.flock(source_fd,fcntl.LOCK_SH|fcntl.LOCK_NB)
        require(signature(os.fstat(source_fd))==plan['archive_stat5']==signature(archive.lstat()),'opened source identity differs')
        reserve_fd=os.open(reserve,os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC);fcntl.flock(reserve_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(details(os.fstat(reserve_fd))==plan['reserve_stat']==details(reserve.lstat()),'locked reserve identity differs')
        probe_marker_commit(namespace)
        absent(candidate);absent(completion);absent(temp);candidate.mkdir(mode=0o700);state['candidate_created']=True
        write_json(evidence/'admission.json',{'status':'ROOT_metadata_stat_admitted_not_fresh_archive_payload_verified',
            'plan_sha256':plan['_admitted_plan_sha256'],'authority_sha256':plan['_admitted_authority_sha256'],
            'publication_mode':PUBLICATION_MODE,'candidate':str(candidate),
            'source_stat5':plan['archive_stat5'],'original_reserve_stat':plan['reserve_stat'],
            'capacity_is_not_an_atomic_cross_file_guarantee':True})
        require(signature(os.fstat(source_fd))==plan['archive_stat5']==signature(archive.lstat()),'source changed before reserve release')
        require(details(os.fstat(reserve_fd))==plan['reserve_stat']==details(reserve.lstat()),'reserve changed before release')
        os.unlink(reserve);state['reserve_consumed']=True;os.close(reserve_fd);reserve_fd=None;fsync_dir(reserve.parent)
        result=extract_and_check(source_fd,archive,plan['archive_stat5'],candidate,plan)
        state.update(result)
        write_json(evidence/'verified_before_publication.json',result)
        chown_file_sync(namespace/'.restore.lock')
        for item in evidence.iterdir(): chown_file_sync(item)
        os.chown(evidence,TARGET_UID,TARGET_GID);fsync_dir(evidence)
        os.chown(namespace,TARGET_UID,TARGET_GID);fsync_dir(namespace);fsync_dir(namespace.parent)
        marker={**result,'status':'verified_tree_ready_marker','publication_mode':PUBLICATION_MODE,
            'namespace':str(namespace),'candidate':str(candidate),'atomic_directory_publication':False,
            'archive_path':str(archive),'archive_sha256':plan['archive_sha256'],'archive_stat5':plan['archive_stat5'],
            'plan_sha256':plan['_admitted_plan_sha256'],'authority_sha256':plan['_admitted_authority_sha256'],
            'tool_sha256':plan['_admitted_tool_sha256'],
            'input_receipt_sha256':{k:plan[k]['sha256'] for k in ('publication','ROOT_publication_acceptance','adjacent_receipt','reserve_receipt')},
            'all_content_owner_uid_gid':[TARGET_UID,TARGET_GID],
            'tree_and_parent_fsync_before_marker_completed':True,'reserve_consumed':True,
            'original_archive_retained':True,'original_workspace_untouched':True,'workspace_symlink_modified':False,
            'atomic_marker_protocol':'fsync_new_temp_then_hardlink_noreplace_unlink_temp_fsync_parent',
            'completion_requires':'immutable marker + external strict int rc0 + ROOT actual acceptance; directory or marker alone insufficient',
            'post_link_parent_fsync_is_attested_by_external_rc0_not_predicted_in_this_marker':True,
            'wall_seconds_before_marker_publication':time.monotonic()-start}
        write_json(temp,marker);chown_file_sync(temp)
        marker_sha=digest(temp.read_bytes());absent(completion)
        link_marker_noreplace(temp,completion)
        state['completion_marker_present']=True;state['completion_marker_durability']='uncertain_until_parent_fsync'
        temp.unlink();fsync_dir(evidence)
        state.update(status='passed_verified_new_editable_tree_restored',**result,wall_seconds=time.monotonic()-start,
            completion_marker=str(completion),completion_marker_sha256=marker_sha,
            completion_marker_durability='parent_fsynced',prepared_marker_temp_removed=True,
            all_content_owner_uid_gid=[TARGET_UID,TARGET_GID],original_archive_retained=True,
            original_workspace_untouched=True,tree_and_parent_fsync_before_marker_completed=True)
        return state
    except BaseException as error:
        try: completion.lstat()
        except FileNotFoundError: pass
        else:
            state['completion_marker_present']=True
            if state['completion_marker_durability']=='not_published':
                state['completion_marker_durability']='unknown_after_failed_publication'
        state.update(status='failed_candidate_retained_do_not_switch_workspace',error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start,
            instruction='Keep candidate/source/logs; no automatic retry or switch. A present marker after failure requires ROOT review; its durability field is explicit. Released reserve is not recreated.')
        sys.stderr.write(json.dumps(state,ensure_ascii=False)+'\n');sys.stderr.flush()
        try: write_json(evidence/'failure.json',state)
        except OSError as log_error: sys.stderr.write('Failure receipt could not fsync: '+str(log_error)+'\n')
        raise
    finally:
        for fd in (reserve_fd,source_fd,lock_fd):
            if fd is not None: os.close(fd)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--authority',required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    require(args.execute,'preparation only; explicit --execute plus future ROOT authority required')
    print(json.dumps(restore(args.plan,args.authority),ensure_ascii=False))

if __name__=='__main__': main()
