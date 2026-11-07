"""ROOT one-shot bounded advice for own completed, published scratch TAR."""
from pathlib import Path
import datetime as dt, fcntl, hashlib, json, os, runpy, time
R = Path(__file__).resolve().parents[3]
O = Path(__file__).resolve().parent
helper = R/'artifacts/provenance/C2native_verification_hint_preparation_v1/hint_once.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest() == '33120a9e18a64e5084512bc2536d85f5e8d20524591592c57da1c290d49b0501'
m = runpy.run_path(str(helper))
sources = [
    (R/'docs/C2p8B_final_weights_durable_publication.json', '5f17f9aaf68663005592be42aa16c68946b85cfa923ceff8f7303a3b53fe050c'),
    (Path('/tmp/grad-research-20261106/delivery/extended_C_final_v1/pythia-2.8b/completion.json'), 'ede86767b3cd03e94c73d2d14eae62c41d815490a02453107ec77d77b135ca19'),
    (R/'artifacts/provenance/C2p8B_final_publication_actual_v1/ROOT_acceptance.json', '8de7ca0da5f62645d9840c6d52d37b16ba293304d713912bb1a25218615a2f28')]
values = [json.loads(m['small'](dict(path=str(p),sha256=h))) for p,h in sources]
pub, done, accepted = values
source = Path('/tmp/grad-research-20261106/delivery/extended_C_final_v1/pythia-2.8b/extended_C_2p8b_final_weights_lossless_v1.tar')
assert pub['source'] == done['tar_path'] == str(source)
assert pub['status'] == done['status'] == 'passed' and accepted['status'] == 'accepted'
assert pub['source_retained'] is True and done['file_fsync_completed'] is True and done['parent_directory_fsync_completed'] is True
assert done['tar_sha256'] == pub['source_sha256'] == pub['destination_sha256'] == '7ca63d1fd4aab0ae491df8577ca2f16483d1e92445e70feaff46a342c837d313'
before = m['target_stat'](source)
assert all(before[k] == v for k,v in pub['source_stable_stat'].items() if k != 'allocated_bytes')
assert (before['device'],before['inode'],before['bytes']) == (139,40925633577,120382699520)
assert not (O/'receipt.json').exists()
item = dict(path=str(source), before=before)
result = dict(authorized_by='ROOT', source=str(source), source_before=before,
              source_proofs=[dict(path=str(p),sha256=h) for p,h in sources],
              helper_source=dict(path=str(helper),sha256=hashlib.sha256(helper.read_bytes()).hexdigest()),
              actual_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              operation_executed=False, offset=0, maximum_length_bytes=64<<30,
              required_memory_bytes=48698310656, memory_before=m['memory'](),
              payload_bytes_read=0, bytes_modified_or_deleted=False, global_drop_caches=False,
              original_CP_base_ZIP_or_durable_TAR_opened=False, exact_reclaimed_bytes=None,
              restore_authorized=False)
start=time.monotonic()
fd=None
try:
    if result['memory_before']['conservative_available_bytes'] >= 48698310656:
        result['status']='skipped_memory_already_sufficient'
    else:
        result['references_before_open']=m['references']([item])
        fd=os.open(source,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
        fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB)
        assert m['signature'](os.fstat(fd)) == m['target_stat'](source) == before
        result['references_after_open']=m['references']([item],{fd})
        result['memory_before_advice']=m['memory']()
        if result['memory_before_advice']['conservative_available_bytes'] >= 48698310656:
            result['status']='skipped_memory_recovered_before_advice'
        else:
            assert result['memory_before_advice']['events']['oom'] == result['memory_before_advice']['events']['oom_kill'] == 0
            t=time.monotonic();os.fsync(fd);result['same_readonly_fd_fsync_region_seconds']=time.monotonic()-t
            result['operation_executed']=True
            t=time.monotonic();os.posix_fadvise(fd,0,64<<30,os.POSIX_FADV_DONTNEED)
            result['advice_region_seconds']=time.monotonic()-t
            result['source_after']=m['signature'](os.fstat(fd))
            assert result['source_after'] == m['target_stat'](source) == before
            result['status']='passed_one_64GiB_completed_published_scratch_TAR_hint'
except Exception as e:
    result.update(status='failed_guard_or_operation',error=repr(e));raise
finally:
    if fd is not None:os.close(fd)
    result['memory_after']=m['memory']()
    result['operation_region_seconds']=time.monotonic()-start
    result['at_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    result['limitations']='Advice is not guaranteed reclaim or a reservation. Proc observations and per-file shared flock are not a producer lease. Whole child timer is inclusive; regions are nested.'
    m['write_new'](O/'receipt.json',result)
print(json.dumps({k:result[k] for k in ['status','operation_executed','memory_after']}))
