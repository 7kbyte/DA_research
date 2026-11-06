import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import stat_bound_digest_cache as cache


def fixture(tmp_path):
    path=tmp_path/'opaque.pt';path.write_bytes(b'opaque checkpoint bytes')
    return path,cache.StatBoundHashes(directory=tmp_path/'cache')


def test_full_read_then_persistent_stat_hit_and_sha_bound_readonly_snapshot(tmp_path):
    path,h=fixture(tmp_path);expected=hashlib.sha256(path.read_bytes()).hexdigest()
    assert h.digest(path)==expected
    assert h.counters=={'full_reads':1,'full_read_bytes':path.stat().st_size,'cache_hits':0,'invalidations':0}
    restarted=cache.StatBoundHashes(directory=tmp_path/'cache')
    assert restarted.digest(path)==expected
    assert restarted.counters['cache_hits']==1 and restarted.counters['full_read_bytes']==0
    proof=restarted.snapshot(tmp_path/'snapshot.json')
    reader=cache.StatBoundHashes(snapshot=proof['path'],snapshot_sha256=proof['sha256'])
    assert reader.verify(path,expected)==(str(path),expected)
    assert reader.counters['cache_hits']==1 and reader.counters['full_reads']==0


def test_missing_entry_requires_fresh_read_even_readonly(tmp_path):
    path,h=fixture(tmp_path);proof=h.snapshot(tmp_path/'snapshot.json')
    reader=cache.StatBoundHashes(snapshot=proof['path'],snapshot_sha256=proof['sha256'])
    assert reader.digest(path)==hashlib.sha256(path.read_bytes()).hexdigest()
    assert reader.counters['full_reads']==1 and reader.counters['cache_hits']==0


def test_wrong_expected_sha_rejected_on_fresh_and_cached_paths(tmp_path):
    path,h=fixture(tmp_path)
    for _ in range(2):
        with pytest.raises(ValueError,match='SHA mismatch'):h.verify(path,'0'*64)


def test_stat_change_invalidates_and_reads_full_bytes(tmp_path):
    path,h=fixture(tmp_path);old=h.digest(path);path.write_bytes(b'new file')
    assert h.digest(path)!=old
    assert h.counters['full_reads']==2 and h.counters['invalidations']==1


def test_completed_checkpoint_byte_change_is_fatal(tmp_path):
    path,h=fixture(tmp_path);h.digest(path,authority={'kind':'completed_training_checkpoint'})
    path.write_bytes(b'changed after rc0')
    with pytest.raises(ValueError,match='Completed checkpoint bytes changed'):h.digest(path)


def test_missing_file_does_not_reuse_historical_digest(tmp_path):
    path,h=fixture(tmp_path);h.digest(path);path.unlink()
    with pytest.raises(FileNotFoundError):h.digest(path)


def test_receipt_corruption_and_snapshot_sha_mismatch_rejected(tmp_path):
    path,h=fixture(tmp_path);h.digest(path);proof=h.snapshot(tmp_path/'snapshot.json')
    with pytest.raises(ValueError,match='SHA-bound'):cache.StatBoundHashes(snapshot=proof['path'],snapshot_sha256='0'*64)
    entry=h.entries[str(path)];Path(entry['path']).write_bytes(b'{}')
    other=cache.StatBoundHashes(directory=tmp_path/'cache')
    with pytest.raises(ValueError,match='receipt SHA mismatch'):other.digest(path)


def test_receipt_requires_full_fd_stability_not_just_digest(tmp_path):
    path,h=fixture(tmp_path);h.digest(path);entry=h.entries[str(path)]
    payload=json.loads(Path(entry['path']).read_bytes());payload['fd_after']['size']+=1
    # Write a new SHA-bound receipt, testing semantic validation rather than hash corruption.
    raw=cache.encoded(payload);receipt=tmp_path/'bad_receipt.json';receipt.write_bytes(raw)
    index=h.index();index['entries'][str(path)]={'path':str(receipt),'sha256':cache.digest_bytes(raw)}
    cache.atomic(tmp_path/'cache/index.json',index)
    with pytest.raises(ValueError,match='stable-fd'):cache.StatBoundHashes(directory=tmp_path/'cache').digest(path)


def test_world_writable_cache_evidence_rejected(tmp_path):
    path,h=fixture(tmp_path);h.digest(path)
    os.chmod(tmp_path/'cache/index.json',0o666)
    with pytest.raises(ValueError,match='owned by this UID'):cache.StatBoundHashes(directory=tmp_path/'cache')


def test_path_replacement_between_stat_and_open_is_rejected(tmp_path,monkeypatch):
    path,h=fixture(tmp_path);original=Path.open;replacement=tmp_path/'replacement';replacement.write_bytes(b'replacement')
    def changing_open(self,*args,**kwargs):
        if self==path and args and args[0]=='rb':os.replace(replacement,path)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',changing_open)
    with pytest.raises(ValueError,match='between stat and open'):h.digest(path)
    assert h.counters['full_reads']==0


def test_fd_change_during_read_is_rejected(tmp_path,monkeypatch):
    path,h=fixture(tmp_path);original=os.fstat;calls=0
    def unstable(fd):
        nonlocal calls
        value=original(fd);calls+=1
        if calls==2:return SimpleNamespace(**{'st_'+k:getattr(value,'st_'+k)+(1 if k=='mtime_ns' else 0) for k in cache.FIELDS})
        return value
    monkeypatch.setattr(os,'fstat',unstable)
    with pytest.raises(ValueError,match='complete fd hash'):h.digest(path)


def test_symlink_retarget_during_read_is_rejected(tmp_path,monkeypatch):
    path,h=fixture(tmp_path);link=tmp_path/'alias';link.symlink_to(path);other=tmp_path/'other';other.write_bytes(b'other')
    original=Path.open
    def swapping(self,*args,**kwargs):
        value=original(self,*args,**kwargs)
        if self==path and args and args[0]=='rb':link.unlink();link.symlink_to(other)
        return value
    monkeypatch.setattr(Path,'open',swapping)
    with pytest.raises(ValueError,match='complete fd hash'):h.digest(link)


def test_atomic_index_fsyncs_file_before_replace_and_directory_after(tmp_path,monkeypatch):
    events=[];sync=os.fsync;replace=os.replace
    def fsync(fd):events.append('fsync');return sync(fd)
    def renamed(a,b):events.append('replace');return replace(a,b)
    monkeypatch.setattr(os,'fsync',fsync);monkeypatch.setattr(os,'replace',renamed)
    cache.atomic(tmp_path/'atomic.json',{'safe':True})
    assert events==['fsync','replace','fsync']
    assert json.loads((tmp_path/'atomic.json').read_bytes())=={'safe':True}


def test_cache_hit_still_calls_checkpoint_completion_guard(tmp_path):
    path=tmp_path/'run/checkpoints/latest.pt';path.parent.mkdir(parents=True);path.write_bytes(b'opaque')
    calls=[]
    def guard(value):calls.append(value);return {'kind':'completed_training_checkpoint'}
    h=cache.StatBoundHashes(directory=tmp_path/'cache',checkpoint_guard=guard);h.digest(path);h.digest(path)
    assert len(calls)==3
    def rejected(value):raise FileNotFoundError('rc0 pending')
    h.checkpoint_guard=rejected
    with pytest.raises(FileNotFoundError):h.digest(path)


def test_immutable_snapshot_cannot_be_overwritten(tmp_path):
    _,h=fixture(tmp_path);h.snapshot(tmp_path/'snapshot.json')
    with pytest.raises(FileExistsError):h.snapshot(tmp_path/'snapshot.json')
