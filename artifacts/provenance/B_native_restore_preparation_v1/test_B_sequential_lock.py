"""Only temporary flock fixtures; never call the real extraction/decode body."""
import fcntl
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('B_restore_lock_under_test',Path(__file__).with_name('run_B_restore.py'))
r = importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def test_existing_C_native_lock_refuses_before_QA_creation(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'QA',tmp_path/'newQA')
    monkeypatch.setattr(r,'authorize',lambda *_: ({},{},{}))
    called=[]
    monkeypatch.setattr(r,'body_locked',lambda *_:called.append(True))
    with (tmp_path/'.native_Cfinal_restore.lock').open('a') as native_C_lock:
        fcntl.flock(native_C_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):r.body(None,None)
    assert not called and not r.QA.exists()


def test_same_lock_is_held_for_entire_body_and_released_after(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'QA',tmp_path/'newQA')
    monkeypatch.setattr(r,'authorize',lambda *_: ({},{},{}))
    called=[]
    def fake_body(*_):
        with (tmp_path/'.native_Cfinal_restore.lock').open('a') as competing_C_lock:
            with pytest.raises(BlockingIOError):fcntl.flock(competing_C_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        called.append(True)
    monkeypatch.setattr(r,'body_locked',fake_body)
    r.body(None,None)
    with (tmp_path/'.native_Cfinal_restore.lock').open('a') as C_after:
        fcntl.flock(C_after,fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert called==[True] and not r.QA.exists()
