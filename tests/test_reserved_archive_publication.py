"""Small real-file tests for reserved-inode publication, using temporary files."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat

import pytest

SPEC = importlib.util.spec_from_file_location("reserved_archive_publish", Path(__file__).resolve().parents[1] / "scripts/publish_reserved_archive.py")
PUBLISH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISH)


def files(tmp_path):
    source, reserved, destination = [tmp_path / n for n in ("source.tar", "reserve", "published.tar")]
    payload = bytes(range(256)) * 127 + b"tail"
    source.write_bytes(payload)
    with reserved.open("wb") as stream:
        stream.write(b"R" * (len(payload) + 16384))
    reserved.chmod(0o600)
    options = dict(receipt_path=tmp_path / "receipt.json", progress_path=tmp_path / "publication.progress.json", block_bytes=1024)
    return source, reserved, destination, payload, options


def test_normal_roundtrip_retains_inode_permissions_and_fsync_evidence(tmp_path):
    source, reserved, destination, payload, options = files(tmp_path)
    original = reserved.stat()
    receipt = PUBLISH.publish(source, hashlib.sha256(payload).hexdigest(), len(payload), reserved, destination, **options)
    assert destination.read_bytes() == payload and source.read_bytes() == payload
    assert not reserved.exists()
    assert destination.stat().st_ino == original.st_ino
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert receipt["parent_directory_fsync"] and receipt["file_fsync_after_copy"] and receipt["file_fsync_after_truncate"]
    assert json.loads(Path(options["progress_path"]).read_text())["status"] == "passed"


@pytest.mark.parametrize("failure", ["bad_sha", "insufficient", "existing_destination", "source_changed"])
def test_preflight_rejections_do_not_modify_the_reservation(tmp_path, monkeypatch, failure):
    source, reserved, destination, payload, options = files(tmp_path)
    expected = hashlib.sha256(payload).hexdigest()
    if failure == "bad_sha":
        expected = "0" * 64
    elif failure == "insufficient":
        reserved.unlink()
        with reserved.open("wb") as stream:
            stream.truncate(len(payload) + 16384)  # Sparse logical capacity is insufficient allocation.
    elif failure == "existing_destination":
        destination.write_bytes(b"do not overwrite")
    elif failure == "source_changed":
        original_hash = PUBLISH.hash_prefix
        def changing_source(*args):
            result = original_hash(*args)
            source.write_bytes(payload + b"changed")
            return result
        monkeypatch.setattr(PUBLISH, "hash_prefix", changing_source)
    before, inode = reserved.read_bytes(), reserved.stat().st_ino
    with pytest.raises((ValueError, FileExistsError)):
        PUBLISH.publish(source, expected, len(payload), reserved, destination, **options)
    assert reserved.read_bytes() == before and reserved.stat().st_ino == inode
    if failure == "existing_destination":
        assert destination.read_bytes() == b"do not overwrite"
    else:
        assert not destination.exists()
    assert not Path(options["receipt_path"]).exists()


def test_atomic_rename_refuses_a_destination_created_after_preflight(tmp_path, monkeypatch):
    source, reserved, destination, payload, options = files(tmp_path)
    original = PUBLISH.rename_noreplace
    def racing_destination(old, new):
        Path(new).write_bytes(b"racing file")
        original(old, new)
    monkeypatch.setattr(PUBLISH, "rename_noreplace", racing_destination)
    with pytest.raises(FileExistsError):
        PUBLISH.publish(source, hashlib.sha256(payload).hexdigest(), len(payload), reserved, destination, **options)
    assert destination.read_bytes() == b"racing file"
    assert reserved.read_bytes() == payload  # Verified prefix retained for explicit recovery.


def test_unsupported_rename_falls_back_to_atomic_same_inode_link(tmp_path, monkeypatch):
    source, reserved, destination, payload, options = files(tmp_path)
    inode = reserved.stat().st_ino
    monkeypatch.setattr(PUBLISH.ctypes, 'CDLL', lambda *args, **kwargs: object())
    receipt = PUBLISH.publish(source, hashlib.sha256(payload).hexdigest(), len(payload), reserved, destination, **options)
    assert receipt['publication_method'] == 'atomic_hard_link_noreplace_fsync_then_unlink'
    assert receipt['atomic_destination_commit_noreplace'] and not receipt['atomic_rename_noreplace']
    assert destination.stat().st_ino == inode and destination.read_bytes() == payload and not reserved.exists()
    reserved.write_bytes(b'other')
    with pytest.raises(FileExistsError):
        PUBLISH.rename_noreplace(reserved, destination)
    assert destination.read_bytes() == payload and reserved.read_bytes() == b'other'


def test_verified_interrupted_commit_can_resume_without_rewriting(tmp_path, monkeypatch):
    source, reserved, destination, payload, options = files(tmp_path)
    original_commit = PUBLISH.rename_noreplace
    def unavailable(*args):
        raise OSError(22, 'Invalid argument')
    monkeypatch.setattr(PUBLISH, 'rename_noreplace', unavailable)
    with pytest.raises(OSError):
        PUBLISH.publish(source, hashlib.sha256(payload).hexdigest(), len(payload), reserved, destination, **options)
    failure = tmp_path / 'failure.json'
    failure.write_bytes(Path(options['progress_path']).read_bytes())
    monkeypatch.setattr(PUBLISH, 'rename_noreplace', original_commit)
    before = reserved.stat()
    receipt = PUBLISH.resume_verified_publication(failure, **options)
    assert receipt['status'] == 'passed' and destination.stat().st_ino == before.st_ino
    assert destination.read_bytes() == payload and receipt['recovery_of_failure']
