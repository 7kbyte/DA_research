"""Small actual-file checks for append-only reservation expansion."""
import datetime as dt
import errno
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

SPEC = importlib.util.spec_from_file_location("grow_reserved_file", Path(__file__).parents[1] / "scripts/grow_reserved_file.py")
growth = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(growth)
NOW = dt.datetime(2026, 11, 6, 10, tzinfo=dt.timezone.utc)


def setup_plan(tmp_path, *, additional=12_288):
    reserve = tmp_path / "reserve"
    reserve.write_bytes(bytes(range(256)) * 16)
    reserve.chmod(0o600)
    observed = growth.file_details(reserve.stat())
    plan = dict(path=str(reserve), identity={k: observed[k] for k in ("device", "inode", "uid", "gid", "mode")},
                initial_bytes=4096, target_bytes=4096 + additional, minimum_free_bytes=4096,
                check_bytes=1024, round_bytes=4096, poll_seconds=.01,
                deadline_utc="2026-11-07T02:00:00+00:00")
    return reserve, plan


def execute(tmp_path, plan, **kwargs):
    return growth.expand_reservation(plan, progress_path=tmp_path / "progress.json", receipt_path=tmp_path / "receipt.json",
                                     now_fn=lambda: NOW, available_fn=lambda _: 1 << 30, **kwargs)


def test_incremental_append_preserves_prefix_inode_and_round_bound(tmp_path):
    reserve, plan = setup_plan(tmp_path)
    original = reserve.read_bytes()
    inode = reserve.stat().st_ino
    waits = []
    result = execute(tmp_path, plan, wait_fn=waits.append)
    assert result["status"] == "complete"
    assert reserve.read_bytes() == original + bytes(plan["target_bytes"] - len(original))
    assert reserve.stat().st_ino == inode
    assert result["maximum_round_appended_bytes"] == 4096
    assert result["rounds"] == 3 and len(waits) == 2
    assert result["free_space_checks"] == 12
    assert result["actual_stat"]["allocated_bytes"] >= plan["target_bytes"]
    assert result["final_file_fsync_completed"] and result["lock_released"]
    assert json.loads((tmp_path / "receipt.json").read_text())["status"] == "complete"


def test_insufficient_capacity_waits_without_modifying_reservation(tmp_path):
    reserve, plan = setup_plan(tmp_path)
    original = reserve.read_bytes()
    stop = threading.Event()
    result = growth.expand_reservation(plan, progress_path=tmp_path / "progress.json", receipt_path=tmp_path / "receipt.json",
        now_fn=lambda: NOW, available_fn=lambda _: plan["minimum_free_bytes"], stop_event=stop,
        wait_fn=lambda _: stop.set())
    assert result["status"] == "stopped_signal" and result["bytes_appended"] == 0
    assert reserve.read_bytes() == original
    assert result["final_file_fsync_completed"]


def test_enospc_after_short_write_keeps_exact_partial_and_waits(tmp_path, monkeypatch):
    reserve, plan = setup_plan(tmp_path)
    original = reserve.read_bytes()
    real_write = os.write
    calls = 0
    stop = threading.Event()
    def short_then_full(fd, value):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(fd, value[:137])
        raise OSError(errno.ENOSPC, "synthetic full disk")
    monkeypatch.setattr(growth.os, "write", short_then_full)
    result = execute(tmp_path, plan, stop_event=stop, wait_fn=lambda _: stop.set())
    assert result["status"] == "stopped_signal" and result["bytes_appended"] == 137
    assert result["partial_fsync_succeeded"] and result["last_enospc"]
    assert reserve.read_bytes() == original + bytes(137)


def test_enospc_can_recover_without_reopening_or_replacing_inode(tmp_path, monkeypatch):
    reserve, plan = setup_plan(tmp_path)
    original = reserve.read_bytes()
    inode = reserve.stat().st_ino
    real_write = os.write
    calls = 0
    waits = []
    def full_once(fd, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.ENOSPC, "temporary full disk")
        return real_write(fd, value)
    monkeypatch.setattr(growth.os, "write", full_once)
    result = execute(tmp_path, plan, wait_fn=waits.append)
    assert result["status"] == "complete" and waits
    assert reserve.stat().st_ino == inode
    assert reserve.read_bytes() == original + bytes(plan["target_bytes"] - len(original))


def test_fsync_enospc_waits_and_recovers_before_complete(tmp_path, monkeypatch):
    reserve, plan = setup_plan(tmp_path, additional=1024)
    original = reserve.read_bytes()
    real_sync = os.fsync
    failures = 0
    waits = []
    def transient_sync(fd):
        nonlocal failures
        if os.fstat(fd).st_ino == reserve.stat().st_ino and failures < 2:
            failures += 1
            raise OSError(errno.ENOSPC, "delayed allocation full disk")
        return real_sync(fd)
    monkeypatch.setattr(growth.os, "fsync", transient_sync)
    result = execute(tmp_path, plan, wait_fn=waits.append)
    assert result["status"] == "complete" and result["partial_fsync_succeeded"]
    assert result["final_file_fsync_completed"] and failures == 2
    assert reserve.read_bytes() == original + bytes(1024)


@pytest.mark.parametrize("field", ["inode", "device", "uid", "gid", "mode"])
def test_bound_identity_change_rejected_before_append(tmp_path, field):
    reserve, plan = setup_plan(tmp_path)
    original = reserve.read_bytes()
    plan["identity"][field] += 1
    with pytest.raises(ValueError, match="identity|inode/device/owner/permissions"):
        execute(tmp_path, plan)
    assert reserve.read_bytes() == original


def test_bound_initial_length_rejected(tmp_path):
    reserve, plan = setup_plan(tmp_path)
    original = reserve.read_bytes()
    plan["initial_bytes"] -= 1
    with pytest.raises(ValueError, match="length changed"):
        execute(tmp_path, plan)
    assert reserve.read_bytes() == original


def test_exclusive_lock_rejects_second_writer(tmp_path):
    reserve, plan = setup_plan(tmp_path)
    original = reserve.read_bytes()
    progress = tmp_path / "progress.json"
    progress.write_text('{"status":"another writer owns this progress"}')
    previous_progress = progress.read_bytes()
    with reserve.open("r+b") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            execute(tmp_path, plan)
    assert reserve.read_bytes() == original
    assert progress.read_bytes() == previous_progress
    assert not (tmp_path / "receipt.json").exists()


def test_path_replacement_between_rounds_stops_original_inode(tmp_path):
    reserve, plan = setup_plan(tmp_path)
    old = tmp_path / "original_inode"
    def replace(_):
        reserve.rename(old)
        reserve.write_bytes(b"replacement must remain untouched")
    with pytest.raises(ValueError, match="inode/device/owner/permissions"):
        execute(tmp_path, plan, wait_fn=replace)
    assert old.stat().st_size == 8192
    assert reserve.read_bytes() == b"replacement must remain untouched"


@pytest.mark.parametrize("reason", ["signal", "deadline"])
def test_stop_boundary_syncs_without_append(tmp_path, reason):
    reserve, plan = setup_plan(tmp_path)
    original = reserve.read_bytes()
    stop = threading.Event()
    if reason == "signal":
        stop.set()
    else:
        plan["deadline_utc"] = NOW.isoformat()
    result = execute(tmp_path, plan, stop_event=stop)
    assert result["status"] == "stopped_" + reason
    assert result["bytes_appended"] == 0 and reserve.read_bytes() == original
    assert result["final_file_fsync_completed"] and result["lock_released"]


def test_signal_during_write_syncs_only_completed_append(tmp_path, monkeypatch):
    reserve, plan = setup_plan(tmp_path)
    original = reserve.read_bytes()
    real_write = os.write
    stop = threading.Event()
    def write_then_stop(fd, value):
        count = real_write(fd, value)
        stop.set()
        return count
    monkeypatch.setattr(growth.os, "write", write_then_stop)
    result = execute(tmp_path, plan, stop_event=stop)
    assert result["status"] == "stopped_signal" and result["bytes_appended"] == 1024
    assert reserve.read_bytes() == original + bytes(1024)
    assert result["final_file_fsync_completed"]


def test_other_io_failure_stops_without_retry(tmp_path, monkeypatch):
    reserve, plan = setup_plan(tmp_path)
    original = reserve.read_bytes()
    def fail(*_):
        raise OSError(errno.EIO, "synthetic IO failure")
    monkeypatch.setattr(growth.os, "write", fail)
    with pytest.raises(OSError, match="synthetic IO failure"):
        execute(tmp_path, plan)
    assert reserve.read_bytes() == original
    assert json.loads((tmp_path / "receipt.json").read_text())["status"] == "failed"


def test_symlink_rejected_without_touching_target(tmp_path):
    reserve, plan = setup_plan(tmp_path)
    original = reserve.read_bytes()
    alias = tmp_path / "link"
    alias.symlink_to(reserve)
    plan["path"] = str(alias)
    with pytest.raises(ValueError, match="non-symlink"):
        execute(tmp_path, plan)
    assert reserve.read_bytes() == original


def test_cap_and_timezone_validation(tmp_path):
    _, plan = setup_plan(tmp_path)
    for key, value in (("check_bytes", (256 << 20) + 1), ("round_bytes", (2 << 30) + 1),
                       ("deadline_utc", "2026-11-07T02:00:00")):
        modified = {**plan, key: value}
        with pytest.raises(ValueError):
            execute(tmp_path, modified)


def test_cli_plan_sha_rejected_without_writing_reservation(tmp_path):
    reserve, plan = setup_plan(tmp_path)
    original = reserve.read_bytes()
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan))
    result = subprocess.run([sys.executable, str(SPEC.origin), "--plan", str(plan_file),
        "--expected-plan-sha256", "0" * 64, "--progress", str(tmp_path / "p.json"),
        "--receipt", str(tmp_path / "r.json")], capture_output=True, text=True)
    assert result.returncode == 2 and "plan SHA mismatch" in result.stderr
    assert reserve.read_bytes() == original


def test_cli_term_releases_lock_after_fsync_and_writes_receipt(tmp_path):
    import hashlib
    reserve, plan = setup_plan(tmp_path)
    original = reserve.read_bytes()
    # More headroom than this filesystem can have makes the child wait without
    # writing even one byte; only its real SIGTERM handler ends that wait.
    plan["minimum_free_bytes"] = 1 << 62
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan))
    progress, receipt = tmp_path / "p.json", tmp_path / "r.json"
    child = subprocess.Popen([sys.executable, str(SPEC.origin), "--plan", str(plan_file),
        "--expected-plan-sha256", hashlib.sha256(plan_file.read_bytes()).hexdigest(),
        "--progress", str(progress), "--receipt", str(receipt)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        started = time.monotonic()
        while not progress.exists():
            if child.poll() is not None:
                pytest.fail(child.stderr.read())
            assert time.monotonic() - started < 5
            time.sleep(.01)
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=5) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    result = json.loads(receipt.read_text())
    assert result["status"] == "stopped_signal" and result["final_file_fsync_completed"]
    assert result["lock_released"] and reserve.read_bytes() == original
    with reserve.open("r+b") as output:
        fcntl.flock(output.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
