#!/usr/bin/env python3
"""Append real zeros to one explicitly bound reservation; never truncate it.

The CLI is intentionally a separate action from preparation of a plan. It does
not publish an archive, allocate a sparse extent, or modify other reservations.
"""
from __future__ import annotations

import argparse
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import threading
import time

GIB = 1 << 30
MAX_CHECK_BYTES = 256 << 20
MAX_ROUND_BYTES = 2 * GIB


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def parse_utc(value):
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() != dt.timedelta(0):
        raise ValueError("deadline must explicitly use UTC")
    return result


def file_details(value):
    return dict(device=value.st_dev, inode=value.st_ino, uid=value.st_uid,
                gid=value.st_gid, mode=stat.S_IMODE(value.st_mode),
                bytes=value.st_size, allocated_bytes=value.st_blocks * 512,
                links=value.st_nlink)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("x") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp, path)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def free_bytes(fd):
    value = os.fstatvfs(fd)
    return value.f_bavail * value.f_frsize


def validate_plan(plan):
    required = {"path", "identity", "initial_bytes", "target_bytes", "minimum_free_bytes",
                "check_bytes", "round_bytes", "poll_seconds", "deadline_utc"}
    if not required <= plan.keys():
        raise ValueError("incomplete reservation growth plan")
    for key in ("initial_bytes", "target_bytes", "minimum_free_bytes", "check_bytes", "round_bytes"):
        if type(plan[key]) is not int or plan[key] <= 0:
            raise ValueError(f"positive integer required: {key}")
    if plan["target_bytes"] < plan["initial_bytes"]:
        raise ValueError("target cannot shrink the reservation")
    if plan["check_bytes"] > MAX_CHECK_BYTES or plan["round_bytes"] > MAX_ROUND_BYTES:
        raise ValueError("space checks must be <=256MiB and rounds <=2GiB")
    if not 0 < plan["poll_seconds"] <= 30:
        raise ValueError("poll interval must be in (0,30] seconds")
    if set(plan["identity"]) != {"device", "inode", "uid", "gid", "mode"}:
        raise ValueError("explicit device/inode/uid/gid/mode identity required")
    if any(type(v) is not int for v in plan["identity"].values()):
        raise ValueError("identity fields must be integers")
    if not Path(plan["path"]).is_absolute():
        raise ValueError("reservation path must be absolute")
    return parse_utc(plan["deadline_utc"])


def check_identity(value, identity, expected_bytes):
    if not stat.S_ISREG(value.st_mode) or stat.S_ISLNK(value.st_mode):
        raise ValueError("reservation must be a regular non-symlink file")
    actual = file_details(value)
    if any(actual[k] != v for k, v in identity.items()):
        raise ValueError("reservation inode/device/owner/permissions changed")
    if actual["bytes"] != expected_bytes:
        raise ValueError("reservation length changed outside this writer")
    if actual["links"] != 1:
        raise ValueError("reservation must have exactly one pathname")


def expand_reservation(plan, *, progress_path, receipt_path, stop_event=None,
                       now_fn=utc_now, available_fn=free_bytes, wait_fn=None):
    """Hold an exclusive lock through every write, wait, and final fsync.

    ENOSPC and insufficient free capacity leave the actual partial inode intact
    and wait. Other errors terminate. Injected functions support tiny CPU tests.
    """
    deadline = validate_plan(plan)
    path = Path(plan["path"])
    if os.path.abspath(progress_path) == os.path.abspath(receipt_path):
        raise ValueError("progress and final receipt must be different paths")
    for output in (progress_path, receipt_path):
        if os.path.abspath(output) == str(path):
            raise ValueError("progress/receipt cannot replace the reservation")
        if Path(output).exists() and os.path.samestat(Path(output).stat(), path.lstat()):
            raise ValueError("progress/receipt cannot alias the reservation")
    stop = stop_event if stop_event is not None else threading.Event()
    wait = wait_fn if wait_fn is not None else stop.wait
    started = time.monotonic()
    state = dict(status="starting", pid=os.getpid(), plan=plan, started_at_utc=now_fn().isoformat(),
                 gpu_used=False, append_only=True, truncate_used=False, published=False,
                 bytes_appended=0, rounds=0, free_space_checks=0, fsync_count=0,
                 lock_released=False, source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    cursor, fd, locked, error = plan["initial_bytes"], None, False, None

    def record(status, **extra):
        state.update(status=status, updated_at_utc=now_fn().isoformat(), wall_seconds=time.monotonic() - started, **extra)
        if fd is not None:
            state["actual_stat"] = file_details(os.fstat(fd))
            state["bytes_appended"] = state["actual_stat"]["bytes"] - plan["initial_bytes"]
        write_json(progress_path, state)

    def stopped():
        if stop.is_set():
            return "stopped_signal"
        if now_fn() >= deadline:
            return "stopped_deadline"
        return None

    def stable():
        check_identity(os.fstat(fd), plan["identity"], cursor)
        check_identity(path.lstat(), plan["identity"], cursor)

    def sync():
        os.fsync(fd)
        state["fsync_count"] += 1
        state["last_file_fsync_at_utc"] = now_fn().isoformat()

    try:
        check_identity(path.lstat(), plan["identity"], cursor)
        fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        stable()
        if os.fstat(fd).st_blocks * 512 < cursor:
            raise ValueError("initial reservation has insufficient real allocated blocks")
        allocation_granule = max(1, os.fstatvfs(fd).f_frsize)
        record("locked")
        zeroes = bytes(min(8 << 20, plan["check_bytes"]))
        while True:
            stable()
            reason = stopped()
            if reason:
                record(reason)
                break
            if state.get("partial_fsync_succeeded") is False:
                try:
                    sync()
                    state["partial_fsync_succeeded"] = True
                except OSError as failure:
                    if failure.errno != errno.ENOSPC:
                        raise
                    record("waiting_enospc", last_enospc=repr(failure))
                    wait(plan["poll_seconds"])
                    continue
            if cursor == plan["target_bytes"]:
                if os.fstat(fd).st_blocks * 512 < cursor:
                    raise ValueError("target length reached without sufficient allocated blocks")
                record("complete")
                break
            state["rounds"] += 1
            round_start = cursor
            waiting = False
            while cursor - round_start < plan["round_bytes"] and cursor < plan["target_bytes"]:
                stable()
                reason = stopped()
                if reason:
                    break
                capacity = available_fn(fd)
                state["free_space_checks"] += 1
                state["last_free_bytes"] = capacity
                # Round free capacity down to a filesystem allocation unit so
                # a short final append cannot consume the promised floor.
                safe_capacity = max(0, capacity - plan["minimum_free_bytes"]) // allocation_granule * allocation_granule
                writable = min(plan["check_bytes"], plan["round_bytes"] - (cursor - round_start),
                               plan["target_bytes"] - cursor, safe_capacity)
                if writable <= 0:
                    try:
                        sync()
                        record("waiting_free_space")
                    except OSError as failure:
                        if failure.errno != errno.ENOSPC:
                            raise
                        state["partial_fsync_succeeded"] = False
                        record("waiting_enospc", last_enospc=repr(failure))
                    waiting = True
                    break
                end = cursor + writable
                try:
                    while cursor < end:
                        if stopped():
                            break
                        count = os.write(fd, memoryview(zeroes)[:min(len(zeroes), end - cursor)])
                        if count <= 0:
                            raise OSError(errno.EIO, "zero append made no progress")
                        cursor += count
                    stable()
                    sync()
                    record("growing", last_check_append_bytes=writable,
                           current_round_appended_bytes=cursor - round_start)
                except OSError as failure:
                    if failure.errno != errno.ENOSPC:
                        raise
                    # A short write may precede ENOSPC. Keep that exact partial.
                    stable()
                    try:
                        sync()
                        state["partial_fsync_succeeded"] = True
                    except OSError as sync_failure:
                        if sync_failure.errno != errno.ENOSPC:
                            raise
                        state["partial_fsync_succeeded"] = False
                    record("waiting_enospc", last_enospc=repr(failure))
                    waiting = True
                    break
                if stopped():
                    break
            state["maximum_round_appended_bytes"] = max(state.get("maximum_round_appended_bytes", 0), cursor - round_start)
            if stopped() or cursor == plan["target_bytes"]:
                continue
            if not waiting:
                try:
                    sync()
                    record("round_complete_waiting", current_round_appended_bytes=cursor - round_start)
                except OSError as failure:
                    if failure.errno != errno.ENOSPC:
                        raise
                    state["partial_fsync_succeeded"] = False
                    record("waiting_enospc", last_enospc=repr(failure))
            wait(plan["poll_seconds"])
    except BaseException as failure:
        error = failure
        state.update(status="failed", error=repr(failure))
    finally:
        if fd is not None:
            if locked:
                try:
                    sync()
                    state["final_file_fsync_completed"] = True
                except BaseException as failure:
                    state["final_file_fsync_completed"] = False
                    state["final_fsync_error"] = repr(failure)
                    if error is None:
                        error = failure
                        state["status"] = "failed"
            state["actual_stat"] = file_details(os.fstat(fd))
            state["bytes_appended"] = state["actual_stat"]["bytes"] - plan["initial_bytes"]
            os.close(fd)
            fd = None
            state["lock_released"] = locked
        state.update(finished_at_utc=now_fn().isoformat(), wall_seconds=time.monotonic() - started,
                     exit_confirmed_by_caller_required_before_publication=True)
        # A rejected second writer must not replace the active writer's
        # progress/receipt names either. No output is owned before flock.
        if locked:
            write_json(progress_path, state)
            write_json(receipt_path, state)
    if error is not None:
        raise error
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    raw = args.plan.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.expected_plan_sha256:
        parser.error("growth plan SHA mismatch")
    plan = json.loads(raw)
    validate_plan(plan)
    if plan["minimum_free_bytes"] < 2 * GIB:
        parser.error("CLI requires at least 2GiB free-space headroom")
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda _signal, _frame: stop.set())
    result = expand_reservation(plan, progress_path=args.progress, receipt_path=args.receipt, stop_event=stop)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
