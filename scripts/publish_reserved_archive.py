#!/usr/bin/env python3
"""Publish a SHA-bound archive through its preallocated inode, without overwrite.

Use renameat2(RENAME_NOREPLACE), or atomic hard-link then unlink on filesystems
without that operation. No destination is committed until
the entire preallocated-file prefix has been read back and verified.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def signature(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def details(value):
    return dict(device=value.st_dev, inode=value.st_ino, bytes=value.st_size, allocated_bytes=value.st_blocks * 512,
                mode=oct(stat.S_IMODE(value.st_mode)), uid=value.st_uid, gid=value.st_gid,
                mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def rename_noreplace(source, destination):
    """Commit without overwrite; hard-link fallback keeps the same inode.

    The fallback's link is atomic and refuses an existing destination. The new
    directory entry is fsynced before removing the reservation name, so a crash
    can leave both names but cannot remove the only durable name.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    error = errno.ENOSYS
    if hasattr(libc, "renameat2"):
        operation = libc.renameat2
        operation.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        operation.restype = ctypes.c_int
        if not operation(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
            return "renameat2_RENAME_NOREPLACE"
        error = ctypes.get_errno()
    if error not in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        raise OSError(error, os.strerror(error), str(destination))
    os.link(source, destination, follow_symlinks=False)
    parent_fd = os.open(Path(destination).parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    os.unlink(source)
    return "atomic_hard_link_noreplace_fsync_then_unlink"


def hash_prefix(fd, length, block_bytes):
    digest, offset = hashlib.sha256(), 0
    while offset < length:
        value = os.pread(fd, min(block_bytes, length - offset), offset)
        if not value:
            raise ValueError("unexpected EOF while verifying archive prefix")
        digest.update(value)
        offset += len(value)
    return digest.hexdigest()


def _regular(path):
    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise ValueError("source and reservation must be regular, non-symlink files")
    return observed


def _destination_absent(path):
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise FileExistsError(errno.EEXIST, "refusing to replace any existing destination", str(path))


def publish(source, expected_sha256, expected_bytes, reserved, destination, *, receipt_path, progress_path, block_bytes=8 << 20):
    source, reserved, destination = [Path(os.path.abspath(p)) for p in (source, reserved, destination)]
    if len({str(source), str(reserved), str(destination)}) != 3:
        raise ValueError("source, reserved and destination must be different paths")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or type(expected_bytes) is not int or expected_bytes <= 0 or block_bytes < 1:
        raise ValueError("explicit lowercase SHA256, positive byte size and block size required")
    started = time.monotonic()
    progress = dict(status="running", pid=os.getpid(), source=str(source), reserved=str(reserved), destination=str(destination),
                    expected_sha256=expected_sha256, expected_bytes=expected_bytes, bytes_written=0,
                    gpu_used=False, source_retained=True, published=False, started_at_utc=utc())
    events = []
    def stage(name, **values):
        progress.update(stage=name, updated_at_utc=utc(), **values)
        write_json(progress_path, progress)
    source_fd = reserve_fd = None
    try:
        _destination_absent(destination)
        source_before, reserve_before = _regular(source), _regular(reserved)
        parent_before = destination.parent.stat()
        if source_before.st_size != expected_bytes:
            raise ValueError("source byte size differs from expected frozen archive")
        if reserve_before.st_dev != parent_before.st_dev:
            raise ValueError("reservation and destination parent must be on the same filesystem")
        if reserve_before.st_ino == source_before.st_ino and reserve_before.st_dev == source_before.st_dev:
            raise ValueError("reservation cannot alias the source inode")
        if reserve_before.st_size < expected_bytes or reserve_before.st_blocks * 512 < expected_bytes:
            raise ValueError("insufficient preallocated reservation size or allocated blocks")
        source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        fcntl.flock(source_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        def source_stable():
            if signature(os.fstat(source_fd)) != signature(source_before) or signature(source.lstat()) != signature(source_before):
                raise ValueError("source changed during publication; destination not committed")
        source_stable()
        stage("verify_entire_source_sha_before_reservation_write")
        actual_sha = hash_prefix(source_fd, expected_bytes, block_bytes)
        source_stable()
        if actual_sha != expected_sha256:
            raise ValueError("source SHA differs; reservation was not modified")
        events.append(dict(event="source_sha_and_stable_stat_verified", utc=utc(), sha256=actual_sha))
        _destination_absent(destination)
        # O_TRUNC is deliberately absent: retain the allocation and original inode.
        reserve_fd = os.open(reserved, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
        fcntl.flock(reserve_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if signature(os.fstat(reserve_fd)) != signature(reserve_before) or signature(reserved.lstat()) != signature(reserve_before):
            raise ValueError("reservation changed before first write")
        source_stable()
        stage("write_existing_preallocated_inode", reserved_original=details(reserve_before))
        copied_hash, offset, last_update = hashlib.sha256(), 0, time.monotonic()
        while offset < expected_bytes:
            block = os.pread(source_fd, min(block_bytes, expected_bytes - offset), offset)
            if not block:
                raise ValueError("source became truncated while copying")
            copied_hash.update(block)
            written = 0
            while written < len(block):
                count = os.pwrite(reserve_fd, memoryview(block)[written:], offset + written)
                if count <= 0:
                    raise OSError("archive write made no progress")
                written += count
            offset += len(block)
            if time.monotonic() - last_update >= 5:
                stage("write_existing_preallocated_inode", bytes_written=offset)
                last_update = time.monotonic()
        source_stable()
        if copied_hash.hexdigest() != expected_sha256:
            raise ValueError("copied source stream SHA differs")
        if os.fstat(reserve_fd).st_size != reserve_before.st_size:
            raise ValueError("reservation size changed before prefix verification")
        os.fsync(reserve_fd)
        events.append(dict(event="full_prefix_write_fsync_completed", utc=utc(), bytes=offset))
        stage("read_back_entire_valid_prefix_sha", bytes_written=offset)
        prefix_sha = hash_prefix(reserve_fd, expected_bytes, block_bytes)
        source_stable()
        if prefix_sha != expected_sha256:
            raise ValueError("read-back valid prefix SHA differs; reservation retained")
        events.append(dict(event="entire_valid_prefix_sha_verified_before_truncate", utc=utc(), sha256=prefix_sha,
                           reservation_size_before_truncate=os.fstat(reserve_fd).st_size))
        _destination_absent(destination)
        current = reserved.lstat()
        if (current.st_dev, current.st_ino) != (reserve_before.st_dev, reserve_before.st_ino):
            raise ValueError("reservation path no longer points to the original inode")
        stage("truncate_verified_tail_and_fsync")
        os.ftruncate(reserve_fd, expected_bytes)
        os.fsync(reserve_fd)
        events.append(dict(event="truncate_to_verified_length_and_file_fsync_completed", utc=utc(), bytes=expected_bytes))
        source_stable()
        stage("atomic_rename_noreplace_and_parent_fsync")
        publication_method = rename_noreplace(reserved, destination)
        progress["published"] = True
        for parent in {reserved.parent, destination.parent}:
            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            events.append(dict(event="parent_directory_fsync_completed", utc=utc(), path=str(parent)))
        final = destination.lstat()
        if ((final.st_dev, final.st_ino, stat.S_IMODE(final.st_mode), final.st_uid, final.st_gid) !=
                (reserve_before.st_dev, reserve_before.st_ino, stat.S_IMODE(reserve_before.st_mode), reserve_before.st_uid, reserve_before.st_gid) or
                final.st_size != expected_bytes):
            raise ValueError("published inode/permissions/ownership/length differs")
        receipt = dict(status="passed", source=str(source), destination=str(destination), source_sha256=actual_sha,
            destination_sha256=prefix_sha, bytes=expected_bytes, original_reservation=str(reserved),
            original_reservation_stat=details(reserve_before), destination_stat=details(final),
            original_inode_preserved=True, permissions_and_ownership_preserved=True, overwrite=False,
            source_stable_stat=details(source_before), source_retained=source.exists(), gpu_used=False,
            file_fsync_after_copy=True, prefix_sha_verified_before_truncate=True, file_fsync_after_truncate=True,
            atomic_rename_noreplace=publication_method == "renameat2_RENAME_NOREPLACE",
            atomic_destination_commit_noreplace=True, publication_method=publication_method,
            parent_directory_fsync=True, durability_events=events,
            started_at_utc=progress["started_at_utc"], finished_at_utc=utc(), wall_seconds=time.monotonic() - started,
            publisher_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        write_json(receipt_path, receipt)
        stage("complete", status="passed", bytes_written=expected_bytes, receipt_path=str(receipt_path), wall_seconds=receipt["wall_seconds"])
        return receipt
    except BaseException as error:
        stage("failed", status="failed", error=repr(error), durability_events=events)
        raise
    finally:
        for fd in (reserve_fd, source_fd):
            if fd is not None:
                os.close(fd)


def resume_verified_publication(failed_progress, *, receipt_path, progress_path, block_bytes=8 << 20):
    """Rehash an already-written inode and finish only its interrupted commit."""
    previous = json.loads(Path(failed_progress).read_text())
    required_events = {"source_sha_and_stable_stat_verified", "full_prefix_write_fsync_completed",
                       "entire_valid_prefix_sha_verified_before_truncate", "truncate_to_verified_length_and_file_fsync_completed"}
    if (previous.get("published") or previous.get("status") != "failed" or
            not required_events <= {e["event"] for e in previous.get("durability_events", [])}):
        raise ValueError("recovery requires verified, truncated, uncommitted previous publication")
    source, reserved, destination = map(Path, [previous[k] for k in ("source", "reserved", "destination")])
    size, expected = previous["expected_bytes"], previous["expected_sha256"]
    original = previous["reserved_original"]
    _destination_absent(destination)
    source_stat, reserve_stat = _regular(source), _regular(reserved)
    if (source_stat.st_size != size or reserve_stat.st_size != size or reserve_stat.st_blocks * 512 < size or
        (reserve_stat.st_dev, reserve_stat.st_ino, oct(stat.S_IMODE(reserve_stat.st_mode)), reserve_stat.st_uid, reserve_stat.st_gid) !=
        (original["device"], original["inode"], original["mode"], original["uid"], original["gid"]) or
        reserve_stat.st_dev != destination.parent.stat().st_dev):
        raise ValueError("recovery source/reservation size, original inode, permissions or filesystem differs")
    started = time.monotonic()
    progress = dict(previous, status="running", pid=os.getpid(), stage="recovery_rehash_source_and_reserved_prefix",
                    previous_failure=previous["error"], recovery_started_at_utc=utc())
    write_json(progress_path, progress)
    sf = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    rf = os.open(reserved, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(sf, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(rf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if signature(os.fstat(sf)) != signature(source_stat) or signature(os.fstat(rf)) != signature(reserve_stat):
            raise ValueError("recovery input changed before hash")
        if hash_prefix(sf, size, block_bytes) != expected or hash_prefix(rf, size, block_bytes) != expected:
            raise ValueError("recovery full source or reserved prefix SHA differs")
        if (signature(os.fstat(sf)) != signature(source_stat) or signature(source.lstat()) != signature(source_stat) or
                signature(os.fstat(rf)) != signature(reserve_stat) or signature(reserved.lstat()) != signature(reserve_stat)):
            raise ValueError("recovery input changed during hash")
        os.fsync(rf)
        events = previous["durability_events"] + [dict(event="recovery_full_source_and_reserved_sha_verified_file_fsync", utc=utc(), sha256=expected)]
        method = rename_noreplace(reserved, destination)
        progress["published"] = True
        fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        events.append(dict(event="parent_directory_fsync_completed", utc=utc(), path=str(destination.parent)))
        final = destination.lstat()
        if (final.st_dev, final.st_ino, final.st_size) != (original["device"], original["inode"], size):
            raise ValueError("recovery destination inode or length differs")
        receipt = dict(status="passed", source=str(source), destination=str(destination), source_sha256=expected,
            destination_sha256=expected, bytes=size, original_reservation=str(reserved), original_reservation_stat=original,
            destination_stat=details(final), original_inode_preserved=True, permissions_and_ownership_preserved=True,
            overwrite=False, source_stable_stat=details(source_stat), source_retained=True, gpu_used=False,
            file_fsync_after_copy=True, prefix_sha_verified_before_truncate=True, file_fsync_after_truncate=True,
            atomic_rename_noreplace=method == "renameat2_RENAME_NOREPLACE", atomic_destination_commit_noreplace=True,
            publication_method=method, parent_directory_fsync=True, durability_events=events,
            recovery_of_failure=previous["error"], recovered_progress_sha256=hashlib.sha256(Path(failed_progress).read_bytes()).hexdigest(),
            started_at_utc=previous["started_at_utc"], finished_at_utc=utc(), recovery_wall_seconds=time.monotonic()-started,
            publisher_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        write_json(receipt_path, receipt)
        progress.update(status="passed", stage="complete", receipt_path=str(receipt_path), durability_events=events, updated_at_utc=utc())
        write_json(progress_path, progress)
        return receipt
    except BaseException as error:
        progress.update(status="failed", stage="recovery_failed", error=repr(error), updated_at_utc=utc())
        write_json(progress_path, progress)
        raise
    finally:
        os.close(sf)
        os.close(rf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--reserved", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    args = parser.parse_args()
    receipt = publish(args.source, args.expected_sha256, args.expected_bytes, args.reserved, args.destination,
                      receipt_path=args.receipt, progress_path=args.progress)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
