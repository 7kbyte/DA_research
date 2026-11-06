"""Full-file SHA receipts reusable only while canonical paths and five stat fields agree."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid

FIELDS = ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns')
METHOD = 'sha256_entire_open_fd_to_eof_v1'


def signature(value):
    return {key: getattr(value, 'st_' + key) for key in FIELDS}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def small_proof(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest_bytes(path.read_bytes())}


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('xb') as stream:
        os.chmod(temp, 0o600)
        stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path); fsync_directory(path.parent)


def secure_owner(path):
    info = Path(path).stat()
    if info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise ValueError('Digest cache evidence must be owned by this UID and not group/world writable')


class StatBoundHashes:
    """Compatible with frozen Hashes.digest/proof, without changing that class.

    One CPU preparation process owns the writable directory. A successor may
    use a SHA-bound index snapshot read-only. A hit is a stat-bound reuse of a
    historical full read, never a claim that this invocation reread the bytes.
    """
    def __init__(self, directory=None, snapshot=None, snapshot_sha256=None, checkpoint_guard=None):
        self.directory = Path(directory).resolve() if directory else None
        self.readonly = snapshot is not None
        self.source = small_proof(__file__)
        self.checkpoint_guard = checkpoint_guard
        self.entries, self.cache = {}, {}
        self.counters = {'full_reads': 0, 'full_read_bytes': 0, 'cache_hits': 0, 'invalidations': 0}
        self.events = []
        if self.readonly:
            path = Path(snapshot).resolve(); secure_owner(path); raw = path.read_bytes()
            if not snapshot_sha256 or digest_bytes(raw) != snapshot_sha256:
                raise ValueError('A SHA-bound cache snapshot is required')
            value = json.loads(raw); self._validate_index(value); self.entries = value['entries']
            self.snapshot_proof = {'path': str(path), 'sha256': snapshot_sha256}
        else:
            if self.directory is None:
                raise ValueError('Writable digest cache needs an explicit directory')
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700); secure_owner(self.directory)
            (self.directory / 'receipts').mkdir(exist_ok=True, mode=0o700)
            path = self.directory / 'index.json'
            if path.exists():
                secure_owner(path); value = json.loads(path.read_bytes())
                self._validate_index(value); self.entries = value['entries']

    def _validate_index(self, value):
        if value.get('schema_version') != 1 or value.get('method') != METHOD or value.get('reader_source') != self.source:
            raise ValueError('Unsupported or unreviewed digest cache reader/index')
        if not isinstance(value.get('entries'), dict):
            raise ValueError('Cache index needs an explicit entry mapping')

    def _receipt(self, key):
        if key in self.cache:
            return self.cache[key]
        if key not in self.entries:
            return None
        proof = self.entries[key]; path = Path(proof['path']).resolve(); secure_owner(path); raw = path.read_bytes()
        if digest_bytes(raw) != proof['sha256']:
            raise ValueError('Digest receipt SHA mismatch')
        value = json.loads(raw)
        if (value.get('schema_version') != 1 or value.get('method') != METHOD or
                value.get('reader_source') != self.source or value.get('canonical_path') != key or
                not re.fullmatch(r'[0-9a-f]{64}', value.get('sha256', '')) or
                not isinstance(value.get('stat_before'), dict) or set(value['stat_before']) != set(FIELDS)):
            raise ValueError('Malformed full-read receipt')
        stat = value['stat_before']
        if any(type(stat[k]) is not int or stat[k] < 0 for k in FIELDS):
            raise ValueError('Malformed stat signature')
        if (any(value.get(name) != stat for name in ['fd_before', 'fd_after', 'stat_after']) or
                value.get('read_bytes') != stat['size'] or value.get('complete_to_eof') is not True):
            raise ValueError('Receipt does not establish a complete stable-fd read')
        self.cache[key] = value
        return value

    def _save(self, key, value):
        if self.readonly:
            self.cache[key] = value
            return
        content = encoded(value); path = self.directory / 'receipts' / (digest_bytes(content) + '.json')
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError('Immutable digest receipt collision')
        else:
            atomic(path,value)
        self.entries[key] = {'path': str(path), 'sha256': digest_bytes(content)}
        self.cache[key] = value
        atomic(self.directory / 'index.json', self.index())

    def index(self):
        return {'schema_version': 1, 'method': METHOD, 'reader_source': self.source,
                'entries': self.entries, 'statistics': self.counters,
                'test_or_behavior_scores_read': False, 'tensor_payloads_deserialized': False}

    def snapshot(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError('Choose a new immutable cache snapshot path')
        atomic(path, self.index())
        return small_proof(path)

    def digest(self, path, authority=None):
        requested = Path(path); path = requested.resolve(strict=True); key = str(path)
        if self.checkpoint_guard and '/checkpoints/' in key and path.suffix == '.pt':
            authority = self.checkpoint_guard(path)
        before = signature(path.stat()); previous = self._receipt(key)
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError('Only regular files may be hashed')
        if previous and previous['stat_before'] == before:
            old_authority=previous.get('source_authority',{})
            if old_authority.get('kind')=='completed_training_checkpoint' and authority is not None and authority!=old_authority:
                raise ValueError('Completed checkpoint source authority changed since full verification')
            if signature(path.stat()) != before or requested.resolve(strict=True) != path:
                raise ValueError('Source changed during cached stat verification')
            self.counters['cache_hits'] += 1
            return previous['sha256']
        if previous:
            self.counters['invalidations'] += 1
            self.events.append({'path': key, 'reason': 'five_field_stat_changed'})
        started = dt.datetime.now(dt.timezone.utc).isoformat(); timer = time.perf_counter()
        digest = hashlib.sha256(); length = 0
        with path.open('rb') as stream:
            fd_before = signature(os.fstat(stream.fileno()))
            if before != fd_before:
                raise ValueError('Path changed between stat and open')
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block); length += len(block)
            fd_after = signature(os.fstat(stream.fileno()))
        after = signature(path.stat())
        if before != fd_after or before != after or length != before['size'] or requested.resolve(strict=True) != path:
            raise ValueError('Source changed during complete fd hash')
        if self.checkpoint_guard and '/checkpoints/' in key and path.suffix == '.pt':
            if self.checkpoint_guard(path) != authority:
                raise ValueError('Completed checkpoint source authority changed during hash')
        value = digest.hexdigest()
        old_authority = previous.get('source_authority', {}) if previous else {}
        if previous and old_authority.get('kind') == 'completed_training_checkpoint' and value != previous['sha256']:
            raise ValueError('Completed checkpoint bytes changed after earlier full verification')
        record = {'schema_version': 1, 'method': METHOD, 'canonical_path': key,
                  'stat_before': before, 'fd_before': fd_before, 'fd_after': fd_after, 'stat_after': after,
                  'read_bytes': length, 'complete_to_eof': True, 'sha256': value, 'reader_source': self.source,
                  'started_at_utc': started, 'finished_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'full_read_seconds': time.perf_counter() - timer, 'source_authority': authority or old_authority}
        self.counters['full_reads'] += 1; self.counters['full_read_bytes'] += length
        self._save(key, record)
        return value

    def proof(self, value):
        expected = value['sha256']; path = Path(value['path']).resolve()
        if not re.fullmatch(r'[0-9a-f]{64}', expected) or self.digest(path) != expected:
            raise ValueError(f'Frozen source SHA mismatch: {path}')
        return str(path), expected

    def verify(self, path, expected):
        return self.proof({'path': path, 'sha256': expected})
