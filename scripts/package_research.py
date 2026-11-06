#!/usr/bin/env python3
"""Create a verified research archive, including run records but excluding external weights."""
import argparse, hashlib, io, json, os, tarfile
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_LOCK_NAMES = {'requirements-report.lock', 'uv.lock', 'poetry.lock', 'Pipfile.lock', 'pdm.lock'}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''): h.update(data)
    return h.hexdigest()

def stable_read(path):
    for _ in range(3):
        before = path.stat(); data = path.read_bytes(); after = path.stat()
        if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns): return data, after
    raise RuntimeError(f'File changed during snapshot: {path}')

def collect_paths(root, runs_root=None, *, require_run_records=False, fail_on_symlinks=False):
    """Enumerate without opening payloads; keep data tensors and dependency locks.

    Run metadata lives outside the worktree. Explicit final-delivery flags make
    its omission or a future unmaterialized symlink a preflight failure.
    """
    root = Path(root)
    if require_run_records and runs_root is None:
        raise ValueError('Final package requires an explicit --runs-root')
    if runs_root is not None:
        runs_root = Path(runs_root)
        if not runs_root.is_dir() or runs_root.is_symlink():
            raise ValueError('runs-root must be an existing non-symlink directory')
    paths, omitted, external = [], [], []
    selected_run_records = 0
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(x in {'__pycache__', '.pytest_cache', '.venv'} for x in relative.parts): continue
        if path.is_symlink(): omitted.append({'path': str(relative), 'target': os.readlink(path)})
        elif path.is_file() and (path.suffix not in {'.pid', '.lock', '.tmp'} or path.name in DEPENDENCY_LOCK_NAMES):
            paths.append((path, str(relative)))
    if runs_root:
        for path in sorted(runs_root.rglob('*')):
            relative = path.relative_to(runs_root)
            if path.is_symlink():
                omitted.append({'path': 'run_records/' + str(relative), 'target': os.readlink(path)})
                continue
            if not path.is_file(): continue
            if path.suffix in {'.json', '.jsonl', '.log'} and 'checkpoints' not in relative.parts:
                paths.append((path, 'run_records/' + str(relative)))
                selected_run_records += 1
            elif path.suffix in {'.pt', '.safetensors', '.bin'}:
                stat = path.stat()
                external.append({'source': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                 'sha256': None, 'bundled': False})
    if fail_on_symlinks and omitted:
        raise ValueError('Unmaterialized symlinks would be omitted: ' + ', '.join(r['path'] for r in omitted))
    names = [name for _, name in paths]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate archive member names')
    if require_run_records and not selected_run_records:
        raise ValueError('No external run records selected')
    return paths, omitted, external

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--runs-root', type=Path)
    p.add_argument('--require-run-records', action='store_true')
    p.add_argument('--fail-on-symlinks', action='store_true')
    p.add_argument('--label', default='intermediate_work_in_progress')
    args = p.parse_args(); output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.exists(): raise ValueError('Use a new archive outside the worktree')
    paths, omitted, external = collect_paths(ROOT, args.runs_root, require_run_records=args.require_run_records,
                                            fail_on_symlinks=args.fail_on_symlinks)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_archive(args, output, paths, omitted, external)

def write_archive(args, output, paths, omitted, external):
    temporary = output.with_name(output.name + '.partial')
    if temporary.exists(): raise FileExistsError(temporary)
    stamp = datetime.now(timezone.utc).isoformat(); records = []
    with tarfile.open(temporary, 'w:gz', compresslevel=6) as archive:
        for path, name in paths:
            data, stat = stable_read(path)
            entry = tarfile.TarInfo('research/' + name)
            entry.size, entry.mode, entry.mtime = len(data), stat.st_mode & 0o777, int(stat.st_mtime)
            archive.addfile(entry, io.BytesIO(data))
            records.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                            'source_mtime_ns': stat.st_mtime_ns})
        manifest = {'created_at_utc': stamp, 'label': args.label, 'source_workspace': str(ROOT),
                    'consistency': 'Each file stable while read; not a simultaneous process snapshot.',
                    'files': records, 'omitted_symlinks': omitted, 'external_weights_not_included': external}
        data = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode()
        entry = tarfile.TarInfo('research/PACKAGE_MANIFEST.json'); entry.size, entry.mode = len(data), 0o644
        archive.addfile(entry, io.BytesIO(data))
    with tarfile.open(temporary, 'r:gz') as archive:
        for record in records:
            data = archive.extractfile('research/' + record['path']).read()
            if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                raise RuntimeError('Archive payload verification failed: ' + record['path'])
    with temporary.open('rb') as stream: os.fsync(stream.fileno())
    temporary.rename(output)
    receipt = {'archive': str(output), 'bytes': output.stat().st_size, 'sha256': sha(output),
               'files': len(records), 'external_weights': len(external), 'created_at_utc': stamp,
               'label': args.label, 'verified_all_payloads': True}
    output.with_name(output.name + '.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))

if __name__ == '__main__': main()

