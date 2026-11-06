#!/usr/bin/env python3
"""Create a verified research archive, including run records but excluding external weights."""
import argparse, hashlib, io, json, os, tarfile
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

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

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--runs-root', type=Path)
    p.add_argument('--label', default='intermediate_work_in_progress')
    args = p.parse_args(); output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.exists(): raise ValueError('Use a new archive outside the worktree')
    output.parent.mkdir(parents=True, exist_ok=True)
    paths, omitted, external = [], [], []
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if any(x in {'__pycache__', '.pytest_cache', '.venv'} for x in relative.parts): continue
        if path.is_symlink(): omitted.append({'path': str(relative), 'target': os.readlink(path)})
        elif path.is_file() and path.suffix not in {'.pid', '.lock', '.tmp'}: paths.append((path, str(relative)))
    if args.runs_root:
        for path in sorted(args.runs_root.rglob('*')):
            if not path.is_file(): continue
            relative = path.relative_to(args.runs_root)
            if path.suffix in {'.json', '.jsonl', '.log'} and 'checkpoints' not in relative.parts:
                paths.append((path, 'run_records/' + str(relative)))
            elif path.suffix in {'.pt', '.safetensors', '.bin'}:
                stat = path.stat()
                external.append({'source': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                 'sha256': None, 'bundled': False})
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

