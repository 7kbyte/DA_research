#!/usr/bin/env python3
"""Verify frozen behavior inputs before dispatch; no frozen scorer changes."""
import argparse
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def verified_job(path, manifest_sha256, record_id):
    if sha(path) != manifest_sha256:
        raise ValueError('Frozen behavioral manifest SHA mismatch')
    manifest = json.loads(Path(path).read_text())
    matches = [job for job in manifest['jobs'] if job['id'] == record_id]
    if len(matches) != 1:
        raise ValueError('Exactly one frozen job ID required')
    job = matches[0]
    for source, expected in {**manifest['common_input_sha256'], **job['input_sha256']}.items():
        if sha(source) != expected:
            raise ValueError(f'Frozen behavioral input SHA mismatch: {source}')
    if job['kind'] != 'initial':
        initial = next(row for row in manifest['jobs'] if row['kind'] == 'initial')
        baseline = json.loads(Path(initial['output']).read_text())
        if baseline.get('complete') is not True or baseline.get('checkpoint') is not None:
            raise ValueError('Complete initial behavioral baseline required')
        if (baseline['model'], baseline['revision']) != (manifest['inference_policy']['model'], manifest['inference_policy']['revision']):
            raise ValueError('Initial behavioral model/revision differs from manifest')
    return manifest, job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--record-id', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--verify-only', action='store_true', help='CPU hash verification; never load model or infer')
    args = parser.parse_args()
    begin = time.perf_counter()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest, job = verified_job(args.manifest, args.manifest_sha256, args.record_id)
    verification_seconds = time.perf_counter() - begin
    if args.verify_only:
        print(json.dumps({'verified': True, 'job': job['id'], 'model_inference': False}))
        return
    argv = list(job['argv'])
    argv[argv.index('--device') + 1] = args.device
    inference_begin = time.perf_counter()
    subprocess.run(argv, check=True)
    output = Path(job['output'])
    outputs = [output, *[output.with_name(output.stem + f'_{task}.jsonl') for task in manifest['tasks']]]
    record = {'manifest': str(args.manifest), 'manifest_sha256': args.manifest_sha256,
              'job_id': job['id'], 'checkpoint_sha256': job.get('checkpoint_sha256'),
              'verified_frozen_inputs': True, 'actual_argv': argv,
              'started_at_utc': started, 'finished_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'verification_wall_seconds': verification_seconds,
              'inference_subprocess_wall_seconds': time.perf_counter() - inference_begin,
              'output_sha256': {str(path): sha(path) for path in outputs}}
    output.with_name(output.stem + '_execution.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
