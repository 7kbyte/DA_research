#!/usr/bin/env python3
"""Retained synthetic-only bundle reproduction from the reviewed source snapshot."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import torch
from pytest import MonkeyPatch

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(exist_ok=False)
fixture = HERE / 'source_snapshot/tests/test_additional_endpoint_export.py'
spec = importlib.util.spec_from_file_location('retained_tiny_fixture', fixture)
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
e = tests.e
with MonkeyPatch.context() as patch:
    item, base, baseline, saved = tests.tiny.__wrapped__(OUT, patch)
    folder = OUT / 'delivery'
    record = e.export_one('tiny', item, folder, base, e.file_sha256(base), baseline)
    scope = tests.put(OUT / 'synthetic_scope.json', {'synthetic_only': True})
    patch.setitem(e.PACKAGES, 'tiny', dict(logical=1, limit_bytes=32 << 20,
                                       model='tiny', kind='best', slot='slot1'))
    phase = {'rows': [item], 'base_files': [dict(path=str(base),
             member='base/tiny-final/model.safetensors', sha256=e.file_sha256(base))]}
    class Cache:
        def stable(self):
            pass
    receipt = e.assemble('tiny', phase, [record], folder, scope, Cache(), {'sources': []})
    unpack = OUT / 'unpacked'
    unpack.mkdir()
    with tarfile.open(receipt['tar_path']) as package:
        package.extractall(unpack)
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    command = [sys.executable, str(unpack / 'tools/scripts/archive_model_weights.py'),
               '--threads', '2', 'decode', '--archive', str(unpack / record['archive_member']),
               '--base', str(unpack / 'base/tiny-final/model.safetensors'),
               '--output', str(unpack / 'portable.pt')]
    child = subprocess.run(command, cwd=unpack, env=env, capture_output=True, text=True)
    (OUT / 'portable_cli_stdout.json').write_text(child.stdout)
    (OUT / 'portable_cli_stderr.txt').write_text(child.stderr)
    if child.returncode:
        raise RuntimeError(f'portable CLI failed: {child.returncode}')
    restored = torch.load(unpack / 'portable.pt', map_location='cpu', weights_only=False)['model_state']
    assert set(restored) == set(saved['model_state'])
    rows = []
    for key, original in sorted(saved['model_state'].items()):
        source = original.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        decoded = restored[key].contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        assert source == decoded
        rows.append(dict(key=key, dtype=str(original.dtype), shape=list(original.shape),
                         elements=original.numel(), bytes=len(source),
                         source_raw_sha256=hashlib.sha256(source).hexdigest(),
                         portable_decoded_raw_sha256=hashlib.sha256(decoded).hexdigest(),
                         all_bytes_equal=True))
    report = dict(status='passed_synthetic_full_bundle_native_base_CLI', argv=command,
                  cwd=str(unpack), PYTHONPATH_removed=True,
                  source_exporter=dict(path=str(Path(e.__file__).resolve()),
                  sha256=e.file_sha256(e.__file__)), source_checkpoint=str(item['bound']['checkpoint']),
                  source_checkpoint_sha256=e.file_sha256(item['bound']['checkpoint']),
                  base_sha256=e.file_sha256(base), tensor_count=len(rows),
                  model_elements=sum(x['elements'] for x in rows), tensors=rows,
                  includes_NaN_and_signed_zero_bits=True, package_receipt=receipt,
                  actual_study_payload_read=False, GPU_used=False)
    (OUT / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(status=report['status'], tensor_count=len(rows), output=str(OUT))))
