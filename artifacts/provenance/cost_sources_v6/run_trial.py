"""One score-blind invocation of the retained cost collector, fully timed."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def proof(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    source = ROOT / 'configs/cost_inventory_final_sources_v6.json'
    collector = ROOT / 'scripts/inventory_study_costs_final_v2.py'
    if proof(source)['sha256'] != 'e4c99db9b1803681ad8706bb90480ede0be584975324040df979c1c9966b17e0':
        raise ValueError('Explicit prepared v6 source changed')
    if proof(collector)['sha256'] != '864effb89f20dbfce3ba6ce51db8eb8351b087c2ac92d255274fe6267753f6ae':
        raise ValueError('Retained collector changed')
    output = ROOT / 'results/cost_inventory_final_v6_cpu_trial_20261106'
    if output.exists():
        raise FileExistsError('Immutable new trial output required')
    argv = [sys.executable, str(collector), '--sources', str(source), '--output', str(output)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    with (HERE / 'trial_stdout.log').open('xb') as stdout, (HERE / 'trial_stderr.log').open('xb') as stderr:
        child = subprocess.run(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    timing = {'schema_version': 1, 'argv': argv, 'returncode': child.returncode,
              'whole_child_wall_seconds': elapsed,
              'whole_child_user_cpu_seconds': after.ru_utime - before.ru_utime,
              'whole_child_sys_cpu_seconds': after.ru_stime - before.ru_stime,
              'collector': proof(collector), 'source_spec': proof(source),
              'stdout': proof(HERE / 'trial_stdout.log'), 'stderr': proof(HERE / 'trial_stderr.log'),
              'scope': 'This one complete cost-collector child spawn-to-exit; not automatically added to its own ledger or future unmeasured engineering.',
              'GPU_used': False, 'score_semantics_read': False}
    with (HERE / 'trial_whole_child_timing.json').open('x') as f:
        json.dump(timing, f, ensure_ascii=False, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    if child.returncode:
        raise RuntimeError('Collector failed; preserve the failed attempt before a separately named retry')
    print(json.dumps({'timer': proof(HERE / 'trial_whole_child_timing.json'),
                      'inventory': proof(output / 'inventory.json'), 'receipt': proof(output / 'collection_receipt.json')}))


if __name__ == '__main__':
    main()
