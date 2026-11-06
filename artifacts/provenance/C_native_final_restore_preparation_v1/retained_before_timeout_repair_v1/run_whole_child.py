"""Future inclusive spawn-to-exit timer; never enables authority itself."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'plan-sha256', 'authority', 'authority-sha256', 'evidence-dir'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    plan_raw = Path(args.plan).read_bytes()
    if hashlib.sha256(plan_raw).hexdigest() != args.plan_sha256:
        raise ValueError('Bound plan SHA differs')
    plan = json.loads(plan_raw)
    cap = plan['resource_policy']['max_wall_seconds']
    if type(cap) is not int or not 0 < cap <= 1200:
        raise ValueError('Invalid inclusive whole-child wall cap')
    folder = Path(args.evidence_dir).absolute()
    if folder.exists() or folder.is_symlink():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONUSERBASE'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', PYTHONNOUSERSITE='1')
    runner = Path(__file__).resolve().with_name('restore_native_final.py')
    argv = [sys.executable, '-I', str(runner), 'execute', '--plan', str(Path(args.plan).absolute()),
            '--plan-sha256', args.plan_sha256, '--authority', str(Path(args.authority).absolute()), '--authority-sha256', args.authority_sha256]
    (folder / 'launch.json').write_text(json.dumps(dict(argv=argv, cwd=str(folder), PYTHONPATH_absent=True,
         GPU_hidden=True, wrapper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + '\n')
    before = resource.getrusage(resource.RUSAGE_CHILDREN); started = time.monotonic()
    timed_out = False
    with (folder / 'stdout.log').open('xb') as out, (folder / 'stderr.log').open('xb') as err:
        child = subprocess.Popen(argv, cwd=folder, env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            rc = child.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                rc = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); rc = child.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = dict(returncode=rc, inclusive_wall_cap_seconds=cap, timed_out=timed_out,
        whole_child_process_wall_seconds=time.monotonic()-started,
        whole_child_user_cpu_seconds=after.ru_utime-before.ru_utime,
        whole_child_sys_cpu_seconds=after.ru_stime-before.ru_stime,
        child_peak_RSS_bytes=after.ru_maxrss*1024,
        accounting='Inclusive recovery child spawn-to-exit, including its imports and descendants; runner body/stages/checker/decode are nested. Wrapper preparation excluded.')
    (folder / 'whole_child_timing.json').write_text(json.dumps(result, indent=2) + '\n')
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
