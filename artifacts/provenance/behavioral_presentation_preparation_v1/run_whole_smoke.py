"""Inclusive parent timer for the hand-made five-phase presentation smoke."""
from pathlib import Path
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
argv=[sys.executable,str(HERE/'synthetic_smoke.py')]
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.perf_counter()
with (HERE/'whole_smoke_stdout.log').open('xb') as stdout,(HERE/'whole_smoke_stderr.log').open('xb') as stderr:
    child=subprocess.run(argv,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
after=resource.getrusage(resource.RUSAGE_CHILDREN)
value={'argv':argv,'returncode':child.returncode,'wall_seconds':time.perf_counter()-started,
    'user_cpu_seconds':after.ru_utime-before.ru_utime,'system_cpu_seconds':after.ru_stime-before.ru_stime,
    'scope':'Complete synthetic fixture generation + all five presentation children, source/CSV checks, PNG/PDF/SVG output and hashes',
    'nested_phase_presentation_timers_not_additive':True,'CPU_threads':2,'CUDA_VISIBLE_DEVICES':'',
    'actual_behavior_score_or_prediction_read':False,'actual_original_check_analyze_or_inference_run':False,
    'source_sha256':hashlib.sha256((HERE/'synthetic_smoke.py').read_bytes()).hexdigest()}
with (HERE/'whole_smoke_timing.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
raise SystemExit(child.returncode)
