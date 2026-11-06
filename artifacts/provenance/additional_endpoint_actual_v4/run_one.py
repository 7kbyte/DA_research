"""Time one independently authorized CPU exporter child; no automatic restart."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

spec_path = Path(sys.argv[1])
spec = json.loads(spec_path.read_text())
folder = Path(spec['logs'])
folder.mkdir(parents=True, exist_ok=False)
env = dict(os.environ, **spec['env'])
assert env['CUDA_VISIBLE_DEVICES'] == ''

def write(name, payload):
    with (folder / name).open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

before = resource.getrusage(resource.RUSAGE_CHILDREN)
started = time.monotonic()
with (folder / 'stdout.log').open('x') as out, (folder / 'stderr.log').open('x') as err:
    child = subprocess.Popen(spec['argv'], cwd=spec['cwd'], env=env, stdout=out, stderr=err)
    write('child_launch.json', dict(pid=child.pid, wrapper_pid=os.getpid(), argv=spec['argv'],
          spec_path=str(spec_path), spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
          started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    code = child.wait()
after = resource.getrusage(resource.RUSAGE_CHILDREN)
write('whole_child_timing.json', dict(returncode=code, wall_seconds=time.monotonic()-started,
      user_cpu_seconds=after.ru_utime-before.ru_utime, sys_cpu_seconds=after.ru_stime-before.ru_stime,
      child_peak_rss_bytes=after.ru_maxrss*1024,
      finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
      timing_scope='Whole single exporter child spawn through exit; startup/endpoint/assembly timers are nested, not additive',
      GPU_used=False, publication_performed=False))
raise SystemExit(code)
