"""Bounded toy-only process-group probe; no actual bind/execute or package read."""
from pathlib import Path
import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
source=ROOT/'artifacts/provenance/C_native_final_restore_preparation_v1/run_whole_child.py'
before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='native_restore_review_timeout_') as temp:
    path=Path(temp)
    shutil.copyfile(source,path/'run_whole_child.py')
    # This is a fake runner, never the actual restore runner.
    fake='''import os,signal,subprocess,sys,time\nfrom pathlib import Path\np=Path(__file__).resolve().parent\nsignal.signal(signal.SIGTERM,lambda *_:sys.exit(0))\nchild_code="import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path("+repr(str(p/'child_ready.json'))+").write_text(str(os.getpid()));time.sleep(30)"\nchild=subprocess.Popen([sys.executable,'-c',child_code])\n(p/'group.json').write_text(__import__('json').dumps({'pgid':os.getpgrp(),'child_pid':child.pid}))\nwhile not (p/'child_ready.json').exists():time.sleep(.01)\ntime.sleep(30)\n'''
    (path/'restore_native_final.py').write_text(fake)
    raw=json.dumps({'resource_policy':{'max_wall_seconds':1}}).encode()
    (path/'plan.json').write_bytes(raw);(path/'authority.json').write_text('{"authorized":false,"SYNTHETIC_ONLY":true}')
    argv=[sys.executable,str(path/'run_whole_child.py'),'--plan',str(path/'plan.json'),'--plan-sha256',hashlib.sha256(raw).hexdigest(),
        '--authority',str(path/'authority.json'),'--authority-sha256',hashlib.sha256((path/'authority.json').read_bytes()).hexdigest(),'--evidence-dir',str(path/'evidence')]
    group=None
    try:
        child=subprocess.run(argv,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True,timeout=10)
        group=json.loads((path/'group.json').read_text())
        proc=Path('/proc')/str(group['child_pid'])/'stat'
        state=proc.read_text().split()[2] if proc.exists() else 'absent'
        timer=json.loads((path/'evidence/whole_child_timing.json').read_text())
        result={'synthetic_only':True,'actual_restore_runner_or_bind_execute_called':False,'actual_payload_bytes_read':0,
            'wrapper_source':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
            'fake_runner_source':fake,'wrapper_returncode':child.returncode,'whole_child_timing':timer,
            'descendant_state_after_wrapper_return':state,'descendant_survived':state not in ('absent','Z'),
            'observation':'SIGTERM-exiting runner can leave a SIGTERM-ignoring descendant alive after wrapper returns; current timeout exit can also be zero.'}
        assert timer['timed_out'] and child.returncode==0 and result['descendant_survived']
    finally:
        if group is None and (path/'group.json').exists():group=json.loads((path/'group.json').read_text())
        if group:
            try:os.killpg(group['pgid'],signal.SIGKILL)
            except ProcessLookupError:pass
            for _ in range(100):
                proc=Path('/proc')/str(group['child_pid'])/'stat'
                state=proc.read_text().split()[2] if proc.exists() else 'absent'
                if state in ('absent','Z'):break
                time.sleep(.01)
            result['probe_cleanup_descendant_state']=state
            assert state in ('absent','Z')
after=resource.getrusage(resource.RUSAGE_CHILDREN)
result['whole_probe_parent_timing']={'wall_seconds':time.perf_counter()-start,'user_cpu_seconds':after.ru_utime-before.ru_utime,'system_cpu_seconds':after.ru_stime-before.ru_stime,
    'scope':'Complete synthetic wrapper process and cleanup; no actual recovery'}
with (OUT/'timeout_probe.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({k:result[k] for k in ('wrapper_returncode','descendant_survived','probe_cleanup_descendant_state','whole_probe_parent_timing')}))
