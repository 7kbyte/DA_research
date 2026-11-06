"""Re-run the original isolated toy against the fixed wrapper, plus pure gates."""
from pathlib import Path
import datetime as dt
import hashlib
import importlib.util
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
source=ROOT/'artifacts/provenance/C_native_final_restore_preparation_v1/run_whole_child.py'
expected_sha='2a2d09daac8504b099a5cd9682dff722bbad0519922ba39beedf4e96e9498266'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected_sha
before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.perf_counter()
old=json.loads((OUT/'timeout_probe.json').read_text())
with tempfile.TemporaryDirectory(prefix='native_restore_review_fixed_timeout_') as temp:
    path=Path(temp);shutil.copyfile(source,path/'run_whole_child.py')
    # Exactly the same independent fake runner as the first failure probe.
    (path/'restore_native_final.py').write_text(old['fake_runner_source'])
    # Its path is resolved at runtime; no old temporary directory is reused.
    policy={'max_wall_seconds':1,'deadline_utc':'2026-11-07T02:56:32+00:00','admission_margin_seconds':300}
    raw=json.dumps({'resource_policy':policy}).encode()
    (path/'plan.json').write_bytes(raw);(path/'authority.json').write_text('{"authorized":false,"SYNTHETIC_ONLY":true}')
    argv=[sys.executable,str(path/'run_whole_child.py'),'--plan',str(path/'plan.json'),'--plan-sha256',hashlib.sha256(raw).hexdigest(),
        '--authority',str(path/'authority.json'),'--authority-sha256',hashlib.sha256((path/'authority.json').read_bytes()).hexdigest(),'--evidence-dir',str(path/'evidence')]
    group=None
    try:
        child=subprocess.run(argv,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True,timeout=20)
        group=json.loads((path/'group.json').read_text())
        timer=json.loads((path/'evidence/whole_child_timing.json').read_text())
        assert child.returncode==timer['returncode']==124 and timer['timed_out'] is True
        assert timer['direct_child_returncode']==0
        cleanup=timer['process_group_cleanup']
        assert cleanup['whole_group_absence_confirmed'] is True
        assert cleanup['final_observation']['killpg_0_exists'] is False and cleanup['final_observation']['members']==[]
        assert [s['signal'] for s in cleanup['signals']]==['SIGTERM','SIGKILL']
        assert not Path('/proc',str(group['child_pid'])).exists()
        assert any(p['pid']==group['child_pid'] for p in cleanup['adopted_zombies_reaped'])
        assert timer['whole_child_process_wall_seconds']>=1+cleanup['cleanup_wall_seconds']-.05
        assert timer['whole_child_process_wall_seconds']<13
        fixed={'returncode':child.returncode,'whole_child_timing':timer,'independently_observed_descendant_absent':True}
    finally:
        if group is None and (path/'group.json').exists():group=json.loads((path/'group.json').read_text())
        if group:
            try:os.killpg(group['pgid'],signal.SIGKILL)
            except ProcessLookupError:pass

spec=importlib.util.spec_from_file_location('independent_fixed_wrapper',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
gates=[]
for model,cap in [('pythia-1b',600),('pythia-2.8b',1200)]:
    policy={'max_wall_seconds':cap,'deadline_utc':module.PROTECTED_END.isoformat(),'admission_margin_seconds':300}
    boundary=module.PROTECTED_END-dt.timedelta(seconds=cap+300)
    try:module.admission(policy,boundary)
    except ValueError:rejected=True
    else:rejected=False
    assert rejected
    admitted=module.admission(policy,boundary-dt.timedelta(seconds=1))
    assert admitted['teardown_budget_seconds']==12 and admitted['required_remaining_seconds']==cap+300
    gates.append({'model':model,'execution_cap_seconds':cap,'strict_latest_start_utc':boundary.isoformat(),
        'at_boundary_rejected':True,'one_second_before_admitted':True,'admission':admitted,
        'cap_plus_teardown_ends_by_utc':(boundary+dt.timedelta(seconds=cap+12)).isoformat()})

# A simulated persistent kernel D-state; no real stuck I/O or process is made.
class FakeChild:
    pid=999999999
    def poll(self):return 0
observation={'killpg_0_exists':True,'members':[{'pid':999999998,'state':'D','ppid':1,'pgid':FakeChild.pid,'session':FakeChild.pid,'start_time_ticks':1}]}
signals=[]
with patch.object(module,'group_observation',return_value=observation),patch.object(module.os,'killpg',side_effect=lambda pgid,number:signals.append(number)):
    unresolved=module.cleanup_group(FakeChild(),term_grace=0,kill_grace=0)
assert unresolved['whole_group_absence_confirmed'] is False and unresolved['status'].startswith('FAILED_')
assert unresolved['final_observation']['members'][0]['state']=='D' and signals==[signal.SIGTERM,signal.SIGKILL]
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected_sha
after=resource.getrusage(resource.RUSAGE_CHILDREN)
result={'synthetic_only':True,'actual_restore_runner_or_bind_execute_called':False,'actual_payload_bytes_read':0,
    'wrapper_source':{'path':str(source),'sha256':expected_sha},'original_failure_probe_sha256':hashlib.sha256((OUT/'timeout_probe.json').read_bytes()).hexdigest(),
    'same_fake_runner_retested':True,'fixed_timeout_probe':fixed,'admission_boundaries':gates,
    'simulated_D_state_unconfirmed_cleanup':unresolved,
    'limits':'No real hung NFS test. Uninterruptible D-state cleanup is not guaranteed; unresolved groups explicitly fail and CPU totals can be incomplete.',
    'whole_probe_parent_timing':{'wall_seconds':time.perf_counter()-start,'user_cpu_seconds':after.ru_utime-before.ru_utime,'system_cpu_seconds':after.ru_stime-before.ru_stime,
        'scope':'Complete independent toy wrapper subprocess, pure admission checks, mock D-state and cleanup; wrapper child time is nested, not additive'}}
with (OUT/'timeout_fix_probe.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'passed':True,'returncode':fixed['returncode'],'descendant_absent':True,'whole_probe_parent_timing':result['whole_probe_parent_timing']}))
