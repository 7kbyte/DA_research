"""Synthetic process-group teardown only: no real restore/bind/model/archive access."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('native_restore_wrapper', HERE/'run_whole_child.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def toy_wrapper(tmp_path, runner_source):
    shutil.copyfile(HERE/'run_whole_child.py',tmp_path/'run_whole_child.py')
    (tmp_path/'restore_native_final.py').write_text(runner_source)
    plan=tmp_path/'plan.json'; authority=tmp_path/'authority.json'; evidence=tmp_path/'evidence'
    plan.write_text(json.dumps(dict(resource_policy=dict(max_wall_seconds=1, deadline_utc=m.PROTECTED_END.isoformat(),
                                                       admission_margin_seconds=300))))
    authority.write_text('{}')  # Ignored by fake script only; real execute is NEVER called.
    argv=[sys.executable,str(tmp_path/'run_whole_child.py'),'--plan',str(plan),'--plan-sha256',hashlib.sha256(plan.read_bytes()).hexdigest(),
          '--authority',str(authority),'--authority-sha256',hashlib.sha256(authority.read_bytes()).hexdigest(),'--evidence-dir',str(evidence)]
    child=subprocess.run(argv,capture_output=True,timeout=20)
    result=json.loads((evidence/'whole_child_timing.json').read_text())
    return child,result


@pytest.mark.parametrize('leader_ignores_TERM',[False,True])
def test_end_to_end_timeout_cleans_TERM_ignoring_descendant_and_never_exits_zero(tmp_path,leader_ignores_TERM):
    source="""import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
HANDLER
code="import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path("+repr(str(root/'child_ready'))+").write_text(str(os.getpid()));time.sleep(60)"
child=subprocess.Popen([sys.executable,'-c',code])
(root/'group.json').write_text(json.dumps(dict(pgid=os.getpgrp(),child_pid=child.pid)))
while not (root/'child_ready').exists():time.sleep(.01)
time.sleep(60)
""".replace('HANDLER',"signal.signal(signal.SIGTERM,signal.SIG_IGN)" if leader_ignores_TERM else "signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))")
    group=None
    try:
        child,result=toy_wrapper(tmp_path,source)
        group=json.loads((tmp_path/'group.json').read_text())
        assert child.returncode==result['returncode']==124
        assert result['timed_out'] is True
        assert result['direct_child_returncode']==(-signal.SIGKILL if leader_ignores_TERM else 0)
        cleanup=result['process_group_cleanup']
        assert cleanup['subreaper_enabled'] and cleanup['whole_group_absence_confirmed']
        assert [r['signal'] for r in cleanup['signals']]==['SIGTERM','SIGKILL']
        assert not cleanup['final_observation']['killpg_0_exists'] and cleanup['final_observation']['members']==[]
        assert not Path('/proc',str(group['child_pid'])).exists(), 'Descendant including zombie must be absent'
        assert any(r['pid']==group['child_pid'] for r in cleanup['adopted_zombies_reaped'])
        assert result['whole_child_process_wall_seconds']>=1+m.TERM_GRACE_SECONDS
        assert result['whole_child_process_wall_seconds']<1+m.TERM_GRACE_SECONDS+m.KILL_GRACE_SECONDS+2
        assert result['admission']['teardown_budget_seconds']<=result['admission']['admission_margin_seconds']
    finally:
        if group is None and (tmp_path/'group.json').exists():group=json.loads((tmp_path/'group.json').read_text())
        if group:
            try:os.killpg(group['pgid'],signal.SIGKILL)
            except ProcessLookupError:pass


def test_completed_toy_child_without_descendants_remains_success(tmp_path):
    child,result=toy_wrapper(tmp_path,"print('synthetic direct child only')\n")
    assert child.returncode==result['returncode']==0 and result['timed_out'] is False
    assert result['process_group_cleanup']['whole_group_absence_confirmed']
    assert result['process_group_cleanup']['signals']==[]


def test_unresolved_D_state_is_an_explicit_failed_cleanup_not_absence(monkeypatch):
    class Child:
        pid=999999999
        def poll(self):return 0
    observation=dict(killpg_0_exists=True,members=[dict(pid=999999998,ppid=1,state='D',pgid=Child.pid,session=Child.pid,start_time_ticks=1)])
    monkeypatch.setattr(m,'group_observation',lambda _:observation)
    signals=[];monkeypatch.setattr(m.os,'killpg',lambda pid,number:signals.append(number))
    result=m.cleanup_group(Child(),term_grace=0,kill_grace=0)
    assert result['whole_group_absence_confirmed'] is False and result['status'].startswith('FAILED_')
    assert result['final_observation']['members'][0]['state']=='D'
    assert signals==[signal.SIGTERM,signal.SIGKILL]


def test_admission_counts_group_teardown_within_protected_tail():
    policy=dict(max_wall_seconds=600,deadline_utc=m.PROTECTED_END.isoformat(),admission_margin_seconds=300)
    with pytest.raises(ValueError,match='protected tail'):
        m.admission(policy,m.PROTECTED_END-dt.timedelta(seconds=900))
    admitted=m.admission(policy,m.PROTECTED_END-dt.timedelta(seconds=901))
    assert admitted['required_remaining_seconds']==900 and admitted['teardown_budget_seconds']==12
    with pytest.raises(ValueError):m.admission(dict(policy,admission_margin_seconds=9),m.PROTECTED_END-dt.timedelta(hours=1))
    with pytest.raises(ValueError):m.admission(dict(policy,deadline_utc=(m.PROTECTED_END+dt.timedelta(seconds=1)).isoformat()))
