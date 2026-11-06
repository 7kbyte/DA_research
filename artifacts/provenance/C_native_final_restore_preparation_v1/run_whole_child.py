"""Future inclusive spawn-to-exit timer; never enables authority itself."""
import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

PROTECTED_END = dt.datetime(2026, 11, 7, 2, 56, 32, tzinfo=dt.timezone.utc)
TERM_GRACE_SECONDS = 5.0
KILL_GRACE_SECONDS = 5.0
TEARDOWN_OVERHEAD_ALLOWANCE_SECONDS = 2.0


def admission(policy, now=None):
    cap = policy['max_wall_seconds']
    if type(cap) is not int or not 0 < cap <= 1200:
        raise ValueError('Invalid inclusive whole-child wall cap')
    deadline = dt.datetime.fromisoformat(policy['deadline_utc'])
    margin = policy['admission_margin_seconds']
    teardown = TERM_GRACE_SECONDS + KILL_GRACE_SECONDS + TEARDOWN_OVERHEAD_ALLOWANCE_SECONDS
    if deadline.tzinfo is None or deadline > PROTECTED_END or type(margin) is not int or margin < 300:
        raise ValueError('Protected delivery deadline/admission margin differs')
    now = now or dt.datetime.now(dt.timezone.utc)
    required = cap + max(margin, teardown)
    if (deadline - now).total_seconds() <= required:
        raise ValueError('Insufficient cap plus whole-process-group teardown protected tail')
    return dict(cap_seconds=cap, TERM_grace_seconds=TERM_GRACE_SECONDS, KILL_grace_seconds=KILL_GRACE_SECONDS,
                teardown_overhead_allowance_seconds=TEARDOWN_OVERHEAD_ALLOWANCE_SECONDS,
                teardown_budget_seconds=teardown, admission_margin_seconds=margin,
                required_remaining_seconds=required, deadline_utc=deadline.isoformat())


def enable_subreaper():
    """Adopt orphaned grandchildren so this wrapper, rather than PID1, can reap them."""
    libc = ctypes.CDLL(None, use_errno=True)
    enabled = ctypes.c_int()
    if libc.prctl(36, 1, 0, 0, 0) != 0 or libc.prctl(37, ctypes.byref(enabled), 0, 0, 0) != 0 or enabled.value != 1:
        raise OSError(ctypes.get_errno(), 'Linux child subreaper setup failed; refuse to launch')


def group_observation(pgid):
    try:
        os.killpg(pgid, 0)
        exists = True
    except ProcessLookupError:
        exists = False
    members = []
    if exists:
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if int(fields[2]) == pgid:
                members.append(dict(pid=int(path.name), state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]),
                                    session=int(fields[3]), start_time_ticks=int(fields[19])))
    return dict(killpg_0_exists=exists, members=sorted(members, key=lambda r: r['pid']))


def cleanup_group(child, term_grace=TERM_GRACE_SECONDS, kill_grace=KILL_GRACE_SECONDS):
    """Direct-child exit never substitutes for checking the entire original process group."""
    pgid = child.pid; reaped = []; snapshots = []; signals = []
    def inspect():
        child.poll()  # Popen owns/reaps the direct child's status.
        observation = group_observation(pgid)
        for item in observation['members']:
            if item['pid'] != child.pid and item['ppid'] == os.getpid() and item['state'] == 'Z':
                try:
                    pid, status = os.waitpid(item['pid'], os.WNOHANG)
                except ChildProcessError:
                    continue
                if pid:
                    reaped.append(dict(pid=pid, wait_status=status, returncode=os.waitstatus_to_exitcode(status)))
        observation = group_observation(pgid)
        snapshots.append(dict(since_cleanup_start_seconds=time.monotonic()-started, **observation))
        return not observation['killpg_0_exists'] and not observation['members']
    def send(number):
        try:
            os.killpg(pgid, number); outcome = 'sent_to_whole_group'
        except ProcessLookupError:
            outcome = 'already_absent'
        signals.append(dict(signal=signal.Signals(number).name, outcome=outcome,
                            since_cleanup_start_seconds=time.monotonic()-started))
    def await_absence(seconds):
        until = time.monotonic() + seconds
        while True:
            if inspect():
                return True
            remaining = until-time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))
    started = time.monotonic()
    if inspect():
        status = 'group_absent_no_cleanup_needed'
    else:
        send(signal.SIGTERM)
        if await_absence(term_grace):
            status = 'whole_group_absence_confirmed_after_TERM'
        else:
            # Even if the direct runner already exited0, remaining descendants must get KILL.
            send(signal.SIGKILL)
            status = ('whole_group_absence_confirmed_after_KILL' if await_absence(kill_grace)
                      else 'FAILED_group_absence_unconfirmed_live_D_or_unreaped_members')
    return dict(status=status, whole_group_absence_confirmed=status != 'FAILED_group_absence_unconfirmed_live_D_or_unreaped_members',
                pgid=pgid, subreaper_enabled=True, signals=signals, snapshots=snapshots,
                adopted_zombies_reaped=reaped, final_observation=snapshots[-1],
                cleanup_wall_seconds=time.monotonic()-started,
                boundary='SIGKILL cannot guarantee immediate removal of uninterruptible D-state tasks; unresolved groups are explicit failures, never success')


def run_group(argv, cwd, env, stdout, stderr, cap, term_grace=TERM_GRACE_SECONDS, kill_grace=KILL_GRACE_SECONDS):
    enable_subreaper()
    before = resource.getrusage(resource.RUSAGE_CHILDREN); started = time.monotonic(); timed_out = False
    child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    try:
        child.wait(timeout=cap)
    except subprocess.TimeoutExpired:
        timed_out = True
    cleanup = cleanup_group(child, term_grace, kill_grace)
    direct_rc = child.poll()
    orphan_cleanup_needed = bool(cleanup['signals'])
    if not cleanup['whole_group_absence_confirmed']:
        rc = 125
    elif timed_out:
        rc = 124
    elif orphan_cleanup_needed:
        rc = 125
    else:
        rc = direct_rc
    if type(rc) is not int:
        rc = 125
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(returncode=rc, direct_child_returncode=direct_rc, timed_out=timed_out,
        inclusive_execution_cap_seconds=cap, execution_plus_cleanup_budget_seconds=cap+term_grace+kill_grace,
        process_group_cleanup=cleanup, whole_child_process_wall_seconds=time.monotonic()-started,
        whole_child_user_cpu_seconds=after.ru_utime-before.ru_utime,
        whole_child_sys_cpu_seconds=after.ru_stime-before.ru_stime, child_peak_RSS_bytes=after.ru_maxrss*1024,
        accounting='Inclusive child spawn-to-exit plus full process-group cleanup/reaping. Descendant CPU is attributed when reaped; failed unresolved groups may have incomplete CPU totals. Body/stages are nested.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'plan-sha256', 'authority', 'authority-sha256', 'evidence-dir'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    plan_raw = Path(args.plan).read_bytes()
    if hashlib.sha256(plan_raw).hexdigest() != args.plan_sha256:
        raise ValueError('Bound plan SHA differs')
    plan = json.loads(plan_raw)
    admitted = admission(plan['resource_policy'])
    cap = admitted['cap_seconds']
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
         GPU_hidden=True, admission=admitted, wrapper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + '\n')
    # Recheck immediately before spawn; cap + bounded cleanup lies within the protected margin.
    admission(plan['resource_policy'])
    with (folder / 'stdout.log').open('xb') as out, (folder / 'stderr.log').open('xb') as err:
        result = run_group(argv, folder, env, out, err, cap)
    result['admission'] = admitted
    (folder / 'whole_child_timing.json').write_text(json.dumps(result, indent=2) + '\n')
    raise SystemExit(result['returncode'])


if __name__ == '__main__':
    main()
