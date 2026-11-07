"""Run SHA-bound, already-tested reservation growers in strict priority order."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-sequence-sha256', required=True)
    args = parser.parse_args()
    plan_path = HERE / 'sequence.json'
    if sha(plan_path) != args.expected_sequence_sha256:
        raise ValueError('Sequence SHA mismatch')
    plan = json.loads(plan_path.read_text())
    child = None
    stopped = False
    state = {'pid': os.getpid(), 'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'sequence_sha256': args.expected_sequence_sha256,
             'source_sha256': sha(__file__), 'status': 'starting', 'completed': [],
             'publication_authorized': False, 'gpu_used': False}

    def stop(_signal, _frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        for item in plan['entries']:
            if stopped:
                state['status'] = 'stopped_signal'
                break
            grower = ROOT / plan['grower_source']
            if sha(grower) != plan['grower_sha256']:
                raise ValueError('Reviewed grower source changed')
            if Path(item['receipt']).exists() or Path(item['progress']).exists():
                raise FileExistsError('A reservation was already attempted; no implicit restart')
            argv = ['python', str(grower), '--plan', item['plan'],
                    '--expected-plan-sha256', item['plan_sha256'],
                    '--progress', item['progress'], '--receipt', item['receipt']]
            child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL)
            if stopped:
                child.send_signal(signal.SIGTERM)
            state.update(status='growing_one_reservation', active_name=item['name'],
                         active_pid=child.pid, argv=argv)
            save(HERE / 'sequence.progress.json', state)
            returncode = child.wait()
            child = None
            if returncode != 0:
                raise RuntimeError(f"Grower failed for {item['name']}: rc={returncode}")
            receipt = json.loads(Path(item['receipt']).read_text())
            if not receipt.get('lock_released') or not receipt.get('final_file_fsync_completed'):
                raise ValueError('Grower did not finish synchronization and release its lock')
            if receipt['status'] != 'complete':
                state.update(status='stopped_incomplete_reservation', stopped_name=item['name'],
                             child_status=receipt['status'])
                break
            state['completed'].append({'name': item['name'], 'receipt': item['receipt'],
                                       'receipt_sha256': sha(item['receipt'])})
        else:
            state['status'] = 'complete_all_capacity_targets'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait()
        state.update(active_pid=None, finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                     no_child_left_running=True)
        save(HERE / 'sequence.progress.json', state)
        save(HERE / 'sequence_receipt.json', state)


if __name__ == '__main__':
    main()
