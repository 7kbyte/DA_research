"""ROOT-only metadata snapshot of one complete frozen analysis closure."""
from pathlib import Path
import argparse, datetime as dt, hashlib, json, os, sys

R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(R / 'scripts'))
from schedule_jobs import digest_job

p = argparse.ArgumentParser()
p.add_argument('--plan', required=True)
p.add_argument('--plan-sha256', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
plan_path = Path(a.plan).resolve()
raw_plan = plan_path.read_bytes()
assert hashlib.sha256(raw_plan).hexdigest() == a.plan_sha256
plan = json.loads(raw_plan)
assert not Path(plan['first_read_receipt_path']).exists(), 'First read already recorded'
source = R / 'artifacts/provenance/final_evaluation_handoff_actual_v1/successor/state.json'
queue_path = source.with_name('queue.json')
raw_queue = queue_path.read_bytes()
assert hashlib.sha256(raw_queue).hexdigest() == '5468a5641cfcc097f2c84c2224df78bdf94fea92a42411b8ef077b951772b8f7'
queue = {j['id']: j for j in json.loads(raw_queue)}
assert len(queue) == 514

def signature(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]

before = signature(source)
raw = source.read_bytes()
after = signature(source)
assert before == after, 'Actual scheduler state changed during read'
state = json.loads(raw)
selected = {}
for item in plan['closure_jobs']:
    j = item['definition']
    assert queue[j['id']] == j
    v = state['jobs'][j['id']]
    assert v['status'] == 'completed' and type(v['returncode']) is int and v['returncode'] == 0
    assert not any('stop' in k.lower() for k in v) and v['missing_outputs'] == []
    assert v['argv'] == j['argv'] and v['definition_sha256'] == digest_job(j)
    selected[j['id']] = v
assert len(selected) == plan['closure_job_count']
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=False)
snapshot = out / 'actual_successor_state_immutable.json'
with snapshot.open('xb') as f:
    f.write(raw); f.flush(); os.fsync(f.fileno())
receipt = dict(status='captured_actual_complete_phase_closure_metadata_only',
               utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               contract_id=plan['contract_id'], plan=dict(path=str(plan_path), sha256=a.plan_sha256),
               source=str(source), source_stat_before=before, source_stat_after=after,
               immutable_state=dict(path=str(snapshot), sha256=hashlib.sha256(raw).hexdigest()),
               actual_queue=dict(path=str(queue_path), sha256=hashlib.sha256(raw_queue).hexdigest()),
               selected_completed_job_count=len(selected), full_executor_job_count=514,
               other_phases_may_be_running=True, selected_completed_records=selected,
               selected_definition_argv_strict_int0_no_any_stop_checked=True,
               score_or_attestation_payload_bytes_read=0)
with (out / 'ROOT_state_capture_receipt.json').open('x') as f:
    json.dump(receipt, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
os.fsync(fd); os.close(fd)
print(json.dumps({k: receipt[k] for k in ['contract_id', 'immutable_state', 'selected_completed_job_count']}))
