"""ROOT manually invokes one phase only after its original complete check.

This does not poll, select methods, change statistics, or authorize other phases.
"""
from pathlib import Path
import argparse, datetime as dt, hashlib, json, os, time

R = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument('--phase', required=True)
args = parser.parse_args()
started = time.monotonic()
verified = {}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def pin(proof):
    p = Path(proof['path'])
    assert p.stat().st_size < 50_000_000, 'ROOT metadata review will not open large payloads'
    b = p.read_bytes()
    assert hashlib.sha256(b).hexdigest() == proof['sha256'], str(p)
    verified[str(p)] = dict(sha256=proof['sha256'], bytes=len(b))
    return b

runbook = R / 'results/C_pending_analysis_metadata_runbook_v1/future_commands_PREPARED_ONLY.json'
all_commands = json.loads(pin(dict(path=str(runbook), sha256='fdefaeb3d9abb059d4431a5b047d9804861c69635f0d2dc31eb1823996150996')))
profile = next(x for x in all_commands['phases'] if x['contract_id'] == args.phase)
operation = Path(profile['ROOT_future_operation_directory_not_created'])
plan = json.loads(pin(profile['plan']))
repair = json.loads(pin(profile['operational_repair']))
authority = json.loads(pin(profile['FALSE_template']))
assert authority['allow_score_analysis'] is False and authority['authorized_by'] == 'root'
assert not Path(plan['first_read_receipt_path']).exists()
assert not Path(profile['new_output_path']).exists()
check_path = operation / 'check_readiness.json'
check = json.loads(check_path.read_bytes())
assert check['ready'] is True and check['score_json_parsed'] is False and not check['pending_job_ids']
assert check['all_source_baseline_alias_output_integrity_verified'] is True
assert check['plan'] == profile['plan'] and check['operational_repair'] == profile['operational_repair']
assert check['counts'] == profile['expected_counts'] == plan['counts']
assert check['closure_job_count'] == plan['closure_job_count']
timing_path = operation / 'check_timing/whole_child_timing.json'
timing = json.loads(timing_path.read_bytes())
assert type(timing['returncode']) is int and timing['returncode'] == 0
state_path = operation / 'state_capture/actual_successor_state_immutable.json'
assert check['scheduler_state'] == str(state_path)
assert sha(state_path) == check['scheduler_state_sha256']
capture = json.loads((operation / 'state_capture/ROOT_state_capture_receipt.json').read_bytes())
assert capture['plan'] == profile['plan'] and capture['selected_definition_argv_strict_int0_no_any_stop_checked'] is True
assert capture['immutable_state'] == dict(path=str(state_path), sha256=sha(state_path))
assert capture['selected_completed_job_count'] == plan['closure_job_count']
assert timing['argv'][timing['argv'].index('--state') + 1] == str(state_path)
for key in ('protocol', 'ready', 'manifest'):
    pin(plan[key])
protocol = json.loads(Path(plan['protocol']['path']).read_bytes())
for mapping in (protocol['analysis_source_sha256'], repair['active_source_sha256'], repair['retained_source_sha256']):
    for path, digest in mapping.items():
        pin(dict(path=path, sha256=digest))
for row in check['attestations']:
    pin(row['attestation'])
for row in check['verification_work']['checkpoint_events']:
    s = Path(row['path']).stat()
    assert [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns] == row['five_stat']
authority.update(allow_score_analysis=True, authorized_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 operational_repair=profile['operational_repair'],
                 actual_successor_immutable_state=dict(path=str(state_path), sha256=sha(state_path),
                     pending_actual_completed_phase_snapshot=False, same_exact_source_required_for_check_and_analyze=True),
                 whole_phase_check=dict(path=str(check_path), sha256=sha(check_path)),
                 ROOT_review=dict(fresh_small_source_proofs=verified,
                     checkpoint_current_five_stat_checks=len(check['verification_work']['checkpoint_events']),
                     source_or_score_model_payload_read_by_ROOT=False,
                     original_complete_check_parent=dict(path=str(timing_path), sha256=sha(timing_path)),
                     root_preparation_body_seconds=time.monotonic()-started, manual_parent_seconds=None))
authority_path = Path(profile['future_ROOT_authority_path_not_created'])
with authority_path.open('x') as f:
    json.dump(authority, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
command = profile['analyze_argv_REFERENCE_ONLY']
command[command.index('--state') + 1] = str(state_path)
command[command.index('--authority-sha256') + 1] = sha(authority_path)
with (operation / 'ROOT_analyze_launch.json').open('x') as f:
    json.dump(dict(argv=command, authority_sha256=sha(authority_path),
                   source_sha256=sha(__file__), utc=authority['authorized_at_utc']), f, indent=2)
    f.write('\n')
env = dict(os.environ)
env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
wrapped = ['python', str(R / 'artifacts/provenance/final_evaluation_handoff_actual_v1/time_child.py'),
           str(operation / 'analyze_timing'), *command]
print(json.dumps(dict(phase=args.phase, authority_sha256=sha(authority_path), state_sha256=sha(state_path))), flush=True)
os.execvpe(wrapped[0], wrapped, env)
