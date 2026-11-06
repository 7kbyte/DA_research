"""One bounded metadata-only prerequisite review; never prepare/start/score/hash payloads."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sources = {}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def small(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    require(path.suffix in {'.json', '.py'}, 'Only explicit metadata/source files may be read')
    before = path.stat()
    require(before.st_size <= 32 << 20, 'Metadata read bound exceeded')
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'Metadata changed during read')
    digest = hashlib.sha256(raw).hexdigest()
    require(expected is None or digest == expected, 'Small source SHA differs: ' + str(path))
    sources[str(path)] = dict(path=str(path), sha256=digest, bytes=len(raw))
    saved = OUT / 'small_source_bytes' / (digest + path.suffix)
    saved.parent.mkdir(exist_ok=True)
    if not saved.exists():
        saved.write_bytes(raw)
    return json.loads(raw) if path.suffix == '.json' else raw

def proof(ref):
    return small(ref['path'], ref['sha256'])

def digest_job(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()

def signature(path):
    value = Path(path).stat()
    return {name: getattr(value, 'st_' + name) for name in ['dev', 'ino', 'size', 'mtime_ns', 'ctime_ns']}

def proc_snapshot():
    result = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            result[int(path.parent.name)] = dict(state=fields[0], ppid=int(fields[1]), pgrp=int(fields[2]), starttime=fields[19])
        except (OSError, ValueError, IndexError):
            continue
    return result

observed_start = dt.datetime.now(dt.timezone.utc).isoformat()
queue = small('configs/supplemental_v1_queue.json')
state = small('logs/supplemental_job_state.json')
catalog = small('configs/supplemental_evaluation_preparation_v5.json')
require(len(queue) == len(state['jobs']) == 607, 'Exact607 cohort required')
require(state.get('active_jobs') == [] and state.get('stopping') is False, 'Master must be idle and not stopping')
jobs = {j['id']: j for j in queue}
require(len(jobs) == 607 and set(jobs) == set(state['jobs']), 'Queue/state IDs differ')
output_count = 0
for name, job in jobs.items():
    record = state['jobs'][name]
    require(record.get('status') == 'completed' and type(record.get('returncode')) is int and record['returncode'] == 0, 'Strict rc0/completed required: ' + name)
    require(not any('stop' in key.lower() for key in record), 'Any stop field rejects: ' + name)
    require(record.get('missing_outputs') == [], 'Missing-output metadata rejects: ' + name)
    require(record.get('argv') == job['argv'] and record.get('definition_sha256') == digest_job(job), 'Actual argv/definition differs: ' + name)
    for value in job.get('outputs', []):
        path = Path(value)
        require((path if path.is_absolute() else ROOT / path).is_file(), 'Declared output absent: ' + value)
        output_count += 1

freeze = small('results/extended_formal_freeze.json')
audit = small('results/extended_run_audit.json')
require(audit.get('status') == 'passed' and audit.get('completed_runs') == audit.get('expected_runs') == 63, 'Real full63 audit required')
require(audit['freeze_manifest_sha256'] == sources[str((ROOT/'results/extended_formal_freeze.json').resolve())]['sha256'], 'Audit freeze binding differs')
phase_rows = []
for phase in freeze['phases']:
    bound = next(p for p in audit['phases'] if p['phase'] == phase['phase'])
    require(bound['status'] == 'passed' and bound['completed_runs'] == bound['expected_runs'] == phase['runs'] and not bound['violations'], 'Model audit incomplete')
    proof(dict(path=ROOT/phase['snapshot'], sha256=phase['snapshot_sha256']))
    registry = small(phase['active_registry'], phase['active_registry_sha256'])
    require({r['id'] for r in registry} == {r['id'] for r in bound['runs']}, 'Audit registry IDs differ')
    phase_rows.append(dict(phase=phase['phase'], model=phase['model'], completed=bound['completed_runs'], expected=phase['runs']))

index = small('artifacts/evaluation_ready_v5/ready_index.json')
require(index['ready_phase_count'] == 9 and index['all_requested_phases_ready'] is True and len(index['phases']) == 9, 'All9 actual READY required')
proof(index['catalog'])
ready_rows = []
checkpoint_expected = {}
for row in index['phases']:
    require(row['status'] == 'ready' and row['ready'] is True, 'Phase not READY')
    ready = small(ROOT/'artifacts/evaluation_ready_v5'/row['id']/'READY.json')
    require(ready['id'] == row['id'] and ready['status'] == 'ready' and ready['ready'] is True, 'READY identity differs')
    loaded = {}
    for key in ['phase_spec', 'queue', 'manifest']:
        require(row[key] == ready[key], 'Index/READY reference differs')
        loaded[key] = proof(row[key])
    manifest = loaded['manifest']
    require(len(loaded['queue']) == row['workload']['total_queue_jobs'], 'Frozen queue workload differs')
    for record in manifest['records']:
        cp = str(Path(record['checkpoint']).resolve())
        old = checkpoint_expected.setdefault(cp, record['checkpoint_sha256'])
        require(old == record['checkpoint_sha256'], 'Conflicting expected CP SHA')
    ready_rows.append(dict(id=row['id'], physical_jobs=len(loaded['queue']), logical_endpoints=row['logical_endpoints'], manifest=row['manifest'], queue=row['queue']))
    equivalence = row.get('source_model_equivalence', {})
    for key in ['completion', 'audit']:
        if key in equivalence:
            proof(equivalence[key])
    for ref in equivalence.get('proofs', []):
        payload = proof(ref)
        require(len(payload['comparisons']) == 3, 'Expected real three-seed proof')
        for comparison in payload['comparisons']:
            require(type(comparison['bitwise_model_state_equal']) is bool, 'Explicit bool equality required')

progress = small('artifacts/evaluation_ready_v5/cache_progress.json')
require(progress['ready_phase_count'] == 9 and progress['prewarm']['pending'] == 0 and progress['prewarm']['visited'] == 534, 'Cache prewarm not complete')
cache_ref = progress['cache_snapshot']
cache = proof(cache_ref)
proof(cache['reader_source'])
require(cache['method'] == 'sha256_entire_open_fd_to_eof_v1', 'Cache method differs')
cp_entries = {p:r for p,r in cache['entries'].items() if '/checkpoints/' in p and p.endswith('.pt')}
require(len(cp_entries) == 534, '534 distinct checkpoint receipts required')
prewarm = {r['id']: r for r in catalog['prewarm_runs']}
require(len(prewarm) == 267, '267 physical train runs expected')
cp_rows = []
for path, ref in cp_entries.items():
    record = proof(ref)
    require(record['canonical_path'] == path == str(Path(path).resolve()), 'CP canonical path differs')
    require(record['method'] == cache['method'] and record['reader_source'] == cache['reader_source'] and record['complete_to_eof'] is True, 'Full stable read evidence missing')
    value = record['stat_before']
    require(set(value) == {'dev','ino','size','mtime_ns','ctime_ns'} and all(type(v) is int and v >= 0 for v in value.values()), 'Malformed five stat')
    require(record['fd_before'] == record['fd_after'] == record['stat_after'] == value and record['read_bytes'] == value['size'] == signature(path)['size'], 'Historical complete stable FD evidence differs')
    require(signature(path) == value and not Path(path).is_symlink(), 'Current CP stat invalidates cache')
    authority = record['source_authority']; run = prewarm[authority['run_id']]
    require(authority['kind'] == 'completed_training_checkpoint' and type(authority['returncode']) is int and authority['returncode'] == 0 and authority['job_definition_sha256'] == run['definition_sha256'], 'Checkpoint authority differs')
    if path in checkpoint_expected:
        require(checkpoint_expected[path] == record['sha256'], 'Manifest CP SHA differs from cache')
    cp_rows.append(dict(path=path, sha256=record['sha256'], receipt=ref, stat=value, run_id=authority['run_id'], endpoint=authority['endpoint']))
require(set(checkpoint_expected).issubset(cp_entries), 'Some READY checkpoint lacks cache coverage')

processes = proc_snapshot()
pid = state['scheduler_pid']
require(pid == 2988506, 'Unexpected old master PID')
cmdline = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if pid in processes else []
cmdline = [x.decode() for x in cmdline if x]
expected_argv = ['/usr/bin/python',str(ROOT/'scripts/schedule_jobs.py'),'--queue','configs/supplemental_v1_queue.json','--state','logs/supplemental_job_state.json','--gpus','0,1,2,3','--deadline','2026-11-07T00:00:00Z']
require(not cmdline or cmdline == expected_argv, 'Old master cmdline differs')
child_groups = {r['pid'] for r in state['jobs'].values() if type(r.get('pid')) is int}
remaining = {p:r for p,r in processes.items() if p != pid and (p in child_groups or r['pgrp'] in child_groups or r['ppid'] == pid)}
gpu = subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
apps = subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
gpu_rows = [r.strip().split(', ') for r in gpu.stdout.splitlines() if r.strip()]
require(len(gpu_rows) == 4 and all(float(r[2]) == 0 and float(r[3]) == 0 for r in gpu_rows) and not apps.stdout.strip(), 'Four GPUs not actually idle')
latest_progress = json.loads((ROOT/'artifacts/evaluation_ready_v5/cache_progress.json').read_bytes())
require(latest_progress['cache_snapshot'] == cache_ref, 'Latest cache changed; review again')
latest_queue = json.loads((ROOT/'configs/supplemental_v1_queue.json').read_bytes())
latest_state = json.loads((ROOT/'logs/supplemental_job_state.json').read_bytes())
require(latest_queue == queue and latest_state['jobs'] == state['jobs'] and latest_state['active_jobs'] == [] and latest_state['scheduler_pid'] == pid, 'Final queue/job state raced')
result = dict(schema_version=1,status='passed_content_prerequisites_old_master_still_blocks_start' if pid in processes else 'passed_content_prerequisites_recheck_exit_lock_before_start',observed_start=observed_start,observed_end=dt.datetime.now(dt.timezone.utc).isoformat(),master=dict(queue=sources[str((ROOT/'configs/supplemental_v1_queue.json').resolve())],state_snapshot=sources[str((ROOT/'logs/supplemental_job_state.json').resolve())],jobs=607,all_strict_integer_rc0=True,no_stop_fields=True,no_missing_outputs=True,declared_output_paths_stat_checked=output_count,actual_argv_definition_equal=True,active_jobs=[]),C_full_audit=dict(source=sources[str((ROOT/'results/extended_run_audit.json').resolve())],runs=63,phases=phase_rows),ready=dict(count=9,phases=ready_rows),digest_cache=dict(snapshot=cache_ref,checkpoint_entries=534,all_current_five_stat_match=True,all_full_read_receipts_validated=True,ready_checkpoint_paths_covered=len(checkpoint_expected),historically_full_read_bytes=sum(x['stat']['size'] for x in cp_rows),current_CP_payload_read_bytes=0),process=dict(old_scheduler_pid=pid,old_scheduler_alive=pid in processes,old_scheduler_cmdline=cmdline,old_scheduler_proc=processes.get(pid),remaining_historical_child_or_PG_members=remaining),gpu=dict(all_four_idle=True,query_gpu_stdout=gpu.stdout,query_compute_apps_stdout=apps.stdout),recommendations=['ROOT must gracefully terminate and reap old scheduler only after its own final approval; this review did not send signals.','Before frozen handoff prepare/start, recheck old PID/children/PG absent and old scheduler lock released.','Use the bound immutable cache snapshot, never mutable progress or a directory; real workers retain fresh input SHA.','Frozen handoff must independently verify output attestations and SHA; this review only stat-checked their existence.','Recompute final actual-source queue closure and per-job launch admission/tail; no new inference authority is granted.'],payload_policy=dict(CP_base_TAR_ZIP_bytes_read=0,opaque_score_output_bytes_read=0,score_semantics_parsed=False),source_proofs=list(sources.values()),checkpoint_metadata=cp_rows,review_whole_timer='whole_child_timing.json; all internal work nested')
(OUT/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','observed_end','process','digest_cache']},ensure_ascii=False))
