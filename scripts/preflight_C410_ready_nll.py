#!/usr/bin/env python3
"""Read-only, actual-READY C410 NLL timing preflight. Never enqueue or authorize."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from handoff_evaluation import job_definition
from reconcile_main_behavior_runtime import fifo_projection
from schedule_jobs import digest_job, validate_queue

PINS = {
    'FIFO.py': ('scripts/reconcile_main_behavior_runtime.py', '17d9b7839aef14e8c073124489c455d67512033f6de7658029411e88047a6f17'),
    'scheduler.py': ('scripts/schedule_jobs.py', 'c5c264b058ff655315ef896ccb0a89b471637c6612013a51999fc6f48b2d16d9'),
    'grammar.py': ('scripts/handoff_evaluation.py', 'bbe5a14e16bde55362aa3a7edc044b35ed8a7e2d0966a8b40692513b6c3a748a'),
    'budget.json': ('results/evaluation_timing_budget_v5.json', '940a0a0651cc7dd20da16d46321d8d4cac6b4dbbf7dd3e1ba6fbdd787e0b8f7d'),
    'catalog.json': ('configs/supplemental_evaluation_preparation_v5.json', '90163f7a77529434c872f52f41d45151bf09826ddd17ab58ea48d21d16f3d186'),
    'freeze.json': ('results/extended_formal_freeze.json', '50786c00724008eed6a96cb753b442295152f9dea96d73a2c75f8fcf16033528'),
    'ROOT_gates.json': ('docs/C410_early_NLL_admission_ROOT_decision.json', '885f0532301cc7a09f2b736763651a7216579378afaca1f7f55eabc9d0cfac02'),
}
PLAN = 'extended_pythia_410m_100m_nll'
OBSERVER_SHA = 'aaf4c774b47d00bef8794ac4151f1194ad04c6d6d6dce830ceceb337f331d0f4'
utc = lambda text: dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
sha = lambda blob: hashlib.sha256(blob).hexdigest()


class Inputs:
    def __init__(self):
        self.blobs, self.proofs, self.paths, self.stamps, self.aliases = {}, {}, {}, {}, {}

    @staticmethod
    def stable_bytes(path):
        stamp = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        with path.open('rb') as stream:
            before = os.fstat(stream.fileno())
            if before.st_size > 32 << 20:
                raise ValueError('Only small metadata inputs are permitted: ' + str(path))
            blob = stream.read((32 << 20) + 1)
            after = os.fstat(stream.fileno())
        if stamp(before) != stamp(after) or stamp(after) != stamp(path.stat()) or len(blob) > 32 << 20:
            raise ValueError('Unstable or non-small metadata input: ' + str(path))
        return blob, stamp(after)

    def read(self, name, path, expected=None, binary=False):
        alias = Path(path).absolute(); path = alias.resolve()
        blob, stamp = self.stable_bytes(path)
        if expected and sha(blob) != expected:
            raise ValueError('Input SHA differs: ' + str(path))
        self.blobs[name] = blob; self.paths[name] = path
        self.aliases[name] = alias
        self.stamps[name] = stamp
        self.proofs[name] = {'path': str(path), 'sha256': sha(blob), 'stat_dev_ino_size_mtime_ns_ctime_ns': list(stamp)}
        return blob if binary else json.loads(blob)

    def unchanged(self):
        for name, path in self.paths.items():
            blob, stamp = self.stable_bytes(path)
            if blob != self.blobs[name] or stamp != self.stamps[name] or self.aliases[name].resolve() != path:
                raise ValueError('Input changed during preflight: ' + str(path))


def price_jobs(queue, manifest, rates):
    """Price actual canonical jobs; internal aliases consume no extra forwards."""
    entries = {row['id']: row for row in manifest['jobs']}
    durations = {}
    for job in queue:
        row = entries[job['id']]
        kinds = {record['kind'] for record in manifest['records'] if record['inference_job_id'] == job['id']}
        if row['kind'] == 'initial':
            durations[job['id']] = rates['initial']
        elif row['kind'] == 'adapted' and kinds and kinds <= {'final', 'best'}:
            durations[job['id']] = max(rates[kind] for kind in kinds)
        else:
            raise ValueError('Unexpected reuse/external scope lacks a frozen C410 rate; ROOT review required')
    return durations


def timing_gates(starts, finish, cutoff, stop, unfinished):
    rows = [row for row in starts if row['id'] in unfinished]
    headroom = min(((cutoff - utc(row['start_utc'])).total_seconds() for row in rows), default=None)
    finish_headroom = (stop - finish).total_seconds()
    return {'unfinished_C_launch_headroom_seconds': headroom,
            'all_unfinished_C_launches_at_least300seconds_before_strict_cutoff': headroom is None or headroom >= 300,
            'whole_master_finish_headroom_before_stop_seconds': finish_headroom,
            'whole_master_finish_at_least900seconds_before_stop': finish_headroom >= 900,
            'both_ROOT_time_gates': (headroom is None or headroom >= 300) and finish_headroom >= 900}


def run(args):
    if args.output.exists():
        raise FileExistsError('New immutable output directory required')
    source = Inputs()
    fixed = {key: source.read(key, ROOT / path, digest, binary=not key.endswith('.json')) for key, (path, digest) in PINS.items()}
    ready = source.read('READY.json', args.ready, args.ready_sha256)
    manifest = source.read('manifest.json', args.manifest, args.manifest_sha256)
    phase = source.read('phase_queue.json', args.phase_queue, args.phase_queue_sha256)
    if not (ready.get('ready') is True and ready.get('status') == 'ready' and ready.get('id') == PLAN and ready.get('mode') == 'nll'):
        raise ValueError('An actual model-specific READY is required')
    for key, path, digest in [('queue', args.phase_queue, args.phase_queue_sha256), ('manifest', args.manifest, args.manifest_sha256)]:
        if ready[key] != {'path': str(path.resolve()), 'sha256': digest}:
            raise ValueError('READY does not bind the explicit actual ' + key)
    spec = source.read('phase_spec.json', ready['phase_spec']['path'], ready['phase_spec']['sha256'])
    plan = next(row for row in fixed['catalog.json']['plans'] if row['id'] == PLAN)
    formal = {row['model']: row for row in fixed['freeze.json']['phases']}
    model, registry = {}, {}
    frozen_jobs = {}
    for tag, frozen in formal.items():
        rows = source.read(tag + '_registry.json', frozen['active_registry'], frozen['active_registry_sha256'])
        jobs = source.read(tag + '_training_queue.json', frozen['queue'], frozen['queue_sha256'])
        if {row['id'] for row in rows} != {job['id'] for job in jobs}:
            raise ValueError('Frozen phase queue and registry differ')
        frozen_jobs.update({job['id']: job for job in jobs})
        registry[tag] = rows; model.update({row['id']: tag for row in rows})
    c410 = {row['id'] for row in registry['pythia-410m']}
    if len(model) != 63 or len(c410) != 24 or manifest.get('schema_version') != 2 or manifest.get('mode') != 'nll':
        raise ValueError('The frozen63 cohort and actual NLL manifest are required')
    if len(spec['phases']) != 1 or spec['phases'][0]['data_family'] != 'extended_v1':
        raise ValueError('Matched extended data family required')
    gate = plan['phase']['model_equivalence_gate']; bound = spec['phases'][0]['source_proof_files']
    completion = source.read('phase_completion.json', gate['completion_path'], bound[gate['completion_path']])
    audit = source.read('phase_audit.json', gate['audit_path'], bound[gate['audit_path']])
    audit_gate = audit['source_state_complete_gate']
    if (completion.get('status') != 'passed' or audit.get('status') != 'passed'
            or any(row.get('expected_runs') != 24 or row.get('completed_runs') != 24
                   or row.get('phase') != gate['phase'] or row.get('model') != 'pythia-410m' for row in (completion, audit))
            or completion['source_registry'] != plan['phase']['registry'] or completion['input_snapshot'] != gate['source_snapshot']
            or completion['audit']['sha256'] != bound[gate['audit_path']] or audit.get('source_violations') != []
            or audit_gate.get('ready') is not True or audit_gate.get('expected_runs') != 24 or audit_gate.get('completed_runs') != 24
            or audit_gate.get('missing_runs') != [] or audit_gate.get('failed_runs') != []):
        raise ValueError('Actual complete24 audit/equivalence gate required')
    for kind in ('final', 'best'):
        item = completion['proofs'][kind]
        proof = source.read(kind + '_complete_equivalence.json', item['path'], item['sha256'])
        if (str(Path(item['path']).resolve()) != gate['proof_paths'][kind] or item['sha256'] != bound[gate['proof_paths'][kind]]
                or proof.get('status') != 'passed' or proof.get('endpoint') != kind
                or proof.get('algorithm') != 'sha256_sorted_key_dtype_shape_contiguous_uint8_v1'
                or {key: proof.get('algorithm_source', {}).get(key) for key in ('path', 'sha256')} != gate['algorithm_source']
                or proof.get('phase') != gate['phase'] or proof.get('model') != 'pythia-410m'
                or len(proof['comparisons']) != 3 or {pair['seed'] for pair in proof['comparisons']} != {11, 22, 33}
                or any(type(pair.get('bitwise_model_state_equal')) is not bool or pair.get('endpoint') != kind for pair in proof['comparisons'])):
            raise ValueError('Complete actual three-seed final/best equivalence metadata required')
    if manifest.get('phase_spec_sha256') != ready['phase_spec']['sha256'] or Path(manifest['phase_spec']).resolve() != Path(ready['phase_spec']['path']).resolve():
        raise ValueError('Manifest is not bound to the actual READY phase spec')
    if len(manifest['records']) != 48 or {(row['run_id'], row['kind']) for row in manifest['records']} != {(jid, kind) for jid in c410 for kind in ('final', 'best')}:
        raise ValueError('Full24 final+best scope required; no subset or best-only selection')
    groups = list(manifest['baseline_groups'].values())
    initial = [row for row in manifest['jobs'] if row['kind'] == 'initial']
    wanted_tests = {key: str(Path(value['path']).resolve()) for key, value in plan['phase']['test_data'].items()}
    if len(groups) != 1 or len(initial) != 1 or groups[0].get('initial_origin') or {key: str(Path(path).resolve()) for key, path in groups[0]['test_paths'].items()} != wanted_tests:
        raise ValueError('One new matched-extended initial is required; primary-v3 baseline cannot be reused')
    validate_queue(phase)
    if {row['id'] for row in phase} != {row['id'] for row in manifest['jobs']} or len(phase) != len(manifest['jobs']):
        raise ValueError('Actual queue and manifest physical scopes differ')
    if any(row['kind'] != 'initial' and row.get('depends_on') != [initial[0]['id']] for row in manifest['jobs']):
        raise ValueError('All actual adapted jobs must depend on the new matched initial')
    authorities = {(str(args.manifest.resolve()), args.manifest_sha256): manifest}; hashes = {}
    checkpoint_metadata = {}
    for job in phase:
        job_definition(job, authorities, hashes)  # Grammar/definitions only; never verified_job/full input hashing.
    for index, record in enumerate(manifest['records']):
        if record['data_family'] != 'extended_v1' or record['model_tag'] != 'pythia-410m':
            raise ValueError('Mixed model/data scope')
        cppath = Path(record['checkpoint']); stat = cppath.stat()
        stamp = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        if stat.st_size != record['checkpoint_bytes'] or stat.st_mtime_ns != record['checkpoint_mtime_ns']:
            raise ValueError('Actual checkpoint metadata changed since READY')
        checkpoint_metadata[str(cppath)] = {'canonical_path': str(cppath.resolve()), 'sha256_bound_by_manifest': record['checkpoint_sha256'],
                                           'stat_dev_ino_size_mtime_ns_ctime_ns': stamp, 'payload_read_by_this_preflight': False}
        evidence = record.get('source_model_equivalence')
        if evidence:
            proof = source.read('equivalence_' + str(index) + '.json', evidence['proof'], evidence['proof_sha256'])
            pair = proof['comparisons'][evidence['comparison_index']]
            if (not pair.get('bitwise_model_state_equal') or record['checkpoint_sha256'] not in {pair['left']['checkpoint_sha256'], pair['right']['checkpoint_sha256']}
                    or record['weights_identity'] != 'model_state:' + evidence['fingerprint']['sha256_model_state']
                    or any(pair[side]['model_state_fingerprint'] != evidence['fingerprint'] for side in ('left', 'right'))):
                raise ValueError('Manifest model-equivalence identity lacks its true complete-state proof')
    raw = source.read('runtime.json', args.runtime)
    state = source.read('state.json', args.state)
    master = source.read('master_queue.json', args.master_queue, args.master_queue_sha256)
    cmdline = source.read('master_cmdline.bin', Path('/proc') / str(args.master_pid) / 'cmdline', binary=True)
    argv = [value.decode() for value in cmdline.split(b'\0') if value]
    if state['scheduler_pid'] != args.master_pid or '--deadline' not in argv or argv[argv.index('--deadline') + 1] != '2026-11-07T00:00:00Z':
        raise ValueError('Actual original master/deadline differs')
    if '--launch-margin-seconds' in argv or '--stop-margin-seconds' in argv or Path(argv[1]).resolve() != (ROOT / 'scripts/schedule_jobs.py').resolve():
        raise ValueError('Original scheduler defaults/source required; no unreviewed margin change')
    for flag, wanted in [('--queue', args.master_queue), ('--state', args.state)]:
        if flag not in argv or (ROOT / argv[argv.index(flag) + 1]).resolve() != wanted.resolve():
            raise ValueError('Actual master argv does not bind the captured ' + flag)
    if '--gpus' not in argv or argv[argv.index('--gpus') + 1] != '0,1,2,3':
        raise ValueError('The original four-slot master is required')
    lookup = {row['id']: row for row in master}
    if len(lookup) != len(master) or any(state['jobs'][jid]['definition_sha256'] != digest_job(job) for jid, job in lookup.items()):
        raise ValueError('Master definitions differ from state')
    if any(lookup[jid] != job for jid, job in frozen_jobs.items()):
        raise ValueError('A formal C training job differs from its frozen original definition')
    if any(row['id'] in lookup or row['id'] in state['jobs'] for row in phase):
        raise ValueError('Phase already present; this tool only previews a new insertion')
    if any(state['jobs'][jid]['status'] != 'completed' or state['jobs'][jid].get('returncode') != 0 or 'stop_requested_at' in state['jobs'][jid] or state['jobs'][jid].get('missing_outputs') or any(not Path(path).is_file() for path in lookup[jid].get('outputs', [])) for jid in c410):
        raise ValueError('Actual24 successful completed training jobs required')
    current = dt.datetime.now(dt.timezone.utc); age = (current - utc(raw['updated_at_utc'])).total_seconds()
    c_rows = [row for update in raw['extended_model_runtime_updates'].values() for row in update['jobs']]
    unfinished = {jid for jid, row in state['jobs'].items() if row['status'] != 'completed'}
    if (not 0 <= age <= 90 or raw.get('observer_source_sha256') != OBSERVER_SHA or raw['status'] != 'monitoring' or raw['failed_or_blocked_jobs'] or not all(raw['frozen_scientific_core_matches'].values())
            or raw['master_queue_jobs'] != len(master) or {row['id'] for row in c_rows} != set(model)
            or any(state['jobs'][row['id']]['status'] != row['scheduler_status'] for row in c_rows)
            or len(c_rows) != 63
            or unfinished != {row['id'] for row in c_rows if row['scheduler_status'] != 'completed'}
            or any(state['jobs'][jid]['status'] not in ('pending', 'running') for jid in unfinished)
            or any(type(row['total_seconds']) not in (int, float) or not math.isfinite(row['total_seconds']) or row['total_seconds'] < 0 for row in c_rows)):
        raise ValueError('Fresh, healthy, C-only, status-consistent snapshot required')
    running = [jid for jid in unfinished if state['jobs'][jid]['status'] == 'running']
    if len(running) != 4 or len({state['jobs'][jid]['gpu'] for jid in running}) != 4 or any(not Path('/proc', str(state['jobs'][jid]['pid'])).exists() for jid in running):
        raise ValueError('Four actual live occupied slots required')
    for index, jid in enumerate(sorted(running)):
        child = source.read('running_' + str(index) + '_cmdline.bin', Path('/proc') / str(state['jobs'][jid]['pid']) / 'cmdline', binary=True)
        if [value.decode() for value in child.split(b'\0') if value] != lookup[jid]['argv']:
            raise ValueError('Live occupied child differs from its frozen job argv')
    if any(lookup[jid]['expected_seconds'] != 3600 for jid in model):
        raise ValueError('Frozen C allowance changed')
    high = next(row for row in fixed['budget.json']['sensitivity_scenarios'] if row['name'] == 'high')
    rates = {row['id'].rsplit('_', 1)[-1]: row['per_job_whole_slot_seconds_proxy'] for row in high['per_phase'] if row['id'].startswith('C_pythia_410m_nll_')}
    if set(rates) != {'initial', 'final', 'best'} or high['whole_wall_sensitivity_factor'] != 1.5:
        raise ValueError('Frozen v5 high model/endpoint rates required')
    priced = price_jobs(phase, manifest, rates); scenarios = []
    current = dt.datetime.now(dt.timezone.utc)  # Anchor after metadata checks; no unmeasured zero READY delay.
    cutoff = utc('2026-11-06T22:30:00Z'); stop = utc('2026-11-06T23:50:00Z')
    for label, jobs, durations in [('current_C_only', [], {}), ('actual_READY_C410_NLL_priority_high', phase, priced)]:
        prediction = fifo_projection(master, state, jobs, durations, raw, raw['remaining_slot_hours'] * 3600, True)
        if prediction['status'] != 'complete_time_simulation' or abs(prediction['unassigned_conservative_nonphase_slot_seconds']) > .01:
            raise ValueError('Unpriced work in actual master')
        future = {row['id']: row for row in prediction['future_dispatch_timing_rows']}
        starts = [{'id': jid, 'model': model[jid], 'status': state['jobs'][jid]['status'],
                   'basis': 'forecast_pending_dispatch' if jid in future else 'actual_already_started',
                   'start_utc': (current + dt.timedelta(seconds=future[jid]['start_seconds'])).isoformat() if jid in future else state['jobs'][jid]['started_at_utc']} for jid in sorted(model)]
        finish = current + dt.timedelta(seconds=prediction['makespan_seconds_with_10pct'])
        envelope = finish + dt.timedelta(seconds=high['wall_seconds_with_dispatch_tail_and_handoff_proxy'])
        scenarios.append({'scenario': label, 'all63_training_starts': starts, 'finish_utc': finish.isoformat(),
            **timing_gates(starts, finish, cutoff, stop, unfinished), 'FIFO': prediction,
            'unchanged_full_v5_tail_envelope_finish_utc': envelope.isoformat(),
            'envelope_margin_before30minute_delivery_tail_seconds': (utc('2026-11-07T02:56:32Z') - envelope).total_seconds()})
    source.read('producer.py', Path(__file__).resolve(), binary=True)
    source.unchanged()  # Includes entire live state/runtime/queue and proc argv; any race fails before output.
    for path, proof in checkpoint_metadata.items():
        value = Path(path); stat = value.stat()
        if (str(value.resolve()) != proof['canonical_path'] or
                [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns] != proof['stat_dev_ino_size_mtime_ns_ctime_ns']):
            raise ValueError('Checkpoint metadata changed during preflight')
    result = {'schema_version': 1, 'status': 'read_only_actual_READY_preflight_not_authority', 'captured_at_utc': current.isoformat(),
        'runtime_age_seconds': age, 'sources': source.proofs, 'checkpoint_metadata_only': checkpoint_metadata,
        'actual_READY_delay_already_elapsed_not_simulated_as_zero': True,
        'forecast_anchor_is_current_capture_not_old_runtime_timestamp': True, 'running_progress_since_runtime_not_credited_conservative': True,
        'physical_scope_from_actual_manifest': {'queue_jobs': len(phase), 'initial': 1, 'adapted': len(phase) - 1, 'logical_endpoints': 48},
        'actual_high_rates_already_contain_factor1p5': rates, 'per_job_rate_seconds': priced,
        'remaining_FIFO_durations_receive_factor1p1_once': True, 'current_running_ids_not_preempted': running,
        'strict_actual_training_cutoff_utc': cutoff.isoformat(), 'ROOT_extra_launch_gate_utc': '2026-11-06T22:25:00Z',
        'ROOT_finish_gate_utc': '2026-11-06T23:35:00Z', 'scenarios': scenarios,
        'actual_phase_completion_tail_credit': 0, 'full_tail_envelope_intentionally_repeats_same_phase_scope_not_total_cost': True,
        'execution_requires_separate_ROOT_authority_and_fresh_actual_manifest_review': True,
        'no_queue_or_order_or_state_or_frozen_helper_changed': True, 'score_or_CP_payload_read': False, 'GPU_or_watcher_started': False}
    args.output.mkdir(); (args.output / 'sources').mkdir()
    for name, blob in source.blobs.items():
        with (args.output / 'sources' / name).open('xb') as stream:
            stream.write(blob); stream.flush(); os.fsync(stream.fileno())
    target = args.output / 'preflight.json'
    with target.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'preflight': str(target), 'sha256': sha(target.read_bytes()), 'both_time_gates': scenarios[-1]['both_ROOT_time_gates'], 'not_authority': True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ready', 'manifest', 'phase-queue'):
        parser.add_argument('--' + name, type=Path, required=True); parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--runtime', type=Path, default=ROOT / 'results/runtime_budget_latest.json')
    parser.add_argument('--state', type=Path, default=ROOT / 'logs/supplemental_job_state.json')
    parser.add_argument('--master-queue', type=Path, default=ROOT / 'configs/supplemental_v1_queue.json')
    parser.add_argument('--master-queue-sha256', required=True)
    parser.add_argument('--master-pid', type=int, default=2988506)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
