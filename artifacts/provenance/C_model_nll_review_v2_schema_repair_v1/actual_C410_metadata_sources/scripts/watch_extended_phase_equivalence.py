#!/usr/bin/env python3
"""Publish CPU-only D/B4 final and best proofs after each complete C cohort."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_extended_runs import audit_phase
from audit_formal_runs import sha256
from evaluation_plan import endpoint
from make_experiments import atomic_json
from make_model_equivalence_proofs import ALGORITHM, checkpoint_identity
from schedule_jobs import digest_job

MODELS = {'pythia-410m', 'pythia-1b'}
SEEDS = {11, 22, 33}
GROUPS = {'B0', 'B1', 'B2', 'B3', 'B4', 'D', 'M', 'C'}
DIAGNOSTIC_CHECKS = {'D_B4_exact_numerical_trajectories'}


def read(path):
    return json.loads(Path(path).read_text())


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def proof(path):
    return {'path': str(absolute(path)), 'sha256': sha256(absolute(path))}


def verify_proof(record):
    if sha256(record['path']) != record['sha256']:
        raise ValueError(f"Frozen input or published proof changed: {record['path']}")


def complete_cohort(rows, state, definitions, master):
    """Require all 24 real jobs, exact definitions, successful exits, and outputs."""
    expected = {(seed, group) for seed in SEEDS for group in GROUPS}
    if (len(rows) != 24 or len({row['id'] for row in rows}) != 24 or
            {(row['seed'], row['group']) for row in rows} != expected):
        raise ValueError('A C equivalence cohort must contain all eight groups and three seeds')
    lookup = {job['id']: job for job in definitions}
    current = {job['id']: job for job in master}
    if len(lookup) != 24 or set(lookup) != {row['id'] for row in rows}:
        raise ValueError('Frozen phase queue and registry disagree')
    missing, failed = [], []
    for row in rows:
        name = row['id']
        if current.get(name) != lookup[name]:
            raise ValueError(f'Master job differs from frozen phase job: {name}')
        job = state['jobs'].get(name, {})
        if job.get('definition_sha256') != digest_job(lookup[name]):
            raise ValueError(f'Scheduler definition SHA differs from frozen phase: {name}')
        if job.get('status') in {'failed', 'blocked'}:
            failed.append(name)
        if job.get('status') != 'completed' or job.get('returncode') != 0:
            missing.append(name)
        else:
            for path in [row['summary_path'], *lookup[name].get('outputs', [])]:
                if not absolute(path).is_file():
                    raise FileNotFoundError(f'Completed cohort output missing: {path}')
    return {'ready': not missing, 'expected_runs': 24,
            'completed_runs': 24 - len(missing), 'missing_runs': missing, 'failed_runs': failed}


def checkpoint_side(row, summary, kind):
    """Reuse the published byte-fingerprint algorithm, then bind saved metadata."""
    import torch

    path, exposure = endpoint(summary, kind)
    path = Path(path)
    before = path.stat()
    identity = checkpoint_identity(path)
    # mmap reads metadata without a second full optimizer-state allocation.
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if saved['config'] != summary['config']:
        raise ValueError(f'Checkpoint config disagrees with completed summary: {path}')
    for field, key in [('step', 'step'),
                       ('target_prediction_tokens', 'target_prediction_tokens_at_checkpoint'),
                       ('replay_prediction_tokens', 'replay_prediction_tokens_at_checkpoint')]:
        if saved['counters'][field] != exposure[key]:
            raise ValueError(f'Checkpoint {field} disagrees with actual selected endpoint: {path}')
    if kind == 'best':
        if (saved.get('checkpoint_kind') != 'model_only_best_target_validation' or
                saved.get('selection') != summary['best_target_validation']):
            raise ValueError(f'Best checkpoint selection metadata disagrees with summary: {path}')
        selection = dict(saved['selection'])
    else:
        if (saved.get('best_target_validation') != summary['best_target_validation'] or
                saved.get('validation_trajectory') != summary['validation_trajectory']):
            raise ValueError(f'Final checkpoint validation history disagrees with summary: {path}')
        selection = {'rule': 'fixed_completed_target_exposure', **exposure}
    del saved
    gc.collect()
    after = path.stat()
    signature = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    if signature(before) != signature(after):
        raise ValueError('Checkpoint changed while binding its selected endpoint')
    return {**identity, 'run_id': row['id'], 'kind': kind,
            'model_revision': summary['config']['model_revision'],
            'model_name_or_path': summary['config']['model_name_or_path'],
            'config_path': row['config_path'], 'config_sha256': sha256(row['config_path']),
            'summary_path': row['summary_path'], 'summary_sha256': sha256(row['summary_path']),
            'checkpoint_exposure': exposure, 'actual_selection': selection}


def publish_phase(phase, state, gate, output_root, authority):
    """Audit first, fingerprint sequentially, and publish completion last."""
    if not gate['ready'] or gate['expected_runs'] != 24 or gate['completed_runs'] != 24:
        raise ValueError('No model-state proof may precede the full successful 24-run gate')
    directory = Path(output_root) / phase['phase']
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f'Never overwrite an earlier phase proof attempt: {directory}')
    registry = absolute(phase['active_registry'])
    snapshot = absolute(phase['snapshot'])
    rows = read(registry)
    sources = [proof(registry), proof(snapshot), *authority]
    for row in rows:
        sources.extend([proof(row['summary_path']), proof(row['config_path']),
                        proof(Path(row['output_dir']) / 'events.jsonl')])
    raw_audit = audit_phase(snapshot, state, allow_incomplete=False, compare_weights=False)
    source_violations = [item for item in raw_audit['violations']
                         if item['check'] not in DIAGNOSTIC_CHECKS]
    diagnostics = [item for item in raw_audit['violations']
                   if item['check'] in DIAGNOSTIC_CHECKS]
    audit = {'schema_version': 1, 'audited_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'status': 'failed' if source_violations else 'passed',
             'phase': phase['phase'], 'model': phase['model'], 'expected_runs': 24,
             'completed_runs': raw_audit['completed_runs'], 'snapshot': str(snapshot),
             'snapshot_sha256': sha256(snapshot), 'source_registry': proof(registry),
             'source_state_complete_gate': gate, 'source_violations': source_violations,
             'completed_scheduler_records': {row['id']: state['jobs'][row['id']] for row in rows},
             'numerical_null_control_status': 'different' if diagnostics else 'passed',
             'numerical_null_control_diagnostics': diagnostics, 'phase_audit': raw_audit,
             'test_and_behavior_scores_read': False}
    audit_path = directory / 'audit.json'
    atomic_json(audit_path, audit)
    if source_violations or raw_audit['completed_runs'] != 24:
        raise ValueError(f'Complete C phase source/state audit failed: {audit_path}')
    lookup = {(row['seed'], row['group']): row for row in rows}
    outputs, all_comparisons = {}, []
    for kind in ['final', 'best']:
        comparisons = []
        for seed in sorted(SEEDS):
            sides = []
            for group in ['B4', 'D']:
                row = lookup[(seed, group)]
                sides.append(checkpoint_side(row, read(row['summary_path']), kind))
            equal = sides[0]['model_state_fingerprint'] == sides[1]['model_state_fingerprint']
            comparisons.append({'id': f"{phase['phase']}_D_B4_s{seed}_{kind}",
                                'kind': 'identical_bank_null_control', 'endpoint': kind,
                                'seed': seed, 'left': sides[0], 'right': sides[1],
                                'bitwise_model_state_equal': equal})
        record = {'schema_version': 1, 'status': 'passed', 'phase': phase['phase'],
                  'model': phase['model'], 'endpoint': kind,
                  'computed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'computation_device': 'CPU only', 'torch_num_threads': 2,
                  'checkpoint_loads_are_sequential': True,
                  'algorithm': ALGORITHM,
                  'algorithm_source': {**proof(ROOT / 'scripts/audit_formal_runs.py'),
                                       'function': 'model_fingerprint'},
                  'producer_source': proof(__file__), 'source_audit': proof(audit_path),
                  'source_registry': proof(registry), 'source_snapshot': proof(snapshot),
                  'comparisons': comparisons,
                  'all_pairs_bitwise_equal': all(item['bitwise_model_state_equal'] for item in comparisons),
                  'test_scores_accessed': False, 'checkpoint_files_modified': False}
        path = directory / f'{kind}_model_equivalence_proofs.json'
        atomic_json(path, record)
        outputs[kind] = proof(path)
        all_comparisons.extend(comparisons)
    # Hash sources after every CPU load so no ready receipt binds a moving input.
    for source in sources:
        verify_proof(source)
    completion = {'schema_version': 1, 'status': 'passed', 'phase': phase['phase'],
                  'model': phase['model'], 'completed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'source_registry': proof(registry), 'input_snapshot': proof(snapshot),
                  'audit': proof(audit_path), 'proofs': outputs,
                  'all_pairs_bitwise_equal': all(item['bitwise_model_state_equal'] for item in all_comparisons),
                  'non_equivalent_comparison_ids': [item['id'] for item in all_comparisons
                                                    if not item['bitwise_model_state_equal']],
                  'source_inputs': sources, 'expected_runs': 24, 'completed_runs': 24,
                  'comparisons_computed': 6, 'cpu_threads': 2,
                  'test_and_behavior_scores_read': False, 'gpu_work_launched': False}
    atomic_json(directory / 'completion.json', completion)
    return completion


def existing_completion(path, phase):
    result = read(path)
    if (result.get('status') != 'passed' or result.get('phase') != phase['phase'] or
            result.get('model') != phase['model'] or result.get('comparisons_computed') != 6):
        raise ValueError('Earlier phase completion is incomplete or belongs to another cohort')
    for source in [result['source_registry'], result['input_snapshot'], result['audit'],
                   *result['proofs'].values(), *result['source_inputs']]:
        verify_proof(source)
    for kind, source in result['proofs'].items():
        record = read(source['path'])
        if ({item['seed'] for item in record['comparisons']} != SEEDS or
                len(record['comparisons']) != 3 or record['endpoint'] != kind):
            raise ValueError('Earlier proof does not cover all three actual checkpoint pairs')
        for comparison in record['comparisons']:
            for side in ['left', 'right']:
                value = comparison[side]
                verify_proof({'path': value['checkpoint'], 'sha256': value['checkpoint_sha256']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'results/extended_formal_freeze.json')
    parser.add_argument('--state', type=Path, default=ROOT / 'logs/supplemental_job_state.json')
    parser.add_argument('--master-queue', type=Path, default=ROOT / 'configs/supplemental_v1_queue.json')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/extended_phase_equivalence')
    parser.add_argument('--progress', type=Path, default=ROOT / 'logs/extended_phase_equivalence_watch.json')
    parser.add_argument('--until', default='2026-11-07T00:10:00Z')
    parser.add_argument('--poll-seconds', type=float, default=60.)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error('Polling must be between 1 and 60 seconds')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    lock = args.progress.with_suffix('.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_paths = [args.manifest, Path(__file__), ROOT / 'scripts/audit_extended_runs.py',
                    ROOT / 'scripts/audit_formal_runs.py', ROOT / 'scripts/evaluation_plan.py',
                    ROOT / 'scripts/make_model_equivalence_proofs.py', ROOT / 'scripts/schedule_jobs.py']
    authority = [proof(path) for path in source_paths]
    manifest = read(args.manifest)
    phases = [phase for phase in manifest['phases'] if phase['model'] in MODELS]
    if len(phases) != 2 or {phase['model'] for phase in phases} != MODELS:
        raise ValueError('Expected independent 410M and 1B C phases')
    for phase in phases:
        for field, expected in [('active_registry', 'active_registry_sha256'),
                                ('snapshot', 'snapshot_sha256'), ('queue', 'queue_sha256')]:
            verify_proof({'path': str(absolute(phase[field])), 'sha256': phase[expected]})
            authority.append(proof(phase[field]))
    deadline = dt.datetime.fromisoformat(args.until.replace('Z', '+00:00'))
    results = {}
    while True:
        record = {'updated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'pid': os.getpid(),
                  'status': 'waiting_complete_model_phases', 'cpu_threads': 2, 'phases': {},
                  'gpu_work_launched': False, 'test_and_behavior_scores_read': False}
        try:
            for source in authority:
                verify_proof(source)
            state, master = read(args.state), read(args.master_queue)
            for phase in phases:
                name = phase['phase']
                completion_path = args.output_root / name / 'completion.json'
                if name not in results and completion_path.exists():
                    results[name] = existing_completion(completion_path, phase)
                if name in results:
                    result = results[name]
                    record['phases'][name] = {'status': 'proofs_completed',
                        'completion': proof(completion_path),
                        'all_pairs_bitwise_equal': result['all_pairs_bitwise_equal'],
                        'non_equivalent_comparison_ids': result['non_equivalent_comparison_ids']}
                    continue
                gate = complete_cohort(read(phase['active_registry']), state, read(phase['queue']), master)
                record['phases'][name] = {'status': 'waiting_complete_cohort', 'gate': gate}
                if gate['failed_runs']:
                    raise ValueError(f'Failed or blocked C training jobs: {gate["failed_runs"]}')
                if gate['ready']:
                    record['status'] = 'auditing_and_fingerprinting'
                    record['phases'][name]['status'] = 'auditing_and_fingerprinting'
                    atomic_json(args.progress, record)
                    results[name] = publish_phase(phase, state, gate, args.output_root, authority)
                    result = results[name]
                    record['phases'][name] = {'status': 'proofs_completed',
                        'completion': proof(completion_path),
                        'all_pairs_bitwise_equal': result['all_pairs_bitwise_equal'],
                        'non_equivalent_comparison_ids': result['non_equivalent_comparison_ids']}
                    print(json.dumps(record['phases'][name]), flush=True)
            if len(results) == 2:
                record['status'] = 'all_model_proofs_completed'
            atomic_json(args.progress, record)
            if len(results) == 2:
                return 0
            if dt.datetime.now(dt.timezone.utc) >= deadline:
                record['status'] = 'incomplete_training_at_watch_deadline'
                atomic_json(args.progress, record)
                return 2
        except Exception as error:
            record.update(status='source_or_state_audit_failed_requires_review', error=repr(error))
            atomic_json(args.progress, record)
            raise
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
