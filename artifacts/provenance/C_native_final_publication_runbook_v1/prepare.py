"""Read-only current reserve and future package path inventory; no review/pass."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def proof(path):
    p = Path(path).resolve()
    if p.stat().st_size > 32 << 20 or p.suffix in {'.pt', '.zip', '.tar', '.safetensors'}:
        raise ValueError('Explicit small source metadata only')
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    sp = importlib.util.spec_from_file_location('native_final_metadata_preparation', HERE / 'review_metadata.py')
    r = importlib.util.module_from_spec(sp); sp.loader.exec_module(r)  # stdlib-only, no review() call.
    helpers = {p: {'expected_sha256': expected, **proof(ROOT / p)} for p, expected in r.PIN.items()}
    if any(v['expected_sha256'] != v['sha256'] for v in helpers.values()):
        raise ValueError('Original live helper changed')
    freeze = json.loads((ROOT / 'results/extended_formal_freeze.json').read_text())
    locks = Path('/proc/locks').read_text().splitlines()
    packages = []
    for model, profile in r.PROFILES.items():
        phase = next(p for p in freeze['phases'] if p['model'] == model)
        snapshot = json.loads((ROOT / phase['snapshot']).read_text())
        grow_path = ROOT / 'artifacts/provenance/supplemental_weight_reservations_v1' / ('C_' + profile['tag'] + '_growth_receipt.json')
        grow = json.loads(grow_path.read_text())
        reserve = r.PERSISTENT / ('.extended_C_' + profile['tag'] + '_final_weights_reserve_20261106')
        s = reserve.lstat()
        observed = dict(path=str(reserve), device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                        allocated_bytes=s.st_blocks * 512, uid=s.st_uid, gid=s.st_gid,
                        mode=oct(stat.S_IMODE(s.st_mode)), links=s.st_nlink)
        key = f'{os.major(s.st_dev):02x}:{os.minor(s.st_dev):02x}:{s.st_ino}'
        folder = r.DELIVERY / model
        package_name = 'extended_C_' + profile['tag'] + '_final_weights_lossless_v1.tar'
        paths = {n: str(folder / n) for n in ['completion.json', 'extended_C_final_manifest.json', package_name]}
        packages.append({
            'model': model, 'phase': phase['phase'], 'expected_target_exposure': profile['target'],
            'expected_logical_final_endpoints': profile['logical'],
            'conditional_expected_unique': profile['unique'], 'conditional_expected_aliases': profile['aliases'],
            'condition_for1B21unique': 'All3 actual final D/B4 complete-state proof decisions true; otherwise this expected21 reviewer rejects for ROOT review.',
            'future_paths': paths, 'future_paths_currently_exist': {k: Path(v).exists() for k, v in paths.items()},
            'actual_whole_package_acceptance_status': 'pending_not_reviewed',
            'reserve_current_stat': observed, 'reserve_matching_locks': [line for line in locks if key in line],
            'historical_complete_growth_receipt': proof(grow_path),
            'reserve_identity_matches_historical_receipt': all(observed[k] == grow['actual_stat'][k] for k in ['device', 'inode', 'bytes', 'allocated_bytes', 'uid', 'gid']) and stat.S_IMODE(s.st_mode) == grow['actual_stat']['mode'],
            'old_grower_PID_currently_exists': Path('/proc', str(grow['pid'])).exists(),
            'native_base_files_from_frozen_snapshot_not_fresh_payload_hashes': snapshot['model_files'],
            'source_registry': {'path': phase['active_registry'], 'sha256': phase['active_registry_sha256']},
            'input_snapshot': proof(ROOT / phase['snapshot']),
            'phase_protocol': {'path': phase['protocol'], 'sha256': phase['protocol_sha256']},
            'future_whole_phase_READY': str(ROOT / 'artifacts/evaluation_ready_v5' / profile['ready'] / 'READY.json'),
            'future_source_audit': str(ROOT / 'results/extended_phase_equivalence' / profile['phase'] / 'audit.json') if model == 'pythia-1b' else str(folder / 'fullphase_source_audit.json'),
            'future_final_equivalence_proof': str(ROOT / 'results/extended_phase_equivalence' / profile['phase'] / 'final_model_equivalence_proofs.json') if model == 'pythia-1b' else None,
            'destination': str(r.PERSISTENT / package_name), 'destination_currently_exists': (r.PERSISTENT / package_name).exists(),
        })
    worker_lock = Path('/tmp/grad-research-20261106/delivery/.supplemental_weight_export.lock')
    worker_stat = worker_lock.stat()
    worker_key = f'{os.major(worker_stat.st_dev):02x}:{os.minor(worker_stat.st_dev):02x}:{worker_stat.st_ino}'
    receipt = {
        'schema_version': 1, 'status': 'prepared_future_ROOT_metadata_review_and_runbook_not_package_acceptance',
        'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'producer': proof(__file__),
        'reviewer': proof(HERE / 'review_metadata.py'), 'synthetic_test_source': proof(HERE / 'test_review_metadata.py'),
        'synthetic_tests_whole_child_timing': proof(HERE / 'synthetic_tests_whole_child_timing.json'),
        'runbook': proof(ROOT / 'docs/C1B与C2p8B原final权重包持久发布runbook.md'),
        'preserved_C410_reviewer': proof(ROOT / 'artifacts/provenance/C410_final_publication_runbook_v1/review_metadata.py'),
        'unchanged_original_helper_sources': helpers,
        'frozen_C63': proof(ROOT / 'results/extended_formal_freeze.json'),
        'packages': packages,
        'existing_live_daemon_global_lock_observation': [line for line in locks if worker_key in line],
        'whole_package_reviewer_actually_invoked': False,
        'new_TRUE_authority_or_future_completion_or_ready_created': False,
        'publication_or_new_reservation_or_GPU_or_new_watcher_started': False,
        'score_semantics_or_checkpoint_or_TAR_or_ZIP_or_native_model_payloads_read': False,
        'live_exporter_codec_publisher_or_imported_helpers_modified': False,
        'whole_preparation_parent_wall_CPU_unknown': True,
        'C410_dependency_difference': 'C410 requires the separately durable primary TAR base closure. These two native packages include all six pinned native files, so model restore needs no original checkpoint/scratch base/network.',
        'timer_rule': 'Draft parent operations and assembly-only completion are disjoint; daemon cumulative wall contains waits and those children. Future publisher whole child is separate; its internal publisher and readback/copy/fsync are nested.',
    }
    target = HERE / 'preparation_receipt.json'
    with target.open('x') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps({'receipt': proof(target), 'reviewer': proof(HERE / 'review_metadata.py'), 'runbook': receipt['runbook']}))


if __name__ == '__main__':
    main()
