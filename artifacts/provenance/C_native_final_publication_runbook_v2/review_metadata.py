#!/usr/bin/env python3
"""Future metadata-only review of original Cfinal1B/2.8B packages, never publish."""
import argparse
import datetime as dt
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import stat

ROOT = Path(__file__).resolve().parents[3]
DELIVERY = Path('/tmp/grad-research-20261106/delivery/extended_C_final_v1')
PERSISTENT = Path('/data/202610/7kb/research_delivery')
FREEZE_SHA = '50786c00724008eed6a96cb753b442295152f9dea96d73a2c75f8fcf16033528'
PIN = {
    'scripts/export_C_final_weights.py': 'bc057d04298b91a8189dae58fb6189c19b69faf70c9c83298d7854bf1cc38390',
    'scripts/publish_reserved_archive.py': '9c194743383a16f98313f767fe2e722c309ec57c1aafc65d9b99d08e5407022e',
    'scripts/export_remaining_primary_endpoints.py': 'eb7339e4224062f3d55bd0adcba467599e06035ac84172f53d954b7302a1473d',
    'src/replay_study/weight_archive.py': '3c2bcf9e38773ae4f69c27f85f342356dbefcb53d817be4321da6e9c4e550e1a',
}
PROFILES = {
    'pythia-1b': dict(tag='1b', phase='extended_v2_pythia_1b_m32_formal', ready='extended_pythia_1b_100m_nll',
                      logical=24, unique=21, aliases=3, target=100_000_000, limit=70 << 30, inode=663438738),
    'pythia-2.8b': dict(tag='2p8b', phase='extended_v2_pythia_2p8b_m8_h25m_formal', ready='extended_pythia_2p8b_25m_nll',
                        logical=15, unique=15, aliases=0, target=25_000_000, limit=140 << 30, inode=559646871),
}
EXTRAS = {'extended_C_final_manifest.json', 'RESTORE_C_FINAL.md', 'tools/provenance/export_C_final_weights.py',
          'tools/src/replay_study/weight_archive.py', 'tools/src/replay_study/__init__.py',
          'tools/scripts/archive_model_weights.py', 'provenance/extended_formal_freeze.json',
          'provenance/registry.json', 'provenance/input_snapshot.json', 'provenance/protocol.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signature(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


class Metadata:
    def __init__(self):
        self.evidence = {}; self.observed = {}; self.canonical = {}

    def opaque(self, path, expected=None):
        path = absolute(path); canonical = path.resolve(); stamp = signature(path)
        require(canonical.suffix not in {'.pt', '.zip', '.tar', '.safetensors'}, 'Payload reads forbidden in metadata reviewer')
        require(stamp['bytes'] <= 32 << 20, 'Small metadata only')
        raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
        require(signature(path) == stamp and (expected is None or digest == expected), 'Metadata changed/SHA mismatch: ' + str(path))
        self.evidence[str(canonical)] = dict(sha256=digest, stat=stamp)
        self.observed[str(path)] = stamp; self.canonical[str(path)] = str(canonical)
        return raw

    def read(self, path, expected=None):
        return json.loads(self.opaque(path, expected))

    def bound(self, proof):
        require(isinstance(proof, dict) and re.fullmatch('[0-9a-f]{64}', proof.get('sha256', '')), 'Explicit JSON/source proof required')
        return self.read(proof['path'], proof['sha256'])

    def payload_stat(self, path, expected=None):
        p = absolute(path); stamp = signature(p)
        require(stat.S_ISREG(p.stat().st_mode) and (expected is None or stamp == expected), 'Payload stat differs: ' + str(p))
        self.observed[str(p)] = stamp; self.canonical[str(p)] = str(p.resolve())
        return stamp

    def finish(self):
        for path, stamp in self.observed.items():
            require(signature(path) == stamp and str(Path(path).resolve()) == self.canonical[path], 'Input changed during metadata review')


def validate_headers(completion, manifest, model, profile):
    expected = (profile['logical'], profile['unique'], profile['aliases'])
    require(completion.get('status') == 'passed' and completion.get('model') == model and completion.get('phase') == profile['phase'], 'Complete matching original Cfinal receipt required')
    require(completion.get('fullphase_source_gate_passed') is True, 'Whole-phase source gate required')
    require(manifest.get('status') == 'complete_phase_all_final_roundtrips_verified' and manifest.get('model_tag') == model and manifest.get('phase_name') == profile['phase'], 'Wrong/incomplete Cfinal manifest')
    require(manifest.get('kind') == 'fixed_exposure_final' and manifest.get('role') == 'extended' and manifest.get('data_family') == 'extended_v1', 'Only original matched Cfinal scope accepted')
    require(manifest.get('optimizer_included') is False, 'Optimizer-free delivery expected')
    for value in (completion, manifest):
        require(tuple(value.get(k) for k in ('logical_final_endpoints', 'unique_weights', 'alias_count')) == expected, 'Logical/unique/alias inventory differs; ROOT must review actual proof, never force alias')
    require(all(completion.get(k) is True for k in ('file_fsync_completed', 'parent_directory_fsync_completed')), 'Completed TAR and parent fsync required')
    require(type(completion.get('verified_tar_members')) is int and completion['verified_tar_members'] > 0, 'Complete TAR member verification required')
    require(re.fullmatch('[0-9a-f]{64}', completion.get('tar_sha256', '')), 'Explicit full TAR SHA required')


def native_members(dependency, snapshot, model, revision):
    require(dependency.get('kind') == 'native_base_config_tokenizer_readme_included', 'Native self-contained base required; external primary dependency is not accepted')
    require(dependency.get('model_revision') == revision, 'Native base revision differs')
    expected = {str(Path(path)): dict(record, member='base/' + model + '-final/' + Path(path).name)
                for path, record in snapshot['model_files'].items()}
    files = dependency.get('files', [])
    require(len(files) == len(expected) and len({row.get('source') for row in files}) == len(files), 'Complete unique native input inventory required')
    require({Path(path).name for path in expected} == {'model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'README.md'}, 'All six native model/config/tokenizer/README files required')
    for row in files:
        require(row.get('source') in expected and {key: row.get(key) for key in ('sha256', 'bytes', 'member')} == expected[row['source']], 'Native file SHA/bytes/member differs from frozen model input')
    base = next(v for path, v in expected.items() if Path(path).name == 'model.safetensors')
    require(dependency.get('base_sha256') == base['sha256'], 'Native base model SHA differs')
    return {row['member'] for row in files}


def member_inventory(manifest):
    names = []
    for row in manifest['logical_runs']:
        names.append(row['metadata_member'])
        if 'alias_of' not in row:
            names.append(row['archive_member'])
    names.extend(row['member'] for row in manifest['base_dependency']['files'])
    names.extend(EXTRAS)
    sources = []
    for source in manifest['fullphase_gate']['sources']:
        if source['path'] not in sources:
            sources.append(source['path'])
            names.append('provenance/gate/' + str(len(sources) - 1) + '_' + Path(source['path']).name)
    require(len(names) == len(set(names)), 'Duplicate derived TAR member names')
    return names


def bind_frozen_row_sources(meta, item, identity, frozen, snapshot, source_root=ROOT):
    """Fresh small-file SHA crossbinding; checkpoint payload is never read."""
    source_root = Path(source_root).resolve()
    def canonical(path):
        path = Path(path)
        return (path if path.is_absolute() else source_root / path).resolve()
    require(identity.get('kind') == 'final' and item['run_id'] == identity.get('run_id') == identity.get('actual_run_id') == frozen['id'], 'Frozen final actual-run identity differs')
    require(identity.get('model_tag') == frozen['model'], 'Frozen identity model differs')
    require(canonical(item['source_checkpoint']) == canonical(identity['checkpoint']) == canonical(frozen['checkpoint_path']), 'Checkpoint canonical path differs from frozen final identity')
    for name in ('config_path', 'summary_path'):
        require(canonical(identity[name]) == canonical(frozen[name]), 'Frozen final source canonical path differs: ' + name)
    try:
        relative = str(canonical(frozen['config_path']).relative_to(source_root))
    except ValueError as error:
        raise ValueError('Frozen config is outside the source workspace') from error
    require(relative in snapshot['configs'], 'Frozen snapshot does not cover config')
    for name, digest_name in [('config_path', 'config_sha256'), ('summary_path', 'summary_sha256')]:
        path = canonical(frozen[name])
        matches = [value for key, value in identity['input_sha256'].items() if canonical(key) == path]
        require(len(matches) == 1 and re.fullmatch('[0-9a-f]{64}', matches[0]) and item[digest_name] == matches[0], 'Row source SHA differs from frozen READY: ' + name)
        if name == 'config_path':
            require(snapshot['configs'][relative]['sha256'] == matches[0], 'Config SHA differs between snapshot and frozen READY')
        raw = meta.opaque(frozen[name], matches[0])  # Fresh bytes, no summary JSON/score parsing.
        if name == 'config_path':
            require(len(raw) == snapshot['configs'][relative]['bytes'], 'Config byte length differs from snapshot')
    observed = meta.payload_stat(item['source_checkpoint'], item['source_stat'])
    require(type(identity.get('checkpoint_bytes')) is int and type(identity.get('checkpoint_mtime_ns')) is int and
            observed['bytes'] == identity['checkpoint_bytes'] and observed['mtime_ns'] == identity['checkpoint_mtime_ns'],
            'Checkpoint size/mtime differs from frozen final identity')


def validate_roundtrip_report(item, base_sha256):
    """Prevent mixing individually valid reports from distinct draft sources."""
    encoded, decoded, verified = (item[name] for name in ('encode', 'decode', 'bytewise_verification'))
    fp = item['model_fingerprint']
    require(re.fullmatch('[0-9a-f]{64}', fp.get('sha256_model_state', '')) and type(fp.get('state_elements')) is int and fp['state_elements'] > 0, 'Complete row model fingerprint required')
    require(encoded.get('status') == 'encoded' and decoded.get('status') == 'decoded_and_hash_verified' and verified.get('status') == 'passed', 'All roundtrip stages must be complete')
    require(all(stage.get('model_fingerprint') == fp for stage in (encoded, decoded, verified)), 'Roundtrip stage fingerprints differ from row')
    require(encoded.get('source_checkpoint_sha256') == verified.get('source_checkpoint_sha256') == item['source_checkpoint_sha256'], 'Nested source checkpoint SHA differs')
    require(encoded.get('archive_sha256') == verified.get('archive_sha256') == item['archive_sha256'], 'Nested ZIP SHA differs')
    require(Path(encoded['archive']).resolve() == Path(decoded['archive']).resolve() == Path(item['archive_path']).resolve(), 'Nested ZIP canonical path differs')
    require(encoded.get('archive_bytes') == item['archive_stat']['bytes'] and encoded.get('source_checkpoint_bytes_including_optimizer') == item['source_stat']['bytes'], 'Nested ZIP/checkpoint byte lengths differ')
    require(encoded.get('base_sha256') == base_sha256, 'Nested encode base SHA differs from frozen native base')
    count = decoded.get('tensor_raw_hashes_verified')
    require(type(count) is int and count > 0 and type(verified.get('tensor_count')) is int and count == verified['tensor_count'], 'Complete decoded tensor hash inventory must match bytewise verification')
    require(encoded.get('model_only') is True and encoded.get('lossless_weight_bits') is True and decoded.get('model_only_evaluation_format') is True and verified.get('all_tensor_bytes_equal') is True, 'Complete lossless model-only roundtrip flags required')


def review(model):
    profile = PROFILES[model]; folder = DELIVERY / model; meta = Metadata()
    for path, digest in PIN.items():
        meta.opaque(ROOT / path, digest)
    freeze = meta.read(ROOT / 'results/extended_formal_freeze.json', FREEZE_SHA)
    phase = next(p for p in freeze['phases'] if p['model'] == model)
    require(phase['phase'] == profile['phase'] and phase['runs'] == profile['logical'] and phase['target_prediction_tokens'] == profile['target'], 'Frozen model/horizon differs')
    registry = meta.read(phase['active_registry'], phase['active_registry_sha256'])
    snapshot = meta.read(phase['snapshot'], phase['snapshot_sha256'])
    protocol = meta.read(phase['protocol'], phase['protocol_sha256'])
    definitions = {r['id']: r for r in meta.read(snapshot['queue_path'], snapshot['queue_sha256'])}
    require(len(registry) == profile['logical'] and {r['id'] for r in registry} == set(definitions), 'Whole frozen registry/queue required')
    completion = meta.read(folder / 'completion.json')  # Missing remains pending, never a fabricated pass.
    manifest = meta.read(folder / 'extended_C_final_manifest.json', completion['manifest_sha256'])
    validate_headers(completion, manifest, model, profile)
    require(manifest['phase_source'] == phase and manifest['freeze']['sha256'] == FREEZE_SHA, 'Package frozen phase provenance differs')
    gate = manifest['fullphase_gate']; require(gate.get('status') == 'complete_source_gate_passed', 'Complete source audit required')
    identity = gate['identity_manifest']; ready = meta.bound(identity['published_ready'])
    evaluation = meta.read(identity['path'], identity['sha256'])
    require(ready.get('status') == 'ready' and ready.get('ready') is True and ready.get('id') == profile['ready'] and ready.get('mode') == 'nll', 'Actual complete matched NLL READY required')
    require(ready['manifest'] == dict(path=identity['path'], sha256=identity['sha256']), 'READY/identity manifest differs')
    finals = {r['run_id']: r for r in evaluation['records'] if r['kind'] == 'final'}
    require(len(finals) == profile['logical'] and finals == identity['records'], 'Every frozen final endpoint must be present')
    sources = {str(absolute(p['path']).resolve()): p for p in gate['sources']}
    if model == 'pythia-1b':
        path = ROOT / 'results/extended_phase_equivalence' / profile['phase'] / 'completion.json'
        phase_completion = meta.read(path)
        require(phase_completion.get('status') == 'passed' and phase_completion.get('model') == model and phase_completion.get('phase') == profile['phase'] and phase_completion.get('expected_runs') == phase_completion.get('completed_runs') == 24 and phase_completion.get('comparisons_computed') == 6, 'Actual1B full-phase audit and final/best proof completion required')
        audit = meta.bound(phase_completion['audit']); finalproof = meta.bound(phase_completion['proofs']['final'])
        require(audit.get('status') == 'passed' and audit.get('source_violations') == [] and audit['source_state_complete_gate'].get('ready') is True and audit['source_state_complete_gate'].get('completed_runs') == 24, 'Actual1B full24 source audit required')
        require(finalproof.get('status') == 'passed' and finalproof.get('model') == model and finalproof.get('phase') == profile['phase'] and finalproof.get('endpoint') == 'final' and finalproof.get('algorithm') == 'sha256_sorted_key_dtype_shape_contiguous_uint8_v1', 'Actual complete-state final proof required')
        require(finalproof['source_audit'] == phase_completion['audit'] and finalproof['source_registry'] == phase_completion['source_registry'] and finalproof['source_snapshot'] == phase_completion['input_snapshot'], 'Proof audit/registry/snapshot bindings differ')
        require(phase_completion['source_registry']['sha256'] == phase['active_registry_sha256'] and phase_completion['input_snapshot']['sha256'] == phase['snapshot_sha256'], 'Proof source is not this frozen phase')
        pairs = finalproof['comparisons']
        require(len(pairs) == 3 and {p['seed'] for p in pairs} == {11, 22, 33}, 'Three actual same-seed final decisions required')
    else:
        path = folder / 'fullphase_source_audit.json'
        require(str(path.resolve()) in sources, 'Actual2.8B audit must be SHA-bound in gate sources')
        audit = meta.bound(sources[str(path.resolve())]); pairs = []
        require(audit.get('status') == 'passed' and audit.get('model') == model and audit.get('phase') == profile['phase'] and audit.get('expected_runs') == audit.get('completed_runs') == 15 and audit.get('violations') == [] and audit.get('snapshot_sha256') == phase['snapshot_sha256'], 'Actual2.8B complete15 source audit required')
    rows = manifest['logical_runs']; byid = {r['run_id']: r for r in rows}; frozen_rows = {r['id']: r for r in registry}
    require(len(rows) == len(byid) == profile['logical'] and set(byid) == set(finals) == set(frozen_rows), 'Complete logical Cfinal row inventory required')
    aliases = []
    for item in rows:
        identity_row = finals[item['run_id']]; frozen = frozen_rows[item['run_id']]; definition = definitions[item['run_id']]
        bind_frozen_row_sources(meta, item, identity_row, frozen, snapshot)
        digest = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
        require(item['definition'] == definition and item['definition_sha256'] == digest, 'Frozen training definition differs')
        child = item['completed_scheduler_record']
        require(child.get('status') == 'completed' and type(child.get('returncode')) is int and child['returncode'] == 0 and not child.get('missing_outputs') and 'stop_requested_at' not in child and 'stop_requested' not in child and child.get('argv') == definition['argv'] and child.get('definition_sha256') == digest, 'Completed un-stopped rc0/actual argv required')
        require(item['source_checkpoint_sha256'] == identity_row['checkpoint_sha256'] and item['model_revision'] == protocol['model_revision'] and item['input_snapshot_sha256'] == phase['snapshot_sha256'], 'Final source identity differs')
        require(item['group'] == frozen['group'] and item['seed'] == frozen['seed'] and item['ratio'] == frozen['ratio'] and item['weight'] == frozen['weight'], 'Frozen method/seed/ratio/weight differs')
        exposure = item['actual_exposure']; replay = int(Decimal(str(item['ratio'])) * profile['target'])
        require(exposure['target_prediction_tokens'] == profile['target'] and exposure['replay_prediction_tokens'] == replay and exposure['checkpoint_step'] == identity_row['step'], 'Fixed-final exposure/step differs')
        require(item['all_nonweight_metadata_roundtrip_verified'] is True, 'Full nonoptimizer metadata roundtrip required')
        for key, statkey in [('source_checkpoint', 'source_stat'), ('metadata_path', 'metadata_stat'), ('archive_path', 'archive_stat')]:
            if key in item:
                meta.payload_stat(item[key], item[statkey])
        if 'alias_of' in item:
            parent = byid[item['alias_of']]
            require(model == 'pythia-1b' and item['group'] == 'D' and parent['group'] == 'B4' and item['seed'] == parent['seed'] and item['model_fingerprint'] == parent['model_fingerprint'] and item['additional_encoding_calls'] == 0, 'Alias must be exact same-seed D/B4 final')
            aliases.append(item)
        else:
            require(item['additional_encoding_calls'] == 1 and item['bytewise_verification'].get('all_tensor_bytes_equal') is True, 'Every unique weight needs complete actual bytewise roundtrip')
            validate_roundtrip_report(item, manifest['base_dependency']['base_sha256'])
    require(len(aliases) == profile['aliases'], 'Actual alias count differs')
    for pair in pairs:
        require(pair.get('endpoint') == 'final' and pair.get('kind') == 'identical_bank_null_control' and pair.get('bitwise_model_state_equal') is True and pair['left']['model_state_fingerprint'] == pair['right']['model_state_fingerprint'], 'Expected21unique requires all three true complete-state decisions; do not force alias')
        sides = [pair['left'], pair['right']]
        require({byid[s['run_id']]['group'] for s in sides} == {'B4', 'D'} and all(byid[s['run_id']]['seed'] == pair['seed'] for s in sides), 'Cross-seed/model pair rejected')
        for side in sides:
            require(side['checkpoint_sha256'] == finals[side['run_id']]['checkpoint_sha256'] and side['model_state_fingerprint'] == byid[side['run_id']]['model_fingerprint'], 'Final proof must bind actual checkpoint/fingerprint')
    native_members(manifest['base_dependency'], snapshot, model, protocol['model_revision'])
    for native in manifest['base_dependency']['files']:
        require(meta.payload_stat(native['source'])['bytes'] == native['bytes'], 'Native file size differs')
    names = member_inventory(manifest)
    require(completion['verified_tar_members'] == len(names), 'Actual verified TAR member count differs from derived complete inventory')
    grow = meta.read(ROOT / 'artifacts/provenance/supplemental_weight_reservations_v1' / ('C_' + profile['tag'] + '_growth_receipt.json'))
    reserved = PERSISTENT / ('.extended_C_' + profile['tag'] + '_final_weights_reserve_20261106')
    reserve = reserved.lstat(); expected = grow['actual_stat']
    require(grow.get('status') == 'complete' and grow.get('lock_released') is True and grow.get('final_file_fsync_completed') is True, 'Actual completed unlocked/fsynced growth receipt required')
    require(stat.S_ISREG(reserve.st_mode) and (reserve.st_dev, reserve.st_ino, reserve.st_uid, reserve.st_gid, stat.S_IMODE(reserve.st_mode), reserve.st_nlink) == (106, profile['inode'], 11471, 11471, 0o600, 1), 'Original reserve identity/owner/mode/link count differs')
    require(reserve.st_size == reserve.st_blocks * 512 == profile['limit'] == expected['bytes'] == expected['allocated_bytes'] and expected['inode'] == reserve.st_ino, 'Reserve not fully physically allocated at original ceiling')
    require(not Path('/proc', str(grow['pid'])).exists(), 'Old grower PID exists; ROOT must inspect/reap')
    lock_key = f'{os.major(reserve.st_dev):02x}:{os.minor(reserve.st_dev):02x}:{reserve.st_ino}'
    require(not any(lock_key in line for line in Path('/proc/locks').read_text().splitlines()), 'Reserve currently locked')
    tar = folder / ('extended_C_' + profile['tag'] + '_final_weights_lossless_v1.tar'); destination = PERSISTENT / tar.name
    require(Path(completion['tar_path']) == tar and meta.payload_stat(tar)['bytes'] == completion['tar_bytes'] <= profile['limit'], 'Complete TAR size/path differs or exceeds original ceiling')
    require(not destination.exists() and not destination.is_symlink(), 'Destination already exists')
    meta.finish()
    return {'schema_version': 1, 'status': 'passed_metadata_only_for_ROOT_review_not_publication_authority',
            'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'model': model, 'phase': profile['phase'],
            'metadata_sources': meta.evidence, 'helper_SHA': PIN, 'source': str(tar),
            'expected_sha256': completion['tar_sha256'], 'expected_bytes': completion['tar_bytes'],
            'reserved': str(reserved), 'reserve_original': dict(device=reserve.st_dev, inode=reserve.st_ino, uid=reserve.st_uid, gid=reserve.st_gid, mode='0o600', allocated_bytes=reserve.st_blocks * 512),
            'destination': str(destination), 'logical_final_endpoints': profile['logical'], 'unique_full_roundtrips_reported': profile['unique'],
            'D_aliases': [{'run_id': a['run_id'], 'alias_of': a['alias_of'], 'seed': a['seed']} for a in aliases],
            'derived_verified_TAR_member_count': len(names), 'native_base_dependency': manifest['base_dependency'],
            'native_base_model_payload_rehashed_by_reviewer': False,
            'all_TAR_members_and_full_SHA_verified_by_original_exporter_receipt': True,
            'reviewer_fresh_TAR_or_tensor_verification': False, 'publisher_must_fresh_source_copy_and_reserved_prefix_SHA': True,
            'offline_restore_requires_only_this_durable_package_and_standard_dependencies': True,
            'reserve_lock_observed_free_not_a_lease': True, 'source_payload_stats': meta.observed,
            'shared_writes_or_publish_or_GPU_or_score_or_large_payload_reads': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=sorted(PROFILES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = review(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'output': str(args.output), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(), 'not_authority': True}))


if __name__ == '__main__':
    main()
