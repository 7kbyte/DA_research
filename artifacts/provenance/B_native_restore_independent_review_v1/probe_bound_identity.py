#!/usr/bin/env python3
"""Only small metadata/source and durable TAR lstat. Never execute restoration."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREP = ROOT / 'artifacts/provenance/B_native_restore_preparation_v1'
evidence = {}

def read(path, digest=None):
    path = Path(path)
    assert path.suffix in ('.py', '.json', '.md', '.log')
    assert path.stat().st_size < 32 << 20
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    assert digest is None or digest == actual
    evidence[str(path)] = dict(sha256=actual, bytes=len(raw))
    return raw

source = read(PREP / 'run_B_restore.py', 'e3aa2b08e5361ab7fee92ef7a2f20042df80c1dd57f4d64a58e7285127f37d41')
retained = HERE / 'retained_e3aa_run_B_restore.py'
with retained.open('xb') as f:
    f.write(source)
spec = importlib.util.spec_from_file_location('independent_B_identity_before_lock_fix', retained)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
plan = json.loads(read(PREP / 'bound_plan.json', '903baac7f57172387bbda7d8f31785d3ac5cd29fe52739b5c856e12d6e74663b'))
runner.validate(plan)
values = {name: json.loads(read(ref['path'], ref['sha256'])) for name, ref in plan['evidence'].items()}
manifest = values['manifest']; row = plan['selected']
matches = [r for r in manifest['logical_runs'] if r['run_id'] == runner.RUN]
assert len(matches) == 1 and matches[0] == row
assert manifest['package'] == 'B_all_final' and manifest['kind'] == 'final'
assert manifest['logical_endpoints'] == manifest['unique_weights'] == 60 and manifest['aliases'] == 0
assert row['actual_exposure'] == dict(checkpoint_step=306, target_prediction_tokens=10_000_000,
    replay_prediction_tokens=500_000, full_training_target_prediction_tokens=10_000_000,
    full_training_replay_prediction_tokens=500_000)
scope = values['scope']
expected_names = {f['member'] for f in manifest['base_dependency']['files']}
expected_names |= {'additional_weights_manifest.json', row['archive_member'], row['metadata_member']}
tools = {'tools/scripts/archive_model_weights.py': 'scripts/archive_model_weights.py',
         'tools/src/replay_study/weight_archive.py': 'src/replay_study/weight_archive.py',
         'tools/src/replay_study/__init__.py': 'src/replay_study/__init__.py'}
expected_names.update(tools)
assert set(plan['members']) == expected_names and len(expected_names) == 12
for member, live in tools.items():
    raw = read(ROOT / live, plan['members'][member]['sha256'])
    assert len(raw) == plan['members'][member]['bytes']
for base in manifest['base_dependency']['files']:
    frozen = scope['frozen_files'][str(Path(base['path']).resolve())]
    assert plan['members'][base['member']]['sha256'] == base['sha256'] == frozen['sha256']
    assert plan['members'][base['member']]['bytes'] == frozen['stat']['bytes']
assert plan['members'][row['archive_member']]['sha256'] == row['archive_sha256']
assert plan['members'][row['archive_member']]['bytes'] == row['archive_stat']['bytes']
assert plan['members'][row['metadata_member']]['sha256'] == row['metadata_sha256']
assert sum(r['bytes'] for r in plan['members'].values()) == plan['selected_member_bytes'] == 2058150693

publication, acceptance = values['publication'], values['ROOT_acceptance']
assert acceptance['publication_receipt'] == publication
assert publication['status'] == 'passed' and publication['bytes'] == plan['archive_stat']['bytes']
assert publication['source_sha256'] == publication['destination_sha256'] == plan['historical_full_TAR_sha256']
assert publication['destination'] == plan['archive']
for key in ('device', 'inode', 'bytes', 'uid', 'gid', 'mode', 'mtime_ns', 'ctime_ns'):
    assert plan['archive_stat'][key] == publication['destination_stat'][key] == acceptance['destination_current_stat'][key]
for ref in (acceptance['publication'], acceptance['metadata_review']):
    read(ref['path'], ref['sha256'])

# Only lstat this real large path: no header or payload read.
tar = Path(plan['archive']); stamp = tar.lstat()
assert not tar.is_symlink()
actual = dict(device=stamp.st_dev, inode=stamp.st_ino, bytes=stamp.st_size,
    uid=stamp.st_uid, gid=stamp.st_gid, mode=oct(stamp.st_mode & 0o7777),
    mtime_ns=stamp.st_mtime_ns, ctime_ns=stamp.st_ctime_ns)
assert actual == plan['archive_stat']
for name in ('qa_directory', 'actual_evidence_directory'):
    assert not Path(plan[name]).exists() and not Path(plan[name]).is_symlink()

rejections = []
for label, mutate in [
    ('other_draw', lambda p: p['selected'].update(selection_seed=1002)),
    ('other_model', lambda p: p.update(model='pythia-1b')),
    ('other_full_FP', lambda p: p['selected']['model_fingerprint'].update(sha256_model_state='0'*64)),
    ('other_member_count', lambda p: p['members'].pop('additional_weights_manifest.json')),
    ('less_margin', lambda p: p['resource_policy'].update(admission_margin_seconds=299))]:
    bad = copy.deepcopy(plan); mutate(bad)
    try:
        runner.validate(bad)
    except ValueError:
        rejections.append(label)
    else:
        raise AssertionError(label)

result = dict(status='passed_bound_identity_metadata_and_stat_probe_before_lock_repair',
    fixed_run=runner.RUN, selected_members=12, selected_payload_bytes=2058150693,
    actual_manifest_row_exact=True, actual_published_TAR_stat_matches=True,
    actual_TAR_header_or_payload_read=False, any_restore_execute_called=False,
    original_CP_ZIP_base_or_scores_opened=False, fixed_scope_mutations_rejected=rejections,
    source_evidence=evidence, open_finding='New B body does not acquire shared C native restore lock; author notified')
with (HERE / 'bound_identity_probe.json').open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
print(json.dumps(dict(status=result['status'], small_source_or_metadata_paths=len(evidence),
    TAR_payload_bytes_read=0, rejected_scope_changes=len(rejections))))
