"""Synthetic metadata only; never run review() on actual future packages."""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('native_final_metadata_reviewer', Path(__file__).with_name('review_metadata.py'))
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def headers(model):
    p = r.PROFILES[model]
    counts = dict(logical_final_endpoints=p['logical'], unique_weights=p['unique'], alias_count=p['aliases'])
    completion = dict(counts, status='passed', model=model, phase=p['phase'], fullphase_source_gate_passed=True,
                      file_fsync_completed=True, parent_directory_fsync_completed=True, verified_tar_members=49,
                      tar_sha256='a' * 64)
    manifest = dict(counts, status='complete_phase_all_final_roundtrips_verified', model_tag=model,
                    phase_name=p['phase'], kind='fixed_exposure_final', role='extended', data_family='extended_v1',
                    optimizer_included=False)
    return completion, manifest, p


@pytest.mark.parametrize('model', sorted(r.PROFILES))
@pytest.mark.parametrize('change', ['partial', 'not_fsynced', 'wrong_unique', 'best_not_final'])
def test_complete_matching_headers_required(model, change):
    c, m, p = headers(model)
    r.validate_headers(c, m, model, p)
    if change == 'partial':
        c['fullphase_source_gate_passed'] = False
    elif change == 'not_fsynced':
        c['parent_directory_fsync_completed'] = False
    elif change == 'wrong_unique':
        m['unique_weights'] -= 1
    else:
        m['kind'] = 'model_only_best'
    with pytest.raises(ValueError):
        r.validate_headers(c, m, model, p)


def native(model):
    names = ['model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'README.md']
    snapshot = {'model_files': {'/synthetic/' + model + '/' + name: {'sha256': 'a' * 64, 'bytes': 3} for name in names}}
    files = [dict(source=path, **record, member='base/' + model + '-final/' + Path(path).name)
             for path, record in snapshot['model_files'].items()]
    dep = dict(kind='native_base_config_tokenizer_readme_included', model_revision='fixed', base_sha256='a' * 64, files=files)
    return dep, snapshot


@pytest.mark.parametrize('change', ['external', 'missing_readme', 'sha', 'revision', 'member'])
def test_native_six_file_closure_is_not_external_primary(change):
    model = 'pythia-2.8b'; dep, snap = native(model)
    assert len(r.native_members(dep, snap, model, 'fixed')) == 6
    if change == 'external':
        dep['kind'] = 'durable_external_primary_base_closure'
    elif change == 'missing_readme':
        dep['files'].pop()
    elif change == 'sha':
        dep['base_sha256'] = 'b' * 64
    elif change == 'revision':
        dep['model_revision'] = 'other'
    else:
        dep['files'][0]['member'] = 'base/wrong/model.safetensors'
    with pytest.raises(ValueError):
        r.native_members(dep, snap, model, 'fixed')


def test_member_count_derived_from_inventory_not_145():
    dep, _ = native('pythia-2.8b')
    manifest = dict(logical_runs=[dict(metadata_member=f'metadata/{i}.pt', archive_member=f'weights/{i}.zip') for i in range(15)],
                    base_dependency=dep, fullphase_gate={'sources': [{'path': '/fake/manifest.json'}, {'path': '/fake/READY.json'}, {'path': '/fake/audit.json'}]})
    assert len(r.member_inventory(manifest)) == 15 + 15 + 6 + 10 + 3 == 49
    manifest['logical_runs'][1]['metadata_member'] = manifest['logical_runs'][0]['metadata_member']
    with pytest.raises(ValueError, match='Duplicate'):
        r.member_inventory(manifest)


def test_even_tiny_checkpoint_payload_is_never_read(tmp_path):
    p = tmp_path / 'checkpoint.pt'; p.write_bytes(b'private')
    with pytest.raises(ValueError, match='Payload reads forbidden'):
        r.Metadata().opaque(p)


def test_missing_future_completion_does_not_create_pass_receipt(tmp_path):
    target = tmp_path / 'pending/completion.json'
    with pytest.raises(FileNotFoundError):
        r.Metadata().read(target)
    assert not target.exists()
