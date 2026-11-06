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


def frozen_sources(tmp_path):
    import hashlib
    config = tmp_path / 'configs/run.json'; config.parent.mkdir(); config.write_bytes(b'CONFIG_OPAQUE')
    summary = tmp_path / 'summary.json'; summary.write_bytes(b'SEALED_SUMMARY_NOT_JSON')
    cp = tmp_path / 'checkpoint.pt'; cp.write_bytes(b'CHECKPOINT_MUST_NEVER_BE_READ')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    frozen = dict(id='run', model='pythia-1b', config_path=str(config), summary_path=str(summary), checkpoint_path=str(cp))
    identity = dict(kind='final', run_id='run', actual_run_id='run', model_tag='pythia-1b', config_path=str(config), summary_path=str(summary),
                    checkpoint=str(cp), checkpoint_bytes=cp.stat().st_size, checkpoint_mtime_ns=cp.stat().st_mtime_ns,
                    input_sha256={str(config): sha(config), str(summary): sha(summary)})
    item = dict(run_id='run', source_checkpoint=str(cp), source_stat=r.signature(cp), config_sha256=sha(config), summary_sha256=sha(summary))
    snapshot = dict(configs={'configs/run.json': {'sha256': sha(config), 'bytes': config.stat().st_size}})
    return item, identity, frozen, snapshot


@pytest.mark.parametrize('change', ['config_row_sha', 'summary_row_sha', 'snapshot_config_sha', 'current_config', 'current_summary',
                                   'identity_config_path', 'identity_summary_path', 'row_checkpoint_path', 'identity_checkpoint_path',
                                   'checkpoint_size', 'checkpoint_mtime', 'actual_run', 'best_identity'])
def test_frozen_READY_snapshot_and_current_source_crossbinding_rejects(tmp_path, change, monkeypatch):
    item, identity, frozen, snapshot = frozen_sources(tmp_path)
    # This valid base case proves the deliberately non-JSON summary remains
    # opaque; monkeypatch additionally forbids checkpoint payload reads.
    old_read = Path.read_bytes
    def read(path):
        assert path.suffix != '.pt'
        return old_read(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    r.bind_frozen_row_sources(r.Metadata(), item, identity, frozen, snapshot, tmp_path)
    if change == 'config_row_sha':
        item['config_sha256'] = 'b' * 64
    elif change == 'summary_row_sha':
        item['summary_sha256'] = 'b' * 64
    elif change == 'snapshot_config_sha':
        snapshot['configs']['configs/run.json']['sha256'] = 'b' * 64
    elif change == 'current_config':
        Path(frozen['config_path']).write_bytes(b'CHANGED_CONFIG')
    elif change == 'current_summary':
        Path(frozen['summary_path']).write_bytes(b'CHANGED_SUMMARY')
    elif change in ('identity_config_path', 'identity_summary_path'):
        identity[change.removeprefix('identity_')] = str(tmp_path / 'other.json')
    elif change == 'row_checkpoint_path':
        item['source_checkpoint'] = str(tmp_path / 'other.pt')
    elif change == 'identity_checkpoint_path':
        identity['checkpoint'] = str(tmp_path / 'other.pt')
    elif change == 'checkpoint_size':
        identity['checkpoint_bytes'] += 1
    elif change == 'checkpoint_mtime':
        identity['checkpoint_mtime_ns'] += 1
    elif change == 'actual_run':
        identity['actual_run_id'] = 'other_run'
    else:
        identity['kind'] = 'best'
    with pytest.raises(ValueError):
        r.bind_frozen_row_sources(r.Metadata(), item, identity, frozen, snapshot, tmp_path)


def roundtrip():
    fp = dict(sha256_model_state='a' * 64, state_elements=10)
    return dict(model_fingerprint=fp, source_checkpoint_sha256='b' * 64, archive_sha256='c' * 64,
                archive_path='/synthetic/run.zip', archive_stat={'bytes': 100}, source_stat={'bytes': 400},
                encode=dict(status='encoded', model_fingerprint=fp, source_checkpoint_sha256='b' * 64,
                            archive_sha256='c' * 64, archive='/synthetic/run.zip', archive_bytes=100,
                            source_checkpoint_bytes_including_optimizer=400, base_sha256='d' * 64,
                            model_only=True, lossless_weight_bits=True),
                decode=dict(status='decoded_and_hash_verified', model_fingerprint=fp, archive='/synthetic/run.zip',
                            tensor_raw_hashes_verified=2, model_only_evaluation_format=True),
                bytewise_verification=dict(status='passed', model_fingerprint=fp, source_checkpoint_sha256='b' * 64,
                                           archive_sha256='c' * 64, tensor_count=2, all_tensor_bytes_equal=True))


@pytest.mark.parametrize('change', ['FP', 'cp_sha', 'zip_sha', 'zip_path', 'zip_bytes', 'source_bytes', 'base_sha', 'decode_status', 'hash_count'])
def test_nested_roundtrip_reports_cannot_be_spliced(change):
    item = roundtrip()
    r.validate_roundtrip_report(item, 'd' * 64)
    if change == 'FP':
        item['decode']['model_fingerprint'] = dict(sha256_model_state='e' * 64, state_elements=10)
    elif change == 'cp_sha':
        item['bytewise_verification']['source_checkpoint_sha256'] = 'e' * 64
    elif change == 'zip_sha':
        item['bytewise_verification']['archive_sha256'] = 'e' * 64
    elif change == 'zip_path':
        item['decode']['archive'] = '/synthetic/other.zip'
    elif change == 'zip_bytes':
        item['encode']['archive_bytes'] += 1
    elif change == 'source_bytes':
        item['encode']['source_checkpoint_bytes_including_optimizer'] += 1
    elif change == 'base_sha':
        item['encode']['base_sha256'] = 'e' * 64
    elif change == 'decode_status':
        item['decode']['status'] = 'partial'
    else:
        item['decode']['tensor_raw_hashes_verified'] = 1
    with pytest.raises(ValueError):
        r.validate_roundtrip_report(item, 'd' * 64)


def test_future_publisher_command_explicitly_supplies_required_progress():
    root = Path(__file__).resolve().parents[3]
    doc = (root / 'docs/C1B与C2p8B原final权重包持久发布runbook_v2.md').read_text()
    assert "'--progress', progress" in doc
    assert 'docs/C1B_final_weights_durable_publication.progress.json' in doc
    assert '\n+  docs/' not in doc
