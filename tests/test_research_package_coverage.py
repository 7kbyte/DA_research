import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('package_research', Path(__file__).parents[1] / 'scripts/package_research.py')
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


def create(path, value=b'fixture'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def test_token_tensors_font_and_dependency_lock_kept_runtime_junk_omitted(tmp_path):
    root = tmp_path / 'project'
    wanted = ['artifacts/data/train.pt', 'artifacts/generated/pool_stats.pt',
              'artifacts/reporting/font.otf', 'requirements-report.lock', 'uv.lock',
              'poetry.lock', 'results/supplement_A/selection.json']
    unwanted = ['logs/worker.pid', 'artifacts/cache/sha.lock', 'logs/growing.tmp',
                'src/__pycache__/a.pyc', '.pytest_cache/state', '.venv/lib/module.py']
    for name in wanted + unwanted:
        create(root / name)
    paths, omitted, external = package.collect_paths(root)
    assert {name for _, name in paths} == set(wanted)
    assert not omitted and not external


def test_external_runrecords_explicit_and_weights_stay_external(tmp_path):
    root, runs = tmp_path / 'project', tmp_path / 'runs'
    create(root / 'README.md')
    for name in ['formal/config.json', 'formal/metrics.jsonl', 'formal/worker.log',
                 'extended/checkpoints/latest.pt', 'formal/checkpoints/best.pt']:
        create(runs / name)
    paths, omitted, weights = package.collect_paths(root, runs, require_run_records=True, fail_on_symlinks=True)
    assert {name for _, name in paths} == {'README.md', 'run_records/formal/config.json',
        'run_records/formal/metrics.jsonl', 'run_records/formal/worker.log'}
    assert len(weights) == 2 and all(not r['bundled'] for r in weights)
    assert not omitted
    with pytest.raises(ValueError, match='explicit --runs-root'):
        package.collect_paths(root, require_run_records=True)
    with pytest.raises(ValueError, match='existing non-symlink'):
        package.collect_paths(root, tmp_path / 'missing')


@pytest.mark.parametrize('external', [False, True])
def test_symlink_target_never_silently_flattened_or_omitted_in_final_mode(tmp_path, external):
    root, runs = tmp_path / 'project', tmp_path / 'runs'
    create(root / 'README.md')
    create(runs / 'formal/config.json')
    target = tmp_path / 'outside'
    create(target / 'tokens.pt')
    link = (runs if external else root) / 'linked'
    link.symlink_to(target, target_is_directory=True)
    _, omitted, _ = package.collect_paths(root, runs)
    assert len(omitted) == 1 and omitted[0]['path'] == ('run_records/' if external else '') + 'linked'
    with pytest.raises(ValueError, match='Unmaterialized symlinks'):
        package.collect_paths(root, runs, fail_on_symlinks=True)


def test_empty_runroot_and_member_collision_rejected(tmp_path):
    root, runs = tmp_path / 'project', tmp_path / 'runs'
    create(root / 'run_records/formal/config.json')
    runs.mkdir()
    with pytest.raises(ValueError, match='No external run records'):
        package.collect_paths(root, runs, require_run_records=True)
    create(runs / 'formal/config.json')
    with pytest.raises(ValueError, match='Duplicate archive member'):
        package.collect_paths(root, runs)


def test_synthetic_archive_manifest_covers_every_selected_payload(tmp_path):
    root, runs = tmp_path / 'project', tmp_path / 'runs'
    create(root / 'requirements-report.lock', b'weasyprint==66.0\n')
    create(root / 'artifacts/data/tokens.pt', b'token fixture')
    create(runs / 'A/config.json', b'{"fixture":true}')
    paths, omitted, external = package.collect_paths(root, runs, require_run_records=True, fail_on_symlinks=True)
    output = tmp_path / 'synthetic.tar.gz'
    package.write_archive(SimpleNamespace(label='synthetic_fixture'), output, paths, omitted, external)
    with tarfile.open(output) as archive:
        manifest = json.load(archive.extractfile('research/PACKAGE_MANIFEST.json'))
        assert len(manifest['files']) == 3
        for row in manifest['files']:
            payload = archive.extractfile('research/' + row['path']).read()
            assert len(payload) == row['bytes']
            assert hashlib.sha256(payload).hexdigest() == row['sha256']
    receipt = json.loads(output.with_name(output.name + '.receipt.json').read_text())
    assert receipt['verified_all_payloads']
