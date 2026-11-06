"""Temporary tiny archives only; no real delivery/reservation inputs are opened."""
import gzip
import hashlib
import importlib.util
import io
import json
import os
import stat
import tarfile
from pathlib import Path

import pytest

SOURCE = Path(__file__).with_name('restore_tree.py')
spec = importlib.util.spec_from_file_location('final_tree_restore_toy', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')
    return {'path': str(path), 'sha256': sha(path.read_bytes())}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Test-only owner substitution; real CLI remains pinned to 11471:11471.
    monkeypatch.setattr(m, 'TARGET_UID', os.getuid())
    monkeypatch.setattr(m, 'TARGET_GID', os.getgid())

    def build(entries=None, mutate_manifest=None, suffix=None, expected_sha=None, trailing_tar=b''):
        parent = tmp_path / 'delivery'
        parent.mkdir()
        original = tmp_path / 'original_workspace'
        original.mkdir()
        (original / 'KEEP.txt').write_text('original remains intact')
        entries = entries if entries is not None else [('research/a.txt', b'abc', tarfile.REGTYPE),
                   ('research/sub/run.py', b'print(1)\n', tarfile.REGTYPE)]
        files = [{'path': name.removeprefix('research/'), 'bytes': len(data), 'sha256': sha(data)}
                 for name, data, kind in entries if kind == tarfile.REGTYPE]
        manifest = {'files': files, 'omitted_symlinks': [], 'external_weights_not_included': []}
        if mutate_manifest:
            mutate_manifest(manifest)
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode='w') as tar:
            for name, data, kind in entries:
                entry = tarfile.TarInfo(name)
                entry.type, entry.mode = kind, 0o755
                entry.size = len(data) if kind == tarfile.REGTYPE else 0
                entry.linkname = 'a.txt' if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE) else ''
                tar.addfile(entry, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
            data = json.dumps(manifest).encode()
            entry = tarfile.TarInfo('research/PACKAGE_MANIFEST.json')
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
            if suffix:
                entry = tarfile.TarInfo('research/after_manifest.txt')
                entry.size = len(suffix)
                tar.addfile(entry, io.BytesIO(suffix))
        archive = parent / 'synthetic_future.tar.gz'
        archive.write_bytes(gzip.compress(raw.getvalue()+trailing_tar))
        archive_sha = expected_sha or sha(archive.read_bytes())
        reserve = parent / '.toy_reserve'
        reserve.write_bytes(bytes(1 << 20))  # Real tiny blocks, not sparse.
        reserve.chmod(0o600)
        reserve_stat = m.details(reserve.lstat())
        reservation = save(tmp_path / 'reserve_receipt.json', {
            'status': 'allocated_for_final_editable_tree_only', 'path': str(reserve), **reserve_stat})
        flags = {k: True for k in ('file_fsync_after_copy', 'prefix_sha_verified_before_truncate',
            'file_fsync_after_truncate', 'atomic_destination_commit_noreplace', 'parent_directory_fsync')}
        publication = save(tmp_path / 'publication.json', {
            'status': 'passed', 'destination': str(archive), 'source_sha256': archive_sha,
            'destination_sha256': archive_sha, 'destination_stat': m.details(archive.lstat()),
            'bytes': archive.stat().st_size, **flags})
        acceptance = save(tmp_path / 'acceptance.json', {
            'status': 'accepted_durable_final_snapshot', 'archive': str(archive),
            'sha256': archive_sha, 'bytes': archive.stat().st_size, 'publication': publication})
        adjacent = save(tmp_path / 'adjacent.json', {
            'archive': str(archive), 'sha256': archive_sha, 'bytes': archive.stat().st_size,
            'files': len(files), 'verified_all_payloads': True,
            'publication_receipt_sha256': publication['sha256']})
        namespace = parent / 'new_namespace'
        plan = {'namespace': str(namespace), 'publication_mode': m.PUBLICATION_MODE,
            'archive_path': str(archive), 'archive_sha256': archive_sha,
            'archive_stat5': m.signature(archive.lstat()), 'reserve_path': str(reserve),
            'reserve_stat': reserve_stat, 'protected_workspace': str(original),
            'owner_uid': os.getuid(), 'owner_gid': os.getgid(), 'publication': publication,
            'ROOT_publication_acceptance': acceptance, 'adjacent_receipt': adjacent,
            'reserve_receipt': reservation, 'ROOT_acceptance_status': 'accepted_durable_final_snapshot',
            'expected_payload_files': len(files), 'max_total_regular_bytes': 1 << 19}
        plan_path, auth_path = tmp_path / 'plan.json', tmp_path / 'authority.json'

        def bind():
            save(plan_path, plan)
            auth = {'authorized': True, 'authorized_by': 'root', 'action': 'restore_final_editable_tree',
                'plan_sha256': sha(plan_path.read_bytes()), 'tool_sha256': sha(SOURCE.read_bytes()),
                'publication_mode': plan.get('publication_mode'),
                'namespace': str(namespace), 'destination': str(namespace / 'research'),
                **{k: plan[k] for k in ('archive_path', 'archive_sha256', 'reserve_path', 'reserve_stat',
                                       'owner_uid', 'owner_gid')}}
            auth.update({k + '_sha256': plan[k]['sha256'] for k in
                ('publication', 'ROOT_publication_acceptance', 'adjacent_receipt', 'reserve_receipt')})
            save(auth_path, auth)
            return auth

        bind()
        return {'parent': parent, 'original': original, 'archive': archive, 'reserve': reserve,
            'namespace': namespace, 'plan': plan, 'plan_path': plan_path, 'auth_path': auth_path, 'bind': bind}
    return build


def execute(f):
    return m.restore(str(f['plan_path']), str(f['auth_path']))


def intact(f):
    assert (f['original'] / 'KEEP.txt').read_text() == 'original remains intact'
    assert f['archive'].is_file()


def test_full_stream_success_and_ownership(fixture):
    f = fixture()
    result = execute(f)
    tree = f['namespace'] / 'research'
    assert result['status'] == 'passed_verified_new_editable_tree_restored'
    assert result['compressed_stream_bytes_read'] == f['archive'].stat().st_size
    assert result['compressed_stream_sha256'] == sha(f['archive'].read_bytes())
    assert result['all_member_SHA_verified'] and result['tree_and_parent_fsync_before_marker_completed']
    assert result['atomic_directory_publication'] is False
    assert result['completion_marker_present'] is True and result['completion_marker_durability']=='parent_fsynced'
    marker_path = f['namespace'] / 'restore_evidence/completion.json'
    marker = json.loads(marker_path.read_bytes())
    assert marker['status']=='verified_tree_ready_marker' and marker['candidate']==str(tree)
    assert marker['plan_sha256']==sha(f['plan_path'].read_bytes())
    assert marker['archive_sha256']==sha(f['archive'].read_bytes())
    assert marker_path.lstat().st_nlink==1 and result['completion_marker_sha256']==sha(marker_path.read_bytes())
    assert not (marker_path.parent / '.completion.json.prepared').exists()
    assert result['verified_members']['a.txt'] == {'bytes': 3, 'sha256': sha(b'abc')}
    assert (tree / 'sub/run.py').read_bytes() == b'print(1)\n'
    for path in tree.rglob('*'):
        assert path.lstat().st_uid == os.getuid()
        assert stat.S_IMODE(path.lstat().st_mode) in (0o600, 0o700)
    assert not f['reserve'].exists() and not (f['namespace'] / 'research.staging').exists()
    assert not result['workspace_symlink_modified']
    intact(f)


@pytest.mark.parametrize('name', ['/absolute', 'research/../escape', 'other/x', 'research//x',
                                  'research/./x', 'research\\x'])
def test_unsafe_paths_rejected_and_original_kept(fixture, name):
    f = fixture(entries=[(name, b'a', tarfile.REGTYPE)])
    with pytest.raises(ValueError):
        execute(f)
    assert (f['namespace'] / 'research').is_dir()
    assert not (f['namespace'] / 'restore_evidence/completion.json').exists()
    assert (f['namespace'] / 'restore_evidence/failure.json').is_file()
    intact(f)


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE,
                                  tarfile.DIRTYPE, tarfile.CHRTYPE])
def test_nonregular_members_rejected(fixture, kind):
    f = fixture(entries=[('research/x', b'', kind)])
    with pytest.raises(ValueError, match='regular'):
        execute(f)
    intact(f)


def test_duplicate_archive_names_rejected(fixture):
    f = fixture(entries=[('research/x', b'a', tarfile.REGTYPE), ('research/x', b'b', tarfile.REGTYPE)])
    with pytest.raises(ValueError, match='duplicate'):
        execute(f)
    intact(f)


@pytest.mark.parametrize('change', ['sha', 'bytes', 'missing', 'extra', 'duplicate', 'omitted'])
def test_manifest_inconsistencies_rejected(fixture, change):
    def mutate(manifest):
        if change == 'sha': manifest['files'][0]['sha256'] = '0' * 64
        elif change == 'bytes': manifest['files'][0]['bytes'] += 1
        elif change == 'missing': manifest['files'].pop()
        elif change == 'extra': manifest['files'].append({'path': 'absent', 'bytes': 0, 'sha256': sha(b'')})
        elif change == 'duplicate': manifest['files'].append(manifest['files'][0].copy())
        else: manifest['omitted_symlinks'] = ['unbundled-link']
    f = fixture(mutate_manifest=mutate)
    with pytest.raises(ValueError): execute(f)
    assert (f['namespace'] / 'research').is_dir()
    assert not (f['namespace'] / 'restore_evidence/completion.json').exists()
    intact(f)


def test_manifest_must_be_last(fixture):
    f = fixture(suffix=b'not allowed')
    with pytest.raises(ValueError): execute(f)
    intact(f)


def test_second_tar_after_zero_trailer_is_not_hidden(fixture):
    extra = io.BytesIO()
    with tarfile.open(fileobj=extra, mode='w') as tar:
        row = tarfile.TarInfo('research/hidden.txt'); row.size = 6
        tar.addfile(row, io.BytesIO(b'hidden'))
    f = fixture(trailing_tar=extra.getvalue())
    with pytest.raises(ValueError, match='manifest must be last|MANIFEST must be last'):
        execute(f)
    intact(f)


def test_full_compressed_sha_checked_in_extraction(fixture):
    f = fixture(expected_sha='0' * 64)
    with pytest.raises(ValueError, match='compressed extraction-stream SHA'):
        execute(f)
    failure = json.loads((f['namespace'] / 'restore_evidence/failure.json').read_bytes())
    assert failure['reserve_consumed'] is True and failure['completion_marker_present'] is False
    intact(f)


@pytest.mark.parametrize('change', ['existing_namespace', 'reserve_inode', 'reserve_mode', 'reserve_size',
                                   'source_stat', 'receipt_sha', 'protected_namespace'])
def test_pre_admission_failures_keep_reserve_and_original(fixture, change):
    f = fixture()
    if change == 'existing_namespace': f['namespace'].mkdir()
    elif change == 'reserve_inode':
        f['reserve'].rename(f['parent'] / '.old_reserve')
        f['reserve'].write_bytes(bytes(1 << 20)); f['reserve'].chmod(0o600)
    elif change == 'reserve_mode': f['reserve'].chmod(0o644)
    elif change == 'reserve_size':
        with f['reserve'].open('ab') as stream: stream.write(b'changed')
    elif change == 'source_stat': os.utime(f['archive'], ns=(1, 1))
    elif change == 'receipt_sha': f['plan']['publication']['sha256'] = '0' * 64; f['bind']()
    else: f['plan']['protected_workspace'] = str(f['parent']); f['bind']()
    with pytest.raises((ValueError, FileExistsError)):
        execute(f)
    assert f['reserve'].exists()
    intact(f)


@pytest.mark.parametrize('authorized', [False, None, 1, 'true'])
def test_explicit_true_authority_required(fixture, authorized):
    f = fixture()
    auth = json.loads(f['auth_path'].read_bytes()); auth['authorized'] = authorized
    save(f['auth_path'], auth)
    with pytest.raises(ValueError, match='explicit ROOT'):
        execute(f)
    assert f['reserve'].exists() and not f['namespace'].exists()


def test_unsupported_noreplace_keeps_reserve(fixture, monkeypatch):
    f = fixture()
    def unavailable(*args): raise OSError(38, 'toy unsupported')
    monkeypatch.setattr(m, 'link_marker_noreplace', unavailable)
    with pytest.raises(OSError): execute(f)
    assert f['reserve'].exists()
    assert not (f['namespace'] / 'research').exists()
    intact(f)


def test_marker_noreplace_does_not_overwrite_existing_file(tmp_path):
    source, dest = tmp_path / 'source', tmp_path / 'dest'
    source.write_text('new marker'); dest.write_text('old marker')
    with pytest.raises(FileExistsError): m.link_marker_noreplace(source, dest)
    assert source.read_text()=='new marker' and dest.read_text()=='old marker'


def test_reserve_lock_contention_rejected_before_release(fixture):
    f = fixture()
    with f['reserve'].open('rb') as holder:
        m.fcntl.flock(holder.fileno(), m.fcntl.LOCK_EX | m.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): execute(f)
    assert f['reserve'].exists()
    intact(f)


def test_enospc_after_release_keeps_uncommitted_candidate_and_failure_evidence(fixture, monkeypatch):
    f = fixture()
    def no_space(*args): raise OSError(28, 'toy ENOSPC')
    monkeypatch.setattr(m, 'create_member', no_space)
    with pytest.raises(OSError): execute(f)
    assert not f['reserve'].exists()
    assert (f['namespace'] / 'research').is_dir()
    state = json.loads((f['namespace'] / 'restore_evidence/failure.json').read_bytes())
    assert state['reserve_consumed'] is True and state['compressed_stream_bytes_read'] is None
    assert not (f['namespace'] / 'restore_evidence/completion.json').exists()
    intact(f)


def test_source_changed_during_admission_does_not_consume_reserve(fixture, monkeypatch):
    f = fixture()
    probe = m.probe_marker_commit
    def change_after_probe(namespace):
        probe(namespace)
        os.utime(f['archive'], ns=(1, 1))
    monkeypatch.setattr(m, 'probe_marker_commit', change_after_probe)
    with pytest.raises(ValueError, match='source changed before reserve release'):
        execute(f)
    assert f['reserve'].exists()
    state = json.loads((f['namespace'] / 'restore_evidence/failure.json').read_bytes())
    assert state['reserve_consumed'] is False
    intact(f)


@pytest.mark.parametrize('status', ['failed', 'accepted_durable_intermediate_snapshot'])
def test_only_fixed_final_acceptance_status_is_allowed(fixture, status):
    f = fixture()
    path = Path(f['plan']['ROOT_publication_acceptance']['path'])
    accepted = json.loads(path.read_bytes()); accepted['status'] = status
    f['plan']['ROOT_publication_acceptance'] = save(path, accepted)
    f['plan']['ROOT_acceptance_status'] = status
    f['bind']()
    with pytest.raises(ValueError, match='ROOT/adjacent identity mismatch'):
        execute(f)
    assert f['reserve'].exists() and not f['namespace'].exists()
    intact(f)


@pytest.mark.parametrize('mode', [None, 'atomic_directory_rename'])
def test_explicit_marker_mode_required_before_namespace_or_release(fixture, mode):
    f = fixture();f['plan']['publication_mode']=mode;f['bind']()
    with pytest.raises(ValueError, match='explicit marker publication mode'):
        execute(f)
    assert f['reserve'].exists() and not f['namespace'].exists()


def test_post_link_fsync_failure_preserves_marker_but_never_reports_success(fixture, monkeypatch):
    f = fixture();original=m.fsync_dir;failed=False
    evidence=f['namespace']/'restore_evidence'
    def fail_once_after_marker(path):
        nonlocal failed
        if Path(path)==evidence and (evidence/'completion.json').exists() and not failed:
            failed=True
            raise OSError(5, 'toy parent fsync failed after link')
        return original(path)
    monkeypatch.setattr(m,'fsync_dir',fail_once_after_marker)
    with pytest.raises(OSError): execute(f)
    assert (evidence/'completion.json').is_file()
    failure=json.loads((evidence/'failure.json').read_bytes())
    assert failure['completion_marker_present'] is True
    assert failure['completion_marker_durability']=='uncertain_until_parent_fsync'
    assert failure['status']=='failed_candidate_retained_do_not_switch_workspace'
    intact(f)


def test_foreign_marker_collision_not_overwritten_and_is_explicit(fixture, monkeypatch):
    f=fixture();original=m.link_marker_noreplace
    def inject_collision(source,destination):
        if Path(destination).name=='completion.json': Path(destination).write_bytes(b'foreign marker')
        return original(source,destination)
    monkeypatch.setattr(m,'link_marker_noreplace',inject_collision)
    with pytest.raises(FileExistsError): execute(f)
    evidence=f['namespace']/'restore_evidence'
    assert (evidence/'completion.json').read_bytes()==b'foreign marker'
    failure=json.loads((evidence/'failure.json').read_bytes())
    assert failure['completion_marker_present'] is True
    assert failure['completion_marker_durability']=='unknown_after_failed_publication'
    intact(f)


def test_no_directory_rename_is_used(fixture, monkeypatch):
    f=fixture()
    def forbidden(*args,**kwargs): raise AssertionError('directory rename forbidden')
    monkeypatch.setattr(m.os,'rename',forbidden);monkeypatch.setattr(m.os,'replace',forbidden)
    assert execute(f)['status']=='passed_verified_new_editable_tree_restored'
