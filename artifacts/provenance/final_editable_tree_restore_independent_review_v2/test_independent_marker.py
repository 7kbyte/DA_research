"""Independent small marker sequencing probes; synthetic archives and /tmp fixtures only."""
import json
from pathlib import Path
import pytest
import author_fixture as toy

fixture = toy.fixture
m = toy.m

def test_early_marker_is_rejected_before_capacity_release(fixture, monkeypatch):
    f=fixture(); original=m.probe_marker_commit
    def early(namespace):
        original(namespace)
        (namespace/'restore_evidence/completion.json').write_bytes(b'premature marker')
    monkeypatch.setattr(m,'probe_marker_commit',early)
    with pytest.raises(FileExistsError): toy.execute(f)
    evidence=f['namespace']/'restore_evidence'
    failed=json.loads((evidence/'failure.json').read_bytes())
    assert f['reserve'].exists() and not (f['namespace']/'research').exists()
    assert failed['reserve_consumed'] is False
    assert failed['completion_marker_present'] is True
    assert failed['completion_marker_durability']=='unknown_after_failed_publication'
    assert (evidence/'completion.json').read_bytes()==b'premature marker'
    toy.intact(f)

def test_completion_link_observes_full_verified_candidate_and_bound_marker(fixture, monkeypatch):
    f=fixture(); original=m.link_marker_noreplace; linked=[]
    def inspect(source,destination):
        if Path(destination).name=='completion.json':
            source=Path(source); tree=f['namespace']/'research'
            marker=json.loads(source.read_bytes())
            assert marker['status']=='verified_tree_ready_marker'
            assert marker['archive_sha256']==f['plan']['archive_sha256']
            assert marker['candidate']==str(tree)
            assert marker['plan_sha256']==toy.sha(f['plan_path'].read_bytes())
            assert marker['authority_sha256']==toy.sha(f['auth_path'].read_bytes())
            assert marker['tool_sha256']==toy.sha(toy.SOURCE.read_bytes())
            assert marker['tree_and_parent_fsync_before_marker_completed'] is True
            assert marker['atomic_directory_publication'] is False
            assert marker['compressed_stream_bytes_read']==f['archive'].stat().st_size
            assert marker['verified_payload_files']==2
            actual={p.relative_to(tree).as_posix() for p in tree.rglob('*') if p.is_file()}
            assert actual==set(marker['verified_members'])
            for name,expected in marker['verified_members'].items():
                data=(tree/name).read_bytes()
                assert len(data)==expected['bytes'] and toy.sha(data)==expected['sha256']
            assert not f['reserve'].exists()
            assert (Path(destination).parent/'verified_before_publication.json').is_file()
            assert not Path(destination).exists()
            linked.append(toy.sha(source.read_bytes()))
        return original(source,destination)
    monkeypatch.setattr(m,'link_marker_noreplace',inspect)
    result=toy.execute(f); marker=f['namespace']/'restore_evidence/completion.json'
    assert len(linked)==1 and linked[0]==result['completion_marker_sha256']==toy.sha(marker.read_bytes())
    assert result['completion_marker_durability']=='parent_fsynced'
    assert marker.stat().st_nlink==1
    assert not marker.with_name('.completion.json.prepared').exists()
    toy.intact(f)

def test_failure_immediately_after_link_is_not_success(fixture, monkeypatch):
    f=fixture(); original=m.link_marker_noreplace
    def after_link(source,destination):
        original(source,destination)
        if Path(destination).name=='completion.json': raise OSError(5,'toy interrupted after actual link')
    monkeypatch.setattr(m,'link_marker_noreplace',after_link)
    with pytest.raises(OSError): toy.execute(f)
    evidence=f['namespace']/'restore_evidence'
    marker=json.loads((evidence/'completion.json').read_bytes())
    failure=json.loads((evidence/'failure.json').read_bytes())
    assert marker['status']=='verified_tree_ready_marker'
    assert failure['status']=='failed_candidate_retained_do_not_switch_workspace'
    assert failure['completion_marker_present'] is True
    assert failure['completion_marker_durability']=='unknown_after_failed_publication'
    assert failure['compressed_stream_bytes_read']==f['archive'].stat().st_size
    toy.intact(f)
