"""Four bounded synthetic cases; no real archive/reserve is opened."""
import gzip
from pathlib import Path
import json
import tarfile
import pytest
from fixture_source import fixture, intact, m, save, sha

@pytest.mark.parametrize('entries', [
    [('research/a', b'a', tarfile.REGTYPE), ('research/a/b', b'b', tarfile.REGTYPE)],
    [('research/a/b', b'b', tarfile.REGTYPE), ('research/a', b'a', tarfile.REGTYPE)],
])
def test_file_parent_conflict_cannot_publish(fixture, entries):
    f = fixture(entries=entries)
    with pytest.raises(OSError): m.restore(f['plan_path'], f['auth_path'])
    assert not (f['namespace']/'research').exists()
    assert (f['namespace']/'research.staging').is_dir()
    assert (f['namespace']/'restore_evidence/failure.json').is_file()
    intact(f)

def append_and_rebind_toy(f, suffix):
    f['archive'].write_bytes(f['archive'].read_bytes()+suffix)
    archive_sha = sha(f['archive'].read_bytes())
    pub = json.loads(__import__('pathlib').Path(f['plan']['publication']['path']).read_text())
    pub.update(source_sha256=archive_sha, destination_sha256=archive_sha,
               destination_stat=m.details(f['archive'].lstat()), bytes=f['archive'].stat().st_size)
    pub_ref = save(Path(f['plan']['publication']['path']), pub)
    accepted = json.loads(__import__('pathlib').Path(f['plan']['ROOT_publication_acceptance']['path']).read_text())
    accepted.update(sha256=archive_sha, bytes=f['archive'].stat().st_size, publication=pub_ref)
    acc_ref = save(Path(f['plan']['ROOT_publication_acceptance']['path']), accepted)
    adjacent = json.loads(__import__('pathlib').Path(f['plan']['adjacent_receipt']['path']).read_text())
    adjacent.update(sha256=archive_sha, bytes=f['archive'].stat().st_size, publication_receipt_sha256=pub_ref['sha256'])
    adj_ref = save(Path(f['plan']['adjacent_receipt']['path']), adjacent)
    f['plan'].update(archive_sha256=archive_sha, archive_stat5=m.signature(f['archive'].lstat()),
                     publication=pub_ref, ROOT_publication_acceptance=acc_ref, adjacent_receipt=adj_ref)
    f['bind']()

def test_concatenated_empty_gzip_member_is_in_full_compressed_hash(fixture):
    f = fixture(); append_and_rebind_toy(f, gzip.compress(b''))
    result = m.restore(f['plan_path'], f['auth_path'])
    assert result['compressed_stream_bytes_read'] == f['archive'].stat().st_size
    assert result['compressed_stream_sha256'] == sha(f['archive'].read_bytes())
    assert result['verified_payload_files'] == 2
    intact(f)

def test_non_gzip_trailing_bytes_fail_even_if_toy_authority_hashes_them(fixture):
    f = fixture(); append_and_rebind_toy(f, b'not-a-gzip-member')
    with pytest.raises((gzip.BadGzipFile, tarfile.TarError, ValueError)):
        m.restore(f['plan_path'], f['auth_path'])
    assert not (f['namespace']/'research').exists()
    assert (f['namespace']/'restore_evidence/failure.json').is_file()
    intact(f)
