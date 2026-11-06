#!/usr/bin/env python3
"""One forward gzip/tar traversal: all header identities and 32 selected SHA checks."""
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
import time

ROOT = Path('/data/202610/7kb/research').resolve()
OUT = Path(__file__).resolve().parent


def digest(b):
    return hashlib.sha256(b).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def proof(path):
    path = Path(path)
    b = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': digest(b), 'bytes': len(b)}


def stat5(s):
    return {'device': s.st_dev, 'inode': s.st_ino, 'bytes': s.st_size,
            'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}


def new_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


class CountingReader:
    def __init__(self, stream):
        self.stream, self.bytes_read, self.calls = stream, 0, 0

    def read(self, size=-1):
        b = self.stream.read(size)
        self.bytes_read += len(b)
        self.calls += 1
        return b


def main():
    start = time.monotonic()
    plan = load(OUT / 'review_plan.json')
    archive = Path(plan['archive'])
    evidence = {}
    for key in ('publication', 'ROOT_acceptance', 'adjacent_receipt'):
        p = Path(plan[key]['path'])
        assert proof(p)['sha256'] == plan[key]['sha256']
        evidence[key] = load(p)
    pub, accepted, adjacent = (evidence[k] for k in ('publication', 'ROOT_acceptance', 'adjacent_receipt'))
    expected = plan['archive_SHA_reused_from_existing_publication']
    assert pub['status'] == 'passed' and accepted['status'] == 'accepted_durable_intermediate_snapshot'
    assert pub['destination_sha256'] == accepted['sha256'] == adjacent['sha256'] == expected
    assert adjacent['publication_receipt_sha256'] == accepted['publication']['sha256'] == plan['publication']['sha256']
    assert adjacent['verified_all_payloads'] is True and adjacent['files'] == accepted['payloads_verified_by_packager'] == 17655
    assert adjacent['external_weights'] == accepted['external_weights_indexed_not_in_project_archive'] == 748
    assert adjacent['source_git_head'] == accepted['source_git_head'] == plan['source_git_head']
    assert pub['file_fsync_after_copy'] and pub['file_fsync_after_truncate'] and pub['parent_directory_fsync'] and pub['atomic_destination_commit_noreplace']
    lst = archive.lstat()
    assert stat.S_ISREG(lst.st_mode) and stat.S_IMODE(lst.st_mode) == 0o600
    assert lst.st_uid == lst.st_gid == 11471 and lst.st_ino == 439053984 and lst.st_dev == 106 and lst.st_nlink == 1
    before = stat5(lst)
    assert before['bytes'] == pub['bytes'] == 784476267
    assert all(before[k] == pub['destination_stat'][k] for k in before)
    selected = {r['path']: r for r in plan['selected_payload_members']}
    assert len(selected) == 32
    members, actual, objects, pdf_pages = {}, {}, {}, {}
    manifest = None
    tar_position = 0
    with archive.open('rb', buffering=0) as raw:
        fd_before = stat5(os.fstat(raw.fileno()))
        assert fd_before == before
        counted = CountingReader(raw)
        with tarfile.open(fileobj=counted, mode='r|gz', bufsize=1 << 20) as tar:
            for member in tar:
                pp = PurePosixPath(member.name)
                assert member.name.startswith('research/') and not pp.is_absolute() and '..' not in pp.parts
                name = member.name[len('research/'):]
                assert name and name not in members and member.isfile(), 'duplicate or nonregular entry'
                members[name] = {'bytes': member.size}
                if name == 'PACKAGE_MANIFEST.json' or name in selected:
                    assert member.size < 32 * 1024 * 1024
                    b = tar.extractfile(member).read()
                    assert len(b) == member.size
                    if name == 'PACKAGE_MANIFEST.json':
                        assert manifest is None
                        manifest = json.loads(b)
                        manifest_sha, manifest_bytes = digest(b), len(b)
                        with (OUT / 'PACKAGE_MANIFEST.reviewed.json').open('xb') as f:
                            f.write(b)
                    else:
                        assert selected[name]['bytes'] == len(b) and selected[name]['sha256'] == digest(b), name
                        actual[name] = {'bytes': len(b), 'sha256': digest(b)}
                        if name.endswith('.json'):
                            objects[name] = json.loads(b)
                        if name.endswith('.pdf'):
                            sys.path.insert(0, '/tmp/grad-research-20261106/report_packages')
                            import pypdfium2
                            with pypdfium2.PdfDocument(b) as pdf:
                                pdf_pages[name] = len(pdf)
                tar_position = tar.fileobj.pos
        tail = 0
        while True:
            b = counted.read(1 << 20)
            if not b:
                break
            tail += len(b)
        fd_after = stat5(os.fstat(raw.fileno()))
    after = stat5(archive.lstat())
    assert before == fd_before == fd_after == after and counted.bytes_read == 784476267
    assert manifest is not None and set(actual) == set(selected)
    files = {r['path']: r for r in manifest['files']}
    assert len(files) == len(manifest['files']) == 17655
    assert set(members) == set(files) | {'PACKAGE_MANIFEST.json'} and len(members) == 17656
    assert not manifest['omitted_symlinks'] and len(manifest['external_weights_not_included']) == 748
    for name, row in files.items():
        assert members[name]['bytes'] == row['bytes'], name
    for name, row in actual.items():
        assert row['bytes'] == files[name]['bytes'] and row['sha256'] == files[name]['sha256'], name
    assert all(r['bundled'] is False for r in manifest['external_weights_not_included'])
    assert not any(n.startswith('run_records/') and '/checkpoints/' in n for n in members)
    snap_name = 'artifacts/provenance/extended_v2_pythia_410m_m64_formal_input_snapshot.json'
    snapshot = objects[snap_name]
    assert snapshot['source_repository_commit'] == 'de54262b8deff2c49547bba740d0dbf28a1074a7' and len(snapshot['code']) == 10
    core_checks = []
    for name, row in snapshot['code'].items():
        assert actual[name]['sha256'] == row['sha256'] == files[name]['sha256']
        assert actual[name]['bytes'] == row['bytes']
        core_checks.append({'path': name, **actual[name]})
    report = 'reports/研究报告_工作稿.pdf'
    deck = 'presentations/答辩工作稿_20261106_1730/答辩演示工作稿.pdf'
    assert actual[report]['sha256'] == accepted['report_PDF_SHA256'] == adjacent['working_report_pdf_sha256']
    assert pdf_pages[report] == 16 and pdf_pages[deck] == 12
    visual = objects['artifacts/report_review/delivery_section7_update_v3/visual_review_receipt.json']
    assert visual['status'] == 'passed'
    assert visual['inputs']['source']['sha256'] == actual['研究报告.md']['sha256']
    assert visual['inputs']['pdf']['sha256'] == actual[report]['sha256']
    guide = objects['docs/交付文件与恢复容量指南_来源核对_v6.json']
    assert guide['guide']['sha256'] == actual['docs/交付文件与恢复容量指南.md']['sha256']
    assert guide['README_authorized_navigation_update']['sha256'] == actual['README.md']['sha256']
    assert actual['docs/durable_snapshot_latest.json']['sha256'] == plan['embedded_previous_1640_pointer_sha256']
    self_boundary = {'embedded_pointer_expected_milestone': '1640', 'embedded_pointer_matches_saved_prepublication_bytes': True,
        'embedded_guide_version': 'v6', 'current_external_pointer': proof(ROOT / 'docs/durable_snapshot_latest.json'),
        'own_postbuild_publication_proof_presence': {name: name in files for name in ('docs/snapshot_1918_durable_publication.json', 'artifacts/provenance/snapshot_1918_publication/ROOT_acceptance.json')},
        'note': 'Internal historical pointer/guide retained. External publisher/adjacent receipt establishes this new archive; no self-referential current archive SHA expected inside itself.'}
    receipt = {'status': 'passed_bounded_independent_archive_review', 'reviewed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'archive': str(archive), 'archive_SHA_reused_not_recomputed': expected,
        'publication': proof(plan['publication']['path']), 'ROOT_acceptance': proof(plan['ROOT_acceptance']['path']), 'adjacent_receipt': proof(plan['adjacent_receipt']['path']),
        'archive_path_fd_stat5_before_after': {'path_before': before, 'fd_before': fd_before, 'fd_after': fd_after, 'path_after': after},
        'single_forward_gzip_stream': {'compressed_bytes_read': counted.bytes_read, 'compressed_read_calls': counted.calls,
            'raw_tail_bytes_drained_without_digest': tail, 'decompressed_tar_position_consumed': tar_position,
            'header_regular_payload_bytes_sum': sum(r['bytes'] for r in members.values()),
            'selected_payload_bytes_explicitly_read_and_hashed': sum(r['bytes'] for r in actual.values()),
            'PACKAGE_MANIFEST_bytes_explicitly_read_and_hashed': manifest_bytes,
            'limitations': 'Sequential gzip scans skipped compressed contents. Only selected32 plus manifest are application-read/hashed. No whole archive SHA/all-payload second hash pass, no weight archive/CP/base read.'},
        'PACKAGE_MANIFEST': {'path': str(OUT / 'PACKAGE_MANIFEST.reviewed.json'), 'sha256': manifest_sha,
            'files': len(files), 'members_including_manifest': len(members), 'external_weight_files_indexed_only': 748,
            'all_header_names_sizes_match': True, 'prior_packager_all_payload_SHA_not_repeated': True, 'consistency': manifest['consistency']},
        'selected_members': actual, 'ten_core_actual_payload_SHA_checks': core_checks, 'PDF_page_counts_from_selected_inmemory_bytes': pdf_pages,
        'guide_README_and_report_visual_metadata_crossbindings': True, 'self_reference_boundary': self_boundary,
        'source_git_head_evidence': plan['source_git_head'], 'inside_main_wall_seconds': time.monotonic() - start,
        'manual_engineering_whole_seconds': None, 'sealed_score_payloads_parsed': False,
        'project_or_weights_extracted': False, 'GPU_used': False, 'review_source': proof(__file__), 'review_plan': proof(OUT / 'review_plan.json')}
    new_json(OUT / 'review_receipt.json', receipt)
    print(json.dumps({'status': receipt['status'], 'selected': len(actual), 'header_payloads': len(files),
        'compressed_bytes_read': counted.bytes_read, 'selected_bytes': sum(r['bytes'] for r in actual.values()),
        'PDF_pages': pdf_pages, 'receipt': proof(OUT / 'review_receipt.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
