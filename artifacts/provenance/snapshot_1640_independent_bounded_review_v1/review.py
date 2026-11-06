#!/usr/bin/env python3
"""One forward gzip/tar traversal, 20 payload SHA checks plus package metadata."""
from __future__ import annotations
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time

ROOT = Path('/data/202610/7kb/research').resolve()
OUT = Path(__file__).resolve().parent

def digest(b): return hashlib.sha256(b).hexdigest()
def proof(p):
    p=Path(p);b=p.read_bytes()
    return {'path':str(p.resolve()),'sha256':digest(b),'bytes':len(b)}
def load(p):return json.loads(Path(p).read_text())
def new_json(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def stat5(s):return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def relative(p):return Path(p).resolve().relative_to(ROOT).as_posix()

class CountingReader:
    def __init__(self,f):self.f=f;self.bytes_read=0;self.calls=0
    def read(self,n=-1):
        b=self.f.read(n);self.bytes_read+=len(b);self.calls+=1;return b

def main():
    plan=load(OUT/'review_plan.json'); archive=Path(plan['archive'])
    publication=ROOT/plan['publication_path'];assert proof(publication)['sha256']==plan['publication_sha256']
    pub=load(publication);acceptance=ROOT/'artifacts/provenance/snapshot_1640_publication/ROOT_acceptance.json';acc=load(acceptance)
    adjacent=archive.with_name(archive.name+'.receipt.json');adj=load(adjacent)
    expected_archive=plan['archive_SHA_from_existing_publication_not_recomputed']
    assert pub['status']=='passed' and pub['destination_sha256']==acc['sha256']==adj['sha256']==expected_archive
    assert acc['publication']['sha256']==adj['publication_receipt_sha256']==plan['publication_sha256']
    assert acc['status']=='accepted_durable_intermediate_snapshot' and adj['verified_all_payloads'] is True
    assert adj['source_git_head']==acc['source_git_head']=='92f71576cf6b01c0f230399436651b35d4be01d4'
    assert pub['file_fsync_after_copy'] and pub['file_fsync_after_truncate'] and pub['parent_directory_fsync'] and pub['atomic_destination_commit_noreplace']
    before=stat5(archive.stat())
    assert all(before[k]==pub['destination_stat'][k] for k in before)
    assert before['bytes']==pub['bytes']==adj['bytes']==630111999
    assert adjacent.stat().st_uid==11471 and adjacent.stat().st_gid==11471
    selected={r['path']:r for r in plan['selected_payload_members']};assert len(selected)==20
    members={};actual={};payloads={};manifest=None;manifest_bytes=0;tar_position=0
    t=time.perf_counter()
    with archive.open('rb',buffering=0) as raw:
        fd_before=stat5(os.fstat(raw.fileno()));assert fd_before==before
        counted=CountingReader(raw)
        with tarfile.open(fileobj=counted,mode='r|gz',bufsize=1<<20) as tar:
            for member in tar:
                assert member.name.startswith('research/') and '..' not in PurePosixPath(member.name).parts
                name=member.name[len('research/'):]
                assert name not in members and member.isfile(), 'Unexpected duplicate or nonregular archive entry'
                members[name]={'bytes':member.size}
                if name=='PACKAGE_MANIFEST.json' or name in selected:
                    assert member.size<32_000_000, 'Bounded metadata/payload size exceeded'
                    stream=tar.extractfile(member);b=stream.read();assert len(b)==member.size
                    if name=='PACKAGE_MANIFEST.json':
                        assert manifest is None;manifest=json.loads(b);manifest_bytes=len(b)
                        manifest_sha=digest(b)
                        # Only the requested package metadata is persisted, not arbitrary archive payloads.
                        with (OUT/'PACKAGE_MANIFEST.reviewed.json').open('xb') as f:f.write(b)
                    else:
                        expected=selected[name];assert len(b)==expected['bytes'] and digest(b)==expected['sha256'], 'Selected source/proof SHA differs: '+name
                        actual[name]={'bytes':len(b),'sha256':digest(b),'current_source_expected':expected['sha256']}
                        if name.endswith('.json'):payloads[name]=json.loads(b)
                tar_position=tar.fileobj.pos
        # Tar end-of-archive padding/trailer can leave a tiny raw suffix; one forward drain,
        # without a digest or another decompression pass, makes compressed read scope exact.
        tail=0
        while True:
            b=counted.read(1<<20)
            if not b:break
            tail+=len(b)
        fd_after=stat5(os.fstat(raw.fileno()))
        compressed_bytes=counted.bytes_read;read_calls=counted.calls
    after=stat5(archive.stat());assert before==fd_before==fd_after==after
    assert manifest is not None and set(actual)==set(selected)
    files={r['path']:r for r in manifest['files']}
    assert len(files)==len(manifest['files'])==12822 and len(manifest['external_weights_not_included'])==721
    assert len(members)==12823 and set(members)==set(files)|{'PACKAGE_MANIFEST.json'}
    assert not manifest['omitted_symlinks']
    for name,r in files.items():assert members[name]['bytes']==r['bytes']
    for name,r in actual.items():assert r['sha256']==files[name]['sha256'] and r['bytes']==files[name]['bytes']
    for r in manifest['external_weights_not_included']:assert r['bundled'] is False
    assert not any(n.startswith('run_records/') and '/checkpoints/' in n for n in members)
    oldplan=payloads['artifacts/analysis_prepared_v2/extended_pythia_410m_100m_nll.json']
    complete=payloads['results/supplemental_analysis_v3/extended_pythia_410m_100m_nll/COMPLETE.json']
    journal_name=relative(oldplan['first_read_receipt_path']);journal=payloads[journal_name]
    assert complete['plan']['sha256']==actual['artifacts/analysis_prepared_v2/extended_pythia_410m_100m_nll.json']['sha256']
    assert complete['first_read_receipt']['sha256']==actual[journal_name]['sha256']
    assert journal['plan']==complete['plan'] and journal['before_first_score_parse'] is True
    plan_source_checks=[]
    for path,expected in oldplan['preparation_source_sha256'].items():
        name=relative(path);assert name in files and files[name]['sha256']==expected
        plan_source_checks.append({'path':name,'sha256':expected,'method':'PACKAGE_MANIFEST binding; payload not additionally read unless one of selected20'})
    snapshot_name='artifacts/provenance/extended_v2_pythia_410m_m64_formal_input_snapshot.json'
    snap=payloads[snapshot_name];core_checks=[]
    assert snap['source_repository_commit']=='de54262b8deff2c49547bba740d0dbf28a1074a7' and len(snap['code'])==10
    for name,r in snap['code'].items():
        assert name in files and files[name]['sha256']==r['sha256'] and files[name]['bytes']==r['bytes']
        core_checks.append({'path':name,**r,'selected_payload_rehashed':name in actual})
    config='configs/extended_v2_pythia_410m_m64_formal/extended_v2_pythia_410m_m64_formal_B4_s11_w0p05.json'
    assert actual[config]['sha256']==snap['configs'][config]['sha256']
    assert actual['reports/研究报告_工作稿.pdf']['sha256']==acc['report_PDF_SHA256']==adj['working_report_pdf_sha256']
    visual=payloads['artifacts/report_review/C410_included_v2/visual_review_receipt.json'];assert visual['page_count']==16 and visual['status'].startswith('passed')
    assert visual['inputs']['source']['sha256']==actual['研究报告.md']['sha256'] and visual['inputs']['pdf']['sha256']==actual['reports/研究报告_工作稿.pdf']['sha256']
    assert files['reports/研究报告_工作稿.html']['sha256']==visual['inputs']['html']['sha256']
    native=payloads['artifacts/provenance/C_native_final_publication_runbook_v2/preparation_receipt.json']
    additional=payloads['artifacts/provenance/additional_endpoint_publication_runbook_v1/reviewer_revision_v2.json']
    for r in [native['reviewer'],native['runbook'],additional['source'],additional['test']]:
        name=relative(ROOT/r['path'] if not Path(r['path']).is_absolute() else r['path']);assert files[name]['sha256']==r['sha256']
    # Current publication and acceptance are necessarily later than this archive's creation.
    self_references={name:{'present_in_archive':name in files,'expected':'post-build publication/timer/latest pointer need not self-contain current bytes'} for name in ['docs/snapshot_1640_durable_publication.json','artifacts/provenance/snapshot_1640_publication/ROOT_acceptance.json','artifacts/provenance/snapshot_1640_build/whole_process_timing.json','docs/durable_snapshot_latest.json']}
    latest='docs/durable_snapshot_latest.json'
    if latest in files:self_references[latest]['matches_current_pointer']=files[latest]['sha256']==proof(ROOT/latest)['sha256']
    result={'status':'passed_bounded_independent_archive_review','reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive':str(archive),'archive_SHA_reused_from_fsynced_publisher_not_recomputed':expected_archive,'publication':proof(publication),'ROOT_acceptance':proof(acceptance),'adjacent_receipt':proof(adjacent),'archive_path_fd_stat5_before_after':{'path_before':before,'fd_before':fd_before,'fd_after':fd_after,'path_after':after},'adjacent_receipt_owner_uid_gid':[adjacent.stat().st_uid,adjacent.stat().st_gid],'single_forward_stream':{'compressed_file_bytes_read':compressed_bytes,'compressed_file_read_calls':read_calls,'raw_tail_bytes_drained_without_digest':tail,'decompressed_tar_position_consumed':tar_position,'uncompressed_regular_payload_sizes_header_sum':sum(r['bytes'] for r in members.values()),'selected20_payload_bytes_explicitly_read_and_hashed':sum(r['bytes'] for r in actual.values()),'PACKAGE_MANIFEST_bytes_explicitly_read_and_hashed':manifest_bytes,'inside_traversal_and_binding_wall_seconds':time.perf_counter()-t,'limitations':'Sequential gzip decompression necessarily scans skipped compressed content. Only20 payloads plus PACKAGE_MANIFEST read/hashed at application level; other payloads not semantically parsed or written. No whole archive SHA recalculation or all-payload hash pass.'},'PACKAGE_MANIFEST':{'path':str(OUT/'PACKAGE_MANIFEST.reviewed.json'),'sha256':manifest_sha,'files':len(files),'total_members_including_manifest':len(members),'external_weights_only_indexed':721,'all12822_header_names_and_sizes_match_manifest':True,'all12822_payload_SHA_verification_is_prior_packager_evidence_not_repeated_here':True,'consistency':manifest['consistency']},'selected_members':actual,'old_v2_plan_fixed_journal_in_archive':{'plan':actual['artifacts/analysis_prepared_v2/extended_pythia_410m_100m_nll.json'],'journal':actual[journal_name],'journal_path_matches_PLAN_and_COMPLETE':True},'old_v2_source_freeze_manifest_bindings':plan_source_checks,'ten_training_core_manifest_bindings':core_checks,'C410_config_matches_pretraining_snapshot_identity':True,'new16_page_report_and_current_proof_match':True,'two_native_publication_v2_preparations_present_and_source_bound':True,'self_reference_boundary':self_references,'source_git_head_evidence':{'commit':acc['source_git_head'],'source':'existing ROOT acceptance and durable adjacent receipt; not claimed a new git-object reconstruction'},'sealed_score_payloads_parsed':False,'project_or_weights_extracted':False,'GPU_used':False,'review_source':proof(__file__),'review_plan':proof(OUT/'review_plan.json')}
    new_json(OUT/'review_receipt.json',result)
    print(json.dumps({'status':result['status'],'selected_members':len(actual),'selected_payload_bytes':sum(r['bytes'] for r in actual.values()),'compressed_bytes_read':compressed_bytes,'manifest_payloads':len(files),'source_freeze_manifest_checks':len(plan_source_checks),'core_manifest_checks':len(core_checks),'receipt':proof(OUT/'review_receipt.json')},ensure_ascii=False))

if __name__=='__main__':main()
