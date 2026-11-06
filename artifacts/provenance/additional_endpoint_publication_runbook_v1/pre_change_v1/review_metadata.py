#!/usr/bin/env python3
"""One-shot A/B publication metadata review; no payload hashing or publication."""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = Path('/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4')
SCOPE_SHA = '7adb8f7f0461fd929995dd370ff2a59b2abc5c0d7750a39e745d21d5546d9de3'
EXPORTER_SHA = 'c93e0c1d35fe39b3d71155a126db841b76648ba6b3afd58d475be4eb9c9da853'
PUBLISHER_SHA = '9c194743383a16f98313f767fe2e722c309ec57c1aafc65d9b99d08e5407022e'
CAPACITY_SHA = 'a15321819901d3272a483f724e7613a8afda302ab0fe9a6e05537856a972a62b'
BATCH_SHA = 'd73a773ee4e6015eaa5f9214577affe6c99c00bb09319e019186a010419de2a1'
NAMES = {'A_remaining_final': (93, 'A_remaining_final'), 'B_all_final': (60, 'B_final')}

def signature(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)

def review(name, output):
    started, cpu = time.monotonic(), time.process_time()
    if output.exists(): raise FileExistsError(output)
    folder = BASE/name
    if not (folder/'completion.json').exists():
        raise ValueError('Pending: actual full package completion does not yet exist; no acceptance emitted')
    evidence = {}; observed = {}
    def small(path, expected=None, parse=True):
        path = Path(path).resolve(); stamp = signature(path)
        if stamp['bytes'] > 32 << 20: raise ValueError('Small metadata/source/sidecar only')
        raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
        if signature(path) != stamp or (expected and digest != expected): raise ValueError('Small source changed or SHA differs')
        evidence[str(path)] = dict(sha256=digest, stat=stamp); observed[str(path)] = stamp
        return json.loads(raw) if parse else digest
    small(ROOT/'scripts/export_additional_endpoints.py', EXPORTER_SHA, False)
    small(ROOT/'scripts/publish_reserved_archive.py', PUBLISHER_SHA, False)
    # Import frozen pure metadata validators. Do not call load_scope/prepare/execute.
    spec = importlib.util.spec_from_file_location('additional_metadata_frozen', ROOT/'scripts/export_additional_endpoints.py')
    e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
    scope = small(BASE/'scope.json', SCOPE_SHA); phase = scope['phases'][name]
    cap = small(ROOT/'artifacts/provenance/extra_endpoint_reservations_v1/ROOT_full_capacity_acceptance.json', CAPACITY_SHA)
    batch = small(ROOT/'artifacts/provenance/additional_endpoint_actual_v4/ROOT_A_B_batch_resource_review.json', BATCH_SHA)
    e.require(batch['approved_packages']==['A_remaining_final','B_all_final'] and batch['scope_sha256']==SCOPE_SHA and batch['exporter_sha256']==EXPORTER_SHA, 'Actual batch authority differs')
    for source in scope['code']:
        small(source['path'], source['sha256'], False)
    completion = small(folder/'completion.json')
    manifest = small(folder/'additional_weights_manifest.json', completion['manifest_sha256'])
    # This existing validator stats CP/TAR and hashes only manifest/scope/gate metadata.
    e.validate_completion(name, folder, phase, BASE/'scope.json', completion)
    n, reserve_name = NAMES[name]
    e.require(completion['unique_weights']==n and completion['alias_count']==0 and completion['file_fsync_completed'] and completion['parent_directory_fsync_completed'], 'Exact independent weights or fsync evidence missing')
    e.require(manifest['base_dependency']==dict(kind='native_base_config_tokenizer_readme_included_no_external_package_needed',files=phase['base_files']) and not manifest['optimizer_included'], 'Native base closure/model-only contract differs')
    e.require(completion['verified_tar_members']==2*n+len(phase['base_files'])+7, 'Native base/tools/provenance TAR member coverage differs')
    master = small(ROOT/'configs/supplemental_v1_queue.json'); state = small(ROOT/'logs/supplemental_job_state.json')
    # Record the observed snapshots, but bind completed per-job definitions below;
    # an unrelated evaluation append is not a global queue/state freeze.
    observed.pop(str((ROOT/'configs/supplemental_v1_queue.json').resolve()))
    observed.pop(str((ROOT/'logs/supplemental_job_state.json').resolve()))
    by_id = {x['bound']['run_id']:x for x in phase['rows']}
    for row in manifest['logical_runs']:
        item = by_id[row['run_id']]; rid = row['run_id']
        original_record = small(folder/'records'/(rid+'__final.json'))
        e.require(original_record==row and row['exporter_sha256']==EXPORTER_SHA and row['prepared_scope_sha256']==SCOPE_SHA, 'Endpoint record/code/scope differs')
        for path, digest in [(item['row']['config_path'],row['config_sha256']),(item['row']['summary_path'],row['summary_sha256']),(row['metadata_path'],row['metadata_sha256'])]: small(path,digest,False)
        e.validate_resume(row,item)
        definition, scheduler = e.source_definition(item['bound'],item['row'],master,state)
        e.require(definition==item['definition'] and scheduler==item['scheduler_record']==row['completed_scheduler_record'], 'Completed job authority changed')
        e.require(row['status']=='verified_additional_endpoint_not_yet_assembled' and row['additional_encoding_calls']==1 and row['all_nonweight_metadata_roundtrip_verified'] and row['source_fd_five_stat_before']==row['source_fd_five_stat_after']==row['source_stat'], 'Endpoint full verification evidence missing')
        fp = row['model_fingerprint']; enc,dec,ver = row['encode'],row['decode'],row['bytewise_verification']
        e.require(fp==enc['model_fingerprint']==dec['model_fingerprint']==ver['model_fingerprint'] and fp['state_elements']==405334016, 'Full model fingerprint differs')
        e.require(dec['status']=='decoded_and_hash_verified' and dec['tensor_raw_hashes_verified']==ver['tensor_count']==292 and ver['all_tensor_bytes_equal'], 'All tensor byte/hash evidence missing')
        e.require(enc['source_checkpoint_sha256']==ver['source_checkpoint_sha256']==row['source_checkpoint_sha256']==item['bound']['checkpoint_sha256'] and row['archive_sha256']==enc['archive_sha256']==ver['archive_sha256'], 'Source or ZIP identity differs')
        e.require(enc['base_sha256']==next(f['sha256'] for f in phase['base_files'] if f['path'].endswith('model.safetensors')), 'Native base SHA differs')
        e.require(row['actual_exposure']['target_prediction_tokens']==10000000 and row['actual_exposure']['replay_prediction_tokens']==500000 and row['GPU_used'] is False and row['shared_storage_written'] is False, 'Endpoint exposure or execution scope differs')
        observed[row['source_checkpoint']]=row['source_stat'];observed[row['archive_path']]=row['archive_stat']
    for f in phase['base_files']:
        stamp = scope['frozen_files'][str(Path(f['path']).resolve())]['stat']
        e.require(signature(f['path'])==stamp,'Native base/config/tokenizer stat changed')
        observed[f['path']]=stamp
    whole_path = ROOT/'artifacts/provenance/additional_endpoint_actual_v4'/(name+'_batch/whole_child_timing.json')
    whole = small(whole_path)
    e.require(whole['returncode']==0 and whole['GPU_used'] is False and whole['publication_performed'] is False, 'Whole batch has not completed rc0')
    reserve_row = next(r for r in cap['rows'] if r['name']==reserve_name)
    reserve=Path(reserve_row['path']); rs=reserve.lstat()
    current=dict(device=rs.st_dev,inode=rs.st_ino,uid=rs.st_uid,gid=rs.st_gid,mode=stat.S_IMODE(rs.st_mode),bytes=rs.st_size,allocated_bytes=rs.st_blocks*512,links=rs.st_nlink)
    e.require(stat.S_ISREG(rs.st_mode) and current==reserve_row['actual_stat'],'Original reserved inode/capacity/owner/mode differs')
    lockkey=f'{os.major(rs.st_dev):02x}:{os.minor(rs.st_dev):02x}:{rs.st_ino}'
    e.require(not any(lockkey in line for line in Path('/proc/locks').read_text().splitlines()),'Reserve lock currently held')
    source=Path(completion['tar_path']); destination=reserve.parent/source.name
    e.require(not destination.exists() and not destination.is_symlink() and destination.parent.stat().st_dev==rs.st_dev and source.lstat().st_size==completion['tar_bytes']<=current['allocated_bytes'] and stat.S_ISREG(source.lstat().st_mode),'Destination exists or source/capacity differs')
    observed[str(source)]=signature(source)
    for path,stamp in observed.items(): e.require(signature(path)==stamp,'Input stat changed while reviewing metadata')
    result=dict(schema_version=1,status='passed_metadata_only_for_ROOT_review_not_publication_authority',utc=dt.datetime.now(dt.timezone.utc).isoformat(),package=name,source=str(source),expected_sha256=completion['tar_sha256'],expected_bytes=completion['tar_bytes'],reserved=str(reserve),reserve_original=current,destination=str(destination),logical_endpoints=n,unique_full_roundtrips_reported=n,native_base_included=True,optimizer_included=False,metadata_sources=evidence,source_payload_stats=observed,whole_batch_timing_source=str(whole_path),assembly_timer=completion,independent_sample_child_timer_not_in_batch=True,whole_batch_parent_includes_startup_resume_records_and_assembly_children=True,review_wall_seconds=time.monotonic()-started,review_process_cpu_seconds=time.process_time()-cpu,all_TAR_members_and_full_SHA_verified_by_original_exporter_receipt=True,reviewer_fresh_TAR_or_tensor_verification=False,publisher_must_fresh_source_copy_and_reserved_prefix_SHA=True,reserve_lock_observed_free_not_a_lease=True,shared_writes_or_publish_or_GPU_or_score_or_large_payload_reads=False,ROOT_authorized=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(output=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),not_authority=True)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',required=True,choices=NAMES);p.add_argument('--output',required=True,type=Path);a=p.parse_args();review(a.package,a.output)
