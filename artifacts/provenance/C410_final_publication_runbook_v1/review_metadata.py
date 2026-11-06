#!/usr/bin/env python3
"""Future one-shot C410 final metadata review; never publish/hash large payloads."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[3]
FOLDER = Path('/tmp/grad-research-20261106/delivery/extended_C_final_v1/pythia-410m')
RESERVE = Path('/data/202610/7kb/research_delivery/.extended_C_410m_final_weights_reserve_20261106')
DEST = RESERVE.parent / 'extended_C_410m_final_weights_lossless_v1.tar'
PHASE = 'extended_v2_pythia_410m_m64_formal'
PIN = {'scripts/export_C_final_weights.py': 'bc057d04298b91a8189dae58fb6189c19b69faf70c9c83298d7854bf1cc38390',
       'scripts/publish_reserved_archive.py': '9c194743383a16f98313f767fe2e722c309ec57c1aafc65d9b99d08e5407022e',
       'scripts/export_remaining_primary_endpoints.py': 'eb7339e4224062f3d55bd0adcba467599e06035ac84172f53d954b7302a1473d',
       'src/replay_study/weight_archive.py': '3c2bcf9e38773ae4f69c27f85f342356dbefcb53d817be4321da6e9c4e550e1a'}


def signature(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    evidence = {}; observed = {}
    def read(path, expected=None):
        path = Path(path).resolve(); stamp = signature(path)
        if stamp['bytes'] > 32 << 20: raise ValueError('Small metadata only')
        data = path.read_bytes(); digest = hashlib.sha256(data).hexdigest()
        if signature(path) != stamp or (expected and digest != expected): raise ValueError('Metadata source changed or SHA differs')
        evidence[str(path)] = {'sha256': digest, 'stat': stamp}; observed[str(path)] = stamp
        return json.loads(data)
    for path, expected in PIN.items():
        data = (ROOT/path).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected: raise ValueError('Original helper changed')
    completion = read(FOLDER/'completion.json')
    manifest = read(FOLDER/'extended_C_final_manifest.json', completion['manifest_sha256'])
    assert completion['status']=='passed' and completion['model']=='pythia-410m' and completion['phase']==PHASE
    assert completion['fullphase_source_gate_passed'] is True
    assert (completion['logical_final_endpoints'],completion['unique_weights'],completion['alias_count'])==(24,21,3)
    assert completion['file_fsync_completed'] and completion['parent_directory_fsync_completed'] and completion['verified_tar_members']>0
    assert manifest['status']=='complete_phase_all_final_roundtrips_verified' and manifest['data_family']=='extended_v1'
    assert (manifest['logical_final_endpoints'],manifest['unique_weights'],manifest['alias_count'])==(24,21,3)
    gate=manifest['fullphase_gate']; assert gate['status']=='complete_source_gate_passed'
    identity=gate['identity_manifest']; ready=read(identity['published_ready']['path'],identity['published_ready']['sha256'])
    assert ready['status']=='ready' and ready['ready'] is True and ready['id']=='extended_pythia_410m_100m_nll'
    evaluation=read(identity['path'],identity['sha256']); assert ready['manifest']['sha256']==identity['sha256']
    finals={r['run_id']:r for r in evaluation['records'] if r['kind']=='final'}
    assert len(finals)==24 and finals==identity['records']
    phase_completion=read(ROOT/'results/extended_phase_equivalence'/PHASE/'completion.json')
    audit=read(phase_completion['audit']['path'],phase_completion['audit']['sha256'])
    finalproof=read(phase_completion['proofs']['final']['path'],phase_completion['proofs']['final']['sha256'])
    assert phase_completion['status']=='passed' and phase_completion['expected_runs']==phase_completion['completed_runs']==24
    assert audit['status']=='passed' and audit['source_violations']==[] and audit['source_state_complete_gate']['ready'] is True
    assert finalproof['endpoint']=='final' and finalproof['phase']==PHASE and finalproof['status']=='passed'
    pairs=finalproof['comparisons']; assert len(pairs)==3 and {r['seed'] for r in pairs}=={11,22,33}
    for pair in pairs:
        assert pair['bitwise_model_state_equal'] is True and pair['left']['model_state_fingerprint']==pair['right']['model_state_fingerprint']
        for side in ('left','right'):
            item=pair[side]; assert item['checkpoint_sha256']==finals[item['run_id']]['checkpoint_sha256']
    rows=manifest['logical_runs']; assert len(rows)==24 and {r['run_id'] for r in rows}==set(finals)
    byid={r['run_id']:r for r in rows}; aliases=[]
    for pair in pairs:
        for side in ('left','right'):
            item=pair[side]; assert byid[item['run_id']]['model_fingerprint']==item['model_state_fingerprint']
    for item in rows:
        assert item['source_checkpoint_sha256']==finals[item['run_id']]['checkpoint_sha256']
        assert item['all_nonweight_metadata_roundtrip_verified'] is True
        child=item['completed_scheduler_record']; assert child['status']=='completed' and child['returncode']==0 and 'stop_requested_at' not in child
        assert item['actual_exposure']['target_prediction_tokens']==100000000
        assert item['actual_exposure']['replay_prediction_tokens']==round(100000000*item['ratio'])
        for key,statkey in [('source_checkpoint','source_stat'),('metadata_path','metadata_stat'),('archive_path','archive_stat')]:
            if key in item:
                assert signature(item[key])==item[statkey]; observed[item[key]]=item[statkey]
        if 'alias_of' in item:
            parent=byid[item['alias_of']]; assert item['group']=='D' and parent['group']=='B4' and item['seed']==parent['seed']
            assert item['model_fingerprint']==parent['model_fingerprint'] and item['additional_encoding_calls']==0
            aliases.append({'run_id':item['run_id'],'alias_of':item['alias_of'],'seed':item['seed']})
        else:
            assert item['additional_encoding_calls']==1 and item['bytewise_verification']['all_tensor_bytes_equal'] is True
    assert len(aliases)==3
    dependency=manifest['base_dependency']; assert dependency['kind']=='durable_external_primary_base_closure'
    primary=read(ROOT/'docs/primary_weights_durable_publication.json')
    assert primary['status']=='passed' and primary['destination_sha256']==dependency['tar_sha256']
    base_tar=Path(dependency['tar']); value=base_tar.lstat(); recorded=primary['destination_stat']
    fields=dict(device=value.st_dev,inode=value.st_ino,bytes=value.st_size,mtime_ns=value.st_mtime_ns,ctime_ns=value.st_ctime_ns,
                uid=value.st_uid,gid=value.st_gid,mode=oct(stat.S_IMODE(value.st_mode)))
    assert stat.S_ISREG(value.st_mode) and all(fields[k]==recorded[k] for k in fields)
    assert dependency['base']['sha256']=='3e09001631bb18ec8a6f4522f8e6ab26a53099e69228f9822adc91bfbe27511b'
    observed[str(base_tar)]=signature(base_tar)
    grow=read(ROOT/'artifacts/provenance/supplemental_weight_reservations_v1/C_410m_growth_receipt.json')
    reserve=RESERVE.lstat(); expected=grow['actual_stat']
    assert stat.S_ISREG(reserve.st_mode) and not stat.S_ISLNK(reserve.st_mode) and grow['status']=='complete' and grow['lock_released']
    assert (reserve.st_dev,reserve.st_ino,reserve.st_uid,reserve.st_gid,stat.S_IMODE(reserve.st_mode))==(106,484145481,11471,11471,0o600)
    assert reserve.st_ino==expected['inode'] and reserve.st_size==reserve.st_blocks*512==27<<30
    if Path('/proc',str(grow['pid'])).exists(): raise ValueError('Old grower PID still present; ROOT must inspect/reap')
    locks=Path('/proc/locks').read_text().splitlines(); key=f'{os.major(reserve.st_dev):02x}:{os.minor(reserve.st_dev):02x}:{reserve.st_ino}'
    if any(key in line for line in locks): raise ValueError('Reserve currently locked')
    if DEST.exists() or DEST.is_symlink(): raise FileExistsError(DEST)
    tar=Path(completion['tar_path']); assert tar==FOLDER/'extended_C_410m_final_weights_lossless_v1.tar'
    assert tar.stat().st_size==completion['tar_bytes']<=27<<30 and stat.S_ISREG(tar.lstat().st_mode)
    observed[str(tar)]=signature(tar)
    for path,stamp in observed.items():
        if signature(path)!=stamp: raise ValueError('Input changed while reviewing metadata')
    result={'schema_version':1,'status':'passed_metadata_only_for_ROOT_review_not_publication_authority',
        'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'metadata_sources':evidence,'helper_SHA':PIN,
        'source':str(tar),'expected_sha256':completion['tar_sha256'],'expected_bytes':completion['tar_bytes'],
        'reserved':str(RESERVE),'reserve_original':dict(device=reserve.st_dev,inode=reserve.st_ino,uid=reserve.st_uid,gid=reserve.st_gid,mode='0o600',allocated_bytes=reserve.st_blocks*512),
        'destination':str(DEST),'logical_final_endpoints':24,'unique_full_roundtrips_reported':21,'D_aliases':aliases,
        'all_TAR_members_and_full_SHA_verified_by_original_exporter_receipt':True,'reviewer_fresh_TAR_or_tensor_verification':False,
        'publisher_must_fresh_source_copy_and_reserved_prefix_SHA':True,'external_durable_primary_base':dependency,
        'restore_requires_both_durable_TARs_but_no_scratch_checkpoint_or_network':True,
        'reserve_lock_observed_free_not_a_lease':True,'source_payload_stats':observed,
        'shared_writes_or_publish_or_GPU_or_score_or_large_payload_reads':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'output':str(args.output),'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'not_authority':True}))


if __name__=='__main__': main()
