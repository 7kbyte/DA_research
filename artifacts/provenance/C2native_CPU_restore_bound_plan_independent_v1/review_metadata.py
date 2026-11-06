#!/usr/bin/env python3
"""One bounded metadata-only review; never bind, admit, or open payloads."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

ROOT = Path('/tmp/grad-research-20261106/worktree')
INPUT = ROOT / 'artifacts/provenance/C_native_final_CPU_restore_actual_v1/2p8b'
OUT = Path(__file__).resolve().parent
reads = {}

def demand(ok, why):
    if not ok:
        raise ValueError(why)

def sig(path):
    p = Path(path); s = p.lstat()
    demand(stat.S_ISREG(s.st_mode) and not p.is_symlink(), 'Nonregular metadata/stat target')
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, uid=s.st_uid,
                gid=s.st_gid, mode=oct(stat.S_IMODE(s.st_mode)),
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)

def raw(path, expected=None):
    p = Path(path).resolve()
    demand(p.suffix in {'.json', '.py', '.log'} and p.stat().st_size <= 32 << 20,
           'Only bounded explicit metadata/source reads permitted')
    before = sig(p); b = p.read_bytes(); sha = hashlib.sha256(b).hexdigest()
    demand(before == sig(p), 'Metadata changed while reading')
    demand(expected is None or sha == expected, 'Metadata/source SHA mismatch: ' + str(p))
    reads[str(p)] = dict(sha256=sha, bytes=len(b), stat=before)
    return b

def load(path, expected=None):
    return json.loads(raw(path, expected))

def proof(p):
    return load(p['path'], p['sha256'])

def persist(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

def main():
    plan = load(INPUT/'bound_plan.json', '573c4f4924298408d53ef5fa7b07441549260b89dbae34c5e634dd2baf82a4f2')
    sources = load(INPUT/'sources.json', '296246f17ef2c55fb44f87717e165f420e8b6dce7cde05f0b50f8197f5beb27a')
    complete = load(INPUT/'COMPLETE.json', '8c4e781fa7ce857c28ceb77d38d239dfc29f0a21b5afa4ba8654f8ba47c8f8cd')
    demand(plan['evidence'] == sources, 'Plan/source inventory mismatch')
    demand(complete['bound_plan']['sha256'] == reads[str((INPUT/'bound_plan.json').resolve())]['sha256'], 'Sealing mismatch')
    for f in complete['files']:
        b = raw(f['path'], f['sha256']); demand(len(b) == f['bytes'], 'Sealed metadata size mismatch')
    vals = {k: proof(v) for k,v in sources.items() if k != 'metadata_reviewer'}
    raw(sources['metadata_reviewer']['path'], sources['metadata_reviewer']['sha256'])
    for key in ('runner', 'checker', 'wrapper'):
        raw(complete[key]['path'], complete[key]['sha256'])
    demand(plan['runner'] == complete['runner'] and plan['checker'] == complete['checker'], 'Runtime source pins differ')
    demand(complete['runner']['sha256'] == 'bfc748c776f18dff86c378c5c55ca72dbc63995cfc40ddfb96eb919486610db1', 'Unexpected original runner')
    timer = proof(complete['whole_bind_timing'])
    demand(type(timer['returncode']) is int and timer['returncode'] == 0, 'Bind not strict int rc0')
    inv = load(INPUT/'bind_invocation.json')
    expected_argv = ['/usr/bin/python', complete['runner']['path'], 'bind', '--model', 'pythia-2.8b',
                     '--sources', str(INPUT/'sources.json'), '--sources-sha256', complete['sources']['sha256'],
                     '--output', str(INPUT/'bound_plan.json')]
    demand(inv['argv'] == expected_argv and inv['cwd'] == str(ROOT), 'Actual bind invocation differs')
    demand(plan['status'] == 'prepared_not_execution_authority' and complete['TRUE_authority_created'] is False,
           'Bound plan is not preparation-only')
    manifest = vals['manifest']; identity = vals['identity']; publication = vals['publication']; accepted = vals['root_acceptance']
    demand(manifest['status'] == 'complete_phase_all_final_roundtrips_verified', 'Incomplete source phase')
    demand(vals['completion']['status'] == 'passed' and vals['completion']['fullphase_source_gate_passed'] is True,
           'Whole source completion missing')
    demand((manifest['logical_final_endpoints'], manifest['unique_weights'], manifest['alias_count']) == (15,15,0), 'Wrong scope')
    demand(vals['completion']['manifest_sha256'] == sources['manifest']['sha256'], 'Assembly manifest mismatch')
    demand(accepted['status'] == 'accepted' and accepted['scope'] == 'original_Cfinal_native_package_publication', 'ROOT acceptance missing')
    for k in ('completion','manifest','metadata_review','metadata_reviewer','publication'):
        demand(accepted[k] == sources[k], 'ROOT proof reference mismatch: '+k)
    demand(vals['ready']['ready'] is True and vals['ready']['manifest'] == sources['identity'], 'READY mismatch')
    demand(vals['ready']['source_model_equivalence'] == {'required':False,'proofs':[]}, 'Unexpected D equivalence')
    gate = manifest['fullphase_gate']
    demand(gate['status'] == 'complete_source_gate_passed' and gate['decisions'] == {}, 'Source gate/D proof mismatch')
    audits = [proof(p) for p in gate['sources']]
    audit = next(a for a in audits if a.get('expected_runs') == 15)
    demand(audit['status'] == 'passed' and audit['completed_runs'] == 15 and audit['violations'] == [], 'Full15 audit failed')
    demand(audit['D_B4_null_controls'] == [] and audit['model_weights_compared'] is False, 'Incorrect D claim')
    for key in ('freeze',): proof(manifest[key])
    ps = manifest['phase_source']
    for name in ('active_registry','snapshot','protocol','queue'):
        path = Path(ps[name]); path = ROOT/path if not path.is_absolute() else path
        raw(path, ps[name+'_sha256'])
    finals = {r['run_id']:r for r in identity['records'] if r['kind']=='final'}
    demand(gate['identity_manifest']['records'] == finals and len(finals)==15, 'Identity closure mismatch')
    rows = manifest['logical_runs']; ids = {r['run_id'] for r in rows}
    demand(len(rows)==15 and ids==set(finals) and {(r['group'],r['seed']) for r in rows} ==
           {(g,s) for g in ('B0','B1','B2','B4','M') for s in (11,22,33)}, 'Wrong model-specific factorial scope')
    reviewed_rows=[]
    for r in rows:
        d=r['definition']; c=r['completed_scheduler_record']; f=finals[r['run_id']]
        sha=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
        demand(r['definition_sha256']==c['definition_sha256']==sha, 'Frozen training definition mismatch')
        demand(c['status']=='completed' and type(c['returncode']) is int and c['returncode']==0 and c.get('missing_outputs')==[]
               and not any('stop' in k.lower() for k in c) and c['argv']==d['argv'], 'Non-strict original training status')
        demand('alias_of' not in r and f['source_model_equivalence'] is None and f['is_training_alias'] is False, 'Invented alias')
        demand(r['source_checkpoint']==f['checkpoint'] and r['source_checkpoint_sha256']==f['checkpoint_sha256'], 'CP provenance mismatch')
        demand(r['source_stat']['bytes']==f['checkpoint_bytes'] and r['source_stat']['mtime_ns']==f['checkpoint_mtime_ns'], 'CP frozen metadata mismatch')
        for key,hkey in [('config_path','config_sha256'),('summary_path','summary_sha256')]:
            raw(f[key],r[hkey]); demand(f['input_sha256'][f[key]]==r[hkey], 'Frozen config/summary mismatch')
        demand(r['actual_exposure']['target_prediction_tokens']==25_000_000 and
               r['actual_exposure']['replay_prediction_tokens']==(0 if r['group']=='B0' else 1_250_000), 'Final exposure mismatch')
        reviewed_rows.append(dict(run_id=r['run_id'],definition_sha256=sha,strict_int_rc0_no_stop_actual_argv=True,
                                  source_checkpoint_sha256=r['source_checkpoint_sha256']))
    row=next(r for r in rows if r['run_id']==plan['run_id'])
    demand(plan['selected']==row and row['group']=='B4' and row['seed']==11, 'Fixed sample differs')
    fp=dict(sha256_model_state='0a819cb2921ba45adfd57b80ddf61f10cbdfeddbc4fb0a3d8846d53a384701b2',state_elements=2775208960)
    demand(row['model_fingerprint']==fp and all(row[k]['model_fingerprint']==fp for k in ('encode','decode','bytewise_verification')),
           'Model fingerprint closure mismatch')
    demand(plan['tensor_count']==row['decode']['tensor_raw_hashes_verified']==row['bytewise_verification']['tensor_count']==388,
           'Not full388 tensor verification')
    demand(row['bytewise_verification']['all_tensor_bytes_equal'] is True and row['all_nonweight_metadata_roundtrip_verified'] is True,
           'Original roundtrip proof incomplete')
    demand(row['actual_exposure']==dict(checkpoint_step=765,target_prediction_tokens=25_000_000,replay_prediction_tokens=1_250_000,
           full_training_target_prediction_tokens=25_000_000,full_training_replay_prediction_tokens=1_250_000), 'Not fixed final25M')
    members=plan['members']; native=manifest['base_dependency']['files']
    demand(len(members)==12 and len(native)==6, 'Wrong member/native inventory')
    for n in members:
        p=PurePosixPath(n); demand(not p.is_absolute() and '..' not in p.parts and str(p)==n and '\\' not in n, 'Unsafe TAR name')
    expected={r['member']:dict(sha256=r['sha256'],bytes=r['bytes']) for r in native}
    expected['extended_C_final_manifest.json']=dict(sha256=sources['manifest']['sha256'],bytes=reads[str(Path(sources['manifest']['path']).resolve())]['bytes'])
    expected[row['archive_member']]=dict(sha256=row['archive_sha256'],bytes=row['archive_stat']['bytes'])
    expected[row['metadata_member']]=dict(sha256=row['metadata_sha256'],bytes=row['metadata_stat']['bytes'])
    tool_pins={'tools/scripts/archive_model_weights.py':'28c5841cae8e7fd87d8cb62e7f756dad40ce15cb558710c31e48dfb45fde668a',
       'tools/src/replay_study/weight_archive.py':'3c2bcf9e38773ae4f69c27f85f342356dbefcb53d817be4321da6e9c4e550e1a',
       'tools/src/replay_study/__init__.py':'b2dce8cc5ea80a67a90b1798232163a7684874487b5ea4102e60851c7017f520'}
    for n,h in tool_pins.items():
        raw(ROOT/n.removeprefix('tools/'),h); expected[n]=dict(sha256=h,bytes=None)
    demand(members==expected, 'Selected12 member SHA/size inventory differs')
    demand(manifest['base_dependency']['kind']=='native_base_config_tokenizer_readme_included' and
           {Path(r['member']).name for r in native}=={'model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','README.md'}, 'Native6 mismatch')
    demand(all(str(PurePosixPath(r['member']).parent)=='base/pythia-2.8b-final' for r in native), 'External/foreign base')
    demand(plan['native_base']=='base/pythia-2.8b-final/model.safetensors' and
           members[plan['native_base']]['sha256']==row['encode']['base_sha256']==manifest['base_dependency']['base_sha256'], 'Base provenance mismatch')
    tarstat=sig(plan['archive'])
    demand(tarstat==plan['archive_stat'] and all(tarstat[k]==publication['destination_stat'][k]==accepted['durable_archive_current_stat'][k]
           for k in tarstat), 'Current durable TAR stat mismatch')
    demand(publication['status']=='passed' and publication['source_sha256']==publication['destination_sha256']==
           vals['completion']['tar_sha256']==plan['historical_full_TAR_sha256'], 'Historical wholeTAR SHA binding mismatch')
    demand(all(publication[k] is True for k in ('file_fsync_after_copy','prefix_sha_verified_before_truncate',
           'file_fsync_after_truncate','atomic_destination_commit_noreplace','parent_directory_fsync')), 'Durability evidence missing')
    pubtimer=proof(accepted['publication_whole_child'])
    demand(type(pubtimer['returncode']) is int and pubtimer['returncode']==0, 'Publication child not strict rc0')
    rawbytes=fp['state_elements']*4; known=sum(v['bytes'] or 0 for v in members.values())
    demand(plan['model_raw_bytes']==rawbytes==11100835840, 'Raw tensor size mismatch')
    rp=plan['resource_policy']; demand(rp==dict(required_scratch_bytes=known+rawbytes+(2<<30),
        required_reclaim_aware_memory_bytes=4*rawbytes+(4<<30),cpu_threads=2,max_wall_seconds=1200,
        deadline_utc='2026-11-07T02:56:32+00:00',admission_margin_seconds=300), 'Resource formula differs')
    demand(rp['required_scratch_bytes']==26574055053 and rp['required_reclaim_aware_memory_bytes']==48698310656, 'Resource totals differ')
    latest=dt.datetime.fromisoformat(rp['deadline_utc'])-dt.timedelta(seconds=1500)
    demand(latest.isoformat()=='2026-11-07T02:31:32+00:00', 'Time admission boundary mismatch')
    qa=Path(plan['qa_directory']); demand(not qa.exists(), 'QA directory unexpectedly already exists')
    receipt=dict(schema_version=1,status='accepted_metadata_only_bound_plan_not_execution_authority',
        at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),bound_plan=complete['bound_plan'],source_inventory=complete['sources'],
        source_completion=dict(path=str(INPUT/'COMPLETE.json'),sha256='8c4e781fa7ce857c28ceb77d38d239dfc29f0a21b5afa4ba8654f8ba47c8f8cd'),
        model=plan['model'],run_id=plan['run_id'],archive=plan['archive'],current_archive_stat=tarstat,
        historical_TAR_SHA=plan['historical_full_TAR_sha256'],whole_TAR_payload_rehashed=False,
        fullphase=dict(logical=15,unique=15,aliases=0,groups=['B0','B1','B2','B4','M'],seeds=[11,22,33],
                       original_strict_training_records=reviewed_rows,source_audit_status=audit['status'],D_proof_required=False),
        selected=dict(actual_exposure=row['actual_exposure'],model_fingerprint=fp,tensor_count=388,
                      source_checkpoint_sha256=row['source_checkpoint_sha256'],native_base_files=6,
                      selected_members=members,selected_member_known_bytes=known,
                      tool_member_sizes_historically_unspecified=3,tool_member_sizes_verified_at_future_extraction=True),
        resources=dict(**rp,admission_has_been_run=False,resource_proxy_is_not_peak_RSS_guarantee=True,
                       strict_start_before_utc=latest.isoformat(),execution_cap_seconds=1200,
                       possible_bounded_group_cleanup_seconds=10,teardown_admission_allowance_seconds=12,
                       cleanup_is_covered_by_300_second_margin=True),
        actual_bind_whole_timing=timer,
        boundaries=dict(large_payload_bytes_read=0,score_output_bytes_read=0,score_semantics_read=False,
                        tests_run=False,bind_or_restore_or_resource_gate_or_hint_run=False,
                        TRUE_authority_created=False,QA_directory_created=False,QA_directory_currently_absent=True,
                        restoration_order='ROOT must complete the 1B restore first; later 2.8B requires separate fresh admission and exact authority',
                        not_claimed=['actual extraction or recovery','fresh original CP byte comparison','Adam resume','GPU or retraining QA']),
        source_reads=reads,source_read_unique_bytes=sum(v['bytes'] for v in reads.values()),
        read_scope='Only explicitly named bounded JSON/py/log source metadata; summary bytes only SHA, no score semantic parsing',
        source=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    persist(OUT/'receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],receipt=str(OUT/'receipt.json'),
        receipt_sha256=hashlib.sha256((OUT/'receipt.json').read_bytes()).hexdigest(),unique_metadata_files=len(reads)),indent=2))

if __name__=='__main__': main()
