import copy
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('native_restore_prep', HERE / 'restore_native_final.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fixture(model='pythia-1b'):
    profile = m.PROFILES[model]; run = profile['phase'] + '_B4_s11_w0p05'
    sha = 'a' * 64; fp = dict(sha256_model_state=sha, state_elements=4)
    proofs = {k: dict(path='/tmp/synthetic/' + k + '.json', sha256=sha) for k in
              ('completion', 'manifest', 'metadata_review', 'publication', 'root_acceptance', 'ready', 'identity')}
    proofs['metadata_reviewer'] = dict(path=str(HERE.parent / 'C_native_final_publication_runbook_v2/review_metadata.py'), sha256=m.REVIEWER_SHA)
    stamp = dict(device=1, inode=2, bytes=10240, uid=3, gid=3, mode='0o600', mtime_ns=4, ctime_ns=5)
    completion = dict(status='passed', model=model, phase=profile['phase'], fullphase_source_gate_passed=True,
                      manifest_sha256=sha, logical_final_endpoints=profile['logical'], unique_weights=profile['unique'],
                      alias_count=profile['logical']-profile['unique'], file_fsync_completed=True,
                      parent_directory_fsync_completed=True, verified_tar_members=99, tar_sha256=sha, tar_bytes=stamp['bytes'])
    final = dict(kind='final', run_id=run, checkpoint_sha256=sha)
    finals = {run: final}
    for i in range(profile['logical']-1):
        key = 'synthetic_' + str(i); finals[key] = dict(kind='final', run_id=key, checkpoint_sha256=sha)
    exposure = dict(checkpoint_step=10, target_prediction_tokens=profile['target'], replay_prediction_tokens=profile['target']//20)
    row = dict(run_id=run, group='B4', seed=11, actual_exposure=exposure, source_checkpoint_sha256=sha,
               all_nonweight_metadata_roundtrip_verified=True, model_fingerprint=fp,
               encode=dict(status='encoded', model_fingerprint=fp, source_checkpoint_sha256=sha, archive_sha256=sha, base_sha256=sha),
               decode=dict(status='decoded_and_hash_verified', model_fingerprint=fp, tensor_raw_hashes_verified=1),
               bytewise_verification=dict(status='passed', model_fingerprint=fp, tensor_count=1,
                     source_checkpoint_sha256=sha, archive_sha256=sha, all_tensor_bytes_equal=True),
               archive_member='weights/' + run + '.xor.zip', archive_sha256=sha, archive_stat=dict(bytes=10),
               metadata_member='metadata/' + run + '.pt', metadata_sha256=sha, metadata_stat=dict(bytes=10))
    definition=dict(id=run,argv=['python','run_train.py','--fixed-config'])
    definition_sha=m.sha_bytes(json.dumps(definition,sort_keys=True).encode())
    row.update(definition=definition,definition_sha256=definition_sha,
               completed_scheduler_record=dict(status='completed',returncode=0,argv=definition['argv'],definition_sha256=definition_sha))
    files = [dict(member='base/'+model+'-final/'+name, sha256=sha, bytes=10) for name in
             ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'README.md')]
    manifest = dict(status='complete_phase_all_final_roundtrips_verified', model_tag=model, phase_name=profile['phase'],
                    kind='fixed_exposure_final', optimizer_included=False,
                    logical_final_endpoints=profile['logical'], unique_weights=profile['unique'], alias_count=profile['logical']-profile['unique'],
                    freeze=dict(sha256=m.FREEZE_SHA), fullphase_gate=dict(status='complete_source_gate_passed',
                        identity_manifest=dict(published_ready=proofs['ready'], sha256=sha, records=finals)),
                    logical_runs=[row]+[dict(copy.deepcopy(row),run_id=k) for k in finals if k != run],
                    base_dependency=dict(kind='native_base_config_tokenizer_readme_included', files=files, base_sha256=sha))
    destination = '/data/202610/7kb/research_delivery/extended_C_' + profile['tag'] + '_final_weights_lossless_v1.tar'
    review = dict(status='passed_metadata_only_for_ROOT_review_not_publication_authority', model=model, phase=profile['phase'],
                  derived_verified_TAR_member_count=99, expected_sha256=sha, expected_bytes=stamp['bytes'], destination=destination,
                  metadata_sources={str(Path(proofs[k]['path']).resolve()): dict(sha256=sha) for k in ('completion', 'manifest')})
    publication = dict(status='passed', source_sha256=sha, destination_sha256=sha, bytes=stamp['bytes'], destination=destination, destination_stat=stamp)
    publication.update({k: True for k in ('file_fsync_after_copy', 'prefix_sha_verified_before_truncate', 'file_fsync_after_truncate',
                                        'atomic_destination_commit_noreplace', 'parent_directory_fsync', 'original_inode_preserved')})
    acceptance = dict(status='accepted', model=model, phase=profile['phase'], scope='original_Cfinal_native_package_publication')
    acceptance.update({k: proofs[k] for k in ('completion', 'manifest', 'metadata_review', 'metadata_reviewer', 'publication')})
    values = dict(completion=completion, manifest=manifest, metadata_review=review, publication=publication,
                  root_acceptance=acceptance, ready=dict(status='ready', ready=True, id=profile['ready'], manifest=proofs['identity']),
                  identity=dict(records=list(finals.values())))
    return model, proofs, values, stamp, 100


@pytest.mark.parametrize('model', list(m.PROFILES))
def test_complete_native_plan_binds_fixed_example(model):
    plan = m.build_plan(*fixture(model))
    assert len(plan['members']) == 12 and plan['run_id'].endswith('_B4_s11_w0p05')
    assert plan['whole_TAR_rehash'] is False and plan['model_raw_bytes'] == 16


@pytest.mark.parametrize('mutation', ['pending', 'false_acceptance', 'review_sha', 'stat', 'fsync', 'ready',
                                     'wrong_cp', 'wrong_fp', 'best', 'missing_native', 'incomplete_finals',
                                     'false_rc', 'stop', 'argv'])
def test_source_closure_rejects(mutation):
    model, proofs, v, stamp, size = fixture()
    if mutation == 'pending': v['completion']['status'] = 'waiting'
    if mutation == 'false_acceptance': v['root_acceptance']['status'] = 'pending'
    if mutation == 'review_sha': v['metadata_review']['metadata_sources'][proofs['manifest']['path']]['sha256'] = 'b'*64
    if mutation == 'stat': stamp = dict(stamp, inode=999)
    if mutation == 'fsync': v['publication']['file_fsync_after_copy'] = False
    if mutation == 'ready': v['ready']['ready'] = False
    if mutation == 'wrong_cp': v['manifest']['logical_runs'][0]['source_checkpoint_sha256'] = 'b'*64
    if mutation == 'wrong_fp': v['manifest']['logical_runs'][0]['decode']['model_fingerprint'] = {}
    if mutation == 'best': v['manifest']['kind'] = 'best_target_validation'
    if mutation == 'missing_native': v['manifest']['base_dependency']['files'].pop()
    if mutation == 'incomplete_finals': v['identity']['records'].pop()
    if mutation == 'false_rc': v['manifest']['logical_runs'][-1]['completed_scheduler_record']['returncode']=False
    if mutation == 'stop': v['manifest']['logical_runs'][-1]['completed_scheduler_record']['stop_requested_at']='past'
    if mutation == 'argv': v['manifest']['logical_runs'][-1]['completed_scheduler_record']['argv']=['other']
    with pytest.raises(ValueError): m.build_plan(model, proofs, v, stamp, size)


def tiny_tar(tmp_path, names=('tiny/file.txt',)):
    archive = tmp_path/'tiny.tar'; payload = b'only synthetic bytes\n'
    with tarfile.open(archive, 'w') as tar:
        for name in names:
            info = tarfile.TarInfo(name); info.size = len(payload); tar.addfile(info, io.BytesIO(payload))
    return archive, payload


def test_selected_extraction_hashes_only_selected(tmp_path):
    archive, payload = tiny_tar(tmp_path, ('tiny/file.txt', 'ignored.txt'))
    qa = tmp_path/'qa'; qa.mkdir()
    plan = dict(archive=str(archive), archive_stat=m.signature(archive),
                members={'tiny/file.txt': dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())})
    result = m.extract_selected(plan, qa)
    assert (qa/'tiny/file.txt').read_bytes() == payload and not (qa/'ignored.txt').exists()
    assert result['selected_member_payload_bytes'] == len(payload) and not result['full_TAR_SHA_recomputed']


@pytest.mark.parametrize('problem', ['duplicate', 'bad_sha', 'changed_stat', 'traversal', 'symlink'])
def test_extraction_rejects(tmp_path, problem):
    names = ('tiny/file.txt', 'tiny/file.txt') if problem == 'duplicate' else ('tiny/file.txt',)
    archive, payload = tiny_tar(tmp_path, names); qa = tmp_path/'qa'; qa.mkdir()
    plan = dict(archive=str(archive), archive_stat=m.signature(archive),
                members={'tiny/file.txt': dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())})
    if problem == 'bad_sha': plan['members']['tiny/file.txt']['sha256'] = '0'*64
    if problem == 'changed_stat': plan['archive_stat']['mtime_ns'] -= 1
    if problem == 'traversal': plan['members'] = {'../outside': dict(bytes=1, sha256='0'*64)}
    if problem == 'symlink':
        with tarfile.open(archive, 'w') as tar:
            info=tarfile.TarInfo('tiny/file.txt'); info.type=tarfile.SYMTYPE; info.linkname='/etc/passwd'; tar.addfile(info)
        plan['archive_stat'] = m.signature(archive)
    with pytest.raises(ValueError): m.extract_selected(plan, qa)


def test_false_authority_rejected_before_any_archive_access(tmp_path):
    plan = tmp_path/'plan.json'; authority = tmp_path/'authority.json'
    plan.write_text('{}'); authority.write_text('{"authorized": false}')
    proof=lambda p: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='ROOT authority'):
        m.execute(proof(plan), proof(authority))


def test_whole_child_wrapper_retains_false_authority_failure_timing(tmp_path):
    plan=tmp_path/'plan.json'; authority=tmp_path/'authority.json'; evidence=tmp_path/'evidence'
    plan.write_text(json.dumps(dict(resource_policy=dict(max_wall_seconds=600, deadline_utc=m.PROTECTED_END.isoformat(),
                                                       admission_margin_seconds=300))))
    authority.write_text('{"authorized": false}')
    child=subprocess.run([sys.executable,str(HERE/'run_whole_child.py'),'--plan',str(plan),
        '--plan-sha256',m.sha_bytes(plan.read_bytes()),'--authority',str(authority),
        '--authority-sha256',m.sha_bytes(authority.read_bytes()),'--evidence-dir',str(evidence)],capture_output=True)
    assert child.returncode!=0
    timing=json.loads((evidence/'whole_child_timing.json').read_text())
    assert type(timing['returncode']) is int and timing['returncode']!=0 and not timing['timed_out']
    assert timing['whole_child_process_wall_seconds']>0
    assert 'Explicit exact single-model ROOT authority' in (evidence/'stderr.log').read_text()


def test_memory_gate_handles_clean_cache_without_host_free_assumption(tmp_path):
    cgroup=tmp_path/'cgroup'; cgroup.mkdir()
    (cgroup/'memory.max').write_text('1000000000'); (cgroup/'memory.current').write_text('999999999')
    (cgroup/'memory.stat').write_text('anon 100000000\nshmem 50000000\nfile 800000000\ninactive_file 600000000\nactive_file 200000000\nfile_dirty 1000\nfile_writeback 0\nunevictable 2000\n')
    (cgroup/'memory.events').write_text('oom 0\noom_kill 0\nmax 100\n')
    policy=dict(deadline_utc=m.PROTECTED_END.isoformat(), max_wall_seconds=1200, admission_margin_seconds=300,
                required_scratch_bytes=1, required_reclaim_aware_memory_bytes=500000000)
    result=m.resource_gate(policy,tmp_path,dt.datetime(2026,11,6,tzinfo=dt.timezone.utc),cgroup)
    assert result['memory_current_bytes'] == 999999999 and result['conservative_available_bytes'] == 599997001
    with pytest.raises(ValueError,match='delivery tail'):
        m.resource_gate(policy,tmp_path,m.PROTECTED_END-dt.timedelta(seconds=1500),cgroup)
    policy['required_reclaim_aware_memory_bytes']=700000000
    with pytest.raises(ValueError,match='headroom'): m.resource_gate(policy,tmp_path,dt.datetime(2026,11,6,tzinfo=dt.timezone.utc),cgroup)


@pytest.mark.parametrize('corruption', [None, 'fingerprint', 'source_cp', 'tensor_count'])
def test_isolated_checker_on_synthetic_new_cpu_tensor_file(tmp_path, corruption):
    """No original model/TAR/base/codec decode: tiny synthetic post-decode fixture only."""
    import torch
    checker_spec=importlib.util.spec_from_file_location('native_checker',HERE/'check_restored.py')
    checker=importlib.util.module_from_spec(checker_spec); checker_spec.loader.exec_module(checker)
    state={'matrix':torch.tensor([[1.0,-0.0],[2.0,3.0]]),'scalar':torch.tensor(4.0)}
    fp, raw_bytes=checker.fresh_fingerprint(state)
    assert raw_bytes==20 and fp['state_elements']==5
    assert checker.fresh_fingerprint(dict(reversed(list(state.items()))))[0]==fp
    changed={k:v.clone() for k,v in state.items()}; changed['matrix'][0,1]=0.0
    assert checker.fresh_fingerprint(changed)[0]!=fp  # signed zero bits preserved
    sha='a'*64; run='synthetic_B4_s11'; exposure=dict(checkpoint_step=1,target_prediction_tokens=10,replay_prediction_tokens=1)
    row=dict(model_fingerprint=fp,source_checkpoint_sha256=sha, archive_sha256=sha, encode=dict(base_sha256=sha),
             archive_member='selected.zip',metadata_member='own_metadata.pt',actual_exposure=exposure)
    plan=dict(model='synthetic',selected=row,run_id=run,tensor_count=2,model_raw_bytes=20)
    (tmp_path/'bound_plan.json').write_text(json.dumps(plan))
    saved=dict(model_state=state,checkpoint_kind='model_only_lossless_xor_restored',counters=exposure,
               lossless_delivery_provenance=dict(model_fingerprint=fp,source_checkpoint_sha256=sha,
                 source_checkpoint_kind='full_training_checkpoint',archive_sha256=sha,base_sha256=sha))
    if corruption=='source_cp':saved['lossless_delivery_provenance']['source_checkpoint_sha256']='b'*64
    (tmp_path/'restored').mkdir(); restored=tmp_path/'restored/B4_seed11_final.pt'; torch.save(saved,restored)
    report=dict(status='decoded_and_hash_verified',tensor_raw_hashes_verified=2,model_fingerprint=fp,gpu_used=False,
                restored_file_sha256=hashlib.sha256(restored.read_bytes()).hexdigest())
    if corruption=='fingerprint':report['model_fingerprint']={}
    if corruption=='tensor_count':report['tensor_raw_hashes_verified']=1
    Path(str(restored)+'.decode.json').write_text(json.dumps(report))
    torch.save(dict(logical_identity=dict(run_id=run),actual_exposure=exposure),tmp_path/'own_metadata.pt')
    with zipfile.ZipFile(tmp_path/'selected.zip','w') as archive:
        archive.writestr('metadata.json',json.dumps(dict(source_checkpoint_kind='full_training_checkpoint',
                           source_checkpoint=dict(sha256=sha),model_fingerprint=fp)))
    env=os.environ.copy(); env.pop('PYTHONPATH',None);env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2')
    child=subprocess.run([sys.executable,'-I',str(HERE/'check_restored.py')],cwd=tmp_path,env=env,capture_output=True)
    assert (child.returncode==0)==(corruption is None),child.stderr.decode()
    if corruption is None:
        result=json.loads((tmp_path/'independent_fingerprint.json').read_text())
        assert result['status']=='passed' and not result['original_checkpoint_read'] and not result['GPU_used']
