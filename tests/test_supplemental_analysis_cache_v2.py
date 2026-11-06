"""Explicit v5 checkpoint-only cache; every artifact is synthetic."""
import copy
import importlib.util
import os
from pathlib import Path

import pytest
from replay_study import supplemental_analysis_guard_v2 as gate


ROOT=Path(__file__).resolve().parents[1]


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(gate.encoded(value));return path


def reader():return module(ROOT/'scripts/stat_bound_digest_cache.py','aux_v2_cache_test_reader')


def cache_snapshot(tmp_path,paths):
    h=reader().StatBoundHashes(directory=tmp_path/'cache')
    for p in paths:h.digest(p,authority={'kind':'completed_training_checkpoint','returncode':0})
    return h.snapshot(tmp_path/'snapshot.json'),h


def policy(snapshot):return {'mode':'v5_checkpoint_snapshot','cache_snapshot':snapshot,'cache_reader_source':gate.proof(ROOT/'scripts/stat_bound_digest_cache.py')}


def source(tmp_path):
    p=tmp_path/'checkpoints/latest.pt';p.parent.mkdir(parents=True);p.write_bytes(b'OPAQUE_COMPLETE_CHECKPOINT_NO_DESERIALIZATION');return p


def test_cache_hit_never_rereads_checkpoint_and_reports_historical_not_current_bytes(tmp_path,monkeypatch):
    p=source(tmp_path);expected=gate.sha(p);snapshot,_=cache_snapshot(tmp_path,[p]);original=gate.sha
    def sha(path):
        assert Path(path).resolve()!=p.resolve()
        return original(path)
    monkeypatch.setattr(gate,'sha',sha);v=gate.Verifier(policy(snapshot),{str(p):expected});assert v.bind(p,expected)==expected;v.finish()
    work=v.accounting();assert work['counters']['checkpoint_cache_hits']==1 and work['counters']['checkpoint_cached_bytes']==p.stat().st_size
    assert work['counters']['checkpoint_fresh_bytes']==0 and work['checkpoint_events'][0]['current_bytes_read']==0
    assert not work['cache_hits_are_current_byte_reads']


def test_default_fresh_and_snapshot_missing_entry_both_read_complete_checkpoint(tmp_path):
    p=source(tmp_path);expected=gate.sha(p);snapshot,_=cache_snapshot(tmp_path,[])
    for setup in (None,policy(snapshot)):
        v=gate.Verifier(setup,{str(p):expected});v.bind(p,expected);v.finish()
        assert v.work['checkpoint_fresh_reads']==1 and v.work['checkpoint_fresh_bytes']==p.stat().st_size and v.work['checkpoint_cache_hits']==0


def test_generic_full_read_receipt_without_completed_checkpoint_authority_is_not_a_hit(tmp_path):
    p=source(tmp_path);expected=gate.sha(p);h=reader().StatBoundHashes(directory=tmp_path/'cache');h.digest(p)
    snapshot=h.snapshot(tmp_path/'snapshot.json');v=gate.Verifier(policy(snapshot),{str(p):expected});v.bind(p,expected)
    assert v.work['checkpoint_cache_hits']==0 and v.work['checkpoint_fresh_bytes']==p.stat().st_size


@pytest.mark.parametrize('name',['score.json','alias_execution.json','scorer.py','evaluation.pt'])
def test_noncheckpoint_outputs_attestations_code_and_tensor_are_fresh_even_if_cached(tmp_path,name):
    p=tmp_path/name;p.write_bytes(b'old');expected=gate.sha(p);snapshot,_=cache_snapshot(tmp_path,[p]);p.write_bytes(b'new')
    v=gate.Verifier(policy(snapshot),{})
    with pytest.raises(ValueError,match='SHA mismatch'):v.bind(p,expected)
    assert v.work['checkpoint_cache_hits']==0 and v.work['other_fresh_reads']>=3


@pytest.mark.parametrize('change',['snapshot_sha','receipt_sha','not_eof','source_reader','expected_digest'])
def test_corrupt_or_unbound_historical_evidence_fails_closed(tmp_path,change):
    p=source(tmp_path);expected=gate.sha(p);snapshot,h=cache_snapshot(tmp_path,[p]);s=copy.deepcopy(snapshot)
    if change=='snapshot_sha':s['sha256']='0'*64
    elif change in ('receipt_sha','not_eof'):
        entry=h.entries[str(p.resolve())]
        if change=='receipt_sha':Path(entry['path']).write_bytes(b'changed receipt')
        else:
            r=gate.read(entry['path']);r['complete_to_eof']=False;write(entry['path'],r)
            index=gate.read(snapshot['path']);index['entries'][str(p.resolve())]=gate.proof(entry['path']);write(snapshot['path'],index);s=gate.proof(snapshot['path'])
    elif change=='source_reader':
        index=gate.read(snapshot['path']);index['reader_source']['sha256']='0'*64;write(snapshot['path'],index);s=gate.proof(snapshot['path'])
    else:expected='0'*64
    with pytest.raises(ValueError):
        v=gate.Verifier(policy(s),{str(p):expected});v.bind(p,expected)


def test_changed_inode_same_bytes_falls_back_fresh_and_changed_bytes_are_rejected(tmp_path):
    p=source(tmp_path);expected=gate.sha(p);snapshot,_=cache_snapshot(tmp_path,[p]);replacement=p.with_name('replacement.pt');replacement.write_bytes(p.read_bytes());replacement.replace(p)
    v=gate.Verifier(policy(snapshot),{str(p):expected});v.bind(p,expected);v.finish()
    assert v.work['checkpoint_fresh_reads']==1 and v.work['checkpoint_cache_hits']==0
    assert v.accounting()['cache_reader_counters']['invalidations']==1
    old=p.stat();p.write_bytes(b'X'*old.st_size);os.utime(p,ns=(old.st_atime_ns,old.st_mtime_ns))
    with pytest.raises(ValueError,match='bytes changed'):gate.Verifier(policy(snapshot),{str(p):expected}).bind(p,expected)


def test_symlink_retarget_and_postverification_stat_change_cannot_hide_under_cache(tmp_path):
    p=source(tmp_path);expected=gate.sha(p);snapshot,_=cache_snapshot(tmp_path,[p]);link=tmp_path/'logical.pt';link.symlink_to(p)
    v=gate.Verifier(policy(snapshot),{str(p):expected});v.bind(link,expected)
    other=tmp_path/'other.pt';other.write_bytes(p.read_bytes());link.unlink();link.symlink_to(other)
    with pytest.raises(ValueError,match='symlink changed'):v.finish()
    v=gate.Verifier(policy(snapshot),{str(p):expected});v.bind(p,expected);p.chmod(0o600)
    with pytest.raises(ValueError,match='Source changed'):v.finish()


def prepared(tmp_path,with_cache=True,mode='nll',both=False,alias=True):
    f=module(Path(__file__).with_name('test_supplemental_analysis_guard.py'),'aux_v2_whole_fixture')
    ready,protocol,state,_,_,m=f.fixture(tmp_path,alias=alias,mode=mode,both=both)
    p=gate.read(protocol);p.update(schema_version=2,operational_revision=2,checkpoint_cache_reader_source=gate.proof(ROOT/'scripts/stat_bound_digest_cache.py'))
    p['analysis_source_sha256'][str(Path(gate.__file__).resolve())]=gate.sha(gate.__file__);write(protocol,p)
    cache=cache_snapshot(tmp_path,[r['checkpoint'] for r in m['records']])[0] if with_cache else None
    plan=write(tmp_path/'plan_v2.json',gate.prepare(gate.proof(ready),gate.proof(protocol),'fixture',cache))
    return plan,state,m,f


@pytest.mark.parametrize('mode',['nll','behavior'])
def test_entire_phase_and_aliases_pass_with_explicit_cache_and_score_json_is_not_parsed(tmp_path,mode):
    plan,state,m,_=prepared(tmp_path,mode=mode);result=gate.inspect(gate.proof(plan),state)
    assert result['ready'] and not result['score_json_parsed']
    assert result['verification_work']['counters']['checkpoint_cache_hits']==2 and result['verification_work']['counters']['checkpoint_fresh_bytes']==0
    p=Path(m['records'][1]['output']);p.write_bytes(b'changed sealed score bytes')
    with pytest.raises(ValueError,match='SHA mismatch'):gate.inspect(gate.proof(plan),state)


def test_pending_or_missing_C_best_blocks_first_read_even_with_all_checkpoints_cached(tmp_path):
    plan,state,m,_=prepared(tmp_path,both=True);s=gate.read(state);s['jobs']['shared']['status']='running';write(state,s)
    result=gate.inspect(gate.proof(plan),state)
    assert not result['ready'] and result['verification_work']['counters']['checkpoint_cache_hits']==0
    s['jobs']['shared']['status']='completed';write(state,s);Path(m['records'][-1]['output']).unlink()
    with pytest.raises(FileNotFoundError):gate.inspect(gate.proof(plan),state)
    assert not Path(gate.read(plan)['first_read_receipt_path']).exists()


def test_root_authority_binds_exact_fresh_or_snapshot_policy(tmp_path):
    plan,state,m,f=prepared(tmp_path);a=f.authority(plan,tmp_path/'authority.json');value=gate.read(a)
    with pytest.raises(PermissionError):gate.authorize(gate.proof(plan),gate.proof(a))
    value['checkpoint_verification']=gate.read(plan)['checkpoint_verification'];write(a,value);gate.authorize(gate.proof(plan),gate.proof(a))
    value['checkpoint_verification']={'mode':'fresh','cache_snapshot':None};write(a,value)
    with pytest.raises(PermissionError):gate.authorize(gate.proof(plan),gate.proof(a))


def test_v2_collector_is_called_after_cache_integrity_and_immutable_first_read(tmp_path,monkeypatch):
    plan,state,_,f=prepared(tmp_path);a=f.authority(plan,tmp_path/'authority.json');value=gate.read(a);value['checkpoint_verification']=gate.read(plan)['checkpoint_verification'];write(a,value)
    cli=module(ROOT/'scripts/analyze_supplemental_phase_v2.py','aux_v2_cli')
    from replay_study import nll_v2_analysis as nll
    def collect(*args,**kwargs):
        assert Path(gate.read(plan)['first_read_receipt_path']).exists()
        assert gate.read(tmp_path/'analysis/whole_phase_readiness.json')['verification_work']['counters']['checkpoint_cache_hits']==2
        return {'complete':True,'completed_conditions':2}
    monkeypatch.setattr(nll,'collect_scores',collect);monkeypatch.setattr(nll,'write_bundle',lambda *a:None)
    assert cli.analyze(gate.proof(plan),gate.proof(a),state,tmp_path/'analysis',no_plots=True)['complete']


def test_existing_NLL_preparer_and_collector_never_rehash_checkpoint_payloads(tmp_path,monkeypatch):
    plan,state,m,_=prepared(tmp_path,alias=False)
    f=module(Path(__file__).with_name('test_nll_v2_analysis.py'),'v2_CP_read_audit_fixture');_,kwargs,examples=f.fixture(tmp_path/'examples',role='supplement_B')
    for job,example in zip(m['jobs'],examples['jobs']):
        value=gate.read(example['output']);value['checkpoint']=job.get('checkpoint');out=write(job['output'],value)
        ap=gate.attestation_path(out);a=gate.read(ap);a['output_sha256'][str(out)]=gate.sha(out);write(ap,a)
    assert gate.inspect(gate.proof(plan),state)['ready']
    from replay_study import nll_v2_analysis as nll
    checkpoints={Path(r['checkpoint']).resolve() for r in m['records']};original=nll.sha;seen=[]
    def sha(path):
        assert Path(path).resolve() not in checkpoints
        seen.append(str(path));return original(path)
    monkeypatch.setattr(nll,'sha',sha);p=gate.read(plan)
    existing=nll.prepare_plan(p['manifest']['path'],p['manifest']['sha256'],**kwargs)
    bundle=nll.collect_scores(existing,read_heldout_scores=True,allow_partial=False,n_bootstrap=0)
    assert bundle['complete'] and len(bundle['endpoints'])==2 and seen


def test_full126_A_with_selected18_uses_only_explicit_CP_hits_and_keeps_scores_sealed(tmp_path):
    f=module(Path(__file__).with_name('test_supplemental_A_whole_phase.py'),'cache_v2_A126_fixture')
    ready,protocol,state,_,manifest=f.A_fixture(tmp_path)
    value=gate.read(protocol);value.update(schema_version=2,operational_revision=2,checkpoint_cache_reader_source=gate.proof(ROOT/'scripts/stat_bound_digest_cache.py'));write(protocol,value)
    snapshot,_=cache_snapshot(tmp_path,[r['checkpoint'] for r in manifest['records']])
    plan=write(tmp_path/'plan_v2.json',gate.prepare(gate.proof(ready),gate.proof(protocol),'supplement_A_nll',snapshot))
    assert len(gate.read(plan)['selection']['selected_ids'])==18
    result=gate.inspect(gate.proof(plan),state)
    assert result['ready'] and result['counts']['logical_endpoints']==126 and result['closure_job_count']==127
    assert result['verification_work']['counters']['checkpoint_cache_hits']==126
    assert result['verification_work']['counters']['checkpoint_fresh_bytes']==0 and not result['score_json_parsed']
