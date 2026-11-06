"""Independent tiny artifacts only; no real scores/checkpoints are opened."""
import importlib.util
from pathlib import Path
import pytest
from replay_study import supplemental_analysis_guard_v3 as gate
ROOT=Path('/data/202610/7kb/research').resolve()
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def fixture():return module(ROOT/'tests/test_supplemental_analysis_guard_v3.py','independent_v3_fixture')
def test_copied_plan_path_cannot_inherit_original_repair(tmp_path):
    f=fixture();plan,state,m,rp,_=f.prepared(tmp_path)
    copied=tmp_path/'copied_same_bytes_plan.json';copied.write_bytes(plan.read_bytes())
    with pytest.raises(ValueError,match='non-authorizing operational repair'):
        gate.load_plan(gate.proof(copied),rp)
    assert not Path(gate.read(plan)['first_read_receipt_path']).exists()
def test_changed_checkpoint_inode_requires_fresh_whole_phase_read(tmp_path):
    f=fixture();plan,state,m,rp,_=f.prepared(tmp_path)
    checkpoint=Path(m['records'][0]['checkpoint']);replacement=checkpoint.with_name('replacement.pt')
    replacement.write_bytes(checkpoint.read_bytes());replacement.replace(checkpoint)
    result=gate.inspect(gate.proof(plan),state,rp)
    assert result['ready'] and not result['score_json_parsed']
    work=result['verification_work']['counters']
    assert work['checkpoint_fresh_reads']==1 and work['checkpoint_cache_hits']==1
    assert work['checkpoint_fresh_bytes']==checkpoint.stat().st_size
    assert not Path(gate.read(plan)['first_read_receipt_path']).exists()
def test_signed_cached_pt_outside_checkpoint_allowlist_stays_fresh(tmp_path):
    f=module(ROOT/'tests/test_supplemental_analysis_cache_v2.py','independent_v3_cache_fixture')
    p=tmp_path/'sealed_score.pt';p.write_bytes(b'SYNTHETIC_OUTPUT_NOT_A_CHECKPOINT')
    snapshot,_=f.cache_snapshot(tmp_path,[p])
    verifier=gate.Verifier(f.policy(snapshot),{})
    verifier.bind(p,gate.sha(p));verifier.finish()
    assert verifier.work['checkpoint_cache_hits']==0 and verifier.work['checkpoint_fresh_reads']==0
    assert verifier.work['other_fresh_bytes']>=p.stat().st_size
@pytest.mark.parametrize('key',['stop_requested_at','stop_requested'])
def test_cached_completed_receipt_with_stop_key_is_rejected(tmp_path,key):
    f=module(ROOT/'tests/test_supplemental_analysis_cache_v2.py','independent_v3_cache_stop_fixture')
    p=f.source(tmp_path);h=f.reader().StatBoundHashes(directory=tmp_path/'cache')
    h.digest(p,authority={'kind':'completed_training_checkpoint','returncode':0,key:None})
    snapshot=h.snapshot(tmp_path/'snapshot.json')
    verifier=gate.Verifier(f.policy(snapshot),{str(p):gate.sha(p)})
    with pytest.raises(ValueError,match='stop-requested'):gate.strict_cache_completion_authorities(verifier)
    assert verifier.work['checkpoint_fresh_bytes']==0
@pytest.mark.parametrize('change',['wrong_journal','extra_output'])
def test_journal_override_or_extra_attested_output_is_fail_closed(tmp_path,change):
    f=fixture();plan,state,m,rp,_=f.prepared(tmp_path)
    if change=='wrong_journal':
        cli=module(ROOT/'scripts/analyze_supplemental_phase_v3.py','independent_v3_cli')
        authority=f.authority(tmp_path,plan,rp)
        with pytest.raises(ValueError,match='immutable per phase manifest'):
            cli.analyze(gate.proof(plan),authority,state,tmp_path/'analysis',rp,first_receipt=tmp_path/'wrong.json',no_plots=True)
        assert not (tmp_path/'analysis').exists()
    else:
        ap=gate.attestation_path(m['jobs'][0]['output']);a=gate.read(ap)
        a['output_sha256'][str(tmp_path/'unexpected.json')]='0'*64;f.write(ap,a)
        with pytest.raises(ValueError,match='Complete exact'):gate.inspect(gate.proof(plan),state,rp)
    assert not Path(gate.read(plan)['first_read_receipt_path']).exists()
