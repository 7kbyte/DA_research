"""Entirely synthetic score-blind gates; private outputs are invalid JSON."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from replay_study import supplemental_analysis_guard as guard


def write(path,value):path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(guard.encoded(value));return path
def digest_job(job):return hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()


def fixture(tmp_path,alias=False,mode='nll',both=False):
    p=Path(__file__).with_name('test_nll_v2_analysis.py');sp=importlib.util.spec_from_file_location('supp_guard_fixture',p);f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f)
    mp,_,m=f.fixture(tmp_path,role='supplement_B',alias=alias)
    if both:
        registry=guard.read(m['phases'][0]['registry'])
        for r in registry:r['phase']='formal'
        write(m['phases'][0]['registry'],registry);m['phases'][0].update(role='extended',data_family='extended_v1',expected_phase='formal',registry_sha256=guard.sha(m['phases'][0]['registry']))
        for r in m['records']:r.update(role='extended',data_family='extended_v1',phase='formal')
    group=m['baseline_groups']['base'];group['base_model_assets_sha256']={};group['inference_policy'].update(batch_size=16,max_length=512)
    m['common_input_sha256']={};m['scorer_input_sha256']={};m['mode']=mode
    if mode=='behavior':group['tasks']={'piqa':{},'hellaswag':{}};group['inference_policy']['max_length']=2048
    for record in m['records']:
        cp=tmp_path/(record['run_id']+'_checkpoint.pt');cp.write_bytes(b'identical test weights' if alias else record['run_id'].encode())
        record.update(checkpoint=str(cp),checkpoint_sha256=guard.sha(cp),weights_identity='checkpoint:'+guard.sha(cp),input_sha256={str(cp):guard.sha(cp)})
    if both:
        for r in list(m['records']):
            best={**copy.deepcopy(r),'id':r['id'].replace('__final','__best'),'kind':'best','target_prediction_tokens_at_checkpoint':100,'replay_prediction_tokens_at_checkpoint':5,
                'output':str(tmp_path/(r['run_id']+'_best.json'))}
            m['records'].append(best)
    for job in m['jobs']:
        rr=[r for r in m['records'] if r['inference_job_id']==job['id']]
        job['input_sha256']={};job['argv']=['synthetic_scorer',job['id']]
        if rr:
            first=rr[0];job.update(checkpoint=first['checkpoint'],checkpoint_sha256=first['checkpoint_sha256'],weights_identity=first['weights_identity'],record_ids=[r['id'] for r in rr],depends_on=['initial'])
            for r in rr:job['input_sha256'].update(r['input_sha256'])
    m['workload']={'logical_checkpoint_endpoints':len(m['records']),'total_queue_jobs':len(m['jobs']),'total_inference_jobs':len(m['jobs']),'reuse_materialization_jobs':0}
    write(mp,m);queue=[]
    for job in m['jobs']:
        definition={'id':job['id'],'argv':[sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/run_frozen_evaluation.py'),
            '--manifest',str(mp),'--manifest-sha256',guard.sha(mp),'--job-id',job['id'],'--device','cuda:0'],'outputs':[]}
        if job.get('depends_on'):definition['depends_on']=job['depends_on']
        queue.append(definition)
    qp=write(tmp_path/'queue.json',queue)
    phase=m['phases'][0];contract={'family':'TEST','mode':mode,'phase':{k:phase[k] for k in ('name','role','model_tag','data_family','expected_phase','rows')},
        'checkpoints':['final','best'] if both else ['final'],'logical_endpoints':len(m['records']),'inference_policy_expect':group['inference_policy']}
    protocol=write(tmp_path/'protocol.json',{'schema_version':1,'bootstrap_replicates':5000,'bootstrap_seed':20261106,'confidence':.95,'allow_partial_for_final_conclusions':False,
        'first_read_root':str(tmp_path/'first_reads'),'contracts':{'fixture':contract},'analysis_source_sha256':{str(Path(guard.__file__).resolve()):guard.sha(guard.__file__)},'prior_sources':[]})
    ready=write(tmp_path/'READY.json',{'status':'ready','ready':True,'id':'fixture','manifest':guard.proof(mp),'queue':guard.proof(qp),
        'phase_spec':{'path':m['phase_spec'],'sha256':m['phase_spec_sha256']},'logical_endpoints':len(m['records']),'workload':m['workload']})
    state=write(tmp_path/'state.json',{'jobs':{j['id']:{'status':'completed','returncode':0,'missing_outputs':[],'definition_sha256':digest_job(j)} for j in queue}})
    for job in m['jobs']:
        cases=[(job['output'],None),*[(r['output'],r) for r in m['records'] if r['inference_job_id']==job['id'] and r['output']!=job['output']]]
        for output,record in cases:
            output=Path(output);output.write_bytes(b'PRIVATE_SCORE_NEVER_JSON_PARSED')
            files=[output]
            if mode=='behavior':
                for task in group['tasks']:
                    p=output.with_name(output.stem+'_'+task+'.jsonl');p.write_bytes(b'PRIVATE_PREDICTION_NOT_JSON');files.append(p)
            att={'manifest_sha256':guard.sha(mp),'job_id':job['id'],'baseline_group':'base','verified_frozen_inputs':True,
                'checkpoint_sha256':record['checkpoint_sha256'] if record else job.get('checkpoint_sha256'),
                'weights_identity':record['weights_identity'] if record else job.get('weights_identity'),'actual_argv':job['argv'],
                'inference_alias':bool(record),'inference_subprocess_wall_seconds':0 if record else 1.,'output_sha256':{str(p):guard.sha(p) for p in files}}
            if record:att.update(logical_record_id=record['id'],canonical_output=str(Path(job['output']).resolve()),source_model_equivalence=record.get('source_model_equivalence'))
            write(guard.attestation_path(output),att)
    return ready,protocol,state,mp,qp,m


def prepare_fixture(tmp_path,**kw):
    ready,protocol,state,mp,qp,m=fixture(tmp_path,**kw)
    plan=guard.prepare(guard.proof(ready),guard.proof(protocol),'fixture');pp=write(tmp_path/'analysis_plan.json',plan)
    return pp,state,mp,qp,m


@pytest.mark.parametrize('mode',['nll','behavior'])
def test_complete_phase_checks_hashes_and_aliases_without_parsing_outputs(tmp_path,mode,monkeypatch):
    pp,state,_,_,m=prepare_fixture(tmp_path,alias=True,mode=mode)
    private={j['output'] for j in m['jobs']}|{r['output'] for r in m['records']};original=guard.read
    def read(path):assert str(path) not in private;return original(path)
    monkeypatch.setattr(guard,'read',read)
    result=guard.inspect(guard.proof(pp),state)
    assert result['ready'] and result['score_files_hashed'] and not result['score_json_parsed']
    assert result['counts']['logical_aliases']==1
    assert b'PRIVATE_SCORE' not in guard.encoded(result)


def test_C_requires_both_final_and_best_aliases_before_first_read(tmp_path):
    pp,state,_,_,m=prepare_fixture(tmp_path,both=True)
    Path(m['records'][-1]['output']).unlink()
    with pytest.raises(FileNotFoundError):guard.inspect(guard.proof(pp),state)
    assert not (tmp_path/'first_read.json').exists()


def test_pending_phase_never_probes_or_hashes_any_score(tmp_path,monkeypatch):
    pp,state,_,_,m=prepare_fixture(tmp_path);s=guard.read(state);s['jobs']['M']['status']='running';write(state,s)
    outputs={j['output'] for j in m['jobs']};original=guard.sha;seen=[]
    def sha(path):
        assert str(path) not in outputs;seen.append(str(path));return original(path)
    monkeypatch.setattr(guard,'sha',sha);result=guard.inspect(guard.proof(pp),state)
    assert not result['ready'] and not result['score_files_hashed']


@pytest.mark.parametrize('change',['definition','checkpoint','attestation','alias','score','policy','READY','missing_baseline'])
def test_changed_or_missing_source_rejects_complete_analysis(change,tmp_path):
    pp,state,mp,qp,m=prepare_fixture(tmp_path,alias=True)
    if change=='definition':s=guard.read(state);s['jobs']['shared']['definition_sha256']='wrong';write(state,s)
    elif change=='checkpoint':Path(m['records'][0]['checkpoint']).write_bytes(b'changed weights')
    elif change=='score':Path(m['jobs'][1]['output']).write_bytes(b'changed score bytes')
    elif change=='missing_baseline':Path(m['jobs'][0]['output']).unlink()
    elif change in ('attestation','alias'):
        record=m['records'][1];p=guard.attestation_path(record['output']);a=guard.read(p)
        if change=='attestation':a['manifest_sha256']='wrong'
        else:a['inference_subprocess_wall_seconds']=1.
        write(p,a)
    elif change=='policy':m['baseline_groups']['base']['inference_policy']['batch_size']=8;write(mp,m)
    else:Path(guard.read(pp)['ready']['path']).write_text('{}')
    with pytest.raises((ValueError,FileNotFoundError)):guard.inspect(guard.proof(pp),state)


def authority(pp,path,allowed=True):
    p=guard.read(pp)
    return write(path,{'authorized_by':'root','allow_score_analysis':allowed,'plan':guard.proof(pp),'protocol':p['protocol'],'ready':p['ready'],'manifest':p['manifest'],
        'contract_id':p['contract_id'],'expected_counts':p['counts'],'allow_partial':False,'training_methods_grid_selection_horizons_frozen':True,'test_or_behavior_used_for_selection':False})


def test_explicit_authority_and_immutable_first_read_are_bound_to_whole_phase(tmp_path):
    pp,state,_,_,_=prepare_fixture(tmp_path);a=authority(pp,tmp_path/'authority.json',False)
    with pytest.raises(PermissionError):guard.authorize(guard.proof(pp),guard.proof(a))
    a=authority(pp,a,True);guard.authorize(guard.proof(pp),guard.proof(a));r=write(tmp_path/'readiness.json',guard.inspect(guard.proof(pp),state));journal=tmp_path/'first_read.json'
    guard.first_read(journal,guard.proof(pp),guard.proof(a),guard.proof(r),'initial.json')
    original=journal.read_bytes()
    with pytest.raises(FileExistsError):guard.first_read(journal,guard.proof(pp),guard.proof(a),guard.proof(r),'another.json')
    assert journal.read_bytes()==original and guard.read(journal)['before_first_score_parse'] is True


def test_collector_is_called_only_after_complete_gate_and_persisted_receipts(tmp_path,monkeypatch):
    pp,state,_,_,_=prepare_fixture(tmp_path);a=authority(pp,tmp_path/'authority.json');output=tmp_path/'analysis'
    script=Path(__file__).resolve().parents[1]/'scripts/analyze_supplemental_phase.py';sp=importlib.util.spec_from_file_location('supplemental_analysis_cli',script);cli=importlib.util.module_from_spec(sp);sp.loader.exec_module(cli)
    from replay_study import nll_v2_analysis as nll
    called=[]
    def collect(*args,**kwargs):
        assert Path(guard.read(pp)['first_read_receipt_path']).exists() and guard.read(output/'whole_phase_readiness.json')['ready']
        assert kwargs['allow_partial'] is False and kwargs['n_bootstrap']==5000;called.append(True)
        return {'complete':True,'completed_conditions':2}
    monkeypatch.setattr(nll,'collect_scores',collect);monkeypatch.setattr(nll,'write_bundle',lambda *a:None)
    assert cli.analyze(guard.proof(pp),guard.proof(a),state,output,no_plots=True)['complete'] and called


def imported_fixture(tmp_path):
    _,_,old_state,old_mp,old_qp,old=fixture(tmp_path/'prior',alias=True)
    ready,protocol,state,mp,qp,m=fixture(tmp_path/'target',alias=True)
    origin={'manifest':str(old_mp),'manifest_sha256':guard.sha(old_mp),'job_id':'shared','output':old['jobs'][1]['output']}
    baseline={**origin,'job_id':'initial','output':old['jobs'][0]['output']}
    m['baseline_groups']['base'].update(initial_origin=baseline,initial_output=baseline['output'])
    job=m['jobs'][1];job.update(id='reuse',kind='reuse',output=origin['output'],inference_origin=origin,depends_on=['initial','shared'])
    m['jobs']=[job]
    for record in m['records']:record['inference_job_id']='reuse'
    m['workload'].update(total_queue_jobs=1,total_inference_jobs=0,reuse_materialization_jobs=1)
    write(mp,m)
    definition={'id':'reuse','argv':[sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/run_frozen_evaluation.py'),
        '--manifest',str(mp),'--manifest-sha256',guard.sha(mp),'--job-id','reuse','--device','cuda:0'],'outputs':[],'depends_on':['initial','shared']}
    write(qp,[definition])
    p=guard.read(protocol);p['prior_sources']=[{'queue':guard.proof(old_qp),'manifests':[guard.proof(old_mp)]}];write(protocol,p)
    rr=guard.read(ready);rr.update(manifest=guard.proof(mp),queue=guard.proof(qp),workload=m['workload']);write(ready,rr)
    ss=guard.read(old_state);ss['jobs']['reuse']={'status':'completed','returncode':0,'missing_outputs':[],'definition_sha256':digest_job(definition)};write(state,ss)
    for record in m['records']:
        ap=guard.attestation_path(record['output']);a=guard.read(ap)
        a.update(manifest_sha256=guard.sha(mp),job_id='reuse',inference_alias=True,inference_subprocess_wall_seconds=0,
            logical_record_id=record['id'],canonical_output=origin['output'],source_model_equivalence=None,inference_origin=origin)
        write(ap,a)
    plan=write(tmp_path/'plan.json',guard.prepare(guard.proof(ready),guard.proof(protocol),'fixture'))
    return plan,state,old,protocol,ready


def test_imported_baseline_and_reused_inference_require_complete_original_attestations(tmp_path):
    plan,state,old,_,_=imported_fixture(tmp_path)
    result=guard.inspect(guard.proof(plan),state)
    assert result['ready'] and result['closure_job_count']==3
    assert result['counts']['reuse_jobs']==1 and result['counts']['new_forward_jobs']==0 and result['counts']['imported_baselines']==1
    assert result['counts']['closure_canonical_forward_jobs']==2 and result['counts']['imported_canonical_forward_jobs']==2
    assert len(result['attestations'])==5  # prior baseline/canonical/alias + two new zero-forward aliases
    Path(old['jobs'][0]['output']).write_bytes(b'prior baseline changed')
    with pytest.raises(ValueError,match='SHA mismatch'):guard.inspect(guard.proof(plan),state)


def test_imported_baseline_must_complete_before_any_phase_score_is_hashed(tmp_path,monkeypatch):
    plan,state,old,_,_=imported_fixture(tmp_path);s=guard.read(state);s['jobs']['initial'].update(status='failed',returncode=1);write(state,s)
    original=guard.sha
    def sha(path):
        assert str(path) not in {j['output'] for j in old['jobs']}
        return original(path)
    monkeypatch.setattr(guard,'sha',sha);result=guard.inspect(guard.proof(plan),state)
    assert not result['ready'] and result['pending_job_ids']==['initial'] and not result['score_files_hashed']


def test_reuse_without_explicit_prior_authority_rejects_before_outputs(tmp_path):
    _,_,_,protocol,ready=imported_fixture(tmp_path);p=guard.read(protocol);p['prior_sources']=[];write(protocol,p)
    with pytest.raises(ValueError,match='prior inference closure'):guard.prepare(guard.proof(ready),guard.proof(protocol),'fixture')


def test_no_authority_never_calls_collector_or_creates_first_read(tmp_path,monkeypatch):
    pp,state,_,_,_=prepare_fixture(tmp_path);a=authority(pp,tmp_path/'authority.json',False);output=tmp_path/'analysis'
    script=Path(__file__).resolve().parents[1]/'scripts/analyze_supplemental_phase.py';sp=importlib.util.spec_from_file_location('supplemental_no_authority',script);cli=importlib.util.module_from_spec(sp);sp.loader.exec_module(cli)
    from replay_study import nll_v2_analysis as nll
    monkeypatch.setattr(nll,'collect_scores',lambda *a,**k:pytest.fail('Collector called without authority'))
    with pytest.raises(PermissionError):cli.analyze(guard.proof(pp),guard.proof(a),state,output,no_plots=True)
    assert not output.exists()


@pytest.mark.parametrize('equal',[False,True])
def test_behavior_alias_requires_true_complete_state_proof_bound_to_both_checkpoint_bytes(tmp_path,equal):
    ready,protocol,state,mp,qp,m=fixture(tmp_path,alias=True,mode='behavior')
    algorithm=tmp_path/'fingerprint_algorithm.py';algorithm.write_text('# Synthetic algorithm identity, never applied to real weights\n')
    fp={'sha256_model_state':'f'*64,'state_elements':2};sides=[{'checkpoint':r['checkpoint'],'checkpoint_sha256':r['checkpoint_sha256'],'model_state_fingerprint':fp} for r in m['records']]
    pp=write(tmp_path/'equivalence.json',{'schema_version':1,'status':'passed','algorithm':'sha256_sorted_key_dtype_shape_contiguous_uint8_v1',
        'algorithm_source':guard.proof(algorithm),'comparisons':[{'bitwise_model_state_equal':equal,'left':sides[0],'right':sides[1]}]})
    evidence={'proof':str(pp),'proof_sha256':guard.sha(pp),'comparison_index':0,'fingerprint':fp,
        'left_checkpoint_sha256':sides[0]['checkpoint_sha256'],'right_checkpoint_sha256':sides[1]['checkpoint_sha256']}
    for r in m['records']:r.update(weights_identity='model_state:'+fp['sha256_model_state'],source_model_equivalence=evidence)
    m['jobs'][1].update(weights_identity='model_state:'+fp['sha256_model_state'],source_model_equivalence=evidence);write(mp,m)
    q=guard.read(qp)
    for definition in q:definition['argv'][definition['argv'].index('--manifest-sha256')+1]=guard.sha(mp)
    write(qp,q);s=guard.read(state)
    for definition in q:s['jobs'][definition['id']]['definition_sha256']=digest_job(definition)
    write(state,s)
    rr=guard.read(ready);rr.update(manifest=guard.proof(mp),queue=guard.proof(qp));write(ready,rr)
    for job in m['jobs']:
        for output in [job['output'],*[r['output'] for r in m['records'] if r['inference_job_id']==job['id'] and r['output']!=job['output']]]:
            ap=guard.attestation_path(output);a=guard.read(ap);a['manifest_sha256']=guard.sha(mp)
            if job['kind']!='initial':a.update(weights_identity=job['weights_identity'],source_model_equivalence=evidence)
            write(ap,a)
    plan=write(tmp_path/'plan.json',guard.prepare(guard.proof(ready),guard.proof(protocol),'fixture'))
    if equal:
        assert guard.inspect(guard.proof(plan),state)['ready']
        algorithm.write_text('# changed algorithm\n')
        with pytest.raises(ValueError,match='SHA mismatch'):guard.inspect(guard.proof(plan),state)
    else:
        with pytest.raises(ValueError,match='true complete-state'):guard.inspect(guard.proof(plan),state)


def test_phase_first_read_cannot_be_reset_by_a_new_output_directory(tmp_path,monkeypatch):
    pp,state,_,_,_=prepare_fixture(tmp_path);a=authority(pp,tmp_path/'authority.json')
    script=Path(__file__).resolve().parents[1]/'scripts/analyze_supplemental_phase.py';sp=importlib.util.spec_from_file_location('supplemental_once',script);cli=importlib.util.module_from_spec(sp);sp.loader.exec_module(cli)
    from replay_study import nll_v2_analysis as nll
    monkeypatch.setattr(nll,'collect_scores',lambda *a,**k:{'complete':True,'completed_conditions':2});monkeypatch.setattr(nll,'write_bundle',lambda *a:None)
    cli.analyze(guard.proof(pp),guard.proof(a),state,tmp_path/'first',no_plots=True)
    with pytest.raises(FileExistsError,match='cannot be reset'):cli.analyze(guard.proof(pp),guard.proof(a),state,tmp_path/'second',no_plots=True)
    assert not (tmp_path/'second').exists()


@pytest.mark.parametrize('change',[None,'data_identity','missing_source','registry_bytes'])
def test_original_scientific_registry_and_model_data_are_independently_bound(tmp_path,change):
    ready,protocol,_,mp,qp,m=fixture(tmp_path)
    identity={'data_family':'main_v3','model_tag':'pythia-410m'}
    write(m['phase_spec'],{'science_frozen':True,'phases':[identity]});m['phase_spec_sha256']=guard.sha(m['phase_spec'])
    original={'path':m['phases'][0]['registry'],'sha256':m['phases'][0]['registry_sha256']}
    for r in m['records']:r['input_sha256'][original['path']]=original['sha256']
    p=guard.read(protocol);p['contracts']['fixture'].update(source_registry=original,scientific_identity=identity,source_proof_files={original['path']:original['sha256']})
    if change=='data_identity':p['contracts']['fixture']['scientific_identity']={**identity,'data_family':'extended_v1'}
    elif change=='missing_source':m['records'][-1]['input_sha256'].pop(original['path'])
    elif change=='registry_bytes':Path(original['path']).write_text('[]')
    write(protocol,p);write(mp,m);q=guard.read(qp)
    for d in q:d['argv'][d['argv'].index('--manifest-sha256')+1]=guard.sha(mp)
    write(qp,q);r=guard.read(ready);r.update(manifest=guard.proof(mp),queue=guard.proof(qp),phase_spec=guard.proof(m['phase_spec']));write(ready,r)
    if change is None:assert guard.prepare(guard.proof(ready),guard.proof(protocol),'fixture')['counts']['logical_endpoints']==2
    else:
        with pytest.raises(ValueError):guard.prepare(guard.proof(ready),guard.proof(protocol),'fixture')


def test_real_collector_reads_only_synthetic_scores_after_immutable_whole_phase_receipt(tmp_path):
    pp,state,mp,_,m=prepare_fixture(tmp_path)
    # Recreate hand-computed score JSON from the existing synthetic fixture in
    # a separate directory; no actual experiment output is opened.
    p=Path(__file__).with_name('test_nll_v2_analysis.py');sp=importlib.util.spec_from_file_location('supp_real_collector_fixture',p);f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f)
    _,_,examples=f.fixture(tmp_path/'examples',role='supplement_B')
    for job,example in zip(m['jobs'],examples['jobs']):
        value=guard.read(example['output']);value['checkpoint']=job.get('checkpoint');out=write(job['output'],value)
        ap=guard.attestation_path(out);att=guard.read(ap);att['output_sha256'][str(out)]=guard.sha(out);write(ap,att)
    script=Path(__file__).resolve().parents[1]/'scripts/analyze_supplemental_phase.py';sp=importlib.util.spec_from_file_location('supplemental_end_to_end',script);cli=importlib.util.module_from_spec(sp);sp.loader.exec_module(cli)
    a=authority(pp,tmp_path/'authority.json');output=tmp_path/'analysis'
    result=cli.analyze(guard.proof(pp),guard.proof(a),state,output,no_plots=True)
    assert result['complete'] and (output/'final/endpoints.csv').is_file()
    assert guard.read(output/'COMPLETE.json')['GPU_inference_started'] is False
    assert Path(guard.read(pp)['first_read_receipt_path']).is_file()
