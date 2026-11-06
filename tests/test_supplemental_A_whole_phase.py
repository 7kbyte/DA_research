"""Full126 synthetic A gate; no real experiment artifacts or score parsing."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from replay_study import supplemental_analysis_guard as guard


def module(filename,name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(guard.encoded(value));return path


def A_fixture(tmp_path):
    tests=module('test_selected_policy_analysis.py','A_selection_source');registry,selection,rows,_=tests.grid(tmp_path/'grid')
    for row in rows:row.update(phase='supplement_A_lambda_sweep',active=True)
    write(registry,rows);choice=guard.read(selection);choice['sources_sha256'][str(registry)]=guard.sha(registry);write(selection,choice)
    f=module('test_nll_v2_analysis.py','A_nll_source');_,_,m=f.fixture(tmp_path/'source',role='supplement_A')
    template=m['records'][0];initial=m['jobs'][0];initial.update(input_sha256={},argv=['synthetic','initial'])
    group=m['baseline_groups']['base'];group['base_model_assets_sha256']={};group['inference_policy'].update(batch_size=16,max_length=512)
    spec=write(tmp_path/'spec.json',{'science_frozen':True,'phases':[{}]})
    m.update(phase_spec=str(spec),phase_spec_sha256=guard.sha(spec),common_input_sha256={},scorer_input_sha256={})
    m['phases'][0].update(rows=126,registry=str(registry),registry_sha256=guard.sha(registry));m['records']=[];m['jobs']=[initial]
    for row in rows:
        cp=tmp_path/'weights'/(row['id']+'.pt');cp.parent.mkdir(exist_ok=True);cp.write_bytes(row['id'].encode());digest=guard.sha(cp)
        record={**template,'id':row['id']+'__final','run_id':row['id'],'actual_run_id':row['actual_run_id'],'group':row['group'],
            'weight':row['weight'],'learning_rate':row['learning_rate'],'training_seed':row['seed'],'checkpoint':str(cp),
            'checkpoint_sha256':digest,'weights_identity':'checkpoint:'+digest,'inference_job_id':row['id'],
            'input_sha256':{str(cp):digest,str(registry):guard.sha(registry)},'output':str(tmp_path/'scores'/(row['id']+'.json'))}
        m['records'].append(record);m['jobs'].append(dict(id=row['id'],kind='adapted',baseline_group='base',checkpoint=str(cp),checkpoint_sha256=digest,
            weights_identity=record['weights_identity'],output=record['output'],input_sha256=record['input_sha256'],record_ids=[record['id']],
            depends_on=['initial'],argv=['synthetic',row['id']]))
    m['workload']=dict(logical_checkpoint_endpoints=126,total_queue_jobs=127,total_inference_jobs=127,reuse_materialization_jobs=0)
    mp=write(tmp_path/'manifest.json',m);queue=[]
    for job in m['jobs']:
        d=dict(id=job['id'],argv=[sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/run_frozen_evaluation.py'),
            '--manifest',str(mp),'--manifest-sha256',guard.sha(mp),'--job-id',job['id'],'--device','cuda:0'],outputs=[])
        if job.get('depends_on'):d['depends_on']=job['depends_on']
        queue.append(d)
        out=Path(job['output']);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b'INVALID_PRIVATE_RESULT_NEVER_JSON_PARSE')
        write(guard.attestation_path(out),dict(manifest_sha256=guard.sha(mp),job_id=job['id'],baseline_group='base',verified_frozen_inputs=True,
            checkpoint_sha256=job.get('checkpoint_sha256'),weights_identity=job.get('weights_identity'),actual_argv=job['argv'],
            inference_alias=False,inference_subprocess_wall_seconds=1.,output_sha256={str(out):guard.sha(out)}))
    qp=write(tmp_path/'queue.json',queue);phase=m['phases'][0]
    contract={'family':'A','mode':'nll','phase':{k:phase[k] for k in ('name','role','model_tag','data_family','expected_phase','rows')},
        'checkpoints':['final'],'logical_endpoints':126,'inference_policy_expect':group['inference_policy'],
        'source_registry':guard.proof(registry),'scientific_identity':{},'source_proof_files':{str(registry):guard.sha(registry)}}
    protocol=write(tmp_path/'protocol.json',{'schema_version':1,'bootstrap_replicates':5000,'bootstrap_seed':20261106,'confidence':.95,
        'allow_partial_for_final_conclusions':False,'first_read_root':str(tmp_path/'first_reads'),'analysis_source_sha256':{},'prior_sources':[],
        'contracts':{'supplement_A_nll':contract},'A_registry':guard.proof(registry),'A_selection':{'path':str(selection),'sha256':None}})
    ready=write(tmp_path/'READY.json',dict(status='ready',ready=True,id='supplement_A_nll',manifest=guard.proof(mp),queue=guard.proof(qp),
        phase_spec=guard.proof(spec),logical_endpoints=126,workload=m['workload']))
    state=write(tmp_path/'state.json',{'jobs':{d['id']:{'status':'completed','returncode':0,'missing_outputs':[],
        'definition_sha256':hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()} for d in queue}})
    return ready,protocol,state,selection,m


def test_full_A126_gate_freezes_actual_complete_validation_selection_and_never_parses_scores(tmp_path,monkeypatch):
    ready,protocol,state,selection,m=A_fixture(tmp_path)
    plan=guard.prepare(guard.proof(ready),guard.proof(protocol),'supplement_A_nll')
    assert plan['selection']['selection']['sha256']==guard.sha(selection) and len(plan['selection']['selected_ids'])==18
    pp=write(tmp_path/'plan.json',plan);private={j['output'] for j in m['jobs']};read=guard.read
    def safe_read(path):
        assert str(path) not in private
        return read(path)
    monkeypatch.setattr(guard,'read',safe_read);result=guard.inspect(guard.proof(pp),state)
    assert result['ready'] and result['counts']['logical_endpoints']==126 and result['closure_job_count']==127 and not result['score_json_parsed']
    s=guard.read(state);s['jobs'][m['jobs'][-1]['id']]['status']='running';write(state,s)
    assert not guard.inspect(guard.proof(pp),state)['ready']
    assert not Path(plan['first_read_receipt_path']).exists()


def test_A_selection_cannot_change_after_preparation_even_without_any_score_access(tmp_path):
    ready,protocol,_,selection,_=A_fixture(tmp_path);plan=write(tmp_path/'plan.json',guard.prepare(guard.proof(ready),guard.proof(protocol),'supplement_A_nll'))
    chosen=guard.read(selection);chosen['selections'][0]['weight']=10.;write(selection,chosen)
    with pytest.raises(ValueError,match='SHA mismatch'):guard.load_plan(guard.proof(plan))
