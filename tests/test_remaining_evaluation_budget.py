"""Runtime accounting uses whole-slot receipts and never deserializes scores."""
import importlib.util
import json
from pathlib import Path
import pytest

source=Path(__file__).resolve().parents[1]/'scripts/describe_remaining_evaluation_budget.py'
sp=importlib.util.spec_from_file_location('remaining_eval_budget',source);module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module)


def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n');return path


def measured_fixture(tmp_path):
    output=tmp_path/'private_score.json';output.write_bytes(b'PRIVATE_SCORE_THIS_IS_NOT_JSON')
    job={'id':'measured','kind':'initial','baseline_group':'g','output':str(output)}
    manifest=write(tmp_path/'manifest.json',{'jobs':[job],'records':[],'common_input_sha256':{},'baseline_groups':{'g':{'model_tag':'pythia-410m','inference_policy':{}}}})
    job['input_sha256']={};write(manifest,{'jobs':[job],'records':[],'common_input_sha256':{},'baseline_groups':{'g':{'model_tag':'pythia-410m','inference_policy':{}}}})
    att=write(tmp_path/'private_score_execution.json',{'job_id':'measured','manifest_sha256':module.sha(manifest),'checkpoint_sha256':None,
        'verified_frozen_inputs':True,'output_sha256':{str(output):'unchanged_sealed_score'},'verification_wall_seconds':2.,'inference_subprocess_wall_seconds':5.})
    definition={'id':'measured','argv':['frozen_worker']}
    state={'jobs':{'measured':{'status':'completed','returncode':0,'missing_outputs':[],'definition_sha256':module.digest_job(definition),
        'elapsed_seconds':10.,'started_at_utc':'2026-11-06T00:00:00Z','ended_at_utc':'2026-11-06T00:00:10Z'}}}
    return manifest,state,{'measured':definition},att,output


def test_whole_slot_timing_does_not_read_or_add_private_score_output(tmp_path,monkeypatch):
    manifest,state,definitions,att,output=measured_fixture(tmp_path);original=module.read;seen=[]
    def guarded(path):
        seen.append(Path(path));assert Path(path)!=output;return original(path)
    monkeypatch.setattr(module,'read',guarded)
    _,rows=module.measure(manifest,state,definitions)
    assert rows[0]['whole_slot_wall_seconds']==10 and rows[0]['unclassified_scheduler_wall_seconds']==3.
    assert output not in seen and 'PRIVATE_SCORE' not in json.dumps(rows)


@pytest.mark.parametrize('bad',['pending','failed','definition','manifest','checkpoint','nested_time'])
def test_measurement_rejects_incomplete_or_unbound_receipt(bad,tmp_path):
    manifest,state,definitions,att,_=measured_fixture(tmp_path);a=module.read(att)
    if bad=='pending':state['jobs']['measured']['status']='running'
    elif bad=='failed':state['jobs']['measured']['returncode']=1
    elif bad=='definition':state['jobs']['measured']['definition_sha256']='wrong'
    elif bad=='manifest':a['manifest_sha256']='wrong'
    elif bad=='checkpoint':a['checkpoint_sha256']='wrong'
    else:a['inference_subprocess_wall_seconds']=20.
    write(att,a)
    with pytest.raises(ValueError):module.measure(manifest,state,definitions)


def test_scope_removes_completed_primary_and_keeps_future_aliases_unspent():
    rows=module.physical_scope()
    assert sum(r['remaining_forward_jobs'] for r in rows if r['mode']=='nll')==297
    assert sum(r['remaining_forward_jobs'] for r in rows if r['mode']=='behavior')==147
    assert sum(r['remaining_forward_jobs'] for r in rows if r['id'].startswith('main_nll'))==0
    assert next(r for r in rows if r['id']=='A_nll')['already_proven_aliases']==18
    assert sum(r['physical_forward_jobs'] for r in rows if r['mode']=='nll')==364


def test_dispatch_baseline_dependencies_and_imbalance_are_not_double_polling():
    jobs=[{'id':'baseline','seconds':10.,'initial':True},{'id':'a','seconds':20.,'depends_on':['baseline']},
          {'id':'b','seconds':20.,'depends_on':['baseline']},{'id':'other','seconds':40.}]
    v=module.dispatch(jobs)
    assert v['sum_whole_job_slot_seconds_proxy']==90.
    assert v['baseline_first_LPT_wall_seconds_proxy']==40.
    assert v['LPT_imbalance_and_dependency_tail_seconds_proxy']==17.5


def build_fixture(tmp_path,monkeypatch):
    def measured(path,*unused):
        path=Path(path)
        def row(m,k,t):return {'model':m,'kind':k,'whole_slot_wall_seconds':t,'verification_wall_seconds_nested':1.,'input_bytes':1000}
        if path.name.endswith('_nll_manifest.json'):return {},[row('pythia-410m','initial',30.),*[row('pythia-410m','final',40.) for _ in range(33)],*[row('pythia-410m','best',30.) for _ in range(33)]]
        if path.name.startswith('planned_model'):return {},[row('pythia-1b','initial',90.),row('pythia-2.8b','initial',220.)]
        return {},[row('pythia-410m','initial',100.)]
    monkeypatch.setattr(module,'measure',measured)
    write(tmp_path/'configs/supplemental_v1_queue.json',[])
    write(tmp_path/'results/evaluation_timing_budget_v2.json',{'measured_initials':{m:{'future_final_checkpoint_byte_size_proxy':1000} for m in module.MODELS}})
    groups=[{'model':m,'data_family':family,'prediction_tokens_per_inference':100,'timing_proxy':{'median_seconds_per_prediction_token':rate},'inference_policy':{'batch_size':batch,'max_length':512}}
            for m,family,rate,batch in [('pythia-410m','main_v3',1.,16),('pythia-410m','extended_v1',1.,16),('pythia-1b','extended_v1',2.,16),('pythia-2.8b','extended_v1',3.,8)]]
    write(tmp_path/'results/nll_scope_budget_v1.json',{'groups':groups})
    write(tmp_path/'results/behavioral_budget_preregistered_v2.json',{'scope':{m:{'parameter_count':1,'work_per_pass':{}} for m in module.MODELS}})
    write(tmp_path/'configs/nll_scope_execution_freeze_v1.json',{'logical_endpoints':384})
    write(tmp_path/'configs/behavioral_scope_execution_freeze_v2.json',{'scope':{'primary':{'logical_endpoints':72},'supplement_A':{'maximum_endpoints':18},'extended':{'logical_endpoints_by_model':dict(zip(module.MODELS,(24,24,15)))}}})
    runtime={'updated_at_utc':'2026-11-06T12:00:00Z','four_gpu_remaining_wall_hours_with_10pct':10.,'master_queue_jobs':378,
        'master_status_counts':{},'remaining_primary_nll_slot_seconds':0}
    return module.build(tmp_path,{'jobs':{}},runtime,[])


def test_budget_only_spends_proven_aliases_and_proof_success_is_separate_scenario(tmp_path,monkeypatch):
    v=build_fixture(tmp_path,monkeypatch)
    assert v['totals']['remaining_forward_jobs_max']==444
    assert v['time_window']['midnight_window_seconds_before_delivery_tail']==10592.
    assert v['time_window']['forecast_training_finish_utc_with_existing_10pct']=='2026-11-06T22:00:00+00:00'
    for c in v['sensitivity_scenarios']:
        assert c['four_slot_dispatch']['job_count']==462
        assert c['future_C_all_D_B4_equal_dispatch']['job_count']==444
        assert sum(s['max_avoided_forwards'] for s in c['future_C_all_D_B4_equal_savings'])==18
        assert c['remaining_forward_jobs']==444
        assert all(not r['id'].startswith('main_nll') for r in c['per_phase'])
    assert v['test_or_behavior_scores_parsed'] is False
