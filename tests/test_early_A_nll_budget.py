import copy
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import estimate_early_A_nll as module


def load_handoff_fixture():
    path=Path(__file__).with_name('test_evaluation_handoff.py');spec=importlib.util.spec_from_file_location('handoff_fixture_earlyA',path)
    other=importlib.util.module_from_spec(spec);spec.loader.exec_module(other);return other


def test_fifo_tail_and_dependencies_have_no_second_polling_charge():
    jobs=[{'id':'a','depends_on':['baseline']},{'id':'b','depends_on':['baseline']},{'id':'alias','depends_on':['a']}]
    result=module.fifo_schedule(jobs,{'a':40.,'b':40.,'alias':20.},{'baseline'})
    assert result['sum_slot_seconds']==100.
    assert result['ready_first_FIFO_makespan_seconds']==60.
    assert result['isolated_phase_dispatch_tail_seconds']==35.
    starts={r['id']:r['start'] for r in result['jobs']}
    assert starts['alias']==40.


def test_missing_prior_baseline_blocks_timing_plan():
    with pytest.raises(ValueError,match='dependency blocks'):
        module.fifo_schedule([{'id':'a','depends_on':['missing']}],{'a':40.},set())


def test_existing_baseline_integrity_verified_without_deserializing_private_score(tmp_path,monkeypatch):
    other=load_handoff_fixture();args,plan,jobs,manifest=other.fixture(tmp_path)
    state=module.read(plan['old_master_state']);metadata=copy.deepcopy(manifest);metadata['jobs']=[manifest['jobs'][1]]
    key=(str(Path(jobs[0]['argv'][3]).resolve()),jobs[0]['argv'][5]);authorities={key:manifest}
    original=Path.read_text
    def guarded(path,*a,**kw):
        assert path!=Path(manifest['jobs'][0]['output'])
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',guarded)
    counts=module.count_jobs([jobs[1]],metadata,[jobs[0]],state,authorities)
    assert counts['new_kind_counts']=={'adapted':1}
    assert counts['required_existing_initial_count']==1
    assert counts['completed_prior_integrity'][0]['integrity_verified_without_score_parse']


@pytest.mark.parametrize('bad',['failed','missing_output','definition','same_id'])
def test_invalid_baseline_or_job_overlap_rejected(tmp_path,bad):
    other=load_handoff_fixture();_,plan,jobs,manifest=other.fixture(tmp_path)
    state=module.read(plan['old_master_state']);metadata=copy.deepcopy(manifest);metadata['jobs']=[manifest['jobs'][1]]
    key=(str(Path(jobs[0]['argv'][3]).resolve()),jobs[0]['argv'][5]);authorities={key:manifest};existing=[jobs[0]]
    if bad=='failed':state['jobs']['initial']['returncode']=1
    if bad=='missing_output':Path(manifest['jobs'][0]['output']).unlink()
    if bad=='definition':state['jobs']['initial']['definition_sha256']='0'*64
    if bad=='same_id':
        altered=copy.deepcopy(jobs[1]);altered['expected_seconds']+=1;existing.append(altered)
    with pytest.raises(ValueError):module.count_jobs([jobs[1]],metadata,existing,state,authorities)


def test_moving_A_work_conserves_slot_cost_and_keeps_training_contingency():
    jobs=[{'id':'a','depends_on':['initial']},{'id':'alias','depends_on':['prior']}]
    counts={'entries':{'a':{'kind':'adapted'},'alias':{'kind':'reuse'}},'new_jobs':jobs,'already_in_master':[],
            'completed_prior_integrity':[{'id':'initial'},{'id':'prior'}]}
    v3={'sensitivity_scenarios':[]}
    for name,alias in [('low',10.),('central',20.),('high',40.)]:
        # Old A estimate differs deliberately: it must be replaced before debit.
        v3['sensitivity_scenarios'].append({'name':name,'per_phase':[{'id':'A_nll','sum_slot_seconds_proxy':30.,'remaining_forward_jobs':1,'per_job_whole_slot_seconds_proxy':30.},
            {'id':'other','sum_slot_seconds_proxy':100.,'remaining_forward_jobs':1,'per_job_whole_slot_seconds_proxy':100.}],
            'materialization_only_jobs':1,'A_alias_materialization_wall_allowance_per_job_seconds':alias,
            'four_slot_dispatch':{'sum_whole_job_slot_seconds_proxy':130.+alias},'extra_unmeasured_handoff_preflight_allowance_seconds':300.})
    runtime={'four_gpu_remaining_wall_hours_with_10pct':10.,'updated_at_utc':'2026-11-06T10:00:00+00:00','margin_before_master_2350_stop_minutes':100.}
    values=module.timing_cases(counts,runtime,v3,40.)
    for case in values:
        assert case['total_slot_work_reduction_from_moving_A']==0.
        assert case['final_evaluation_slot_seconds_if_A_completes_early']==100.
        assert case['final_evaluation_slot_seconds_before_A_retimed']==100.+case['additional_slot_seconds']
        assert case['training_plus_early_A_hours_with_10pct_plus_tail']>10.


def test_real_ready_is_required_before_any_metadata_decision(tmp_path):
    with pytest.raises(FileNotFoundError,match='real v5 A-NLL READY'):module.estimate(tmp_path)


def test_reuse_materializes_only_the_exact_complete_canonical_origin(tmp_path):
    other=load_handoff_fixture();_,plan,jobs,prior=other.fixture(tmp_path);state=module.read(plan['old_master_state'])
    canonical=prior['jobs'][1];output=Path(canonical['output']);output.write_bytes(b'NOT JSON PRIVATE NLL')
    other.write(other.module.attestation_path(output),{'manifest_sha256':jobs[1]['argv'][5],'job_id':'endpoint',
        'verified_frozen_inputs':True,'checkpoint_sha256':canonical['checkpoint_sha256'],'actual_argv':canonical['argv'],
        'inference_alias':False,'output_sha256':{str(output):other.module.sha256(output)}})
    state['jobs']['endpoint'].update(status='completed',returncode=0)
    entry={'id':'reuse','kind':'reuse','baseline_group':'same','output':str(output),'argv':None,
        'checkpoint_sha256':canonical['checkpoint_sha256'],'input_sha256':{},'depends_on':['endpoint'],
        'inference_origin':{'job_id':'endpoint','manifest_sha256':jobs[1]['argv'][5],'output':str(output)}}
    alias=tmp_path/'logical_alias.json';metadata={'schema_version':2,'mode':'nll','common_input_sha256':prior['common_input_sha256'],
        'baseline_groups':prior['baseline_groups'],'jobs':[entry],
        'records':[{'id':'logical','inference_job_id':'reuse','output':str(alias),'checkpoint_sha256':canonical['checkpoint_sha256']}]}
    path=other.write(tmp_path/'reuse_manifest.json',metadata);proof=module.small_proof(path)
    job={'id':'reuse','argv':[sys.executable,str(other.module.WORKER),'--manifest',str(path),'--manifest-sha256',proof['sha256'],
         '--job-id','reuse','--device','cuda:0'],'depends_on':['endpoint'],'expected_seconds':180,
         'env':{'CUBLAS_WORKSPACE_CONFIG':':4096:8'},'outputs':[str(alias),str(other.module.attestation_path(alias))]}
    authorities={(str(Path(jobs[0]['argv'][3]).resolve()),jobs[0]['argv'][5]):prior,(str(path),proof['sha256']):metadata}
    result=module.count_jobs([job],metadata,jobs,state,authorities)
    assert result['new_kind_counts']=={'reuse':1}
    assert result['required_existing_adapted_origin_count']==1
    entry['inference_origin']['output']='wrong_canonical_output'
    with pytest.raises(ValueError,match='same completed canonical prior output'):
        module.count_jobs([job],metadata,jobs,state,authorities)
