"""Synthetic only: no real phase state, scored outputs or checkpoints are read."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import describe_remaining_evaluation_budget_v6 as b


def fixture():
    records=[]
    for group in ['B0','B1','B2','B3','B4','D','M','C']:
        for seed in [11,22,33]:
            for endpoint in ['final','best']:
                run=f'{group}_{seed}';canonical=f'{"B4" if group=="D" else group}_{seed}_{endpoint}'
                records.append(dict(run_id=run,kind=endpoint,checkpoint_sha256=run+'_'+endpoint,inference_job_id=canonical))
    comparisons=[]
    for endpoint in ['final','best']:
        for seed in [11,22,33]:
            comparisons.append(dict(endpoint=endpoint,seed=seed,bitwise_model_state_equal=True,left_run_id=f'B4_{seed}',right_run_id=f'D_{seed}',left_checkpoint_sha256=f'B4_{seed}_{endpoint}',right_checkpoint_sha256=f'D_{seed}_{endpoint}'))
    adapted=sorted({r['inference_job_id'] for r in records})
    nll=dict(workload=dict(logical_checkpoint_endpoints=48,initial_baselines=1,adapted_inference_jobs=42,total_queue_jobs=43,avoided_duplicate_inferences=6),jobs=[dict(id='initial',kind='initial')]+[dict(id=i,kind='adapted') for i in adapted],records=records)
    final=[r for r in records if r['kind']=='final']
    behavior=dict(workload=dict(logical_checkpoint_endpoints=24,initial_baselines=0,adapted_inference_jobs=21,total_queue_jobs=21,avoided_duplicate_inferences=3),records=final,jobs=[dict(id='behavior_'+i,kind='adapted') for i in sorted({r['inference_job_id'] for r in final})])
    return nll,behavior,comparisons


def test_scope43plus6_not49_real_and_behavior21():
    nll,behavior,proof=fixture();out=b.validate_scope(nll,behavior,proof)
    assert out['actual_completed_NLL_forwards_required']==43
    assert out['actual_completed_NLL_forwards_required']+out['proven_NLL_avoided_forwards']==49
    assert out['future_C410_behavior_forwards']==21


def test_changed_checkpoint_or_false_pair_rejected():
    nll,behavior,proof=fixture();proof[0]['bitwise_model_state_equal']=False
    with pytest.raises(ValueError):b.validate_scope(nll,behavior,proof)
    nll,behavior,proof=fixture();proof[0]['left_checkpoint_sha256']='wrong'
    with pytest.raises(ValueError):b.validate_scope(nll,behavior,proof)


def test_partial_or_stopped_gate_never_calls_output_proof():
    jobs=[dict(id=str(i)) for i in range(43)]
    state=dict(jobs={j['id']:dict(status='completed',returncode=0,definition_sha256=b.digest_job(j)) for j in jobs})
    state['jobs']['42']['status']='running'
    calls=[]
    with pytest.raises(b.WaitForGate):b.completion_rows(jobs,{},jobs,state,lambda *x:calls.append(x))
    assert calls==[]
    state['jobs']['42']['status']='completed';state['jobs']['0']['stop_requested_at']='x'
    with pytest.raises(ValueError):b.completion_rows(jobs,{},jobs,state,lambda *x:calls.append(x))
    assert calls==[]


def test_generic_completion_accepts_initial_without_actual_run_id(tmp_path):
    output=tmp_path/'initial.json';att=b.attestation_path(output)
    att.write_text(json.dumps(dict(inference_subprocess_wall_seconds=4.,verification_wall_seconds=1.)))
    job=dict(id='initial');binding=dict(entry=dict(kind='initial',output=str(output)),records=[])
    state=dict(jobs={'initial':dict(status='completed',returncode=0,definition_sha256=b.digest_job(job),elapsed_seconds=6.,started_at_utc='a',ended_at_utc='b')})
    rows=b.completion_rows([job],{'initial':binding},[job],state,lambda *args:{str(att):'fake-attestation-sha'})
    assert rows[0]['kind']=='initial' and rows[0]['checkpoint_sha256'] is None


def test_future_scope_no_phantom6_no_other_aliases_and_rates_kept():
    _,behavior,_=fixture()
    primary=dict(jobs=[dict(id='main'+str(i),kind='adapted') for i in range(66)])
    A=dict(jobs=[dict(id='A'+str(i),kind='adapted') for i in range(15)]+[dict(id='reuse'+str(i),kind='reuse',inference_origin={'job_id':'main'+str(i)}) for i in range(3)])
    rows=[]
    def add(identity,count,rate=10.,material=0):rows.append(dict(id=identity,remaining_forward_jobs=count,materialization_jobs=material,per_job_whole_slot_seconds_proxy=rate,sum_slot_seconds_proxy=(count+material)*rate))
    add('main_behavior_adapted',66)
    for tag,n in [('410m',24),('1b',24),('2p8b',15)]:
        for endpoint in ['final','best']:add('C_pythia_'+tag+'_nll_'+endpoint,n)
        add('C_pythia_'+tag+'_nll_initial',1)
        add('C_pythia_'+tag+'_behavior_final',n)
    add('A_selected_behavior',15);add('A_selected_behavior_materialization',0,1.,3)
    v5=dict(sensitivity_scenarios=[dict(name=name,whole_wall_sensitivity_factor=factor,per_phase=copy.deepcopy(rows),handoff_preflight_allowance_seconds_unmeasured=300.) for name,factor in [('low',1.),('central',1.25),('high',1.5)]])
    out=b.future_cases(v5,primary,A,behavior)
    for case in out:
        assert (case['remaining_NLL_forward_jobs'],case['remaining_behavior_forward_jobs'],case['remaining_forward_jobs'],case['remaining_materialization_jobs'])==(80,141,221,3)
        assert case['four_slot_dispatch']['job_count']==224
        byid={r['id']:r for r in case['per_phase']}
        assert not any(k.startswith('C_pythia_410m_nll') for k in byid)
        assert byid['C_pythia_410m_behavior_final']['remaining_forward_jobs']==21
        assert byid['C_pythia_1b_nll_final']['remaining_forward_jobs']==24
        assert byid['C_pythia_1b_behavior_final']['remaining_forward_jobs']==24
        assert all(r['per_job_whole_slot_seconds_proxy']==(1. if r['id']=='A_selected_behavior_materialization' else 10.) for r in byid.values())
