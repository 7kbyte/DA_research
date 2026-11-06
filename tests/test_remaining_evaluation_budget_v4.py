from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import describe_remaining_evaluation_budget_v4 as module
from describe_remaining_evaluation_budget import physical_scope
from stat_bound_digest_cache import StatBoundHashes


def fixture():
    main={'jobs':[{'id':'main'+str(i),'kind':'adapted','output':'old'+str(i),'baseline_group':'g','weights_identity':'state'+str(i)} for i in range(66)]}
    selected={'records':[{} for _ in range(18)],'jobs':[{'id':'selected'+str(i),'kind':'adapted'} for i in range(15)]}
    for i in range(3):selected['jobs'].append({'id':'reuse'+str(i),'kind':'reuse','baseline_group':'g','weights_identity':'state'+str(i),
        'inference_origin':{'job_id':'main'+str(i),'manifest_sha256':'primarySHA','output':'old'+str(i)}})
    v3={'timing_transfer':{'NLL':{'pythia-410m:extended_v1':{'whole_wall_transfer_ratio_assumption':.9}}},'sensitivity_scenarios':[]}
    for name,factor,alias in [('low',1.,10.),('central',1.25,20.),('high',1.5,40.)]:
        rows=[]
        for row in physical_scope():
            if not row['remaining_forward_jobs']:continue
            one=(100. if row['mode']=='behavior' else 20. if row['endpoint']=='initial' else 30. if row['endpoint']=='best' else 40.)*factor
            rows.append({'id':row['id'],'remaining_forward_jobs':row['remaining_forward_jobs'],'per_job_whole_slot_seconds_proxy':one})
        v3['sensitivity_scenarios'].append({'name':name,'whole_wall_sensitivity_factor':factor,'A_alias_materialization_wall_allowance_per_job_seconds':alias,'per_phase':rows})
    completion={'whole_slot_statistics':{'adapted':{'min':39.,'median':40.,'max':41.},'reuse':{'min':10.,'median':10.,'max':20.}},
                'actual_whole_slot_seconds_sum':4531.}
    return v3,completion,selected,main


def test_completed_A_forward_and_materialization_removed_once_and_actual_selected_aliases_kept():
    v3,completion,selected,main=fixture();rows=module.future_cases(v3,completion,selected,main,'primarySHA')
    for row in rows:
        assert row['four_slot_dispatch']['job_count']==336
        assert row['remaining_forward_jobs']==333 and row['remaining_materialization_jobs']==3
        assert not any(p['id']=='A_nll' for p in row['per_phase'])
        assert sum(p['remaining_forward_jobs'] for p in row['per_phase'] if p['id'].endswith('_nll_initial'))==3
        assert row['counterfactual_slot_work_if_A_not_done_early']==row['four_slot_dispatch']['sum_whole_job_slot_seconds_proxy']+4531.
        assert row['total_compute_reduction_due_to_early_A_seconds']==0.


def test_A_NLL_evidence_does_not_rewrite_behavior_or_large_model_extrapolation():
    v3,completion,selected,main=fixture();rows=module.future_cases(v3,completion,selected,main,'primarySHA')
    for row,old in zip(rows,v3['sensitivity_scenarios']):
        actual={p['id']:p for p in row['per_phase']};previous={p['id']:p for p in old['per_phase']}
        assert actual['main_behavior_adapted']['per_job_whole_slot_seconds_proxy']==previous['main_behavior_adapted']['per_job_whole_slot_seconds_proxy']
        assert actual['C_pythia_1b_nll_final']['per_job_whole_slot_seconds_proxy']==previous['C_pythia_1b_nll_final']['per_job_whole_slot_seconds_proxy']
        assert actual['C_pythia_2p8b_nll_final']['per_job_whole_slot_seconds_proxy']==previous['C_pythia_2p8b_nll_final']['per_job_whole_slot_seconds_proxy']
        assert actual['C_pythia_410m_nll_final']['per_job_whole_slot_seconds_proxy']==row['future_410m_NLL_final_whole_slot_seconds_proxy']*.9
        assert row['C_future_D_B4_equality_discount_spent'] is False


def test_selected_alias_requires_same_canonical_model_state_and_policy():
    _,_,selected,main=fixture();selected['jobs'][-1]['weights_identity']='different'
    with pytest.raises(ValueError,match='canonical main inference'):module.A_behavior_scope(selected,main,'primarySHA')


def test_deadline_reserves_full_30_minutes():
    v3,completion,selected,main=fixture();rows=module.future_cases(v3,completion,selected,main,'primarySHA')
    for row in rows:
        latest=module.utc(row['latest_feasible_start_utc_proxy'])
        assert abs((module.utc('2026-11-07T02:56:32Z')-latest).total_seconds()-row['wall_seconds_with_dispatch_tail_and_handoff_proxy'])<1e-5


def test_cache_inventory_stat_verifies_prior_complete_read_without_new_checkpoint_io(tmp_path):
    run=tmp_path/'run';cp=run/'checkpoints/latest.pt';cp.parent.mkdir(parents=True);cp.write_bytes(b'opaque model bytes')
    h=StatBoundHashes(directory=tmp_path/'cache');h.digest(cp,authority={'kind':'completed_training_checkpoint'})
    progress={'cache_snapshot':h.snapshot(tmp_path/'snapshot.json'),'counters_this_process':dict(h.counters)}
    value=module.cache_inventory({'prewarm_runs':[{'output_dir':str(run)}]},progress)
    assert value['registered_final_and_best_checkpoint_paths']==2
    assert value['completed_checkpoint_full_read_receipts_stat_verified']==1
    assert value['not_yet_cached_registered_paths']==1
    assert value['this_inventory_checkpoint_full_byte_reads']==0
    assert value['this_inventory_stat_bound_cache_hits']==1
    cp.write_bytes(b'changed')
    with pytest.raises(ValueError,match='stat changed'):module.cache_inventory({'prewarm_runs':[{'output_dir':str(run)}]},progress)
