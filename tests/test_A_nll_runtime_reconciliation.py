from pathlib import Path
import datetime as dt
import json
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import reconcile_A_nll_runtime as module


def fixture():
    queue=[{'id':'A_forward','expected_seconds':180},{'id':'A_reuse','expected_seconds':180}]
    entries={'A_forward':{'kind':'adapted'},'A_reuse':{'kind':'reuse'}}
    raw={'remaining_slot_hours':1000/3600,'updated_at_utc':'2026-11-06T10:00:00+00:00',
         'requested_matrix':[{'phase':'other_queued_jobs','job_ids':['A_forward','A_reuse','unrelated'],'remaining_gpu_seconds':380}]}
    return raw,queue,entries


def test_only_actual_A_generic_ids_replaced_and_unrelated_allowance_kept():
    raw,queue,entries=fixture();result=module.numbers(raw,queue,entries,queue,queue,[],set(),40.)
    assert result['raw_generic_A_remaining_slot_seconds_removed']==360
    assert result['remaining_non_A_slot_seconds_kept_from_raw_forecast']==640
    central=result['sensitivity_scenarios'][1]
    assert central['remaining_A_slot_seconds']==70
    assert central['per_remaining_forward_whole_slot_seconds']==50
    assert central['per_remaining_reuse_whole_slot_seconds']==20


def test_actual_completion_zeroes_remaining_cost_and_captures_whole_slot_once():
    raw,queue,entries=fixture();samples=[{'id':'A_forward','kind':'adapted','whole_slot_seconds':44.}]
    result=module.numbers(raw,queue,entries,queue,[queue[1]],samples,{'A_forward'},40.)
    assert result['A_jobs_remaining']==1 and result['A_completed_whole_slot_seconds_sum']==44.
    assert result['sensitivity_scenarios'][1]['remaining_A_slot_seconds']==20.
    assert result['running_jobs_receive_partial_progress_credit'] is False


def test_complete_A_phase_has_no_remaining_charge_even_if_raw_observer_lags():
    raw,queue,entries=fixture();samples=[{'kind':'adapted','whole_slot_seconds':42.},{'kind':'reuse','whole_slot_seconds':11.}]
    result=module.numbers(raw,queue,entries,queue,[],samples,{'A_forward','A_reuse'},40.)
    assert result['A_jobs_remaining']==0
    assert all(c['remaining_A_slot_seconds']==0 and c['A_isolated_dispatch_tail_seconds']==0 for c in result['sensitivity_scenarios'])


def test_queue_append_lag_is_explicit_and_new_work_not_silently_dropped():
    raw,queue,entries=fixture();raw['requested_matrix'][0].update(job_ids=['unrelated'],remaining_gpu_seconds=20)
    result=module.numbers(raw,queue,entries,queue,queue,[],set(),40.)
    assert result['raw_generic_A_remaining_slot_seconds_removed']==0
    assert result['remaining_non_A_slot_seconds_kept_from_raw_forecast']==1000
    assert result['sensitivity_scenarios'][1]['remaining_A_slot_seconds']==70


def test_at_least_three_matched_samples_switches_forward_reference_to_median():
    raw,queue,entries=fixture();samples=[{'kind':'adapted','whole_slot_seconds':t} for t in [41.,45.,100.]]
    result=module.numbers(raw,queue,entries,queue,queue,samples,set(),40.)
    assert result['forward_reference_seconds']==45.
    assert result['matched_completed_timing_sample_counts']['adapted']==3


def test_inconsistent_generic_subtotal_is_rejected():
    raw,queue,entries=fixture();raw['requested_matrix'][0]['remaining_gpu_seconds']=100
    with pytest.raises(ValueError,match='subtotal is inconsistent'):module.numbers(raw,queue,entries,queue,queue,[],set(),40.)


def test_after_append_original_initial_still_verified_in_prior_closure(tmp_path,monkeypatch):
    def write(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return path
    queue=[{'id':'A'+str(i),'expected_seconds':180,'argv':['opaque_worker'],'depends_on':['baseline']} for i in range(126)]
    entries=[{'id':j['id'],'kind':'adapted' if i<108 else 'reuse'} for i,j in enumerate(queue)]
    qp=write('queue.json',queue);mp=write('manifest.json',{'jobs':entries})
    cp=write('catalog.json',{'plans':[{'id':'supplement_A_nll','prior_plans':[]}]})
    advice={'queue':module.small_proof(qp),'manifest':module.small_proof(mp),'catalog':module.small_proof(cp),
        'new_kind_counts':{'adapted':108,'reuse':18},'new_initials':0,'timing_reference':{'main_final_median_whole_slot_seconds':40.}}
    ap=write('advice.json',advice);master=write('master.json',[{'id':'baseline','argv':['baseline']},*queue])
    state=write('state.json',{'jobs':{j['id']:{'status':'pending'} for j in queue}})
    raw={'remaining_slot_hours':40000/3600,'updated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'status':'monitoring','failed_or_blocked_jobs':[],'observer_source_sha256':module.EXPECTED_RAW_OBSERVER_SOURCE_SHA256,
        'master_queue_jobs':127,'requested_matrix':[{'phase':'other_queued_jobs','job_ids':[j['id'] for j in queue],'remaining_gpu_seconds':22680}]}
    rp=write('raw.json',raw)
    def verify_full_A_source(incoming,manifest,existing,*args):
        assert [j['id'] for j in existing]==['baseline']
        return {'entries':{j['id']:j for j in entries},'completed_prior_integrity':[{'id':'baseline'}]}
    monkeypatch.setattr(module,'count_jobs',verify_full_A_source)
    result=module.reconcile(ap,module.small_proof(ap)['sha256'],rp,state,master)
    assert result['A_jobs_present_in_master']==126 and result['A_jobs_remaining']==126
    assert result['raw_generic_A_remaining_slot_seconds_removed']==22680
