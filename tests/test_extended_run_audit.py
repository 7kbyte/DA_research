import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/audit_extended_runs.py'
spec=importlib.util.spec_from_file_location('extended_run_audit',SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def trace(ratio,horizon=70_001):
    # 70,001 is deliberately indivisible by both the chunk and replay budgets.
    cfg={'target_prediction_tokens':horizon,'replay_ratio':ratio}
    steps=[];prior_target=prior_replay=0
    for number in range(1,-(-horizon//32704)+1):
        target=min(number*32704,horizon)
        replay=int(target*ratio)
        steps.append({'step':number,'step_counts':{'target_prediction_tokens':target-prior_target,
            'replay_prediction_tokens':replay-prior_replay},'counters':{'target_prediction_tokens':target,
            'replay_prediction_tokens':replay},'target_ce':2.,'replay_loss':0.,'grad_norm':1.})
        prior_target,prior_replay=target,replay
    return cfg,steps


@pytest.mark.parametrize('ratio',[.01,.05,.1])
def test_audit_accepts_partial_final_batch_and_exact_fractional_replay(ratio):
    cfg,steps=trace(ratio)
    result=module.audit_steps(steps,cfg)
    assert result['violations']==[]
    assert result['target_prediction_tokens']==70_001


def test_audit_rejects_consistent_counters_with_biased_per_step_replay_rounding():
    cfg,steps=trace(.01,1_000_001)
    # A per-step rounded replay policy can keep its own counters consistent.
    cumulative=0
    for step in steps:
        step['step_counts']['replay_prediction_tokens']=round(step['step_counts']['target_prediction_tokens']*.01)
        cumulative+=step['step_counts']['replay_prediction_tokens']
        step['counters']['replay_prediction_tokens']=cumulative
    assert any(v['check']=='exact_replay_prefix_floor' for v in module.audit_steps(steps,cfg)['violations'])


def validation_fixture():
    cfg={'target_prediction_tokens':25_000_000,'replay_ratio':.05,'eval_every_steps':153}
    trajectory=[]
    for i,step in enumerate([153,306,459,612,765]):
        target=min(step*32704,25_000_000)
        trajectory.append({'step':step,'target_prediction_tokens':target,'replay_prediction_tokens':target//20,
            'metrics':{name:{'nll':2.+(i-2)**2*.1 if name=='target' else 3.-i*.1,
                'kl':.1,'complete_corpus':True,'precision':'bf16','model_weight_dtype':'torch.float32'}
                for name in ['target','c4','wikitext']}})
    return cfg,{'validation_trajectory':trajectory,'best_target_validation':{'step':459,'nll':2.}}


def test_validation_audit_accepts_25m_fresh_horizon_and_target_only_selection():
    cfg,summary=validation_fixture()
    assert module.audit_validation(summary,cfg)==[]
    summary['best_target_validation']={'step':765,'nll':2.4}
    assert 'best_selected_by_target_validation_only' in module.audit_validation(summary,cfg)


def test_validation_audit_rejects_test_corpus_and_truncated_50m_schedule():
    cfg,summary=validation_fixture()
    summary['validation_trajectory'][0]['metrics']['test']=copy.deepcopy(summary['validation_trajectory'][0]['metrics']['target'])
    assert 'validation_corpora_only' in module.audit_validation(summary,cfg)
    cfg['target_prediction_tokens']=50_000_000
    assert 'complete_fixed_exposure_validation_trajectory' in module.audit_validation(summary,cfg)
