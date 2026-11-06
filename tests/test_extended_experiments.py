import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/make_extended_experiments.py'
spec=importlib.util.spec_from_file_location('extended_make',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@pytest.mark.parametrize('model,count,budget',[('pythia-410m',24,100_000_000),('pythia-1b',24,100_000_000),('pythia-2.8b',15,50_000_000)])
def test_extended_grid_uses_matched_model_data_and_exposure_schedule(tmp_path,model,count,budget):
    for phase,size in [('lr',3),('lambda',16),('formal',count)]:
        configs,registry,queue=module.build_extended(model,phase,tmp_path,tmp_path/'scratch',3e-5,.05)
        assert len(configs)==len(registry)==len(queue)==size
        for cfg in configs.values():
            assert cfg['microbatch_size']*cfg['gradient_accumulation_steps']==64
            assert cfg['deterministic_algorithms'] and cfg['attention_backend']=='auto'
            assert '/data_extended/' in cfg['target_path']
            assert all('/data_extended/' in p for p in cfg['validation_paths'].values())
            if cfg['replay_path']:assert f'/extended/{model}/banks/' in cfg['replay_path']
            if phase=='formal':
                assert cfg['target_prediction_tokens']==budget
                assert cfg['eval_every_steps']*64*511==5_003_712
            else:assert cfg['target_prediction_tokens']==2_000_000 and cfg['seed']==101
        assert all(not r['active'] for r in registry)
    assert not list(tmp_path.iterdir())

def test_extended_lambda_requires_its_model_calibration_choice(tmp_path):
    with pytest.raises(ValueError):module.build_extended('pythia-1b','lambda',tmp_path,tmp_path/'scratch')
    with pytest.raises(ValueError):module.build_extended('pythia-2.8b','lr',tmp_path,tmp_path/'scratch',microbatch=7)

def test_batch_change_keeps_exposure_schedule_and_preserves_prior_configs(tmp_path):
    left,old_registry,_=module.build_extended('pythia-1b','formal',tmp_path,tmp_path/'scratch',3e-5,.05,microbatch=16)
    right,new_registry,_=module.build_extended('pythia-1b','formal',tmp_path,tmp_path/'scratch',3e-5,.05,microbatch=32)
    assert not {r['id'] for r in old_registry}&{r['id'] for r in new_registry}
    assert all(c['gradient_accumulation_steps']==2 and c['microbatch_size']==32 for c in right.values())
    for a,b in zip(left.values(),right.values()):
        for field in ['target_prediction_tokens','seed','learning_rate','eval_every_steps','validation_paths']:
            assert a[field]==b[field]
        assert b['deadline_utc']=='2026-11-07T00:00:00Z'

def test_resource_fallback_keeps_complete_cohort_and_uses_a_fresh_full_horizon(tmp_path):
    from replay_study.train import learning_rate_at_exposure
    old,rows50,_=module.build_extended('pythia-2.8b','formal',tmp_path,tmp_path/'scratch',1e-5,.5,microbatch=8)
    new,rows25,_=module.build_extended('pythia-2.8b','formal',tmp_path,tmp_path/'scratch',1e-5,.5,microbatch=8,
                                      target_prediction_tokens=25_000_000)
    assert len(new)==15 and {(r['group'],r['seed']) for r in rows25}=={(g,s) for g in ['B0','B1','B2','B4','M'] for s in [11,22,33]}
    assert not {r['id'] for r in rows25}&{r['id'] for r in rows50}
    assert all('_h25m_formal_' in r['id'] for r in rows25)
    for cfg in new.values():
        assert cfg['target_prediction_tokens']==25_000_000 and cfg['resume_from'] is None and cfg['max_steps'] is None
        assert cfg['eval_every_steps']==153 and cfg['microbatch_size']*cfg['gradient_accumulation_steps']==64
    a=module.TrainConfig.from_dict(next(iter(old.values())));b=module.TrainConfig.from_dict(next(iter(new.values())))
    assert learning_rate_at_exposure(a,1_000_000)!=learning_rate_at_exposure(b,1_000_000)
    assert module.MODELS['pythia-2.8b']['target_tokens']==50_000_000
    with pytest.raises(ValueError,match='formal target'):
        module.build_extended('pythia-2.8b','lr',tmp_path,tmp_path/'scratch',target_prediction_tokens=25_000_000)
