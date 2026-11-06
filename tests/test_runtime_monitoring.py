import importlib.util
from pathlib import Path

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
def load(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
runtime=load('monitor_runtime_budget');selection=load('watch_lambda_selection')


def test_long_run_rate_requires100_steps_and_excludes_ten_warmup_steps():
    events=[{'event':'train_step','step_seconds':10. if i<10 else 1.,'step_counts':{'target_prediction_tokens':32704}} for i in range(100)]
    assert runtime.steady_sample(events[:99]) is None
    result=runtime.steady_sample(events)
    assert result['sampled_training_seconds']==90
    assert result['sampled_target_prediction_tokens']==90*32704


def test_partial_training_credits_exposure_but_retains_unpaid_validation_and_final_save():
    config={'target_prediction_tokens':70_001,'eval_every_steps':2}
    events=[{'event':'train_step','counters':{'target_prediction_tokens':65408}},
            {'event':'validation','step':2}]
    result=runtime.remaining_formal(config,events,'running',.001,10.,2.,5.,3.)
    assert result['remaining_target_prediction_tokens']==4593
    assert result['remaining_validation_points']==1
    assert abs(result['total_seconds']-(4.593+10+2+5))<1e-9
    assert runtime.remaining_formal(config,events,'completed',.001,10.,2.,5.,3.)['total_seconds']==0


def test_a_selector_waits_for_last_condition_including_parent_aliases():
    rows=[{'id':f'physical{i}','execution_kind':'train'} for i in range(108)]
    rows +=[{'id':f'alias{i}','execution_kind':'alias','alias_of':f'parent{i}'} for i in range(18)]
    physical={'jobs':{f'physical{i}':{'status':'completed','returncode':0} for i in range(108)}}
    parents={'jobs':{f'parent{i}':{'status':'completed','returncode':0} for i in range(18)}}
    assert selection.complete_grid(rows,physical,parents)['ready']
    parents['jobs']['parent17']['status']='pending'
    gate=selection.complete_grid(rows,physical,parents)
    assert not gate['ready'] and gate['missing_actual_runs']==['parent17']


def test_future_extended_evaluation_phase_is_not_silently_counted_as_budgeted_training():
    updates,known=runtime.extended_runtime({'models':{}},{'jobs':{}},{
        'extended_v2_pythia_1b_m32_formal_nll':{'job_ids':['future_nll'],'registry':'not_a_training_registry'}})
    assert updates=={} and known==set()
