import importlib.util
from pathlib import Path
import pytest


def load_script(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make=load_script('make_experiments')
select=load_script('select_hyperparameters')


def protocol():
    return {'seed':101,'groups':['B1','B2','B3','B4'],'lambda_grid':[.05,.5,1.,5.],
        'target_prediction_tokens':2000000,'replay_prediction_tokens':100000,'replay_ratio':.05,
        'microbatch_size':16,'gradient_accumulation_steps':4,'shared_learning_rate':1e-5,
        'lambda_selection_rule':'minimum mean; strict ties smaller weight',
        'lambda_group_score':'delta target + .5*(delta c4 + delta wiki)',
        'lambda_aggregate':'mean of four groups'}


def test_formal_grid_has_36_matched_configs_without_generating_files(tmp_path):
    configs,registry,queue=make.build_experiments('formal',tmp_path,tmp_path/'scratch',1e-5,.5)
    assert len(configs)==len(registry)==len(queue)==36
    assert not list(tmp_path.iterdir())
    assert len({r['id'] for r in registry})==36
    for seed in [11,22,33]:
        rows=[r for r in registry if r['seed']==seed]
        assert len(rows)==12
        for group in ['B4','M']:
            assert {r['ratio'] for r in rows if r['group']==group}=={.01,.05,.1}
    for config in configs.values():
        assert config['target_prediction_tokens']==10000000
        assert config['eval_every_steps']==50 and config['save_every_steps']==0
        assert config['precision']=='bf16' and config['evaluate_teacher_kl']
        assert all('validation' in Path(p).name and 'test' not in Path(p).name for p in config['validation_paths'].values())


def fake_grid(tmp_path):
    p=protocol()
    configs,registry,queue=make.build_experiments('lambda',tmp_path,tmp_path/'scratch',1e-5,protocol=p)
    initial={'corpora':{name:{'nll':value,'valid_prediction_tokens':100} for name,value in [('target',2.),('c4',3.),('wikitext',4.)]}}
    summaries={}
    for record in registry:
        weight=record['weight']
        target_delta,old_delta={.05:(-.1,.2),.5:(-.1,0.),1.:(-.08,0.),5.:(.3,-.1)}[weight]
        metrics={name:{'nll':base['nll']+(target_delta if name=='target' else old_delta),
                       'valid_prediction_tokens':100,'complete_corpus':True,'kl':0.01}
                 for name,base in initial['corpora'].items()}
        summaries[record['id']]={'config':configs[record['config_path']], 'status':'completed',
            'counters':{'target_prediction_tokens':2000000,'replay_prediction_tokens':100000},
            'target_stream':{'seed':101,'trace_hash':'matched'},
            'validation_trajectory':[{'target_prediction_tokens':2000000,'metrics':metrics}]}
    state={'jobs':{r['id']:{'status':'completed'} for r in registry}}
    return p,registry,initial,summaries,state


def test_shared_lambda_uses_all_four_group_scores_and_equal_exposure(tmp_path):
    p,registry,initial,summaries,state=fake_grid(tmp_path)
    result=select.select_shared_lambda(registry,initial,p,summaries,state)
    assert result['selected_shared_lambda']==.5
    assert len(result['runs'])==16 and len(result['aggregates'])==4
    assert result['selected_mean_score']==pytest.approx(-.1)
    assert all(len(row['groups'])==4 for row in result['aggregates'])


def test_strict_ties_choose_smaller_weight_independent_of_grid_order(tmp_path):
    p,registry,initial,summaries,state=fake_grid(tmp_path)
    p['lambda_grid']=[5.,1.,.5,.05]
    for summary in summaries.values():
        for name,metrics in summary['validation_trajectory'][-1]['metrics'].items():
            metrics['nll']=initial['corpora'][name]['nll']
    result=select.select_shared_lambda(registry,initial,p,summaries,state)
    assert result['selected_shared_lambda']==.05


@pytest.mark.parametrize('defect',['missing','budget','stream','failed','missing_kl'])
def test_incomplete_or_unmatched_calibration_is_not_selected(tmp_path,defect):
    p,registry,initial,summaries,state=fake_grid(tmp_path)
    name=registry[-1]['id']
    if defect=='missing':registry.pop()
    elif defect=='budget':summaries[name]['counters']['replay_prediction_tokens']=99999
    elif defect=='stream':summaries[name]['target_stream']['trace_hash']='different'
    elif defect=='failed':state['jobs'][name]['status']='failed'
    elif defect=='missing_kl':del summaries[name]['validation_trajectory'][-1]['metrics']['target']['kl']
    with pytest.raises(ValueError):select.select_shared_lambda(registry,initial,p,summaries,state)


def test_v3_lr_reselection_can_change_shared_lr_from_cached_models(tmp_path):
    import json
    p=protocol()
    p['learning_rate_grid_completed']=[1e-5,3e-5,1e-4]
    p['learning_rate_selection_rule']='minimum target validation NLL'
    initial={'corpora':{name:{'nll':2.,'valid_prediction_tokens':100} for name in ['target','c4','wikitext']}}
    records=[];evaluations={};state={'jobs':{}}
    for label,lr,nll in [('1e-05',1e-5,1.99),('3e-05',3e-5,1.95),('1e-04',1e-4,2.1)]:
        output=tmp_path/label;output.mkdir()
        (output/'summary.json').write_text(json.dumps({'status':'completed','counters':{'target_prediction_tokens':2000000},'target_stream':{'hash':'same'}}))
        records.append({'label':label,'learning_rate':lr,'output_dir':str(output)})
        evaluation_path=output/'evaluation_v3.json'
        metrics={name:{'nll':nll if name=='target' else 2.,'valid_prediction_tokens':100,'complete_corpus':True,'kl':.1} for name in ['target','c4','wikitext']}
        evaluation_path.write_text(json.dumps({'corpora':metrics}))
        evaluations[f'lr_{label}']={'evaluation_path':str(evaluation_path),'job_id':label}
        state['jobs'][label]={'status':'completed'}
    result=select.select_learning_rate(records,initial,p,evaluations,state)
    assert result['selected_learning_rate']==3e-5


def test_formal_configuration_uses_versioned_validation_and_test_metadata(tmp_path):
    p={'validation_paths':{'target':str(tmp_path/'evaluation_v3/finemath_validation.pt')},
       'test_paths':{'target':str(tmp_path/'evaluation_v3/finemath_test.pt')}}
    configs,registry,_=make.build_experiments('formal',tmp_path,tmp_path/'scratch',1e-5,.05,protocol=p)
    assert all(cfg['validation_paths']==p['validation_paths'] for cfg in configs.values())
    assert all(row['test_paths']==p['test_paths'] for row in registry)
