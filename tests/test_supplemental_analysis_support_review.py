"""Synthetic evidence for current support and explicitly documented adapter gaps."""
import copy
import importlib.util
import json
from pathlib import Path

from replay_study import nll_v2_analysis as nll
from replay_study.behavioral_analysis import grouped_seed_summary,paired_seed_summary


def fixture_module():
    path=Path(__file__).with_name('test_nll_v2_analysis.py');sp=importlib.util.spec_from_file_location('nll_support_review_fixture',path)
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m


def test_complete_A_grid_keeps_all_42_lambda_method_cells_and_three_real_seeds(tmp_path):
    f=fixture_module();path,kwargs,manifest=f.fixture(tmp_path,role='supplement_A')
    template=manifest['records'][0];source=nll.read(manifest['phases'][0]['registry'])[0]
    score=nll.read(template['output']);registry=[];records=[];jobs=[manifest['jobs'][0]]
    for group in ('B1','B2','B3','B4','M','C'):
        for weight in (.01,.05,.1,.5,1.,5.,10.):
            for seed in (11,22,33):
                name=f'{group}_w{weight}_s{seed}';registry.append({**source,'id':name,'group':group,'weight':weight,'seed':seed})
                record={**template,'id':name+'__final','run_id':name,'actual_run_id':name,'group':group,'weight':weight,'training_seed':seed,
                    'checkpoint':'checkpoint_'+name,'checkpoint_sha256':name,'weights_identity':'checkpoint:'+name,
                    'inference_job_id':name,'output':str(tmp_path/(name+'.json'))}
                records.append(record);jobs.append({'id':name,'kind':'adapted','baseline_group':'base','weights_identity':record['weights_identity'],
                    'checkpoint':record['checkpoint'],'checkpoint_sha256':name,'output':record['output'],'record_ids':[record['id']]})
    f.write(Path(manifest['phases'][0]['registry']),registry);manifest['phases'][0]['registry_sha256']=nll.sha(manifest['phases'][0]['registry']);manifest['phases'][0]['rows']=126
    manifest['records']=records;manifest['jobs']=jobs;f.write(path,manifest)
    baseline=Path(jobs[0]['output']);a=baseline.with_name(baseline.stem+'_execution.json');att=nll.read(a);att['manifest_sha256']=nll.sha(path);f.write(a,att)
    for record in records:
        value=copy.deepcopy(score);value['checkpoint']=record['checkpoint'];out=Path(record['output']);f.write(out,value)
        f.write(out.with_name(out.stem+'_execution.json'),{'manifest_sha256':nll.sha(path),'job_id':record['inference_job_id'],
            'verified_frozen_inputs':True,'baseline_group':'base','checkpoint_sha256':record['checkpoint_sha256'],'weights_identity':record['weights_identity'],
            'output_sha256':{str(out):nll.sha(out)},'inference_alias':False,'inference_subprocess_wall_seconds':7.})
    plan=nll.prepare_plan(path,nll.sha(path),**kwargs)
    assert plan['expected_conditions']==126 and not plan['score_files_read']
    bundle=nll.collect_scores(plan,read_heldout_scores=True,n_bootstrap=0)
    assert len(bundle['endpoints'])==126 and len(bundle['all_candidates_frontier'])==42
    assert all(c['n']==3 for c in bundle['all_candidates_frontier'])
    assert {r['training_seed'] for r in bundle['endpoints']}=={11,22,33}
    # An auxiliary selected view must join by method's own frozen IDs. A single
    # global weight filter would destroy this intentionally different-lambda view.
    choices=dict(zip(('B1','B2','B3','B4','M','C'),(.01,.05,.1,.5,1.,5.)))
    chosen=[r for r in bundle['endpoints'] if r['weight']==choices[r['group']]]
    assert len(chosen)==18 and len({r['weight'] for r in chosen})==6


def test_existing_behavior_pairs_intentionally_do_not_compare_different_lambdas():
    rows=[]
    for group,weight in (('B4',.05),('M',.5)):
        for seed in (11,22,33):
            rows.append({'phase':'formal','checkpoint_kind':'final','group':group,'ratio':.05,'weight':weight,'learning_rate':1e-5,
                'task':'piqa','metric':'acc_norm','seed':seed,'run_id':f'{group}_{seed}',
                'accuracy':.5,'accuracy_delta':0.,'conditional_forgetting_rate':.1,'conditional_acquisition_rate':.1,
                'target_prediction_tokens_at_checkpoint':100})
    assert all(r['n_training_seeds']==3 for r in grouped_seed_summary(rows))
    pairs,aggregate=paired_seed_summary(rows)
    assert pairs==[] and aggregate==[]  # Documents the need for a selected-policy auxiliary adapter.


def test_NLL_final_partial_rejection_is_not_a_before_first_score_gate(tmp_path):
    f=fixture_module();path,kwargs,manifest=f.fixture(tmp_path,role='supplement_A')
    plan=nll.prepare_plan(path,nll.sha(path),**kwargs);Path(manifest['records'][1]['output']).unlink();read_paths=[]
    import pytest
    with pytest.raises(ValueError,match='incomplete frozen phase'):
        nll.collect_scores(plan,read_heldout_scores=True,n_bootstrap=0,before_score_read=lambda p:read_paths.append(str(p)))
    assert read_paths  # A separate full-phase integrity/authorization gate is required before calling collect_scores.
