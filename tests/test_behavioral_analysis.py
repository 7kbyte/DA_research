"""Hand-computed behavioral inference units and paired uncertainty invariants."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from replay_study.behavioral_analysis import (align,collect,grouped_seed_summary,mean_sd,
    paired_binary_summary,paired_seed_summary,verify_prediction_rows,wilson,write_bundle)


def predictions(flags):
    return [{'task':'piqa','item_id':str(i),'fingerprint':f'fp{i}','gold':0,
             'correct_acc':flag,'correct_acc_norm':flag} for i,flag in enumerate(flags)]


def test_swaps_hidden_by_mean_accuracy_are_preserved_with_paired_bootstrap():
    initial=predictions([1,1,0,0]);adapted=predictions([1,0,1,0])
    result=paired_binary_summary(initial,adapted,'acc_norm',n_bootstrap=1000,seed=17)
    assert result['counts']=={'CC':1,'CW':1,'WC':1,'WW':1}
    assert result['accuracy_delta']==0
    assert result['conditional_forgetting']['estimate']==.5
    assert result['conditional_acquisition']['estimate']==.5
    assert result['delta_ci_low']<0<result['delta_ci_high']
    assert 'does not create training replicates' in result['uncertainty_scope']
    identical=paired_binary_summary(initial,list(reversed(initial)),'acc',n_bootstrap=100,seed=8)
    assert identical['delta_ci_low']==identical['delta_ci_high']==0
    changed=copy.deepcopy(adapted);changed[0]['fingerprint']='other'
    with pytest.raises(ValueError,match='fingerprint'):align(initial,changed)
    assert wilson(0,0)['estimate'] is None
    assert wilson(0,100)['ci_high']<.04
    assert mean_sd([3])['std_ddof1'] is None


def test_seed_differences_pair_before_sd_and_questions_never_supply_n():
    rows=[]
    for seed,value,delta in [(11,.1,.01),(22,.4,.02),(33,.8,.03)]:
        for group,accuracy in [('B4',value),('M',value+delta)]:
            rows.append({'phase':'formal','group':group,'seed':seed,'run_id':f'{group}{seed}',
                'checkpoint_kind':'final','ratio':.05,'weight':.5,'learning_rate':3e-5,
                'task':'piqa','metric':'acc_norm','accuracy':accuracy,'accuracy_delta':accuracy-.5,
                'conditional_forgetting_rate':.1,'conditional_acquisition_rate':.2,
                'target_prediction_tokens_at_checkpoint':100})
    paired,aggregate=paired_seed_summary(rows)
    assert [r['accuracy_left_minus_right'] for r in paired]==pytest.approx([.01,.02,.03])
    assert aggregate[0]['mean']==pytest.approx(.02)
    assert aggregate[0]['std_ddof1']==pytest.approx(.01)
    assert aggregate[0]['n_training_seeds']==3
    groups=grouped_seed_summary(rows)
    assert all(row['n_training_seeds']==3 for row in groups)
    bad=copy.deepcopy(rows);bad[-1]['ratio']=.10
    p,a=paired_seed_summary(bad)
    assert {r['seed'] for r in p}=={11,22}  # different replay budgets never pair


def test_choice_character_normalization_is_verified_separately():
    prepared=[{'task':'piqa','item_id':'0','fingerprint':'x','gold':1}]
    row={**prepared[0],'choices':['a','bbbb'],'choice_character_lengths':[1,4],
         'choice_logprobs':[-1.,-2.],'prediction_acc':0,'correct_acc':0,
         'prediction_acc_norm':1,'correct_acc_norm':1}
    verify_prediction_rows([row],prepared)
    row['choice_character_lengths']=[1,2]
    with pytest.raises(ValueError,match='character length'):verify_prediction_rows([row],prepared)


def test_manifest_collection_requires_attested_full_results_and_exports_both_metrics(tmp_path):
    source=Path(__file__).with_name('test_behavioral_queue.py')
    spec=importlib.util.spec_from_file_location('behavior_queue_fixture',source)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    registry,authority,data,cfg,summary,registry_rows=fixture.fixture(tmp_path)
    # Replace fixture's empty rows with two fingerprinted synthetic questions.
    prepared={}
    for task in fixture.make.TASKS:
        examples=[{'task':task,'item_id':str(i),'fingerprint':f'{task}{i}','gold':i,'choices':['a','bbbb']} for i in (0,1)]
        path=data/f'{task}_validation.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in examples))
        prepared[task]=examples
        meta=json.loads(path.with_suffix('.json').read_text());meta['artifact_sha256']=fixture.make.sha(path)
        fixture.write(path.with_suffix('.json'),meta)
        top=json.loads((data/'data_manifest.json').read_text());top['tasks'][task]=meta
        fixture.write(data/'data_manifest.json',top)
    queue,manifest=fixture.build(tmp_path,registry,authority,data)
    manifest_path=fixture.write(tmp_path/'manifest.json',manifest);manifest_sha=fixture.make.sha(manifest_path)
    policy=manifest['inference_policy']
    execution={**policy,'torch_version':'fixture','cuda_version':'fixture','gpu_name':'fixture',
               'precision':'bf16','attention_backend_requested':policy['attention_backend']}
    for job in manifest['jobs']:
        output=Path(job['output']);output.parent.mkdir(parents=True,exist_ok=True)
        tasks={};outputs={}
        for task,examples in prepared.items():
            records=[]
            for example in examples:
                # Raw and character-normalized metrics disagree, deliberately.
                records.append({**example,'choice_logprobs':[-1.,-2.],'choice_character_lengths':[1,4],
                    'prediction_acc':0,'prediction_acc_norm':1,
                    'correct_acc':int(example['gold']==0),'correct_acc_norm':int(example['gold']==1)})
            pred=output.with_name(output.stem+f'_{task}.jsonl')
            pred.write_text(''.join(json.dumps(x)+'\n' for x in records));outputs[str(pred)]=fixture.make.sha(pred)
            tasks[task]={'complete_split':True,'items':2,'data_sha256':manifest['tasks'][task]['sha256'],
                         'dataset_revision':manifest['tasks'][task]['revision'],'batch_size':16,'max_length':2048,
                         'predictions_path':str(pred),'predictions_sha256':outputs[str(pred)],'acc':.5,'acc_norm':.5}
        result={'complete':True,'evaluation_role':'formal','model':policy['model'],'revision':policy['revision'],
                'checkpoint':job.get('checkpoint'),'precision':'bf16','harness_revision':manifest['harness_revision'],
                'project_primary_metric':'acc_norm','tasks':tasks,'execution_provenance':execution}
        fixture.write(output,result);outputs[str(output)]=fixture.make.sha(output)
        fixture.write(output.with_name(output.stem+'_execution.json'),{'manifest_sha256':manifest_sha,
            'job_id':job['id'],'verified_frozen_inputs':True,'checkpoint_sha256':job.get('checkpoint_sha256'),
            'output_sha256':outputs})
    bundle=collect(manifest_path)
    assert len(bundle['endpoints'])==12  # 3 checkpoints x 2 tasks x 2 scores
    assert all(row['accuracy_delta']==0 for row in bundle['endpoints'])
    assert {row['metric'] for row in bundle['endpoints']}=={'acc','acc_norm'}
    assert all(row['target_prediction_tokens_at_checkpoint']==50 for row in bundle['endpoints'] if row['checkpoint_kind']=='best')
    write_bundle(bundle,tmp_path/'analysis')
    assert (tmp_path/'analysis/endpoints.csv').exists()
    with pytest.raises(ValueError,match='bootstrap settings'):collect(manifest_path,n_bootstrap=100)
    Path(manifest['jobs'][-1]['output']).write_text('{}')
    with pytest.raises(ValueError,match='output SHA'):collect(manifest_path)
