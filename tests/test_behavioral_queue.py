"""CPU-only freeze/eligibility/hash invariants; checkpoints are synthetic bytes."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
make=load('make_behavioral_queue');runner=load('run_frozen_behavioral')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(make.encoded(value));return path

def fixture(tmp_path):
    data=tmp_path/'data';data.mkdir()
    top={'harness_revision':make.HARNESS_REVISION,'tasks':{}}
    for task in make.TASKS:
        path=data/f'{task}_validation.jsonl';path.write_text('{}\n{}\n')
        meta={'harness_revision':make.HARNESS_REVISION,'artifact_sha256':make.sha(path),
              'fewshot':0,'apply_chat_template':False,'items':2,'revision':f'{task}-pinned',
              'ordered_example_fingerprint':f'{task}-fingerprints','license':{'license':'fixture-only'}}
        top['tasks'][task]=meta;write(path.with_suffix('.json'),meta)
    write(data/'data_manifest.json',top)
    source=tmp_path/'official_source';source.write_text('pinned task source')
    write(data/'sources/harness_manifest.json',{'revision':make.HARNESS_REVISION,
        'files':[{'path':str(source),'sha256':make.sha(source)}]})
    final=tmp_path/'latest.pt';final.write_bytes(b'final-checkpoint')
    best=tmp_path/'best.pt';best.write_bytes(b'best-checkpoint')
    model=tmp_path/'fixture-model';model.mkdir();(model/'pytorch_model.bin').write_bytes(b'initial-weights')
    cfg={'model_name_or_path':str(model),'model_revision':'pinned','precision':'bf16',
        'attention_backend':'auto','attention_implementation':'sdpa','allow_tf32':True,
        'deterministic_algorithms':True,'evaluation_batch_size':16,'run_label':'run_B4_s11',
        'seed':11,'replay_ratio':.05,'replay_weight':.5,'learning_rate':3e-5,
        'target_prediction_tokens':100,'device':'cuda:0','output_dir':'original'}
    summary={'status':'completed','config':{**cfg,'device':'cuda:3','output_dir':'resumed','resume_from':'old.pt'},
        'checkpoint_path':str(final),'counters':{'step':4,'target_prediction_tokens':100,'replay_prediction_tokens':5},
        'best_target_validation':{'step':2,'target_prediction_tokens':50,'nll':1.5,'checkpoint_path':str(best)},
        'validation_trajectory':[{'step':2,'target_prediction_tokens':50,'replay_prediction_tokens':2,
                                  'metrics':{'target':{'nll':1.5}}}]}
    config=write(tmp_path/'config.json',cfg);sp=write(tmp_path/'summary.json',summary)
    rows=[{'id':'run_B4_s11','phase':'formal','active':True,'group':'B4','seed':11,
           'ratio':.05,'weight':.5,'learning_rate':3e-5,'config_path':str(config),'summary_path':str(sp)}]
    registry=write(tmp_path/'registry.json',rows)
    authority=write(tmp_path/'authority.json',{'active_registry':str(registry)})
    return registry,authority,data,cfg,summary,rows

def build(tmp_path,registry,authority,data):
    return make.build(registry,tmp_path/'outputs','pythia-test',['final','best'],30,
                      tmp_path/'frozen_manifest.json',data,authority,16,2048)

def test_frozen_jobs_bind_data_checkpoint_policy_and_true_best_exposure(tmp_path):
    registry,authority,data,cfg,summary,rows=fixture(tmp_path)
    queue,manifest=build(tmp_path,registry,authority,data)
    assert len(queue)==3 and len([j for j in manifest['jobs'] if j['kind']=='initial'])==1
    assert manifest['inference_policy']['primary_metric']=='acc_norm'
    assert manifest['inference_policy']['auxiliary_metric']=='acc'
    assert manifest['output_root']==str(tmp_path/'outputs/pythia-test')
    initial,final,best=manifest['jobs']
    assert final['target_prediction_tokens_at_checkpoint']==100
    assert best['target_prediction_tokens_at_checkpoint']==50
    assert best['replay_prediction_tokens_at_checkpoint']==2
    assert best['checkpoint_sha256']==make.sha(summary['best_target_validation']['checkpoint_path'])
    for record in manifest['jobs']:
        argv=record['argv']
        assert argv[argv.index('--batch-size')+1]=='16'
        assert argv[argv.index('--max-length')+1]=='2048'
        assert '--deterministic-algorithms' in argv and '--no-cudnn-benchmark' in argv
        assert not any(flag in argv for flag in ('--smoke','--limit'))
    assert final['argv'][final['argv'].index('--baseline')+1]==initial['output']
    assert queue[-1]['depends_on']==[queue[0]['id']]
    assert len(queue[-1]['outputs'])==4
    manifest_path=write(tmp_path/'frozen_manifest.json',manifest)
    expected=hashlib.sha256(make.encoded(manifest)).hexdigest()
    assert queue[0]['argv'][queue[0]['argv'].index('--manifest-sha256')+1]==expected
    runner.verified_job(manifest_path,expected,initial['id'])
    write(Path(initial['output']),{'complete':True,'checkpoint':None,'model':cfg['model_name_or_path'],'revision':cfg['model_revision']})
    runner.verified_job(manifest_path,expected,best['id'])
    Path(best['checkpoint']).write_bytes(b'mutated')
    with pytest.raises(ValueError,match='input SHA mismatch'):
        runner.verified_job(manifest_path,expected,best['id'])

@pytest.mark.parametrize('change',['inactive','provisional','incomplete','exposure','science','authority','best_exposure'])
def test_ineligible_or_mismatched_runs_cannot_be_frozen(tmp_path,change):
    registry,authority,data,cfg,summary,rows=fixture(tmp_path)
    if change=='inactive':rows[0]['active']=False
    elif change=='provisional':rows[0]['status']='provisional-numerical-noise'
    elif change=='incomplete':summary['status']='deadline_interrupted'
    elif change=='exposure':summary['counters']['target_prediction_tokens']=99
    elif change=='science':summary['config']['learning_rate']=1e-4
    elif change=='authority':write(authority,{'active_registry':str(tmp_path/'other.json')})
    elif change=='best_exposure':summary['best_target_validation']['target_prediction_tokens']=99
    write(registry,rows);write(Path(rows[0]['summary_path']),summary)
    with pytest.raises(ValueError):build(tmp_path,registry,authority,data)
    assert not (tmp_path/'frozen_manifest.json').exists()

def test_models_cannot_be_mixed_and_all_eligibility_is_checked_before_checkpoint_sha(tmp_path):
    registry,authority,data,cfg,summary,rows=fixture(tmp_path)
    cfg2={**cfg,'run_label':'run_B4_s22','seed':22,'model_revision':'other-model-revision'}
    row2={**rows[0],'id':'run_B4_s22','seed':22,'config_path':str(tmp_path/'cfg2.json'),
          'summary_path':str(tmp_path/'sum2.json')}
    write(Path(row2['config_path']),cfg2)
    write(Path(row2['summary_path']),{**summary,'config':cfg2})
    write(registry,[*rows,row2])
    # A missing first checkpoint must not mask a mixed-model registry.
    Path(summary['checkpoint_path']).unlink()
    with pytest.raises(ValueError,match='Different models'):build(tmp_path,registry,authority,data)

def test_same_model_second_registry_reuses_one_initial_plan_with_explicit_dependency(tmp_path):
    registry,authority,data,cfg,summary,rows=fixture(tmp_path)
    queue,manifest=build(tmp_path,registry,authority,data)
    original=write(tmp_path/'original_manifest.json',manifest)
    reused_queue,reused=make.build(registry,tmp_path/'extension','pythia-test-extended',
        ['final','best'],30,tmp_path/'extended_manifest.json',data,authority,16,2048,original)
    assert len(reused_queue)==2
    assert reused['jobs'][0]['external_initial'] is True
    assert reused_queue[0]['depends_on']==[queue[0]['id']]
    assert reused['jobs'][1]['argv'][reused['jobs'][1]['argv'].index('--baseline')+1]==manifest['jobs'][0]['output']
    assert reused['common_input_sha256'][str(original)]==make.sha(original)
    with pytest.raises(ValueError,match='identical model'):
        make.build(registry,tmp_path/'bad','other',['final'],30,tmp_path/'bad_manifest',data,authority,8,2048,original)
