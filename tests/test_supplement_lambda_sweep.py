import copy
import importlib.util
from pathlib import Path
import torch


def module(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


sweep=module('make_lambda_sweep')
audit=module('audit_formal_runs')


def sources(root):
    records=[];configs={}
    for group in sweep.GROUPS:
        for seed in sweep.SEEDS:
            name=f'formal_{group}_s{seed}'
            record={'id':name,'group':group,'seed':seed,'ratio':.05,'weight':.05,
                'config_path':str(root/f'{name}.json'),'output_dir':str(root/name),
                'summary_path':str(root/f'{name}_summary.json'),'test_paths':{'target':'evaluation_v3/math_test.pt'}}
            records.append(record)
            configs[name]={'target_prediction_tokens':10000000,'learning_rate':3e-5,'replay_ratio':.05,
                'replay_weight':.05,'output_dir':record['output_dir'],'run_label':name,
                'microbatch_size':16,'gradient_accumulation_steps':4,'seed':seed,'precision':'bf16',
                'validation_paths':{'target':'evaluation_v3/math_validation.pt'}}
    return records,configs


def test_full_sweep_is_108_new_runs_and_18_provenance_aliases(tmp_path):
    records,configs=sources(tmp_path)
    before=copy.deepcopy(configs)
    new,registry,queue,aliases=sweep.build_sweep(records,configs,tmp_path,tmp_path/'scratch')
    assert len(new)==len(queue)==108 and len(registry)==126 and len(aliases)==18
    assert configs==before and not list(tmp_path.iterdir())
    assert {(r['group'],r['seed'],r['weight']) for r in registry}=={
        (g,s,w) for g in sweep.GROUPS for s in sweep.SEEDS for w in sweep.WEIGHTS}
    indexed={r['id']:r for r in registry}
    assert {indexed[r['id']]['group'] for r in queue[:36]}=={'B4','M'}
    assert all(r['weight']==.05 and r['alias_of']==r['source_formal_run_id'] for r in aliases)
    assert all(not r['counted_as_new_training'] for r in aliases)
    for row in registry:
        if row['execution_kind']=='train':
            original=configs[row['source_formal_run_id']]
            changed={key for key in original if original[key]!=new[row['config_path']][key]}
            assert changed==sweep.IDENTITY_FIELDS


def test_audit_budget_floors_and_model_fingerprint_ignore_checkpoint_metadata(tmp_path):
    assert [audit.expected_replay(10000000,b) for b in [.01,.05,.1]]==[100000,500000,1000000]
    a=tmp_path/'a.pt';b=tmp_path/'b.pt';c=tmp_path/'c.pt'
    state={'weight':torch.tensor([[1.,2.]]),'counter':torch.tensor(3)}
    torch.save({'model_state':state,'metadata':'first'},a)
    torch.save({'model_state':state,'metadata':'second'},b)
    torch.save({'model_state':{'weight':state['weight']+.001,'counter':state['counter']}},c)
    assert audit.model_fingerprint(a)==audit.model_fingerprint(b)
    assert audit.model_fingerprint(a)!=audit.model_fingerprint(c)
