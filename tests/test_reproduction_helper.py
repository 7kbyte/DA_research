import argparse,hashlib,importlib.util,json
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/reproduce_experiments.py'
spec=importlib.util.spec_from_file_location('portable_reproduction',path)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

def fixtures(tmp_path,monkeypatch):
    source=tmp_path/'archive';workspace=tmp_path/'unpacked';scratch=tmp_path/'new-scratch'
    source.mkdir();workspace.mkdir();scratch.mkdir();monkeypatch.setattr(helper,'ROOT',source)
    data={}
    for name in ['train','bank','validation','test']:
        rel=f'artifacts/{name}.pt';p=workspace/rel;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(name.encode());data[rel]={'sha256':helper.sha(p)}
    codes={}
    for relative in helper.CORE_FILES:
        code=workspace/relative;code.parent.mkdir(parents=True,exist_ok=True);code.write_text('frozen '+relative)
        codes[relative]={'sha256':helper.sha(code)}
    model=scratch/'models/pythia-410m-final/model.safetensors';model.parent.mkdir(parents=True);model.write_bytes(b'public pinned model')
    models={'pythia-410m-final':{'revision':'fixed-revision','files':{'model.safetensors':helper.sha(model)}}}
    monkeypatch.setattr(helper,'model_catalog',lambda:models)
    logical=helper.OLD_WORKSPACES[0];physical=helper.OLD_WORKSPACES[1]
    original={'model_name_or_path':helper.OLD_SCRATCH+'/models/pythia-410m-final','model_revision':'fixed-revision',
        'target_path':logical+'/artifacts/train.pt','replay_path':physical+'/artifacts/bank.pt',
        'validation_paths':{'target':physical+'/artifacts/validation.pt'},'output_dir':helper.OLD_SCRATCH+'/runs/original',
        'run_label':'original','seed':11,'learning_rate':3e-5,'target_prediction_tokens':10_000_000,
        'replay_objective':'kl','replay_ratio':.05,'replay_weight':.05,'microbatch_size':16,
        'gradient_accumulation_steps':4,'deterministic_algorithms':True,'deadline_utc':'2026-11-07T03:26:32Z','resume_from':None}
    cfg=source/'configs/original.json';cfg.parent.mkdir();cfg.write_text(json.dumps(original))
    registry=source/'registry.json';registry.write_text(json.dumps([{'id':'original','config_path':logical+'/configs/original.json',
        'group':'B4','seed':11,'test_paths':{'target':logical+'/artifacts/test.pt'}}]))
    snapshot=source/'snapshot.json';snapshot.write_text(json.dumps({'data':data,
        'code':codes,
        'configs':{'configs/original.json':{'sha256':helper.sha(cfg)}}}))
    args=argparse.Namespace(registry=registry,input_manifest=[snapshot],new_workspace=workspace,new_scratch=scratch,
        namespace='portable',run_id=None,deadline=None)
    return args,original

def test_relocation_verifies_both_workspace_prefixes_and_preserves_science(tmp_path,monkeypatch):
    args,original=fixtures(tmp_path,monkeypatch);helper.relocate(args)
    registry=args.new_workspace/'configs/reproductions/portable_registry.json';record=json.loads(registry.read_text())[0]
    cfg=json.loads(Path(record['config_path']).read_text())
    assert helper.scientific_parameters(cfg)==helper.scientific_parameters(original)
    assert cfg['target_path']==str(args.new_workspace/'artifacts/train.pt')
    assert cfg['replay_path']==str(args.new_workspace/'artifacts/bank.pt')
    assert cfg['model_name_or_path']==str(args.new_scratch/'models/pythia-410m-final')
    assert cfg['deadline_utc'] is None and record['reproduction_of']['scientific_parameters_unchanged']
    assert not Path(record['output_dir']).exists()
    monkeypatch.setattr(helper.subprocess,'Popen',lambda *a,**kw:pytest.fail('Dry run started training'))
    helper.reproduce(argparse.Namespace(registry=registry,gpus='0',deadline='2030-01-01T00:00:00Z',execute=False))

@pytest.mark.parametrize('which',['data','model'])
def test_tampered_inputs_fail_before_any_config_write(tmp_path,monkeypatch,which):
    args,_=fixtures(tmp_path,monkeypatch)
    target=args.new_workspace/'artifacts/train.pt' if which=='data' else args.new_scratch/'models/pythia-410m-final/model.safetensors'
    target.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='hash mismatch'):helper.relocate(args)
    assert not (args.new_workspace/'configs').exists()

def test_frozen_source_config_tampering_is_rejected(tmp_path,monkeypatch):
    args,_=fixtures(tmp_path,monkeypatch)
    source=helper.ROOT/'configs/original.json';obj=json.loads(source.read_text());obj['learning_rate']=1e-4;source.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='config hash mismatch'):helper.relocate(args)
    assert not (args.new_workspace/'configs').exists()

def test_existing_reproduction_output_is_not_overwritten(tmp_path,monkeypatch):
    args,_=fixtures(tmp_path,monkeypatch)
    existing=args.new_scratch/'runs/portable__original';existing.mkdir(parents=True)
    marker=existing/'keep.txt';marker.write_text('preserve original')
    with pytest.raises(FileExistsError):helper.relocate(args)
    assert marker.read_text()=='preserve original'
    assert not (args.new_workspace/'configs').exists()
