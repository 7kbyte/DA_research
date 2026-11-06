import copy
import importlib.util
import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file

SPEC=importlib.util.spec_from_file_location('A_export',Path(__file__).parents[1]/'scripts/export_A_selected_weights.py')
A=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(A)


def selection_fixture():
    selected={'B1':.05,'B2':1.,'B3':.1,'B4':5.,'M':5.,'C':1.};registry=[];choices=[]
    for group,weight in selected.items():
        current=[]
        for w in [.01,.05,.1,.5,1.,5.,10.]:
            for seed in [11,22,33]:
                condition=f'A_{group}_{seed}_{w}';alias=w==.05
                row=dict(id=condition,phase='supplement_A_lambda_sweep',group=group,seed=seed,ratio=.05,weight=w,
                    execution_kind='alias' if alias else 'train',alias_of=f'main_{group}_{seed}' if alias else None)
                registry.append(row)
                if w==weight:current.append(row)
        choices.append(dict(group=group,weight=weight,n=3,complete=True,selection_frozen=True,shared_across_seeds=True,
            selected_by_best_checkpoint=False,test_scores_used=False,condition_ids=[r['id'] for r in current],
            actual_run_ids=[r['alias_of'] if r['execution_kind']=='alias' else r['id'] for r in current]))
    return dict(complete=True,completed_conditions=126,expected_conditions=126,selection_uses_test=False,
                test_scores_accessed=False,selections=choices),registry


def test_18_15_3_and_A_logical_kind():
    selection,registry=selection_fixture();rows=A.selected_rows(selection,registry)
    assert len(rows)==18 and sum(r['external_weight_reference'] for r in rows)==3
    assert {r['kind'] for r in rows}=={'validation_selected_lambda_final'}
    assert {r['group'] for r in rows if r['external_weight_reference']}=={'B1'}
    assert all(r['condition_id']!=r['actual_run_id'] for r in rows if r['external_weight_reference'])


@pytest.mark.parametrize('mutation',['partial','best','test','mapping','lambda'])
def test_selection_rejects_invalid_identity(mutation):
    selection,registry=selection_fixture()
    if mutation=='partial':selection['complete']=False
    elif mutation=='best':selection['selections'][0]['selected_by_best_checkpoint']=True
    elif mutation=='test':selection['selection_uses_test']=True
    elif mutation=='mapping':selection['selections'][0]['actual_run_ids'][0]='other'
    else:selection['selections'][0]['weight']=.5
    with pytest.raises(ValueError):A.selected_rows(selection,registry)


def source_fixture(tmp_path,monkeypatch,external=False):
    condition='A_B1_s11' if external else 'A_B2_s11';actual='main_B1_s11' if external else condition
    config=dict(seed=11,replay_ratio=.05,replay_weight=.05 if external else 1.,learning_rate=3e-5,run_label=actual,
        model_revision=A.REVISION,target_path=str(A.ROOT/'artifacts/data/finemath_train.pt'),
        validation_paths={'target':'fixed-validation.pt'},device='cpu',precision='fp32',attention_implementation='sdpa')
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config))
    counters=dict(step=306,target_prediction_tokens=10_000_000,replay_prediction_tokens=500_000)
    point={**counters,'metrics':{'target':{}}};best=dict(step=100,nll=1.2)
    summary=dict(status='completed',config=config,counters=counters,validation_trajectory=[point],best_target_validation=best,provenance={})
    summary_path=tmp_path/'summary.json';summary_path.write_text(json.dumps(summary))
    base=tmp_path/'base.safetensors';save_file({'w':torch.arange(4,dtype=torch.float32)},base)
    checkpoint=tmp_path/'latest.pt'
    saved=dict(config=config,counters=counters,validation_trajectory=[point],best_target_validation=best,
        model_state={'w':torch.tensor([float('nan'),-0.,2.,3.])},optimizer_state={'fixture':True},
        rng_state=torch.tensor([1,2,3]),stream={'epoch':2,'offset':9},checkpoint_kind='full_training_checkpoint',provenance={})
    torch.save(saved,checkpoint)
    row=dict(condition_id=condition,actual_run_id=actual,id=condition,phase='supplement_A_lambda_sweep',kind=A.KIND,
        model_tag='pythia-410m',data_family='main_v3',group='B1' if external else 'B2',seed=11,ratio=.05,
        weight=config['replay_weight'],learning_rate=3e-5,config_path=str(config_path),summary_path=str(summary_path),
        checkpoint=str(checkpoint),source_stat=A.signature(checkpoint),expected_checkpoint_sha256=A.file_sha256(checkpoint),
        external_weight_reference=external)
    baseline=dict(execution_provenance=A.training_validation_execution(summary));protocol=dict(validation_paths=config['validation_paths'])
    row['actual_exposure']=A.validate_source(row,summary,saved,baseline,protocol)
    monkeypatch.setattr(A,'BASE',base);monkeypatch.setattr(A,'BASE_SHA',A.file_sha256(base))
    return row,summary,saved,baseline,protocol


def test_tiny_real_codec_full_bytes_metadata_and_resume(tmp_path,monkeypatch):
    row,_,saved,baseline,protocol=source_fixture(tmp_path,monkeypatch)
    record=A.export_one(row,tmp_path/'output',baseline,protocol)
    assert record['bytewise_verification']['all_tensor_bytes_equal']
    metadata=torch.load(record['metadata_path'],map_location='cpu',weights_only=False)
    assert metadata['logical_identity']['kind']==A.KIND
    assert A.same_metadata(metadata['original_checkpoint_metadata'],A.checkpoint_metadata(saved))
    assert Path(row['checkpoint']).exists() and Path(record['archive_path']).exists()
    reused=A.export_one(row,tmp_path/'output',baseline,protocol)
    assert reused['archive_sha256']==record['archive_sha256'] and reused['record_reused_from_prior_verified_invocation']


def test_frozen_checkpoint_sha_mismatch_cannot_commit_record(tmp_path,monkeypatch):
    row,_,_,baseline,protocol=source_fixture(tmp_path,monkeypatch)
    row['expected_checkpoint_sha256']='0'*64
    with pytest.raises(ValueError,match='frozen A checkpoint SHA'):A.export_one(row,tmp_path/'output',baseline,protocol)
    assert not list((tmp_path/'output'/'records').glob('*.json'))


def test_B1_reference_keeps_own_metadata_without_encoding(tmp_path,monkeypatch):
    row,_,_,baseline,protocol=source_fixture(tmp_path,monkeypatch,external=True)
    row['durable_weight_reference']=dict(source_checkpoint_sha256=row['expected_checkpoint_sha256'],
        tar_sha256=A.PRIMARY_SHA,archive_member='weights/main_B1.zip',model_fingerprint={'fixture':'bound'})
    monkeypatch.setattr(A,'encode_weights',lambda *_a,**_kw:pytest.fail('B1 must not encode'))
    record=A.export_one(row,tmp_path/'output',baseline,protocol)
    assert not record['new_weight_forward_or_encoding']
    metadata=torch.load(record['metadata_path'],map_location='cpu',weights_only=False)
    assert metadata['logical_identity']['condition_id']!=metadata['logical_identity']['actual_run_id']
    assert metadata['original_checkpoint_metadata']['config']['run_label']==row['actual_run_id']


@pytest.mark.parametrize('field',['target_prediction_tokens','replay_prediction_tokens'])
def test_wrong_exposure_cannot_replace_final(tmp_path,monkeypatch,field):
    row,summary,saved,baseline,protocol=source_fixture(tmp_path,monkeypatch)
    saved=copy.deepcopy(saved);saved['counters'][field]-=1
    with pytest.raises(ValueError,match='actual10M/.5M'):A.validate_source(row,summary,saved,baseline,protocol)


def test_status_ledger_only_exact_operational_fields_may_change():
    old=dict(active_registry='fixed.json',stage='before',shared_lambda=.05)
    assert A.operational_ledger_diff(old,{**old,'stage':'after'})==['stage']
    with pytest.raises(ValueError,match='non-operational'):
        A.operational_ledger_diff(old,{**old,'shared_lambda':1.})


def test_primary_receipt_stat_detects_inode_or_modification(tmp_path):
    file=tmp_path/'archive';file.write_bytes(b'fixture');s=file.stat()
    receipt=dict(destination_stat=dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,
        ctime_ns=s.st_ctime_ns,uid=s.st_uid,gid=s.st_gid,mode=oct(s.st_mode&0o777)))
    A.verify_primary_stat(receipt,file)
    file.write_bytes(b'changed-length')
    with pytest.raises(ValueError,match='publication identity'):A.verify_primary_stat(receipt,file)
