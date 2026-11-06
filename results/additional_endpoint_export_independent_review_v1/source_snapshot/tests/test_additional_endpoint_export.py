"""Temporary CPU tensors and hand-built frozen identities; no real exports."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('additional_export',ROOT/'scripts/export_additional_endpoints.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

def put(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x)+'\n');return p

def A_fixture():
    rows=[];identities=[];primary=[];selected=[]
    choices={'B1':.05,'B2':1,'B3':.1,'B4':5,'M':5,'C':1}
    for g in choices:
        for seed in [11,22,33]:
            for w in [.01,.05,.1,.5,1,5,10]:
                rid=f'A_{g}_{seed}_{w}';actual=f'main_{g}_{seed}' if w==.05 else rid;sha='sha_'+actual
                rows.append(dict(id=rid,phase='supplement_A_lambda_sweep',execution_kind='alias' if w==.05 else 'train',alias_of=actual if w==.05 else None))
                identities.append(dict(run_id=rid,actual_run_id=actual,checkpoint_sha256=sha,role='supplement_A',kind='final',opaque_performance=100-w))
                if w==.05:primary.append(dict(weights_actual_run_id=actual,source_checkpoint_sha256=sha))
                if w==choices[g]:selected.append(dict(actual_run_id=actual,source_checkpoint_sha256=sha))
    return rows,identities,{'logical_runs':primary},{'logical_runs':selected}

def test_A93_excluded_by_saved_identity_not_scores():
    args=A_fixture();keep,excluded=e.remaining_A(*args)
    assert len(keep)==93 and len(excluded)==33
    for r in args[1]:r['opaque_performance']=-1e30
    assert {e.endpoint_key(r) for r in e.remaining_A(*args)[0]}=={e.endpoint_key(r) for r in keep}

def test_A_missing_saved_source_SHA_cannot_silently_reselect():
    args=A_fixture();args[2]['logical_runs'][3]['source_checkpoint_sha256']='changed'
    with pytest.raises(ValueError):e.remaining_A(*args)

def test_B_draw_seed_identity_and_complete_grid():
    rows=[];identities=[]
    for g in ['B4','M','C']:
        for k in [20,196]:
            for s in range(1001,1011):
                rid=f'{g}_{k}_{s}';rows.append(dict(id=rid,phase='conditional_bank_variance',group=g,bank_chunks=k,selection_seed=s,training_seed=11,generation_seed=11))
                identities.append(dict(run_id=rid,actual_run_id=rid,role='supplement_B',kind='final',selection_seed=s,training_seed=11,generation_seed=11))
    assert len(e.validate_B(rows,identities))==60
    identities[0]['training_seed']=1001
    with pytest.raises(ValueError):e.validate_B(rows,identities)

@pytest.fixture
def tiny(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');torch.set_num_threads(2)
    rid='tiny_B4_s11';cp=tmp_path/'checkpoint_best.pt'
    cfg=dict(seed=11,replay_ratio=.05,replay_weight=.05,learning_rate=3e-5,run_label=rid,target_prediction_tokens=100,device='cuda:0',precision='bf16',attention_implementation='sdpa')
    provenance=dict(torch_version='synthetic',cuda_version='synthetic',gpu='synthetic',device='cuda:0',precision='bf16',attention_implementation='sdpa',attention_backend_policy='auto',sdpa_backends_enabled=dict.fromkeys(['flash','math','memory_efficient','cudnn'],True),deterministic_algorithms=True,deterministic_warn_only=False,allow_tf32=True,allow_tf32_matmul=True,allow_tf32_cudnn=True,cudnn_deterministic=True,cudnn_benchmark=False,cublas_workspace_config=':4096:8',model_weight_dtype='torch.float32',teacher_weight_dtype='torch.float32')
    point=dict(step=1,target_prediction_tokens=50,replay_prediction_tokens=2,metrics={'target':dict(nll=2.,model_weight_dtype='torch.float32')})
    selected=dict(step=1,target_prediction_tokens=50,nll=2.,checkpoint_path=str(cp))
    summary=dict(status='completed',config=cfg,provenance=provenance,counters=dict(step=2,target_prediction_tokens=100,replay_prediction_tokens=5),best_target_validation=selected,validation_trajectory=[point])
    state={'a.weight':torch.arange(8,dtype=torch.float32),'z.bits':torch.tensor([0,-2147483648,2143289345],dtype=torch.int32).view(torch.float32)}
    base=tmp_path/'model.safetensors';save_file({k:torch.zeros_like(v) for k,v in state.items()},base)
    saved=dict(config=cfg,provenance=provenance,counters=dict(step=1,target_prediction_tokens=50,replay_prediction_tokens=2),checkpoint_kind='model_only_best_target_validation',selection=selected,model_state=state,rng_state={'tensor':torch.tensor([7,8])})
    torch.save(saved,cp)
    config_path=put(tmp_path/'config.json',cfg);summary_path=put(tmp_path/'summary.json',summary)
    row=dict(id=rid,seed=11,group='B4',ratio=.05,weight=.05,learning_rate=3e-5,config_path=str(config_path),summary_path=str(summary_path))
    bound=dict(run_id=rid,actual_run_id=rid,kind='best',model_tag='tiny',checkpoint=str(cp),checkpoint_sha256=e.file_sha256(cp),step=1,target_prediction_tokens_at_checkpoint=50,replay_prediction_tokens_at_checkpoint=2,full_training_committed_path={'target_prediction_tokens':100})
    item=dict(bound=bound,row=row,definition=dict(id=rid,argv=['python','train']),scheduler_record=dict(status='completed',returncode=0),frozen_source_stat=e.signature(cp))
    return item,base,e.training_validation_execution(summary),saved

def test_tiny_actual_roundtrip_metadata_and_source_FD(tiny,tmp_path):
    item,base,baseline,saved=tiny
    result=e.export_one('tiny',item,tmp_path/'delivery',base,e.file_sha256(base),baseline)
    assert result['bytewise_verification']['all_tensor_bytes_equal'] and result['source_fd_five_stat_before']==result['source_fd_five_stat_after']
    metadata=torch.load(result['metadata_path'],map_location='cpu',weights_only=False)
    assert e.same_metadata(metadata['original_checkpoint_metadata'],e.checkpoint_metadata(saved))
    assert result['additional_encoding_calls']==1 and result['all_nonweight_metadata_roundtrip_verified']

def test_tiny_wrong_frozen_CP_SHA_cannot_complete(tiny,tmp_path):
    item,base,baseline,_=tiny;item['bound']['checkpoint_sha256']='0'*64
    with pytest.raises(ValueError,match='Frozen CP'):e.export_one('tiny',item,tmp_path/'delivery',base,e.file_sha256(base),baseline)
    assert not list((tmp_path/'delivery').glob('records/*.json'))

def test_best_selection_metadata_mismatch_fails_before_codec(tiny):
    item,base,baseline,saved=tiny;saved['selection']={**saved['selection'],'step':2}
    with pytest.raises(ValueError,match='validation-selected'):e.validate_payload(item,saved,baseline)

def test_actual_stop_and_argv_definition_must_match(tiny):
    item,_,_,_=tiny;rid=item['bound']['run_id'];job=dict(id=rid,argv=['python','run','--config',item['row']['config_path']]);record=dict(status='completed',returncode=0,missing_outputs=[],argv=job['argv'],definition_sha256=e.digest_job(job))
    assert e.source_definition(item['bound'],item['row'],[job],{'jobs':{rid:record}})[0]==job
    record['stop_requested_at']=None
    with pytest.raises(ValueError):e.source_definition(item['bound'],item['row'],[job],{'jobs':{rid:record}})

def test_partial_assembly_and_unsafe_members_rejected(tmp_path):
    with pytest.raises(ValueError,match='Partial'):e.assemble('C410_best',{'rows':[]},[],tmp_path,tmp_path/'scope.json',None,{})
    for name in ['/absolute','../escape','x/../../escape']:
        with pytest.raises(ValueError):e.safe_members([(tmp_path/'file',name,'sha')])

def test_slots_are_disjoint_and_batch_requires_external_review():
    assert set(e.SLOTS['slot1']).isdisjoint(e.SLOTS['slot2'])
    assert sum(v['logical'] for v in e.PACKAGES.values())==216
    assert sum(v['limit_bytes'] for v in e.PACKAGES.values())==435<<30
    with pytest.raises(ValueError):e.validate_resource_review(None,None,'scope','slot1')

def test_missing_READY_is_pending_without_manifest_read(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'ROOT',tmp_path)
    assert e.published('not_ready',e.FrozenFiles()) is None

def test_pair_true_never_inferred_from_passed_phase(tmp_path):
    with pytest.raises(ValueError,match='best proof'):e.best_pair_decisions({'phase':dict(phase='phase',model='model'),'rows':[]},{'status':'passed'}, {})

def test_best_pair_false_is_independent_and_contradiction_rejected(tmp_path):
    rows=[];identities={};pairs=[]
    for seed in [11,22,33]:
        sides=[]
        for group in ['B4','D']:
            rid=f'{group}_{seed}';cfg=put(tmp_path/(rid+'_cfg.json'),{});summary=put(tmp_path/(rid+'_summary.json'),{})
            cp=str(tmp_path/(rid+'_best.pt'));sha='cp_'+rid;fp=dict(sha256_model_state=rid,state_elements=8)
            rows.append(dict(id=rid,seed=seed,group=group,config_path=str(cfg),summary_path=str(summary)))
            exposure=dict(target_prediction_tokens_at_checkpoint=seed,replay_prediction_tokens_at_checkpoint=1)
            identities[rid]=dict(checkpoint_sha256=sha,checkpoint=cp,**exposure)
            sides.append(dict(run_id=rid,kind='best',model_revision='revision',checkpoint_sha256=sha,checkpoint=cp,config_sha256=e.file_sha256(cfg),summary_sha256=e.file_sha256(summary),checkpoint_exposure=exposure,model_state_fingerprint=fp))
        pairs.append(dict(seed=seed,bitwise_model_state_equal=False,kind='identical_bank_null_control',endpoint='best',left=sides[0],right=sides[1]))
    context=dict(phase=dict(phase='phase',model='model'),rows=rows,protocol={'model_revision':'revision'})
    payload=dict(status='passed',phase='phase',model='model',endpoint='best',algorithm='sha256_sorted_key_dtype_shape_contiguous_uint8_v1',comparisons=pairs)
    assert all(not d['equal'] for d in e.best_pair_decisions(context,payload,identities).values())
    pairs[0]['bitwise_model_state_equal']=True
    with pytest.raises(ValueError,match='contradicts'):e.best_pair_decisions(context,payload,identities)
