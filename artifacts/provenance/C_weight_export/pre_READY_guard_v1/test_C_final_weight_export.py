"""Score-free C draft/complete/alias gates with tiny raw-bit weight fixtures."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file

SPEC=importlib.util.spec_from_file_location('C_export',Path(__file__).parents[1]/'scripts/export_C_final_weights.py')
C=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(C)

def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return str(path)

def run_fixture(tmp_path):
    output=tmp_path/'summary.json';put(output,{})
    row=dict(id='C_B4_s11',summary_path=str(output));definition=dict(id=row['id'],argv=['train'],outputs=[str(output)])
    state=dict(jobs={row['id']:dict(status='completed',returncode=0,definition_sha256=C.digest_job(definition),missing_outputs=[])})
    return row,definition,state

def test_master_append_and_reorder_do_not_mutate_training_identity(tmp_path):
    row,definition,state=run_fixture(tmp_path)
    assert C.run_ready(row,definition,state,[dict(id='new_evaluation'),definition])
    assert C.run_ready(row,definition,state,[definition,dict(id='new_evaluation')])
    with pytest.raises(ValueError,match='definition changed'):
        C.run_ready(row,definition,state,[{**definition,'argv':['other']}])

@pytest.mark.parametrize('field,value',[('returncode',1),('definition_sha256','wrong'),('missing_outputs',['x'])])
def test_rc0_source_definition_and_outputs_required(tmp_path,field,value):
    row,definition,state=run_fixture(tmp_path);state['jobs'][row['id']][field]=value
    with pytest.raises(ValueError):C.run_ready(row,definition,state,[definition])

def test_pending_is_a_draft_wait_not_completion(tmp_path):
    row,definition,state=run_fixture(tmp_path);state['jobs'][row['id']]['status']='running'
    assert C.run_ready(row,definition,state,[definition]) is False
    with pytest.raises(ValueError,match='partial'):
        C.assemble(dict(rows=[row],phase={}),{},[],tmp_path,None)

def pair_fixture(tmp_path,equal=True):
    rows=[];identities={};pairs=[]
    for seed in [11,22,33]:
        sides=[]
        for group in ['B4','D']:
            name=f'C_{group}_s{seed}';cp=tmp_path/(name+'.pt');cp.write_bytes(b'checkpoint')
            config=put(tmp_path/(name+'.config.json'),{});summary=put(tmp_path/(name+'.summary.json'),{})
            row=dict(id=name,seed=seed,group=group,ratio=.05,checkpoint_path=str(cp),config_path=config,summary_path=summary)
            rows.append(row);identities[name]=dict(checkpoint_sha256='sha'+name)
            fingerprint=dict(sha256_model_state='same' if equal or group=='B4' else 'different',state_elements=4)
            sides.append(dict(run_id=name,kind='final',model_revision='revision',checkpoint=str(cp),
                checkpoint_sha256='sha'+name,config_sha256=C.file_sha256(config),summary_sha256=C.file_sha256(summary),
                checkpoint_exposure=dict(target_prediction_tokens_at_checkpoint=100_000_000,replay_prediction_tokens_at_checkpoint=5_000_000),
                model_state_fingerprint=fingerprint))
        pairs.append(dict(id=f'pair{seed}',seed=seed,kind='identical_bank_null_control',endpoint='final',
            left=sides[0],right=sides[1],bitwise_model_state_equal=equal))
    context=dict(rows=rows,phase=dict(phase='C_phase',model='pythia-1b',target_prediction_tokens=100_000_000),
        protocol=dict(model_revision='revision'))
    proof=dict(status='passed',phase='C_phase',model='pythia-1b',endpoint='final',
        algorithm='sha256_sorted_key_dtype_shape_contiguous_uint8_v1',comparisons=pairs)
    return context,proof,identities

@pytest.mark.parametrize('equal',[True,False])
def test_per_pair_equality_controls_alias_and_false_requires_own_model(tmp_path,equal):
    context,proof,identities=pair_fixture(tmp_path,equal)
    decisions=C.validate_final_pairs(context,proof,identities)
    assert len(decisions)==3 and all(d['equal'] is equal for d in decisions.values())

@pytest.mark.parametrize('mutation',['status_only','flag','source','model','seed','best','revision'])
def test_alias_never_inferred_from_passed_status(tmp_path,mutation):
    context,proof,identities=pair_fixture(tmp_path)
    if mutation=='status_only':proof['comparisons']=[]
    elif mutation=='flag':proof['comparisons'][0]['bitwise_model_state_equal']=False
    elif mutation=='source':proof['comparisons'][0]['right']['checkpoint_sha256']='changed'
    elif mutation=='model':proof['model']='pythia-410m'
    elif mutation=='seed':proof['comparisons'][0]['seed']=22
    elif mutation=='best':proof['comparisons'][0]['right']['kind']='best'
    else:proof['comparisons'][0]['right']['model_revision']='other'
    with pytest.raises(ValueError):C.validate_final_pairs(context,proof,identities)

def tiny_draft_fixture(tmp_path,monkeypatch):
    row,definition,state=run_fixture(tmp_path);name=row['id'];config=dict(model_revision='revision')
    row.update(model='pythia-1b',seed=11,group='B4',ratio=.05,weight=.05,output_dir=str(tmp_path),
        config_path=put(tmp_path/'config.json',config),checkpoint_path=str(tmp_path/'latest.pt'))
    put(Path(row['summary_path']),dict(status='completed',config=config))
    put(tmp_path/'events.jsonl',{})
    base=tmp_path/'model.safetensors';save_file({'w':torch.arange(4,dtype=torch.float32)},base)
    saved=dict(model_state={'w':torch.tensor([float('nan'),-0.,2.,3.])},optimizer_state={'state':1},
        config=config,counters=dict(step=4,target_prediction_tokens=100,replay_prediction_tokens=5),
        checkpoint_kind='full_training_checkpoint',selection={'note':'kept'},rng_state=torch.tensor([4,5]),provenance={'source':'fixture'})
    torch.save(saved,row['checkpoint_path'])
    context=dict(phase=dict(phase='C_phase',model='pythia-1b',snapshot_sha256='snapshot'),protocol=dict(model_revision='revision'),
        base=str(base),base_sha=C.file_sha256(base))
    monkeypatch.setattr(C,'validate_source',lambda *args:dict(checkpoint_step=4,target_prediction_tokens=100,replay_prediction_tokens=5))
    return context,row,definition,saved

def test_real_codec_draft_full_bits_metadata_and_frozen_source_sha(tmp_path,monkeypatch):
    context,row,definition,saved=tiny_draft_fixture(tmp_path,monkeypatch)
    record=C.draft_one(context,row,tmp_path/'draft',definition,C.file_sha256(row['checkpoint_path']))
    assert record['status']=='verified_draft_not_deliverable' and record['bytewise_verification']['all_tensor_bytes_equal']
    metadata=torch.load(record['metadata_path'],weights_only=False)
    assert metadata['logical_identity']['role']=='extended' and metadata['logical_identity']['data_family']=='extended_v1'
    assert C.same_metadata(metadata['original_checkpoint_metadata'],C.checkpoint_metadata(saved))
    again=C.draft_one(context,row,tmp_path/'draft',definition,record['source_checkpoint_sha256'])
    assert again['archive_sha256']==record['archive_sha256']
    with pytest.raises(ValueError,match='frozen CP SHA'):
        C.draft_one(context,row,tmp_path/'draft',definition,'wrong')
    assert Path(row['checkpoint_path']).exists() and Path(record['archive_path']).exists()

def test_bad_source_sha_cannot_commit_verified_record(tmp_path,monkeypatch):
    context,row,definition,_=tiny_draft_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='checkpoint identity'):
        C.draft_one(context,row,tmp_path/'draft',definition,'bad')
    assert not (tmp_path/'draft/records'/(row['id']+'.json')).exists()

def test_frozen_file_identity_change_rejected(tmp_path):
    path=tmp_path/'fixed';path.write_bytes(b'one');cache=C.FrozenFiles();cache.bind(path,C.file_sha256(path))
    path.write_bytes(b'two altered bytes')
    with pytest.raises(ValueError):cache.stable()


def test_complete_gate_waits_for_all_runs_before_reading_identity_or_audit(tmp_path,monkeypatch):
    row,definition,state=run_fixture(tmp_path);state['jobs'][row['id']]['status']='running'
    context=dict(rows=[row],definitions={row['id']:definition})
    monkeypatch.setattr(C,'final_identity_manifest',lambda *_:pytest.fail('identity was read before full cohort'))
    assert C.complete_phase_gate(context,state,[definition],None,tmp_path) is None


@pytest.mark.parametrize('gate_kind',['draft','equal','different'])
def test_D_waits_for_final_proof_then_aliases_only_when_true(tmp_path,monkeypatch,gate_kind):
    context,proof,identities=pair_fixture(tmp_path,gate_kind!='different')
    context['definitions']={r['id']:dict(id=r['id']) for r in context['rows']}
    gate=None if gate_kind=='draft' else dict(identity_manifest=dict(records=identities),
        decisions=C.validate_final_pairs(context,proof,identities),sources=[])
    captured={}
    def draft(context,row,folder,definition,expected,fingerprint,alias,completed_record=None):
        captured[row['id']]=alias
        return dict(run_id=row['id'],group=row['group'],seed=row['seed'],model_fingerprint=dict(sha256_model_state='same',state_elements=4))
    monkeypatch.setattr(C,'run_ready',lambda *_:True);monkeypatch.setattr(C,'draft_one',draft)
    monkeypatch.setattr(C,'assemble',lambda *args:dict(status='assembled'))
    cache=type('Cache',(),{'stable':lambda self:None})()
    result=C.process_phase(context,dict(jobs={r['id']:{} for r in context['rows']}),[],cache,tmp_path,gate)
    if gate_kind=='draft':
        assert len(captured)==3 and all('_B4_' in name for name in captured) and result is None
    else:
        assert len(captured)==6 and result['status']=='assembled'
        for name,alias in captured.items():
            assert bool(alias)==('_D_' in name and gate_kind=='equal')


@pytest.mark.parametrize('mutation',['provenance','exposure','model','teacher_dtype'])
def test_per_run_source_metadata_guards_before_draft(tmp_path,monkeypatch,mutation):
    context,row,definition,saved=tiny_draft_fixture(tmp_path,monkeypatch)
    # Exercise the real source validator independently of the pipeline fixture stub.
    spec=importlib.util.spec_from_file_location('C_source_guards',Path(C.__file__))
    guards=importlib.util.module_from_spec(spec);spec.loader.exec_module(guards)
    config=dict(model_revision='revision',seed=11,replay_ratio=.05,replay_weight=.05,learning_rate=3e-5,
                run_label=row['id'],target_prediction_tokens=100)
    put(Path(row['config_path']),config);point=dict(step=1,target_prediction_tokens=100,replay_prediction_tokens=5,metrics={})
    summary=dict(status='completed',config=config,counters={k:point[k] for k in ['step','target_prediction_tokens','replay_prediction_tokens']},
        validation_trajectory=[point],best_target_validation=dict(step=1,nll=1.),provenance=dict(teacher_weight_dtype='torch.float32'))
    saved.update(config=config,counters=copy.deepcopy(summary['counters']),validation_trajectory=copy.deepcopy(summary['validation_trajectory']),
        best_target_validation=copy.deepcopy(summary['best_target_validation']),provenance=copy.deepcopy(summary['provenance']))
    context['phase']['target_prediction_tokens']=100;context['baseline']=dict(execution_provenance={})
    monkeypatch.setattr(guards,'verify_recorded_execution',lambda *_:None)
    monkeypatch.setattr(guards,'training_validation_execution',lambda *_:{})
    if mutation=='provenance':saved['provenance']['changed']=True
    elif mutation=='exposure':saved['counters']['target_prediction_tokens']=99
    elif mutation=='model':row['model']='pythia-410m'
    else:
        saved['provenance']['teacher_weight_dtype']='torch.bfloat16'
        summary['provenance']['teacher_weight_dtype']='torch.bfloat16'
    with pytest.raises(ValueError):guards.validate_source(context,row,summary,saved)
