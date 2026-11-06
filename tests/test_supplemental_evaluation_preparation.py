"""CPU readiness gates never fabricate missing training or validation choices."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

source=Path(__file__).resolve().parents[1]/'scripts/prepare_supplemental_evaluations.py'
spec=importlib.util.spec_from_file_location('supplemental_eval_prepare',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def write(path,value):path.write_text(json.dumps(value)+'\n');return path


def row_fixture(tmp_path):
    cp=tmp_path/'latest.pt';cp.write_bytes(b'only_test_metadata_never_torch_loaded')
    cfg=write(tmp_path/'config.json',{'target_prediction_tokens':100,'replay_ratio':.05})
    summary=write(tmp_path/'summary.json',{'status':'completed','counters':{'step':2,'target_prediction_tokens':100,'replay_prediction_tokens':5},'checkpoint_path':str(cp)})
    return {'id':'r','config_path':str(cfg),'config_sha256':module.sha(cfg),'summary_path':str(summary)},cp


@pytest.mark.parametrize('change',['running','wrong_replay','missing_checkpoint','missing_best','changed_config'])
def test_incomplete_or_changed_training_cannot_publish_scoring_plan(change,tmp_path):
    row,cp=row_fixture(tmp_path);summary=Path(row['summary_path']);value=json.loads(summary.read_text());kinds=['final']
    if change=='running':value['status']='running';write(summary,value)
    elif change=='wrong_replay':value['counters']['replay_prediction_tokens']=4;write(summary,value)
    elif change=='missing_checkpoint':cp.unlink()
    elif change=='missing_best':kinds=['final','best']
    else:Path(row['config_path']).write_text('{}')
    if change=='changed_config':
        with pytest.raises(ValueError,match='SHA'):module.completion_gate([row],kinds,module.Hashes())
    else:
        gate=module.completion_gate([row],kinds,module.Hashes());assert not gate['ready'] and gate['pending_conditions']==1


def test_pending_registry_creates_no_fake_phase_spec_or_checkpoint(tmp_path):
    row,cp=row_fixture(tmp_path);Path(row['summary_path']).unlink();registry=write(tmp_path/'registry.json',[row])
    folder=tmp_path/'must_not_be_created';plan={'phase':{'registry':module.proof(registry),'expected_rows':1},'mode':'nll','checkpoints':['final']}
    value=module.prepare_one({},plan,folder,module.Hashes())
    assert value['status']=='waiting_training' and not folder.exists() and cp.read_bytes().startswith(b'only_test')


def selection_fixture():
    groups=['B1','B2','B3','B4','C','M'];rows=[];choices=[]
    for group in groups:
        for weight in [.01,.05,.1,.5,1.,5.,10.]:
            for seed in [11,22,33]:rows.append({'id':f'{group}_{weight}_{seed}','group':group,'weight':weight,'seed':seed})
        choices.append({'group':group,'weight':.05,'condition_ids':[f'{group}_0.05_{seed}' for seed in [11,22,33]],
                        'selection_frozen':True,'shared_across_seeds':True,'test_scores_used':False,'selected_by_best_checkpoint':False})
    return rows,{'phase':'supplement_A_lambda_sweep','complete':True,'completed_conditions':126,'expected_conditions':126,
                 'selection_uses_test':False,'test_scores_accessed':False,'selections':choices,'all_candidate_scores':[{'complete':True} for _ in range(42)]}


def test_A_behavior_selection_keeps_exact_six_method_three_seed_identity():
    rows,choice=selection_fixture();assert len(module.selected_A_ids(choice,rows))==18


@pytest.mark.parametrize('change',['partial','candidate_gap','uses_test','best_choice','wrong_seed','wrong_lambda'])
def test_A_behavior_rejects_partial_grid_or_unregistered_choice(change):
    rows,s=selection_fixture()
    if change=='partial':s['completed_conditions']=125
    elif change=='candidate_gap':s['all_candidate_scores'].pop()
    elif change=='uses_test':s['selection_uses_test']=True
    elif change=='best_choice':s['selections'][0]['selected_by_best_checkpoint']=True
    elif change=='wrong_seed':rows[3]['seed']=22
    else:s['selections'][0]['weight']=.1
    with pytest.raises(ValueError):module.selected_A_ids(s,rows)


def test_closure_is_metadata_only_and_requires_explicit_original_baseline(tmp_path,monkeypatch):
    path=Path(__file__).with_name('test_evaluation_handoff.py');sp=importlib.util.spec_from_file_location('handoff_fixture_for_preparation',path);f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f)
    _,draft,_,_=f.fixture(tmp_path)
    assert [j['id'] for j in module.closure_metadata(draft,module.Hashes())]==['initial','endpoint']
    draft['prior_sources']=[]
    with pytest.raises(ValueError,match='prior dependency'):module.closure_metadata(draft,module.Hashes())


def equivalence_fixture(tmp_path,unequal=None):
    model=tmp_path/'pinned_model';model.mkdir();algorithm=tmp_path/'fingerprint.py';algorithm.write_text('# fixed synthetic algorithm identity\n')
    snapshot=write(tmp_path/'snapshot.json',{'model':'pythia-1b'});rows=[]
    for seed in (11,22,33):
        for group in ('B0','B1','B2','B3','B4','D','M','C'):
            run=f'{group}_{seed}';cfg=write(tmp_path/(run+'_config.json'),{'model_name_or_path':str(model),'model_revision':'fixed_revision','target_prediction_tokens':100,'replay_ratio':.05})
            final=tmp_path/(run+'_final.pt');final.write_bytes(('synthetic_final_'+run).encode())
            best=tmp_path/(run+'_best.pt');best.write_bytes(('synthetic_best_'+run).encode())
            trajectory=[{'step':1,'target_prediction_tokens':50,'replay_prediction_tokens':2,'metrics':{'target':{'nll':1.}}},
                        {'step':2,'target_prediction_tokens':100,'replay_prediction_tokens':5,'metrics':{'target':{'nll':2.}}}]
            summary=write(tmp_path/(run+'_summary.json'),{'status':'completed','config':module.read(cfg),'checkpoint_path':str(final),
                'counters':{'step':2,'target_prediction_tokens':100,'replay_prediction_tokens':5},'validation_trajectory':trajectory,
                'best_target_validation':{'step':1,'nll':1.,'checkpoint_path':str(best)}})
            rows.append({'id':run,'group':group,'seed':seed,'config_path':str(cfg),'config_sha256':module.sha(cfg),'summary_path':str(summary)})
    registry=write(tmp_path/'registry.json',rows)
    phase={'family':'C','expected_rows':24,'model_tag':'pythia-1b','model':{'path':str(model),'revision':'fixed_revision'},'registry':module.proof(registry),
           'source_proof_files':{},'model_equivalence_gate':{'phase':'model_formal','completion_path':str(tmp_path/'completion.json'),'audit_path':str(tmp_path/'audit.json'),
               'proof_paths':{kind:str(tmp_path/(kind+'_proof.json')) for kind in ('final','best')},'source_snapshot':module.proof(snapshot),'algorithm_source':module.proof(algorithm)}}
    audit=write(tmp_path/'audit.json',{'phase':'model_formal','model':'pythia-1b','status':'passed','expected_runs':24,'completed_runs':24,'snapshot':str(snapshot),'snapshot_sha256':module.sha(snapshot),
        'source_registry':module.proof(registry),'source_violations':[],'test_and_behavior_scores_read':False,
        'source_state_complete_gate':{'ready':True,'expected_runs':24,'completed_runs':24,'missing_runs':[],'failed_runs':[]}})
    lookup={r['id']:r for r in rows};proofs={}
    for kind in ('final','best'):
        pairs=[]
        for seed in (11,22,33):
            sides=[]
            for group in ('B4','D'):
                row=lookup[f'{group}_{seed}'];summary=module.read(row['summary_path']);cp,exposure=module.endpoint(summary,kind)
                state=seed+(100 if kind=='best' else 0)+(1 if unequal==(kind,seed) and group=='D' else 0)
                sides.append({'run_id':row['id'],'kind':kind,'model_revision':'fixed_revision','model_name_or_path':str(model),
                    'checkpoint':cp,'checkpoint_sha256':module.sha(cp),'checkpoint_exposure':exposure,
                    'actual_selection':summary['best_target_validation'] if kind=='best' else {'rule':'fixed_completed_target_exposure',**exposure},
                    'config_path':row['config_path'],'config_sha256':row['config_sha256'],'summary_path':row['summary_path'],'summary_sha256':module.sha(row['summary_path']),
                    'model_state_fingerprint':{'sha256_model_state':f'{state:064x}','state_elements':2}})
            pairs.append({'kind':'identical_bank_null_control','endpoint':kind,'seed':seed,'left':sides[0],'right':sides[1],'bitwise_model_state_equal':unequal!=(kind,seed)})
        p=write(tmp_path/(kind+'_proof.json'),{'schema_version':1,'status':'passed','phase':'model_formal','model':'pythia-1b','endpoint':kind,
            'test_scores_accessed':False,'checkpoint_files_modified':False,'algorithm':'sha256_sorted_key_dtype_shape_contiguous_uint8_v1','algorithm_source':module.proof(algorithm),
            'source_audit':module.proof(audit),'source_registry':module.proof(registry),'source_snapshot':module.proof(snapshot),'comparisons':pairs})
        proofs[kind]=module.proof(p)
    completion=write(tmp_path/'completion.json',{'schema_version':1,'status':'passed','phase':'model_formal','model':'pythia-1b',
        'source_registry':module.proof(registry),'input_snapshot':module.proof(snapshot),'audit':module.proof(audit),'proofs':proofs,'all_pairs_bitwise_equal':unequal is None,
        'source_inputs':[module.proof(algorithm)],'expected_runs':24,'completed_runs':24,'comparisons_computed':6,'test_and_behavior_scores_read':False,'gpu_work_launched':False})
    return phase,rows,completion


def refresh_equivalence_fixture(phase,completion,kind=None,audit=False):
    c=module.read(completion)
    if kind:c['proofs'][kind]=module.proof(phase['model_equivalence_gate']['proof_paths'][kind])
    if audit:c['audit']=module.proof(phase['model_equivalence_gate']['audit_path'])
    write(completion,c)


def test_C_equivalence_gate_requires_complete_audit_before_creating_any_plan(tmp_path):
    phase,rows,completion=equivalence_fixture(tmp_path);completion.unlink();folder=tmp_path/'unpublished'
    plan={'phase':phase,'mode':'behavior','checkpoints':['final']}
    result=module.prepare_one({},plan,folder,module.Hashes())
    assert result['status']=='waiting_C_model_state_audit' and not folder.exists()


def test_C_equivalence_checks_actual_six_endpoints_and_does_not_guess_unequal_pair(tmp_path):
    phase,rows,completion=equivalence_fixture(tmp_path,unequal=('best',22))
    result=module.model_equivalence_gate(phase,rows,module.Hashes())
    assert result['ready'] and not result['all_pairs_bitwise_equal']
    assert result['equal_pairs_by_endpoint']=={'final':3,'best':2}
    assert [(p['endpoint'],p['seed']) for p in result['unequal_pairs']]==[('best',22)]
    # The unchanged frozen consumer skips only the false comparison.
    from evaluation_plan import equivalence_proofs
    assert len(equivalence_proofs({'model_equivalence_proofs':result['proofs']},module.Hashes()))==10


@pytest.mark.parametrize('change',['incomplete_audit','source_violation','scheduler_incomplete','wrong_model','wrong_revision','wrong_registry','wrong_summary','wrong_checkpoint','wrong_exposure','missing_pair','wrong_endpoint','false_equality_flag','completion_flag'])
def test_C_equivalence_rejects_incomplete_or_unbound_evidence(change,tmp_path):
    phase,rows,completion=equivalence_fixture(tmp_path);gate=phase['model_equivalence_gate'];p=Path(gate['proof_paths']['best']);v=module.read(p)
    if change in ('incomplete_audit','source_violation','scheduler_incomplete'):
        a=module.read(gate['audit_path'])
        if change=='incomplete_audit':a['completed_runs']=23
        elif change=='source_violation':a['source_violations']=[{'check':'frozen_config'}]
        else:a['source_state_complete_gate']['ready']=False
        write(Path(gate['audit_path']),a);refresh_equivalence_fixture(phase,completion,audit=True)
    elif change=='wrong_model':
        c=module.read(completion);c['model']='pythia-410m';write(completion,c)
    elif change=='wrong_revision':v['comparisons'][0]['left']['model_revision']='other'
    elif change=='wrong_registry':v['source_registry']=module.proof(write(tmp_path/'other_registry.json',rows))
    elif change=='wrong_summary':v['comparisons'][0]['left']['summary_sha256']='0'*64
    elif change=='wrong_checkpoint':Path(v['comparisons'][0]['left']['checkpoint']).write_bytes(b'changed after independent proof')
    elif change=='wrong_exposure':v['comparisons'][0]['left']['checkpoint_exposure']['step']=2
    elif change=='missing_pair':v['comparisons'].pop()
    elif change=='wrong_endpoint':v['comparisons'][0]['endpoint']='final'
    elif change=='false_equality_flag':v['comparisons'][0]['bitwise_model_state_equal']=False
    else:
        c=module.read(completion);c['all_pairs_bitwise_equal']=False;write(completion,c)
    if change not in ('incomplete_audit','source_violation','scheduler_incomplete','wrong_model','completion_flag'):
        write(p,v);refresh_equivalence_fixture(phase,completion,'best')
    with pytest.raises(ValueError):module.model_equivalence_gate(phase,rows,module.Hashes())


def test_2p8b_without_D_does_not_wait_for_unrelated_model_proofs():
    assert module.model_equivalence_gate({'family':'C','model_tag':'pythia-2.8b'},[],module.Hashes())=={'ready':True,'required':False,'proofs':[],'source_proof_files':{}}
