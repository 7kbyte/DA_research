"""Synthetic accounting proofs; score files deliberately contain invalid JSON."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from replay_study.evaluation_cost_inventory import inventory_evaluations

_spec=importlib.util.spec_from_file_location('cost_inventory_reader',Path(__file__).resolve().parents[1]/'scripts/inventory_study_costs.py')
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n');return path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attest(output,manifest_sha,jid,cp,verify,scorer,**extra):
    output=Path(output);paths=[output,*[output.with_name(output.stem+'_'+task+'.jsonl') for task in ('piqa','hellaswag')]]
    for p in paths:p.write_bytes(b'PRIVATE_SCORES_deliberately_invalid_JSON')
    return write(output.with_name(output.stem+'_execution.json'),{'manifest_sha256':manifest_sha,'job_id':jid,
        'checkpoint_sha256':cp,'verified_frozen_inputs':True,'verification_wall_seconds':verify,
        'inference_subprocess_wall_seconds':scorer,'inference_alias':False,'output_sha256':{str(p):digest(p) for p in paths},**extra})


def fixture(tmp_path):
    initial=tmp_path/'initial.json';adapted=tmp_path/'adapted.json';alias=tmp_path/'D.json';reuse=tmp_path/'reuse.json'
    jobs=[{'id':'initial','kind':'initial','baseline_group':'g','output':str(initial)},
          {'id':'adapted','kind':'adapted','baseline_group':'g','output':str(adapted),'checkpoint_sha256':'a'*64}]
    group={'initial_job_id':'initial','initial_output':str(initial),'tasks':{'piqa':{},'hellaswag':{}}}
    records=[{'id':'B4','run_id':'B4','actual_run_id':'B4','output':str(adapted),'inference_job_id':'adapted','checkpoint_sha256':'a'*64},
             {'id':'D','run_id':'D','actual_run_id':'D','output':str(alias),'inference_job_id':'adapted','checkpoint_sha256':'b'*64,'source_model_equivalence':{'proof_sha':'equiv'}}]
    p=write(tmp_path/'primary.json',{'schema_version':2,'mode':'behavior','baseline_groups':{'g':group},'jobs':jobs,'records':records});psha=digest(p)
    attest(initial,psha,'initial',None,1.,10.)
    ap=attest(adapted,psha,'adapted','a'*64,2.,20.)
    alias_ap=attest(alias,psha,'adapted','b'*64,2.,0.,inference_alias=True,logical_record_id='D',canonical_output=str(adapted.resolve()),canonical_inference_subprocess_wall_seconds=20.,source_model_equivalence={'proof_sha':'equiv'})
    origin={'manifest':str(p),'manifest_sha256':psha,'job_id':'adapted','output':str(adapted)}
    baseline_origin={'manifest':str(p),'manifest_sha256':psha,'job_id':'initial','output':str(initial)}
    reusejob={'id':'reuse','kind':'reuse','baseline_group':'g','output':str(adapted),'checkpoint_sha256':'a'*64,'inference_origin':origin}
    reuse_record={'id':'supplement_alias','output':str(reuse),'inference_job_id':'reuse','checkpoint_sha256':'a'*64}
    p2=write(tmp_path/'supplement.json',{'schema_version':2,'mode':'behavior','baseline_groups':{'g':{**group,'initial_origin':baseline_origin}},'jobs':[reusejob],'records':[reuse_record]});p2sha=digest(p2)
    attest(reuse,p2sha,'reuse','a'*64,.5,0.,inference_alias=True,inference_origin=origin,logical_record_id='supplement_alias',canonical_output=str(adapted.resolve()),canonical_inference_subprocess_wall_seconds=0.)
    state=write(tmp_path/'state.json',{'jobs':{j:{'status':'completed','returncode':0,'elapsed_seconds':t} for j,t in [('initial',14.),('adapted',24.),('reuse',1.)]}})
    first={'path':str(p),'sha256':psha,'category':'primary','scheduler_state':str(state),'verify_output_hashes':True}
    second={'path':str(p2),'sha256':p2sha,'category':'supplement','scheduler_state':str(state),'verify_output_hashes':True}
    return {'evaluation_manifests':[first,dict(first),second]},p,ap,alias_ap,state


def test_duplicate_baseline_state_alias_and_prior_reuse_count_actual_inference_once(tmp_path):
    spec,*_=fixture(tmp_path);reader=_module.Reader();r=inventory_evaluations(spec,reader)
    assert len(r['evaluation_executions'])==3 and len(r['evaluation_canonical_inferences'])==2
    assert sum(x['additional_forward_jobs_observed'] for x in r['evaluation_canonical_inferences'])==2
    assert sum(t['scorer_subprocess_seconds_known_sum'] for t in r['evaluation_category_totals'])==30
    assert sum(t['SHA_and_manifest_verification_seconds_known_sum'] for t in r['evaluation_category_totals'])==3.5
    assert sum(t['known_disjoint_recorded_wall_seconds'] for t in r['evaluation_category_totals'])==33.5
    assert all(not x['result_internal_timers_parsed_or_added'] for x in r['evaluation_executions'])
    assert b'PRIVATE_SCORES' not in json.dumps(r).encode()
    aliases=[x for x in r['evaluation_references'] if x.get('alias_status')=='materialized_zero_forward']
    assert len(aliases)==2 and all(x['extra_model_forward_calls']==0 for x in aliases)
    assert all(info.get('read_purpose')=='SHA_only_no_score_deserialization' for p,info in reader.sources.items() if p.endswith('.jsonl'))


def test_missing_failed_or_external_sources_remain_unknown_instead_of_zero(tmp_path):
    spec,p,ap,alias_ap,state=fixture(tmp_path)
    # The second phase may refer to a prior baseline whose plan is omitted.
    r=inventory_evaluations({'evaluation_manifests':[spec['evaluation_manifests'][-1]]},_module.Reader())
    assert all(x['status']=='external_canonical_not_explicitly_inventoried' for x in r['evaluation_canonical_inferences'])
    ap.unlink();alias_ap.unlink();s=json.loads(state.read_text());s['jobs']['adapted']={'status':'failed','returncode':1,'elapsed_seconds':9.};write(state,s)
    r=inventory_evaluations({'evaluation_manifests':[spec['evaluation_manifests'][0],{'path':str(tmp_path/'missing.json'),'sha256':'f'*64,'category':'diagnostic'}]},_module.Reader())
    failed=next(x for x in r['evaluation_executions'] if x['inference_job_id']=='adapted')
    assert failed['status']=='failed_without_attested_timing' and failed['scorer_subprocess_wall_seconds'] is None
    assert failed['scheduler_elapsed_seconds_nonadditive']==9.
    assert r['evaluation_manifest_status'][-1]['status']=='missing_or_invalid_manifest'
    assert r['evaluation_category_totals'][0]['scorer_timer_missing_records']==1


def test_conflicting_declared_roles_keep_one_physical_cost_and_visible_category(tmp_path):
    spec,*_=fixture(tmp_path)
    spec['evaluation_manifests'].append({**spec['evaluation_manifests'][0],'category':'diagnostic'})
    result=_module.inventory(spec,tmp_path/'no_training_runs')
    conflict=next(r for r in result['category_totals'] if r['category']=='classification_conflict')
    assert conflict['evaluation_known_disjoint_recorded_wall_seconds']==33.
    assert conflict['known_training_preparation_evaluation_wall_seconds']==33.
    assert sum(r['known_disjoint_recorded_wall_seconds'] for r in result['evaluation_category_totals'])==33.5


@pytest.mark.parametrize('change',['manifest','alias_timing','output_bytes','missing_timer'])
def test_changed_sources_or_invalid_alias_proof_cannot_silently_count_cost(change,tmp_path):
    spec,p,ap,alias_ap,state=fixture(tmp_path)
    if change=='manifest':p.write_text(p.read_text()+' ')
    elif change=='output_bytes':(tmp_path/'adapted_piqa.jsonl').write_bytes(b'changed')
    elif change=='alias_timing':a=json.loads(alias_ap.read_text());a['inference_subprocess_wall_seconds']=20.;write(alias_ap,a)
    else:a=json.loads(ap.read_text());a.pop('verification_wall_seconds');write(ap,a)
    if change=='missing_timer':
        r=inventory_evaluations(spec,_module.Reader());row=next(x for x in r['evaluation_executions'] if x['inference_job_id']=='adapted')
        assert row['missing_recorded_timing_fields']==['verification_wall_seconds']
        assert r['evaluation_category_totals'][0]['verification_timer_missing_records']==1
    else:
        with pytest.raises(ValueError):inventory_evaluations(spec,_module.Reader())
