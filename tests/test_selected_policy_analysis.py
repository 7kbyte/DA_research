"""Synthetic-only selected-policy joins, paired units and uncertainty signs."""
import copy
from pathlib import Path

import pytest

from replay_study import selected_policy_analysis as aux
from replay_study import supplemental_analysis_guard as gate
from replay_study.analysis import METRICS,aggregate_endpoints


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(gate.encoded(value));return path


def grid(tmp_path):
    rows=[];sources={};weights=(.01,.05,.1,.5,1.,5.,10.)
    chosen_weights=dict(zip(aux.GROUPS,weights));candidate=[];selected=[]
    for group in aux.GROUPS:
        for weight in weights:
            ids=[]
            for seed in (11,22,33):
                name=f'{group}_w{weight}_s{seed}';ids.append(name)
                cfg=write(tmp_path/'configs'/(name+'.json'),{'group':group,'seed':seed,'weight':weight})
                summary=write(tmp_path/'summaries'/(name+'.json'),{'synthetic_validation_only':True,'completed':True})
                row=dict(id=name,actual_run_id='physical_'+name,group=group,weight=weight,ratio=.05,learning_rate=1e-5,seed=seed,
                    config_path=str(cfg),config_sha256=gate.sha(cfg),summary_path=str(summary))
                rows.append(row);sources.update({str(cfg):gate.sha(cfg),str(summary):gate.sha(summary)})
            c=dict(group=group,weight=weight,condition_ids=ids,actual_run_ids=['physical_'+i for i in ids],n=3,expected_n=3,complete=True,
                source='validation_final_fixed_exposure_only')
            candidate.append(c)
            if weight==chosen_weights[group]:selected.append({**c,'selection_frozen':True,'shared_across_seeds':True,'test_scores_used':False,'selected_by_best_checkpoint':False})
    registry=write(tmp_path/'registry.json',rows);sources[str(registry)]=gate.sha(registry)
    selection=write(tmp_path/'selection.json',{'phase':'supplement_A_lambda_sweep','complete':True,'completed_conditions':126,'expected_conditions':126,
        'selection_uses_test':False,'test_scores_accessed':False,'all_candidate_scores':candidate,'selections':selected,'sources_sha256':sources})
    return registry,selection,rows,chosen_weights


def bound_grid(tmp_path):
    registry,selection,rows,weights=grid(tmp_path)
    return aux.bind_selection(gate.proof(selection),gate.proof(registry),gate.Verifier()),rows,weights


def nll_bundle(rows):
    result=[];corpora={}
    for row in rows:
        i=aux.GROUPS.index(row['group'])+1;s=row['seed']/11
        value=i*.01*s+row['weight']*.001
        r={**row,'run_id':row['id'],'training_seed':row['seed'],'checkpoint':'final','checkpoint_sha256':row['id']+'_sha',
            'model_tag':'pythia-410m','data_family':'main_v3','baseline_group':'base','target_prediction_tokens':10000000,
            'replay_prediction_tokens':500000,'full_training_committed_path':{'target_prediction_tokens':10000000,'training_seconds':999},
            **{m:value for m in METRICS}}
        r['g_new']=-value;r['g_new_equal_document']=-value
        result.append(r);corpora[r['run_id']]={}
        for name in ('target','c4','wikitext'):
            chunks=[{'doc_id':'d0','chunk_index':0,'valid_prediction_tokens':2,'sum_nll':value*2,'sum_kl':value*2},
                {'doc_id':'d0','chunk_index':1,'valid_prediction_tokens':3,'sum_nll':value*3,'sum_kl':value*3},
                {'doc_id':'d1','chunk_index':2,'valid_prediction_tokens':1,'sum_nll':value,'sum_kl':value}]
            corpora[r['run_id']][name]={'chunks':chunks}
    return {'complete':True,'endpoints':result,'endpoint_statistics':aggregate_endpoints(result,rows)},corpora


def test_complete_grid_exact_ID_join_preserves_six_different_lambdas_and_sources(tmp_path):
    selection,rows,weights=bound_grid(tmp_path);bundle,corpora=nll_bundle(rows)
    result=aux.nll_selected(bundle,selection,lambda r:corpora[r['run_id']],n_bootstrap=30)
    assert len(result['selected_endpoints'])==18 and len(result['selected_policy_pairs'])==18
    assert {r['weight'] for r in result['selected_endpoints']}==set(weights.values())
    assert all(r['weight']==weights[r['group']] for r in result['selected_endpoints'])
    assert all(r['target_prediction_tokens']==r['full_training_committed_path']['target_prediction_tokens']==10000000 for r in result['selected_endpoints'])
    assert all(r['actual_run_id'].startswith('physical_') and r['validation_grid_registry_sha256']==selection['source_registry']['sha256'] for r in result['selected_endpoints'])
    assert len(result['lambda_curve_statistics'])==42*len(METRICS)
    assert all(r['n']==3 and r['complete'] for r in result['lambda_curve_statistics'])
    assert all(r['n']==3 and r['replicate_unit']=='training_seed' for r in result['selected_policy_statistics'])
    pair=next(r for r in result['selected_policy_pairs'] if r['comparison']=='M-B4' and r['training_seed']==11)
    assert pair['lambda_a']==weights['M'] and pair['lambda_b']==weights['B4'] and pair['lambda_a']!=pair['lambda_b']
    ci=next(r for r in result['selected_document_bootstrap'] if r['comparison']=='M-B4' and r['training_seed']==11 and r['corpus']=='target' and r['field']=='sum_nll')
    assert ci['token_weighted']['estimate']==pytest.approx(-pair['g_new'])
    assert ci['documents']==2 and ci['prediction_tokens']==6 and ci['bootstrap_seed']==20261106
    assert result['metadata']['additional_model_forward_calls']==0 and not result['metadata']['selection_recomputed']


@pytest.mark.parametrize('change',['missing_summary','changed_config','test_selected','incomplete_grid','duplicate_candidate','wrong_ID','wrong_actual_ID'])
def test_validation_selected_join_rejects_incomplete_or_changed_science(tmp_path,change):
    registry,sp,rows,_=grid(tmp_path);selection=gate.read(sp)
    if change=='missing_summary':Path(rows[-1]['summary_path']).unlink()
    elif change=='changed_config':Path(rows[-1]['config_path']).write_text('{}')
    elif change=='test_selected':selection['selection_uses_test']=True
    elif change=='incomplete_grid':selection['completed_conditions']=125
    elif change=='duplicate_candidate':selection['all_candidate_scores'][-1]=selection['all_candidate_scores'][0]
    elif change=='wrong_ID':selection['selections'][0]['condition_ids'][0]=selection['selections'][1]['condition_ids'][0]
    else:selection['selections'][0]['actual_run_ids'][0]='wrong'
    write(sp,selection)
    with pytest.raises((ValueError,FileNotFoundError)):aux.bind_selection(gate.proof(sp),gate.proof(registry),gate.Verifier())


def test_NLL_pairing_rejects_changed_document_identity_and_duplicate_selected_rows(tmp_path):
    selection,rows,_=bound_grid(tmp_path);bundle,corpora=nll_bundle(rows)
    chosen=next(r for r in bundle['endpoints'] if r['run_id'] in selection['selected_ids'] and r['group']=='M')
    corpora[chosen['run_id']]['target']['chunks'][0]['doc_id']='different'
    with pytest.raises(ValueError,match='identical'):aux.nll_selected(bundle,selection,lambda r:corpora[r['run_id']],n_bootstrap=10)
    subset=aux.selected_rows(bundle['endpoints'],selection)
    with pytest.raises(ValueError,match='Duplicate'):aux.selected_rows(subset+[subset[0]],selection)
    bundle['endpoints'].pop()
    with pytest.raises(ValueError,match='Full126'):aux.nll_selected(bundle,selection,lambda r:None,n_bootstrap=0)


def behavior_bundle(selection,rows):
    result=[];pred={}
    for row in rows:
        if row['id'] not in selection['selected_ids']:continue
        i=aux.GROUPS.index(row['group']);s=row['seed']//11
        for task in ('piqa','hellaswag'):
            items=[{'task':task,'item_id':str(j),'fingerprint':task+str(j),'gold':0,
                'correct_acc':int((i+j+s)%3==0),'correct_acc_norm':int((i+2*j+s)%4<2)} for j in range(5)]
            pred[row['id'],task]=items
            for metric in ('acc_norm','acc'):
                initial=[1,1,0,0,0];after=[p['correct_'+metric] for p in items]
                counts={k:0 for k in ('CC','CW','WC','WW')}
                for a,b in zip(initial,after):counts[('C' if a else 'W')+('C' if b else 'W')]+=1
                result.append({**row,'run_id':row['id'],'training_seed':row['seed'],'checkpoint_kind':'final','checkpoint_sha256':row['id']+'_sha',
                    'model_tag':'pythia-410m','data_family':'main_v3','baseline_group':'base','target_prediction_tokens_at_checkpoint':10000000,
                    'replay_prediction_tokens_at_checkpoint':500000,'task':task,'metric':metric,'accuracy':sum(after)/5,**counts})
    return {'endpoints':result},pred


def test_behavior_paired_questions_keep_both_scores_directions_and_three_training_seeds(tmp_path):
    selection,rows,_=bound_grid(tmp_path);bundle,pred=behavior_bundle(selection,rows)
    load=lambda r:pred[r['run_id'],r['task']]
    a=aux.behavior_selected(bundle,selection,load,n_bootstrap=120);b=aux.behavior_selected(bundle,selection,load,n_bootstrap=120)
    assert a==b and len(a['selected_endpoints'])==72 and len(a['selected_policy_pairs'])==72
    assert all(r['n']==3 for r in a['selected_policy_statistics'])
    assert all(sum(r[k] for k in ('CC','CW','WC','WW'))==5 for r in a['selected_endpoints'])
    assert len({r['bootstrap_seed'] for r in a['selected_question_bootstrap']})==72
    for r in a['selected_question_bootstrap']:
        assert (r['right_wrong_left_correct']-r['right_correct_left_wrong'])/5==pytest.approx(r['accuracy_left_minus_right'])
        assert 'not initial' in r['transition_direction'] and 'exchangeable' in r['question_assumption']
    assert {r['metric'] for r in a['selected_policy_pairs']}=={'acc','acc_norm'}
    compare={(r['comparison'],r['training_seed'],r['task']):{} for r in a['selected_policy_pairs']}
    for r in a['selected_policy_pairs']:compare[r['comparison'],r['training_seed'],r['task']][r['metric']]=r['accuracy_left_minus_right']
    assert any(r['acc']!=r['acc_norm'] for r in compare.values())
    bad=copy.deepcopy(pred);key=next(iter(bad));bad[key][0]['fingerprint']='changed'
    with pytest.raises(ValueError,match='fingerprint'):aux.behavior_selected(bundle,selection,lambda r:bad[r['run_id'],r['task']],n_bootstrap=10)


def test_seven_lambda_curves_render_all_methods_without_filtering_selected_lambda(tmp_path):
    selection,rows,_=bound_grid(tmp_path);bundle,_=nll_bundle(rows)
    aux.plot_lambda_curves(bundle['endpoint_statistics'],tmp_path/'figures',bundle['endpoints'])
    assert all((tmp_path/'figures'/('full_seven_lambda_curves.'+ext)).stat().st_size>100 for ext in ('png','pdf','svg'))
    from PIL import Image
    assert Image.open(tmp_path/'figures/full_seven_lambda_curves.png').size==(1400,1320)
    assert (tmp_path/'figures/full_seven_lambda_endpoints.csv').is_file()
    bad=[r for r in bundle['endpoint_statistics'] if not (r['group']=='B4' and r['weight']==10.)]
    with pytest.raises(ValueError,match='Seven complete'):aux.plot_lambda_curves(bad,tmp_path/'bad',bundle['endpoints'])
