"""Tiny CPU algebra and complete-grid checks, without real experiment reads."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('C_independent_review',Path(__file__).parents[1]/'scripts/review_C_nll_results.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def corpus(a,b):
    return {'complete_corpus':True,'sum_nll':a+b,'valid_prediction_tokens':10,'chunks':[
        {'doc_id':'short','chunk_index':0,'sum_nll':a,'valid_prediction_tokens':1},
        {'doc_id':'long','chunk_index':1,'sum_nll':b,'valid_prediction_tokens':9}]}

def test_independent_token_document_estimands_and_target_sign():
    initial={c:corpus(1.,9.) for c in ('target','c4','wikitext')}
    current={c:corpus(0.,18.) for c in initial}
    r=m.raw_metrics(current,initial)
    assert r['g_new']==pytest.approx(-.8)
    assert r['g_new_equal_document']==pytest.approx(0)
    assert r['f_old_c4']==pytest.approx(.8)
    assert r['f_old_wikitext_equal_document']==pytest.approx(0)

def test_bootstrap_multiplicity_and_G_endpoint_exchange_manual_case():
    names=['a','b'];counts=np.array([1.,3.]);delta=np.array([2.,8.])
    context=m.Resampling(names,counts)
    raw=context.interval(names,counts,delta,1);gain=context.interval(names,counts,delta,-1)
    assert raw['token_weighted']['estimate']==2.5
    assert raw['token_weighted']['ci_low']==2.
    assert raw['token_weighted']['ci_high']==pytest.approx(8/3)
    assert gain['token_weighted']=={'estimate':-2.5,'ci_low':pytest.approx(-8/3),'ci_high':-2.}
    assert gain['equal_document']['estimate']==pytest.approx(-7/3)
    with pytest.raises(ValueError):context.interval(['b','a'],counts,delta)

@pytest.mark.parametrize('change',['document','count','duplicate','partial'])
def test_incomplete_or_unpaired_chunk_identity_is_rejected(change):
    a=corpus(1.,9.);b=corpus(0.,18.)
    if change=='document':b['chunks'][0]['doc_id']='foreign'
    elif change=='count':b['chunks'][0]['valid_prediction_tokens']=2;b['valid_prediction_tokens']=11
    elif change=='duplicate':b['chunks'][1].update(doc_id='short',chunk_index=0)
    else:b['complete_corpus']=False
    with pytest.raises(ValueError):m.delta_vectors(a,b)

def grid():return [{'checkpoint':k,'group':g,'training_seed':str(s)} for k in ('final','best') for g in m.GROUPS for s in m.SEEDS]

@pytest.mark.parametrize('change',['missing_best','duplicate','foreign_seed'])
def test_both_whole_model_endpoints_required(change):
    r=grid();m.validate_grid(r,m.GROUPS)
    if change=='missing_best':r.pop()
    elif change=='duplicate':r[-1]=r[0]
    else:r[-1]['training_seed']='44'
    with pytest.raises(ValueError):m.validate_grid(r,m.GROUPS)

def test_seed_SD_and_metric_directions():
    assert m.stats([1.,2.,3.])=={'mean':2.,'std_ddof1':1.,'n':3}
    assert m.favorable('g_new',.1)==1
    assert m.favorable('f_old_c4_equal_document',.1)==-1
    assert m.favorable('g_new',0)==0
