"""CPU invariants for the independent, already-unsealed B report reviewer."""
import importlib.util
from pathlib import Path
import pytest

SPEC=importlib.util.spec_from_file_location("b_review",Path(__file__).parents[1]/"scripts/review_supplement_B_test.py")
review=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(review)


def inventory():
    return [{"bank_chunks":str(k),"group":g,"selection_seed":str(d),"training_seed":"11","generation_seed":"11",
             "checkpoint":"final","target_prediction_tokens":"10000000","replay_prediction_tokens":"500000","weight":".05","ratio":".05"}
            for k in (20,196) for g in ("B4","M","C") for d in range(1001,1011)]


def test_complete_draw_grid_is_required():
    review.validate_inventory(inventory())
    with pytest.raises(ValueError): review.validate_inventory(inventory()[:-1])
    values=inventory();values[-1]=values[0]
    with pytest.raises(ValueError): review.validate_inventory(values)


@pytest.mark.parametrize("field,value",[("training_seed","22"),("generation_seed","22"),("target_prediction_tokens","9999999"),("weight","5"),("checkpoint","best")])
def test_changed_scientific_identity_rejected(field,value):
    values=inventory();values[0][field]=value
    with pytest.raises(ValueError): review.validate_inventory(values)


def test_token_and_document_estimands_and_G_sign_differ():
    def corpus(a,b):
        chunks=[{"chunk_index":0,"doc_id":"short","sum_nll":a,"valid_prediction_tokens":1},
                {"chunk_index":1,"doc_id":"long","sum_nll":b,"valid_prediction_tokens":9}]
        return {"chunks":chunks,"sum_nll":a+b,"valid_prediction_tokens":10}
    base={c:corpus(1,9) for c in ("target","c4","wikitext")}
    adapted={c:corpus(0,18) for c in base}
    m=review.raw_metrics(adapted,base)
    assert m["g_new"]==pytest.approx(-.8)
    assert m["g_new_equal_document"]==pytest.approx(0)
    assert m["f_old_c4"]==pytest.approx(.8)
    assert m["f_old_c4_equal_document"]==pytest.approx(0)


def test_paired_G_bootstrap_reverses_both_interval_endpoints():
    raw={"token_weighted":{"estimate":2.,"ci_low":1.,"ci_high":4.}}
    assert review.bootstrap_metric(raw,"g_new","token_weighted",True)=={"estimate":-2.,"ci_low":-4.,"ci_high":-1.}
    assert review.bootstrap_metric(raw,"g_new","token_weighted",False)==raw["token_weighted"]


def test_sample_SD_is_not_SD_subtraction_or_SEM():
    assert review.summary([1.,3.])["std_ddof1"]==pytest.approx(2**.5)
    assert review.direction("g_new",.1)==1
    assert review.direction("f_old_c4",.1)==-1
    assert review.direction("f_old_wikitext_equal_document",0.)==0


def test_doc_identity_or_token_count_mismatch_rejected():
    c={"chunks":[{"chunk_index":0,"doc_id":"a","sum_nll":1.,"valid_prediction_tokens":1}],"sum_nll":1.,"valid_prediction_tokens":1}
    b={**c,"chunks":[{**c["chunks"][0],"doc_id":"b"}]}
    with pytest.raises(AssertionError): review.raw_metrics({k:c for k in ("target","c4","wikitext")},{k:b for k in ("target","c4","wikitext")})


def weighting_module():
    spec=importlib.util.spec_from_file_location("b_doc_weights",Path(__file__).parents[1]/"scripts/describe_supplement_B_document_weights.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_explanatory_decomposition_reconstructs_distinct_estimands():
    module=weighting_module();base={"short":(1.,511),"long":(1.,2044)}
    result=module.decomposition(base,{(20,"B4"):[{"short":(0.,511),"long":(2.,2044)}]*10})
    assert sum(r["contribution_to_equal_document_G"] for r in result)==pytest.approx(0.)
    assert sum(r["contribution_to_token_G"] for r in result)==pytest.approx(-.6)
    assert {r["evaluation_chunks_per_doc"] for r in result}=={1,4}


def test_explanatory_decomposition_does_not_drop_missing_documents():
    with pytest.raises(ValueError): weighting_module().decomposition({"a":(1.,511)},{(20,"B4"):[{}]})


def test_v2_preserves_all_arithmetic_functions():
    import ast
    root=Path(__file__).parents[1]/"scripts"
    def functions(file):
        return {n.name:ast.dump(n) for n in ast.parse((root/file).read_text()).body if isinstance(n,ast.FunctionDef) and n.name!="plot"}
    assert functions("review_supplement_B_test.py")==functions("review_supplement_B_test_v2.py")


def test_v2_figures_share_each_metric_y_range_and_explain_labels(tmp_path,monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    spec=importlib.util.spec_from_file_location("b_review_v2",Path(__file__).parents[1]/"scripts/review_supplement_B_test_v2.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    endpoints=[];pairs=[]
    for k in (20,196):
        for g in ("B4","M","C"):
            for d in range(1001,1011):
                endpoints.append({"bank_chunks":str(k),"group":g,"selection_seed":str(d),**{m:(d-1000)*(1 if k==20 else .1) for m in module.METRICS}})
        for comparison in ("M-B4","C-B4"):
            for d in range(1001,1011):pairs.append({"bank_chunks":str(k),"comparison":comparison,"selection_seed":str(d),**{m:(d-1005)*(1 if k==20 else .1) for m in module.METRICS}})
    inspected=[]
    def save(fig,*args,**kwargs):
        for row in range(3):assert fig.axes[row*2].get_ylim()==fig.axes[row*2+1].get_ylim()
        inspected.append([t.get_text() for t in fig.texts])
    monkeypatch.setattr(Figure,"savefig",save)
    module.plot(endpoints,pairs,tmp_path)
    assert len(inspected)==6
    assert any("01-10 = selection seeds 1001-1010" in text for group in inspected for text in group)
