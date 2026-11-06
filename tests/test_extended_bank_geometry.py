"""Hand-computed fixed-pool quota/CDF and exact subset invariants."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
S=importlib.util.spec_from_file_location('extended_geometry',Path(__file__).parents[1]/'scripts/analyze_extended_bank_geometry.py')
A=importlib.util.module_from_spec(S);S.loader.exec_module(A)


def test_integer_quota_expected_bias_is_exact_fixed_pool_not_performance():
    x=A.quota_expectation([0,1,2,3,4],[3,2],[1,1])
    assert x['quota_weight_total_variation']==pytest.approx(.1)
    assert x['quota_expected_cdf_max_distance']==pytest.approx(.1)
    assert x['quota_expected_bank_nll_mean']==pytest.approx(2.25)
    assert x['quota_expected_bank_nll_bias']==pytest.approx(.25)


def test_proportional_integer_weights_recover_exact_pool_expectation():
    x=A.quota_expectation([0,1,2,3,4],[3,2],[3,2])
    assert x['quota_expected_cdf_max_distance']==pytest.approx(0)
    assert x['quota_expected_bank_nll_bias']==pytest.approx(0)


def test_ties_can_make_cdf_bias_smaller_than_stratum_weight_tv():
    x=A.quota_expectation([1,1,1,1,1],[3,2],[1,1])
    assert x['quota_weight_total_variation']==pytest.approx(.1)
    assert x['quota_expected_cdf_max_distance']==pytest.approx(0)
    assert x['quota_expected_bank_nll_bias']==pytest.approx(0)


def test_nearest_feature_and_cdf_hand_calculation():
    x=A.geometry(np.array([0.,1.,2.]),np.eye(3),np.array([0,1]))
    assert x['mean_nearest_normalized_feature_squared_distance']==pytest.approx(2/3)
    assert x['empirical_cdf_max_distance']==pytest.approx(1/3)
    assert x['nll_wasserstein1']==pytest.approx(.5)
    assert x['teacher_nll_bank_sd']==pytest.approx(np.sqrt(.5))


@pytest.mark.parametrize('mutation',['tokens','doc_ids','duplicate_indices'])
def test_bank_must_be_exact_ordered_pool_subset(mutation):
    pool=dict(input_ids=torch.arange(12).reshape(3,4),doc_ids=['a','b','c'])
    bank=dict(input_ids=pool['input_ids'][[2,0]].clone(),doc_ids=['c','a'],selection_indices=torch.tensor([2,0]))
    manifest=dict(selected_indices=[2,0],bank_chunks=2)
    assert A.validate_bank(pool,bank,manifest).tolist()==[2,0]
    if mutation=='tokens':bank['input_ids'][0,0]+=1
    elif mutation=='doc_ids':bank['doc_ids']=['a','c']
    else:manifest['selected_indices']=[2,2]
    with pytest.raises(ValueError):A.validate_bank(pool,bank,manifest)
