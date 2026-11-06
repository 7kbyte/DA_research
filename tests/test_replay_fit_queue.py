"""CPU scientific partition invariants, independent of model scoring."""
import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from make_replay_fit_queue import validate_partitions


def fixture():
    return {"partitions":[{"seed":s,"disjoint_pool_indices":True,"exhaustive_pool_cover":True,
        "cross_partition_exact_token_duplicates":0,"corpora":{
            "bank":{"pool_indices":list(range(196)),"chunks":196,"prediction_tokens":196*511},
            "complement":{"pool_indices":list(range(196,1957)),"chunks":1761,"prediction_tokens":1761*511}}} for s in [11,22,33]]}


def test_partition_exact_cover_and_causal_budgets():
    validate_partitions(fixture())


@pytest.mark.parametrize("change",["overlap","duplicate","nonexhaustive","token_count","copy","seed"])
def test_bad_partition_or_provenance_cannot_be_queued(change):
    x=fixture();p=x["partitions"][0]
    if change=="overlap":p["corpora"]["complement"]["pool_indices"][0]=0
    elif change=="duplicate":p["corpora"]["bank"]["pool_indices"][1]=0
    elif change=="nonexhaustive":p["corpora"]["complement"]["pool_indices"][-1]=1958
    elif change=="token_count":p["corpora"]["bank"]["prediction_tokens"]+=196
    elif change=="copy":p["cross_partition_exact_token_duplicates"]=1
    else:p["seed"]=22
    with pytest.raises(ValueError):validate_partitions(x)
