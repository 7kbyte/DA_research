import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location("replay_fit_prepare", Path(__file__).resolve().parents[1] / "scripts/prepare_replay_fit_diagnostic.py")
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


def test_partition_preserves_unsorted_bank_and_exhaustive_complement():
    pool = dict(input_ids=torch.arange(5 * 512).reshape(5, 512), doc_ids=list("abcde"))
    bank = dict(input_ids=pool["input_ids"][[3, 1]].clone(), doc_ids=["d", "b"], selection_indices=torch.tensor([3, 1]))
    copied, complement = PREP.partition(pool, bank)
    assert copied["pool_indices"].tolist() == [3, 1]
    assert complement["pool_indices"].tolist() == [0, 2, 4]
    assert copied["doc_ids"] == ["d", "b"]
    assert complement["doc_ids"] == ["a", "c", "e"]
    copied["input_ids"][0, 0] = -1
    assert bank["input_ids"][0, 0] != -1
    bank["doc_ids"][0] = "wrong"
    with pytest.raises(ValueError, match="doc IDs"):
        PREP.partition(pool, bank)


def test_partition_rejects_duplicate_tokens_even_at_disjoint_indices():
    pool = dict(input_ids=torch.ones(3, 512, dtype=torch.long), doc_ids=list("abc"))
    bank = dict(input_ids=pool["input_ids"][[1]], doc_ids=["b"], selection_indices=torch.tensor([1]))
    with pytest.raises(ValueError, match="duplicate crosses"):
        PREP.partition(pool, bank)
