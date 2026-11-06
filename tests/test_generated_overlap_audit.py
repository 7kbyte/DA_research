"""Audit boundary/threshold invariants on synthetic tokens; no held-out scores."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("generated_overlap_audit", Path(__file__).resolve().parents[1] / "scripts/audit_extended_generated_overlap.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_threshold_uses_100_of_500_ngrams_and_does_not_cross_chunks():
    source = np.arange(1024).reshape(2,512)
    reference = np.unique(AUDIT.ngram_hashes(source))
    for copied, count in ((111,99),(112,100)):
        query = (10_000+np.arange(512))[None,:]
        query[0,:copied] = source[0,:copied]
        hashes = AUDIT.ngram_hashes(query)
        fraction = AUDIT.matching_ngram_mask(hashes,reference).mean(axis=1)
        assert fraction[0] == count/500
        assert bool(fraction[0]>=.2) == (copied==112)
        confirmed = AUDIT.confirmed_flagged_fractions([source],query,hashes,np.array([0]),reference)
        assert confirmed[0] == count/500
    query = (20_000+np.arange(512))[None,:]
    query[0,:13] = np.arange(506,519)  # Only exists across source chunk boundary.
    assert not AUDIT.matching_ngram_mask(AUDIT.ngram_hashes(query),reference)[0,0]


def test_above_threshold_hash_candidates_require_actual_token_matches():
    source = np.arange(512)[None,:]
    query = source+100_000
    pretend_collision_hashes = AUDIT.ngram_hashes(source)
    reference = np.unique(pretend_collision_hashes)
    assert AUDIT.matching_ngram_mask(pretend_collision_hashes,reference).mean() == 1
    # Even complete rolling-hash collisions must not create confirmed overlap.
    confirmed = AUDIT.confirmed_flagged_fractions([source],query,pretend_collision_hashes,np.array([0]),reference)
    assert confirmed == {0:0.}
