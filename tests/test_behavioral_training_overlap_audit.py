"""Hand-calculated lexical audit invariants, using no behavioral scores."""
import importlib.util
import json
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("behavioral_overlap_audit", Path(__file__).resolve().parents[1] / "scripts/audit_behavioral_training_overlap.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_short_inputs_exact_threshold_item_dedup_and_source_boundaries():
    source = np.arange(1024).reshape(2, 512)
    arrays = [np.arange(12), np.arange(13), np.r_[np.arange(13), 10000 + np.arange(4)],
              np.r_[np.arange(13), 20000 + np.arange(5)], np.arange(506, 519)]
    queries, hashes, offsets = [], [], [0]
    for i, ids in enumerate(arrays):
        h = AUDIT.ngram_hashes(ids[None, :])[0] if len(ids) >= 13 else np.empty(0, dtype=np.uint64)
        queries.append(dict(task="synthetic", item_id="shared-item" if i in (1, 2) else str(i), choice_index=i,
                            ids=ids, tokens=len(ids), ngram_occurrences=len(h)))
        hashes.append(h)
        offsets.append(offsets[-1] + len(h))
    report = AUDIT.audit_reference(queries, np.concatenate(hashes), np.array(offsets), source, ["a", "b"])
    # 13/17-token inputs match respectively 1/1 and 1/5; 18 tokens gives 1/6.
    assert [q["choice_index"] for q in report["confirmed_flagged_choices"]] == [1, 2]
    assert [q["confirmed_fraction"] for q in report["confirmed_flagged_choices"]] == [1., .2]
    assert report["by_task"]["synthetic"]["confirmed_flagged_items"] == 1
    assert report["by_task"]["synthetic"]["choices_shorter_than_13_inapplicable"] == 1
    assert report["confirmed_flagged_choices"][0]["entire_sequence_occurrences"] == 1
    assert report["confirmed_flagged_choices"][1]["entire_sequence_occurrences"] == 0
    assert report["confirmed_13gram_source_witnesses"][0]["tokens"] == list(range(13))
    json.dumps(report, allow_nan=False)


def test_prepare_choice_audit_includes_final_predicted_token():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]
    examples = [dict(task="toy", item_id="toy:0", context="abc", choices=["de"])]
    queries, _, _, _ = AUDIT.encode_queries(Tokenizer(), examples)
    assert queries[0]["ids"].tolist() == list(map(ord, "abc de"))
    assert queries[0]["tokens"] == 6


def test_target_and_replay_union_can_cross_threshold_without_individual_flag():
    query = np.arange(112)
    sources = [(10000 + np.arange(512))[None, :], (20000 + np.arange(512))[None, :]]
    sources[0][0, :22] = query[:22]
    sources[1][0, :22] = query[-22:]
    queries = [dict(task="toy", item_id="toy:0", choice_index=0, ids=query, tokens=112)]
    hashes = AUDIT.ngram_hashes(query[None, :])[0]
    masks = []
    for source in sources:
        report = AUDIT.audit_reference(queries, hashes, np.array([0, 100]), source, ["doc"], matched_masks=masks)
        assert report["confirmed_flagged_choices"] == []
    assert [int(mask.sum()) for mask in masks] == [10, 10]
    assert int(np.logical_or.reduce(masks).sum()) == 20
    combined = AUDIT.audit_reference(queries, hashes, np.array([0, 100]), np.concatenate(sources), ["target", "replay"])
    assert combined["confirmed_flagged_choices"][0]["confirmed_fraction"] == .2


def test_a_hash_collision_flag_does_not_become_confirmed_overlap():
    source = np.arange(512)[None, :]
    ids = 100000 + np.arange(112)
    queries = [dict(task="toy", item_id="toy:0", choice_index=0, ids=ids, tokens=112)]
    collision_hashes = AUDIT.ngram_hashes(source[:, :112])[0]
    report = AUDIT.audit_reference(queries, collision_hashes, np.array([0, 100]), source, ["source"])
    assert report["confirmed_flagged_choices"] == []
    assert report["hash_collision_rejections"] == [{"query_index": 0, "confirmed_matches": 0, "hashed_matches": 100}]
