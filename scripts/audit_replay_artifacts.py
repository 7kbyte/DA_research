#!/usr/bin/env python3
"""Verify replay provenance, exact bank budgets, and shared-pool invariants."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.evaluate import load_tensor_data, prediction_mask, write_json
from replay_study.selection import chunk_hashes


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/generation_summary.json")
    parser.add_argument("--output", default="results/replay_artifact_audit.json")
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text())
    if not summary.get("complete"):
        raise ValueError("generation pipeline has not completed")
    results = {"passed": True, "pools": {}, "assertions": [
        "artifact SHA256 matches", "all generated chunks start with BOS and have fixed length",
        "teacher statistics refer to exact source pool hash", "bank tokens equal indexed source chunks",
        "every bank has identical valid prediction-token budget", "deduplicated banks have unique chunks",
        "random and dedup-random coincide when pool has no duplicate chunks",
        "NLL strata are proportionally represented", "no EOS stopping or generated padding"]}
    for seed, record in summary["pools"].items():
        pool_path = Path(record["generation"]["config"]["output"])
        generation = json.loads(pool_path.with_suffix(".json").read_text())
        assert digest(pool_path) == generation["artifact_sha256"]
        assert generation["eos_stopping_disabled_in_both_explicit_and_model_generation_config"]
        assert generation["pad_token_id"] is None
        pool = load_tensor_data(pool_path)
        ids = pool["input_ids"]
        assert (ids[:, 0] == generation["bos_token_id"]).all()
        assert ids.shape == (generation["chunks"], generation["sequence_length"])
        stats_path = pool_path.with_name(pool_path.stem + "_stats.pt")
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        assert stats["source_pool_sha256"] == digest(pool_path)
        assert len(stats["features"]) == len(ids) == len(stats["teacher_nll"])
        assert (stats["prediction_token_counts"] == ids.shape[1] - 1).all()
        pool_unique = len(set(chunk_hashes(pool)))
        bank_results, budgets = {}, set()
        for name, bank_record in record["banks"].items():
            path = Path(bank_record["artifact"])
            manifest = json.loads(path.with_suffix(".json").read_text())
            assert digest(path) == manifest["artifact_sha256"]
            bank = load_tensor_data(path)
            if name == "c4_random":
                source = load_tensor_data(manifest["source_pool"])
                assert digest(manifest["source_pool"]) == manifest["source_pool_sha256"]
            else:
                source = pool
                assert manifest["source_pool_sha256"] == digest(pool_path)
            selected = torch.tensor(manifest["selected_indices"])
            assert torch.equal(bank["selection_indices"], selected)
            assert torch.equal(bank["input_ids"], source["input_ids"][selected])
            valid = prediction_mask(bank.get("attention_mask", torch.ones_like(bank["input_ids"]))).sum().item()
            assert valid == manifest["prediction_tokens"]
            assert bank["input_ids"].numel() == manifest["raw_input_tokens"]
            unique_bank = len(set(chunk_hashes(bank)))
            assert unique_bank == manifest["bank_unique_chunks"]
            if name not in ("random", "c4_random"):
                assert unique_bank == len(selected)
            if name == "nll_stratified":
                # Disjoint equal-frequency strata retain stable original-index ties.
                unique_candidates = []
                seen = set()
                for index, value in enumerate(chunk_hashes(pool)):
                    if value not in seen:
                        unique_candidates.append(index)
                        seen.add(value)
                ordered = sorted(unique_candidates, key=lambda i: (stats["teacher_nll"][i].item(), i))
                groups = torch.tensor_split(torch.tensor(ordered), manifest["n_strata"])
                observed = [sum(i in set(group.tolist()) for i in selected.tolist()) for group in groups]
                assert observed == manifest["stratum_bank_sizes"]
            bank_results[name] = {"prediction_tokens": valid, "raw_input_tokens": bank["input_ids"].numel(),
                                  "unique_chunks": unique_bank, "sha256": manifest["artifact_sha256"]}
            budgets.add(valid)
        assert len(budgets) == 1
        if pool_unique == len(ids):
            assert record["banks"]["random"]["selected_indices"] == record["banks"]["dedup_random"]["selected_indices"]
        eos = generation["eos_token_id"]
        after_eos = int(((ids[:, 1:-1] == eos) & (ids[:, 2:] != eos)).sum()) if eos is not None else None
        results["pools"][seed] = {"role": record["role"], "chunks": len(ids), "unique_chunks": pool_unique,
                                  "post_eos_non_eos_transitions": after_eos, "banks": bank_results}
    write_json(args.output, results)
    print(json.dumps({"passed": True, "pools": list(results["pools"]), "output": args.output}))


if __name__ == "__main__":
    main()
