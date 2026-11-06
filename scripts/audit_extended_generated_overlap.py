#!/usr/bin/env python3
"""Audit token-only generated-pool overlap; never filter frozen inputs or read scores."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_data_ngrams import ngram_hashes, matching_ngram_mask


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_tokens(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    ids = data["input_ids"].numpy()
    if ids.ndim != 2 or ids.shape[1] != 512 or ("attention_mask" in data and not bool(data["attention_mask"].bool().all())):
        raise ValueError("audit protocol requires complete unpadded 512-token chunks")
    exact = [hashlib.sha256(np.asarray(row, dtype="<i8").tobytes()).hexdigest() for row in ids]
    return data, ids, exact


def confirmed_flagged_fractions(source_arrays, query_ids, query_hashes, hashed_flags, ref_hashes):
    """Resolve every above-threshold rolling-hash flag with actual token tuples."""
    if not len(hashed_flags):
        return {}
    wanted = np.unique(query_hashes[hashed_flags][matching_ngram_mask(query_hashes[hashed_flags], ref_hashes)])
    candidates = set()
    for ids in source_arrays:
        hashes = ngram_hashes(ids, 13)
        rows, columns = np.nonzero(np.isin(hashes, wanted))
        for row, column in zip(rows, columns):
            candidates.add(tuple(map(int, ids[row, column:column + 13])))
    width = query_ids.shape[1] - 12
    return {int(i): sum(tuple(map(int, query_ids[i, j:j + 13])) in candidates for j in range(width)) / width
            for i in hashed_flags}


def main():
    torch.set_num_threads(2)
    started = time.monotonic()
    output = ROOT / "artifacts/assets/extended_generated_heldout_overlap_audit.json"
    progress = output.with_suffix(".progress.json")
    corpora = ("finemath_validation", "finemath_test", "c4_validation", "c4_test",
               "wikitext103_validation", "wikitext103_test")
    heldouts = {}
    input_hashes = {}
    for version, directory in (("evaluation_v3", "artifacts/evaluation_v3"), ("extended_v1", "artifacts/data_extended")):
        for name in corpora:
            path = ROOT / directory / (name + ".pt")
            data, ids, exact = read_tokens(path)
            heldouts[version, name] = {"ids": ids, "hashes": ngram_hashes(ids), "exact": exact,
                                      "doc_ids": data["doc_ids"], "path": str(path)}
            input_hashes[str(path)] = sha(path)
    reports = []
    groups = {}
    for model in ("pythia-410m", "pythia-1b", "pythia-2.8b"):
        pools = []
        for seed in (101, 11, 22, 33):
            directory = "artifacts/generated" if model == "pythia-410m" else f"artifacts/extended/{model}/generated"
            path = ROOT / directory / f"pool_seed{seed}.pt"
            metadata = json.loads(path.with_suffix(".json").read_text())
            actual_sha = sha(path)
            if metadata["artifact_sha256"] != actual_sha:
                raise ValueError("generated pool SHA differs from its frozen source manifest")
            data, ids, exact = read_tokens(path)
            pools.append({"seed": seed, "ids": ids, "exact": exact, "path": str(path),
                          "revision": metadata["revision"], "role": metadata.get("experiment_role", metadata.get("config", {}).get("role"))})
            input_hashes[str(path)] = actual_sha
            input_hashes[str(path.with_suffix(".json"))] = sha(path.with_suffix(".json"))
        groups[model] = pools
    for model, pools in groups.items():
        cases = [(f"seed{p['seed']}", [p]) for p in pools] + [("union_seeds101_11_22_33", pools)]
        for label, sources in cases:
            arrays = [p["ids"] for p in sources]
            reference = np.unique(np.concatenate([ngram_hashes(ids).ravel() for ids in arrays]))
            exact_reference = set(value for p in sources for value in p["exact"])
            versions = ("evaluation_v3", "extended_v1") if model == "pythia-410m" else ("extended_v1",)
            for version in versions:
                for name in corpora:
                    query = heldouts[version, name]
                    matches = matching_ngram_mask(query["hashes"], reference)
                    fractions = matches.mean(axis=1)
                    flagged = np.flatnonzero(fractions >= .2)
                    exact_flags = [i for i, value in enumerate(query["exact"]) if value in exact_reference]
                    confirmed = confirmed_flagged_fractions(arrays, query["ids"], query["hashes"], flagged, reference)
                    confirmed_flags = [i for i, value in confirmed.items() if value >= .2]
                    row = {"model": model, "pool": label, "seeds": [p["seed"] for p in sources],
                           "roles": [p["role"] for p in sources], "revisions": sorted({p["revision"] for p in sources}),
                           "data_version": version, "heldout": name, "heldout_chunks": len(query["ids"]),
                           "reference_unique_13grams": len(reference), "exact_full_chunk_matches": len(exact_flags),
                           "exact_matching_chunk_indices": exact_flags,
                           "exact_matching_doc_ids": sorted({query["doc_ids"][i] for i in exact_flags}),
                           "ngram_fraction_max": float(fractions.max()), "matching_ngram_occurrences": int(matches.sum()),
                           "hashed_threshold_flagged_chunks": flagged.tolist(),
                           "confirmed_13gram_threshold_flagged_chunks": confirmed_flags,
                           "confirmed_flag_fractions": confirmed,
                           "flagged_doc_ids": sorted({query["doc_ids"][i] for i in confirmed_flags}),
                           "scope": "union diagnostic; no model trained on union" if len(sources) > 1 else "single generated pool",
                           "extra_scope": model == "pythia-410m" and len(sources) == 1 and sources[0]["seed"] != 101}
                    reports.append(row)
                    if exact_flags or len(flagged):
                        print(json.dumps({"overlap_flag": row}, ensure_ascii=False), flush=True)
                    write(progress, {"status": "running", "completed_comparisons": len(reports),
                                     "latest": {k: row[k] for k in ("model", "pool", "data_version", "heldout")},
                                     "test_scores_accessed": False, "model_inference_performed": False})
            print(json.dumps({"completed_pool": f"{model}/{label}", "comparisons": len(reports)}), flush=True)
    exact_total = sum(r["exact_full_chunk_matches"] for r in reports)
    confirmed_total = sum(len(r["confirmed_13gram_threshold_flagged_chunks"]) for r in reports)
    hashed_total = sum(len(r["hashed_threshold_flagged_chunks"]) for r in reports)
    result = {"audited_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "status": "overlap_detected" if exact_total or confirmed_total else ("hash_collision_warning" if hashed_total else "passed"),
              "ngram_tokens": 13, "fixed_chunk_fraction_threshold": .2,
              "direction": "heldout chunk ngram occurrences matched against generated source-pool ngrams",
              "boundary_policy": "within each unpadded full512 chunk only, no cross-chunk/document ngrams",
              "threshold_rule": "at least 100 of 500 within-chunk 13grams; any flagged chunk identifies its whole document",
              "exact_rule": "SHA256 of full canonical int64 token rows; masks verified all-valid",
              "collision_policy": "rolling uint64 hashes as original protocol; all threshold flags confirmed with exact 13-token tuples",
              "test_scores_accessed": False, "model_inference_performed": False, "frozen_inputs_modified": False,
              "comparisons": len(reports), "exact_match_comparison_events": exact_total,
              "hashed_threshold_flag_comparison_events": hashed_total,
              "confirmed_threshold_flag_comparison_events": confirmed_total,
              "global_max_ngram_fraction": max(r["ngram_fraction_max"] for r in reports),
              "input_sha256": input_hashes, "script_sha256": sha(__file__),
              "ngram_implementation_sha256": sha(ROOT / "scripts/audit_data_ngrams.py"),
              "seconds": time.monotonic() - started, "results": reports,
              "interpretation": "Data-only overlap diagnostic; zero flags do not establish semantic independence or absence from teacher pretraining. Union reports do not add experimental replicates."}
    write(output, result)
    write(progress, {"status": result["status"], "completed_comparisons": len(reports), "output": str(output),
                     "test_scores_accessed": False, "model_inference_performed": False})
    print(json.dumps({k: result[k] for k in ("status", "comparisons", "exact_match_comparison_events",
          "confirmed_threshold_flag_comparison_events", "global_max_ngram_fraction", "seconds")}))


if __name__ == "__main__":
    main()
