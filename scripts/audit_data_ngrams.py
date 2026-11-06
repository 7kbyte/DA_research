#!/usr/bin/env python3
"""Report fixed-threshold token 13-gram overlap without filtering by model loss."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.data import load_token_dataset


def ngram_hashes(ids: np.ndarray, n: int = 13) -> np.ndarray:
    """Order-sensitive polynomial rolling hashes modulo 2**64, within chunks.

    Hash matches are overlap diagnostics (very low, nonzero collision risk), not
    proof of semantic duplication. No ngram crosses a chunk/document boundary.
    """
    ids = np.asarray(ids, dtype=np.uint64)
    if ids.ndim != 2 or ids.shape[1] < n:
        raise ValueError("input must be a matrix of chunks at least n tokens long")
    width = ids.shape[1] - n + 1
    result = np.zeros((ids.shape[0], width), dtype=np.uint64)
    base = np.uint64(65537)
    with np.errstate(over="ignore"):
        for offset in range(n):
            result = result * base + ids[:, offset:offset + width] + np.uint64(1)
    return result


def matching_ngram_mask(hashes: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    if not len(sorted_reference):
        return np.zeros(hashes.shape, dtype=bool)
    positions = np.searchsorted(sorted_reference, hashes)
    return ((positions < len(sorted_reference)) &
            (sorted_reference[np.minimum(positions, len(sorted_reference) - 1)] == hashes))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    p.add_argument("--ngram", type=int, default=13)
    p.add_argument("--fraction-threshold", type=float, default=.2)
    args = p.parse_args()
    started = time.monotonic()
    reference_names = ["finemath_train", "c4_candidate", "finemath_validation"]
    references = {}
    for name in reference_names:
        ids = load_token_dataset(args.data_dir / f"{name}.pt")["input_ids"].numpy()
        references[name] = np.unique(ngram_hashes(ids, args.ngram))
    reports = {}
    for path in sorted(args.data_dir.glob("*.pt")):
        name = path.stem
        if name in ["finemath_train", "c4_candidate"]:
            continue
        data = load_token_dataset(path)
        hashes = ngram_hashes(data["input_ids"].numpy(), args.ngram)
        details = {}
        for reference, ref_hashes in references.items():
            if reference == name:
                continue
            matches = matching_ngram_mask(hashes, ref_hashes)
            fraction = matches.mean(axis=1)
            flagged = np.flatnonzero(fraction >= args.fraction_threshold)
            details[reference] = {"matching_ngram_occurrences": int(matches.sum()),
                                  "total_ngram_occurrences": int(matches.size),
                                  "matching_occurrence_fraction": float(matches.mean()),
                                  "chunks_at_or_above_threshold": int(len(flagged)),
                                  "chunk_fraction_max": float(fraction.max()),
                                  "flagged_chunk_indices": flagged.tolist(),
                                  "flagged_document_ids": sorted({data["doc_ids"][i] for i in flagged})}
        reports[name] = details
        print(name, {k: v["matching_occurrence_fraction"] for k, v in details.items()}, flush=True)
    result = {"method": "within-chunk order-sensitive polynomial rolling uint64 token ngram hashes",
              "ngram_tokens": args.ngram, "fixed_chunk_match_fraction_threshold": args.fraction_threshold,
              "policy": "diagnostic report only; no loss measurements or filtering", "reports": reports,
              "reference_unique_ngram_counts": {k: len(v) for k, v in references.items()},
              "seconds": time.monotonic() - started}
    (args.data_dir / "ngram_overlap_audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
