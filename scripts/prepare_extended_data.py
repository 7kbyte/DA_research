#!/usr/bin/env python3
"""Optional 20M/10M coverage expansion, isolated from all primary artifacts."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from transformers import AutoTokenizer

from prepare_data import Builder, parquet_documents, c4_documents, round_robin
from prepare_evaluation_v3 import primary_source_documents, reuse_wikitext, ngram_reference
from download_assets import FINEMATH_REVISION, C4_REVISION
from replay_study.data import hash_partition


def comparisons(output, reference):
    result = {}
    for path in sorted(output.glob("*.pt")):
        old_path = reference / path.with_suffix(".json").name
        if not old_path.exists():
            continue
        new = json.loads(path.with_suffix(".json").read_text())
        old = json.loads(old_path.read_text())
        a, b = set(new["chunk_sha256"]), set(old["chunk_sha256"])
        x = {d["sha256"] for d in new["documents"]}
        y = {d["sha256"] for d in old["documents"]}
        weights = [d["chunks_used"] for d in new["documents"]]
        result[path.stem] = {"old_chunks": len(b), "extended_chunks": len(a), "shared_exact_chunks": len(a & b),
            "all_old_chunks_in_extended": b <= a, "old_documents": len(y), "extended_documents": len(x),
            "shared_normalized_documents": len(x & y), "old_sha256": old["sha256"], "extended_sha256": new["sha256"],
            "maximum_document_chunk_weight": max(weights) / sum(weights),
            "effective_documents_from_length_weights": sum(weights)**2 / sum(w*w for w in weights)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch", type=Path, default=Path("/tmp/grad-research-20261106"))
    p.add_argument("--output", type=Path, default=Path("artifacts/data_extended"))
    p.add_argument("--primary", type=Path, default=Path("artifacts/data"))
    p.add_argument("--primary-evaluation", type=Path, default=Path("artifacts/evaluation_v3"))
    p.add_argument("--target-chunks", type=int, default=39140)
    p.add_argument("--candidate-chunks", type=int, default=19570)
    p.add_argument("--evaluation-chunks", type=int, default=979)
    args = p.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.scratch / "models/pythia-410m-final", local_files_only=True)
    tokenizer.model_max_length = 10**12
    builder = Builder(tokenizer, args.output, args.scratch, 512, "extended-v1")
    raw = args.scratch / "raw"
    paths = sorted(raw.glob("finemath-4plus-train-*.parquet"))
    if len(paths) != 4:
        raise RuntimeError("Expected four pinned FineMath shards")
    sampling = {"shards": [path.name for path in paths], "document_order": "round-robin source-row order",
        "split_policy": "unchanged normalized-SHA256 train<.8, validation<.9, else test",
        "selection_bias": "bounded prefix expansion of four of 64 shards; not a uniform full-corpus draw",
        "role": "optional robustness assets only"}
    def fine_docs():
        return round_robin(*(parquet_documents(path, "HuggingFaceTB/finemath:finemath-4plus") for path in paths))
    builder.collect("finemath", fine_docs(), {"train": args.target_chunks}, lambda d, h: hash_partition(h),
        "HuggingFaceTB/finemath:finemath-4plus", FINEMATH_REVISION, sampling)
    builder.collect("c4", c4_documents(raw / "c4-en-train-00000.json.gz", "train"),
        {"candidate": args.candidate_chunks}, lambda d, h: "candidate", "allenai/c4:en", C4_REVISION,
        {"shards": ["en/c4-train.00000-of-01024.json.gz"], "split_policy": "official train; global exact deduplication",
         "selection_bias": "bounded prefix of one of 1024 train shards", "role": "optional robustness assets only"})
    training_reference = np.union1d(ngram_reference(args.output / "finemath_train.pt"),
                                   ngram_reference(args.output / "c4_candidate.pt"))
    for corpus in ["finemath", "c4"]:
        for split in ["validation", "test"]:
            builder.reference_ngrams = training_reference
            builder.reference_names = ["finemath_train", "c4_candidate"]
            if split == "test":
                builder.reference_ngrams = np.union1d(training_reference,
                    ngram_reference(args.output / (corpus + "_validation.pt")))
                builder.reference_names.append(corpus + "_validation")
            if corpus == "finemath":
                docs, source, revision, sample = fine_docs(), "HuggingFaceTB/finemath:finemath-4plus", FINEMATH_REVISION, sampling
                partition = lambda d, h: hash_partition(h)
            else:
                docs, source, revision = c4_documents(raw / "c4-en-validation-00000.json.gz", "validation"), "allenai/c4:en", C4_REVISION
                partition = lambda d, h: "validation" if int(h[:16], 16) / 2**64 < .5 else "test"
                sample = {"shards": ["en/c4-validation.00000-of-00008.json.gz"],
                    "split_policy": "same official validation source and SHA256 50/50 as primary",
                    "selection_bias": "bounded prefix of one of eight official validation shards", "role": "optional robustness assets only"}
            builder.collect(corpus, docs, {split: args.evaluation_chunks}, partition, source, revision, sample, .2, 4)
    reuse_wikitext(builder, args.primary, primary_source_documents(args.primary))
    builder.audit_and_finish()
    report = {"version": "extended-v1", "role": "optional robustness assets; no training started",
        "independent_directory": True, "independent_population_replication": False,
        "comparisons_with_primary_data_v2": comparisons(args.output, args.primary),
        "comparisons_with_primary_evaluation_v3": comparisons(args.output, args.primary_evaluation),
        "decontamination": "all FineMath/C4 held-out source documents filtered against expanded target+candidate 13grams, test also against its new validation; >=.2 rejects whole source document before cap",
        "evaluation_cap": "max four uniformly doc-SHA-seeded full chunks, matching v3 document balancing; last doc fills exact quota",
        "limitations": ["Same bounded source shards and hash partitions; training contains primary data and is an expansion, not an independent draw",
            "Only four of 64 FineMath shards and one C4 train/validation shard are sampled in source order",
            "Full-chunk preprocessing excludes short documents and suffixes; expanded evaluation is capped and not natural token-frequency weighting",
            "Primary held-out decontamination used the smaller primary training; use expanded held-out evaluation with expanded training",
            "WikiText is byte-identical reuse, not a new replication; historical pretraining overlap remains possible"],
        "wall_seconds": time.monotonic() - started, "preparation_stages": builder.preparation_timings}
    (args.output / "primary_relationship.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "VERSION").write_text("extended-v1\n")


if __name__ == "__main__":
    main()
