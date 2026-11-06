#!/usr/bin/env python3
"""Document-capped evaluation revision, chosen from distribution diagnostics only.

Reads primary data v2 and pinned raw sources. Never writes primary training data,
banks, or primary evaluation. No model predictions or test losses are consumed.
"""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from transformers import AutoTokenizer

from prepare_data import Builder, parquet_documents, c4_documents, round_robin
from download_assets import FINEMATH_REVISION, C4_REVISION
from audit_data_ngrams import ngram_hashes
from replay_study.data import hash_partition, load_token_dataset


def ngram_reference(path):
    return np.unique(ngram_hashes(load_token_dataset(path)["input_ids"].numpy()))


def primary_source_documents(primary):
    audit = json.loads((primary / "split_audit.json").read_text())
    with gzip.open(audit["documents_path"], "rt", encoding="utf-8") as stream:
        return {doc["doc_id"]: doc for doc in map(json.loads, stream)}


def register_primary_training(builder, primary, source_documents):
    for name in ["finemath_train", "c4_candidate"]:
        manifest = json.loads((primary / (name + ".json")).read_text())
        ref_name = "primary:" + name
        builder.chunk_hashes[ref_name] = manifest["chunk_sha256"]
        for digest in manifest["chunk_sha256"]:
            builder.chunk_seen[digest] = ref_name
        for item in manifest["documents"]:
            doc = copy.deepcopy(source_documents[item["doc_id"]])
            doc["artifact"] = ref_name
            builder.seen[doc["sha256"]] = {"artifact": ref_name, "doc_id": doc["doc_id"]}
            builder.documents.append(doc)


def reuse_wikitext(builder, primary, source_documents):
    for split in ["validation", "test"]:
        name = "wikitext103_" + split
        destination = builder.output / (name + ".pt")
        shutil.copy2(primary / destination.name, destination)
        manifest = json.loads((primary / (name + ".json")).read_text())
        for item in manifest["documents"]:
            if item["sha256"] in builder.seen:
                raise RuntimeError("Unchanged WikiText document collides with another artifact")
            builder.seen[item["sha256"]] = {"artifact": name, "doc_id": item["doc_id"]}
            builder.documents.append(copy.deepcopy(source_documents[item["doc_id"]]))
        for digest in manifest["chunk_sha256"]:
            if digest in builder.chunk_seen:
                raise RuntimeError("Unchanged WikiText token chunk collides with another artifact")
            builder.chunk_seen[digest] = name
        builder.chunk_hashes[name] = manifest["chunk_sha256"]
        manifest.update(data_version=builder.data_version, path=str(destination),
            primary_relationship={"policy": "byte-identical official-split article full-chunk reuse",
                                  "primary_path": str(primary / destination.name), "primary_sha256": manifest["sha256"]})
        destination.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        builder.results[name] = manifest


def relationship_report(output, primary):
    comparisons = {}
    for path in sorted(output.glob("*.pt")):
        name = path.stem
        current = json.loads(path.with_suffix(".json").read_text())
        original = json.loads((primary / (name + ".json")).read_text())
        a = {doc["sha256"] for doc in current["documents"]}
        b = {doc["sha256"] for doc in original["documents"]}
        x, y = set(current["chunk_sha256"]), set(original["chunk_sha256"])
        weights = [doc["chunks_used"] for doc in current["documents"]]
        comparisons[name] = {"old_documents": len(b), "new_documents": len(a), "shared_document_hashes": len(a & b),
            "old_chunks": len(y), "new_chunks": len(x), "shared_exact_chunks": len(x & y),
            "all_old_chunks_in_new": y <= x, "old_sha256": original["sha256"], "new_sha256": current["sha256"],
            "document_chunk_count_histogram": dict(Counter(weights)), "max_chunks_per_document": max(weights),
            "max_document_token_weight": max(weights) / sum(weights),
            "top5_document_token_weight": sum(sorted(weights, reverse=True)[:5]) / sum(weights),
            "effective_documents_from_length_weight_concentration": sum(weights)**2 / sum(w*w for w in weights)}
    result = {"version": "evaluation-v3", "primary_data_version": "v2",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trigger": "data-only review: primary FineMath validation single-document weight 25.4%; no test loss or model scores used",
        "policy": {"finemath_and_c4_chunks_per_split": 1024, "sequence_length": 512, "max_chunks_per_document": 4,
            "long_document_sampling": "uniform without replacement using NumPy PCG64 seeded by int(normalized-document SHA256,16); sort indices to restore source order",
            "near_duplicate_filter": "reject whole source document if any full chunk has >=20% matching token13grams against training references; test additionally references new validation",
            "filter_before_cap": True, "source_document_hash_partitions_unchanged": True,
            "wikitext": "all primary official-split full article chunks unchanged"},
        "comparisons": comparisons,
        "limitations": ["Same bounded source shards and same hash partitions; this is a changed evaluation design, not a fully independent population draw",
            "Full-chunk preprocessing still excludes documents shorter than 511 tokenizer tokens and discards suffixes",
            "Capping changes the estimand toward more balanced document coverage; NLL is still token-weighted over retained chunks",
            "13gram filtering does not prove semantic independence or remove all historical pretraining overlap"],
        "primary_training_unchanged": {name: hashlib.sha256((primary / (name + ".pt")).read_bytes()).hexdigest()
                                       for name in ["finemath_train", "c4_candidate"]}}
    (output / "version_and_relationship.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch", type=Path, default=Path("/tmp/grad-research-20261106"))
    p.add_argument("--primary", type=Path, default=Path("artifacts/data"))
    p.add_argument("--output", type=Path, default=Path("artifacts/evaluation_v3"))
    args = p.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.scratch / "models/pythia-410m-final", local_files_only=True)
    tokenizer.model_max_length = 10**12
    builder = Builder(tokenizer, args.output, args.scratch, 512, "evaluation-v3")
    source_documents = primary_source_documents(args.primary)
    register_primary_training(builder, args.primary, source_documents)
    raw = args.scratch / "raw"
    fine_paths = sorted(raw.glob("finemath-4plus-train-*.parquet"))
    fine_reference = ngram_reference(args.primary / "finemath_train.pt")
    for split in ["validation", "test"]:
        builder.reference_ngrams = fine_reference
        builder.reference_names = ["primary:finemath_train"]
        if split == "test":
            builder.reference_ngrams = np.union1d(fine_reference, ngram_reference(args.output / "finemath_validation.pt"))
            builder.reference_names.append("finemath_validation")
        documents = round_robin(*(parquet_documents(path, "HuggingFaceTB/finemath:finemath-4plus") for path in fine_paths))
        builder.collect("finemath", documents, {split: 1024}, lambda d, h: hash_partition(h),
            "HuggingFaceTB/finemath:finemath-4plus", FINEMATH_REVISION,
            {"shards": [path.name for path in fine_paths], "document_order": "round-robin source-row order",
             "split_policy": "unchanged normalized-SHA256 train<.8, validation<.9, else test",
             "selection_bias": "capped bounded-prefix subset of four of 64 shards"}, .2, 4)
    c4_reference = ngram_reference(args.primary / "c4_candidate.pt")
    for split in ["validation", "test"]:
        builder.reference_ngrams = c4_reference
        builder.reference_names = ["primary:c4_candidate"]
        if split == "test":
            builder.reference_ngrams = np.union1d(c4_reference, ngram_reference(args.output / "c4_validation.pt"))
            builder.reference_names.append("c4_validation")
        builder.collect("c4", c4_documents(raw / "c4-en-validation-00000.json.gz", "validation"),
            {split: 1024}, lambda d, h: "validation" if int(h[:16], 16) / 2**64 < .5 else "test",
            "allenai/c4:en", C4_REVISION,
            {"shards": ["en/c4-validation.00000-of-00008.json.gz"],
             "split_policy": "unchanged official validation source SHA256 first64bits 50/50",
             "selection_bias": "capped bounded prefix of one of eight official validation shards"}, .2, 4)
    reuse_wikitext(builder, args.primary, source_documents)
    builder.audit_and_finish()
    relationship_report(args.output, args.primary)
    (args.output / "VERSION").write_text("v3\n")
    (args.output / "preparation_timing.json").write_text(json.dumps({"wall_seconds": time.monotonic() - started,
        "stages": builder.preparation_timings, "test_loss_read": False, "training_started": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
