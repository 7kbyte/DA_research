#!/usr/bin/env python3
"""Prepare the pinned FineMath, C4 and WikiText103 document-disjoint subsets."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.data import (Document, chunks_from_token_ids, document_hash,
                              hash_partition, reconstruct_wikitext_articles,
                              save_token_dataset, token_chunk_hash)
from download_assets import (MODEL_REVISION, FINEMATH_REVISION, C4_REVISION,
                             WIKITEXT_REVISION, sha256_file)
from audit_data_ngrams import ngram_hashes, matching_ngram_mask


def parquet_documents(path: Path, source: str, split="train"):
    index = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            yield Document(row["text"], f"{source}:{path.name}:{index}", source, split,
                           {k: v for k, v in row.items() if k != "text"})
            index += 1


def round_robin(*iterables):
    sentinel = object()
    for group in itertools.zip_longest(*iterables, fillvalue=sentinel):
        yield from (x for x in group if x is not sentinel)


def c4_documents(path: Path, split: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for index, line in enumerate(f):
            row = json.loads(line)
            yield Document(row["text"], f"c4:en:{split}:{path.name}:{index}",
                           "allenai/c4:en", split,
                           {k: v for k, v in row.items() if k != "text"})


class Builder:
    def __init__(self, tokenizer, output: Path, scratch: Path, seq_len=512, data_version="v2"):
        self.tokenizer, self.output, self.seq_len = tokenizer, output, seq_len
        self.data_version = data_version
        self.started = time.monotonic()
        self.preparation_timings = []
        self.seen = {}
        self.documents = []
        self.duplicates = []
        self.results = {}
        self.chunk_seen = {}
        self.chunk_hashes = {}
        self.reference_ngrams = None
        self.reference_names = []
        self.ngram_document_rejections = []
        self.scratch = scratch

    def collect(self, name: str, docs, quotas: dict[str, int | None], partition,
                source: str, revision: str, sampling: dict,
                reject_ngram_fraction: float | None = None):
        collect_started = time.monotonic()
        chunks = {s: [] for s in quotas}
        doc_ids = {s: [] for s in quotas}
        stats = {s: Counter() for s in quotas}
        scanned = 0
        for doc in docs:
            if all(limit is not None and len(chunks[s]) >= limit for s, limit in quotas.items()):
                break
            scanned += 1
            doc_hash = doc.sha256
            split = partition(doc, doc_hash)
            if split not in quotas or (quotas[split] is not None and len(chunks[split]) >= quotas[split]):
                continue
            if doc_hash in self.seen:
                self.duplicates.append({"sha256": doc_hash, "rejected": doc.doc_id,
                                        "existing": self.seen[doc_hash], "dataset": name, "split": split})
                continue
            tokens = self.tokenizer.encode(doc.text, add_special_tokens=False)
            rows, dropped = chunks_from_token_ids(tokens, self.seq_len, self.tokenizer.eos_token_id)
            if not rows:
                stats[split]["short_documents_dropped"] += 1
                stats[split]["short_document_tokens_dropped"] += len(tokens) + 1
                continue
            if reject_ngram_fraction is not None:
                if self.reference_ngrams is None:
                    raise RuntimeError("A training ngram reference must be prepared first")
                overlap = matching_ngram_mask(ngram_hashes(np.asarray(rows), 13), self.reference_ngrams).mean(axis=1)
                if overlap.max() >= reject_ngram_fraction:
                    self.ngram_document_rejections.append({"doc_id": doc.doc_id, "split": split,
                        "sha256": doc_hash, "references": list(self.reference_names),
                        "maximum_chunk_matching_13gram_fraction": float(overlap.max())})
                    stats[split]["near_duplicate_documents_rejected"] += 1
                    continue
            limit = quotas[split]
            artifact_name = f"{name}_{split}"
            used = []
            for row in rows:
                if limit is not None and len(used) + len(chunks[split]) >= limit:
                    break
                chunk_hash = token_chunk_hash(row)
                if chunk_hash in self.chunk_seen:
                    stats[split]["exact_token_chunks_rejected"] += 1
                    continue
                self.chunk_seen[chunk_hash] = artifact_name
                self.chunk_hashes.setdefault(artifact_name, []).append(chunk_hash)
                used.append(row)
            if not used:
                continue
            self.seen[doc_hash] = {"doc_id": doc.doc_id, "artifact": artifact_name}
            record = {"doc_id": doc.doc_id, "sha256": doc_hash, "artifact": artifact_name,
                      "source": doc.source, "source_split": doc.source_split,
                      "source_metadata": doc.metadata, "source_tokens": len(tokens),
                      "chunks_used": len(used), "source_text": doc.text}
            self.documents.append(record)
            chunks[split].extend(used)
            doc_ids[split].extend([doc.doc_id] * len(used))
            stats[split]["source_document_tokens"] += len(tokens)
            stats[split]["suffix_tokens_dropped"] += dropped
            stats[split]["quota_chunks_discarded"] += len(rows) - len(used)
            if len(self.documents) % 200 == 0:
                print(name, "scanned", scanned, "chunks", {s: len(c) for s, c in chunks.items()}, flush=True)
        for split in quotas:
            limit = quotas[split]
            if limit is not None and len(chunks[split]) < limit:
                raise RuntimeError(f"{name}/{split}: only {len(chunks[split])}/{limit} chunks")
            key = f"{name}_{split}"
            selected = [{k: v for k, v in d.items() if k != "source_text"}
                        for d in self.documents if d["artifact"] == key]
            manifest = {"data_version": self.data_version, "source": source, "revision": revision, "split": split,
                        "tokenizer": {"repo": "EleutherAI/pythia-410m", "revision": MODEL_REVISION},
                        "sampling": {**sampling, "source_documents_scanned": scanned},
                        "preprocessing": {"sequence_length": self.seq_len, "document_boundary":
                            "tokenize each document independently, append one EOS, nonoverlapping full chunks, discard suffix",
                            "cross_document_chunks": False, "padding": False,
                            "duplicate_normalization": "NFKC, collapse all whitespace; SHA256",
                            "global_exact_duplicate_policy": "keep first encountered retained document; reject all subsequent matches"},
                        "statistics": dict(stats[split]), "documents": selected}
            manifest["chunk_sha256"] = self.chunk_hashes[key]
            manifest["preprocessing"]["token_chunk_duplicate_policy"] = "global exact SHA256 over little-endian uint32 IDs; retain first"
            if reject_ngram_fraction is not None:
                manifest["preprocessing"]["near_duplicate_policy"] = {
                    "references": list(self.reference_names), "ngram_tokens": 13,
                    "chunk_overlap_fraction_threshold": reject_ngram_fraction,
                    "action": "reject whole source document if any full chunk reaches threshold; refill from same fixed hash split",
                    "loss_based_filtering": False}
            self.results[key] = save_token_dataset(self.output / f"{key}.pt", chunks[split], doc_ids[split], manifest)
            print("saved", key, "chunks", len(chunks[split]), flush=True)
        self.preparation_timings.append({"dataset": name, "splits": list(quotas),
                                         "seconds": time.monotonic() - collect_started,
                                         "source_documents_scanned": scanned})

    def audit_and_finish(self):
        by_artifact = {}
        for doc in self.documents:
            by_artifact.setdefault(doc["artifact"], set()).add(doc["sha256"])
        cross = {}
        cross_chunks = {}
        for a, b in itertools.combinations(sorted(by_artifact), 2):
            cross[f"{a} :: {b}"] = len(by_artifact[a] & by_artifact[b])
            cross_chunks[f"{a} :: {b}"] = len(set(self.chunk_hashes[a]) & set(self.chunk_hashes[b]))
        if any(cross.values()) or any(cross_chunks.values()):
            raise RuntimeError("Cross-artifact document leakage detected")
        doc_path = self.scratch / f"raw/selected_documents_{self.output.name}.jsonl.gz"
        with gzip.open(doc_path, "wt", encoding="utf-8") as f:
            for doc in self.documents:
                f.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")
        audit = {"data_version": self.data_version, "cross_artifact_normalized_exact_document_overlap": cross,
                 "cross_artifact_exact_token_chunk_overlap": cross_chunks,
                 "all_pairs_disjoint": not any(cross.values()) and not any(cross_chunks.values()),
                 "duplicate_rejections": self.duplicates,
                 "near_duplicate_document_rejections": self.ngram_document_rejections,
                 "selected_documents": len(self.documents), "documents_path": str(doc_path),
                 "documents_sha256": sha256_file(doc_path)}
        (self.output / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        summary = {"data_version": self.data_version,
                   "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "preprocessing_seconds": time.monotonic() - self.started,
                   "preparation_stages": self.preparation_timings,
                   "artifacts": {k: {m: v[m] for m in ["path", "sha256", "chunks", "raw_input_tokens", "prediction_tokens", "document_count"]}
                                 for k, v in self.results.items()}, "audit": str(self.output / "split_audit.json")}
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch", type=Path, default=Path("/tmp/grad-research-20261106"))
    p.add_argument("--output", type=Path, default=Path("artifacts/data"))
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--target-chunks", type=int, default=4096)
    p.add_argument("--evaluation-chunks", type=int, default=264)
    p.add_argument("--candidate-chunks", type=int, default=2048)
    p.add_argument("--finemath-eval-max-overlap", type=float, default=.2,
                   help="Reject eval source documents with any chunk reaching this 13gram fraction against target train; 0 disables")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.scratch / "models/pythia-410m-final", local_files_only=True)
    tokenizer.model_max_length = 10**12
    builder = Builder(tokenizer, args.output, args.scratch, args.sequence_length,
                      data_version="v2" if args.finemath_eval_max_overlap else "v1")
    raw = args.scratch / "raw"
    fine_paths = sorted(raw.glob("finemath-4plus-train-*.parquet"))
    if len(fine_paths) != 4:
        raise RuntimeError("Expected all four fixed FineMath shards 0/16/32/48")
    fine_sampling = {"shards": [p.name for p in fine_paths], "document_order": "round_robin source-row order across four shards",
                     "split_policy": "SHA256 normalized document, first64bits/2**64, train<.8, validation<.9, otherwise test",
                     "selection_bias": "deterministic bounded subset of four of 64 shards; not a full-corpus uniform sample"}
    def fine_documents():
        return round_robin(*(parquet_documents(p, "HuggingFaceTB/finemath:finemath-4plus") for p in fine_paths))
    builder.collect("finemath", fine_documents(), {"train": args.target_chunks},
                    lambda d, h: hash_partition(h), "HuggingFaceTB/finemath:finemath-4plus", FINEMATH_REVISION, fine_sampling)
    if args.finemath_eval_max_overlap:
        from replay_study.data import load_token_dataset
        train_ids = load_token_dataset(args.output / "finemath_train.pt")["input_ids"].numpy()
        builder.reference_ngrams = np.unique(ngram_hashes(train_ids))
        builder.reference_names = ["finemath_train"]
    for split in ["validation", "test"]:
        builder.collect("finemath", fine_documents(), {split: args.evaluation_chunks},
                        lambda d, h: hash_partition(h), "HuggingFaceTB/finemath:finemath-4plus", FINEMATH_REVISION,
                        fine_sampling, args.finemath_eval_max_overlap or None)
        if split == "validation" and args.finemath_eval_max_overlap:
            val_ids = load_token_dataset(args.output / "finemath_validation.pt")["input_ids"].numpy()
            builder.reference_ngrams = np.union1d(builder.reference_ngrams, np.unique(ngram_hashes(val_ids)))
            builder.reference_names = ["finemath_train", "finemath_validation"]
    builder.collect("c4", c4_documents(raw / "c4-en-validation-00000.json.gz", "validation"),
                    {"validation": args.evaluation_chunks, "test": args.evaluation_chunks},
                    lambda d, h: "validation" if int(h[:16], 16) / 2**64 < .5 else "test",
                    "allenai/c4:en", C4_REVISION,
                    {"shards": ["en/c4-validation.00000-of-00008.json.gz"],
                     "split_policy": "official validation shard; SHA256 normalized doc first64bits, 50/50 validation/test",
                     "selection_bias": "source-order bounded subset of one of eight official validation shards"})
    for split in ["validation", "test"]:
        lines = pq.read_table(raw / f"wikitext-103-raw-{split}.parquet", columns=["text"])["text"].to_pylist()
        builder.collect("wikitext103", reconstruct_wikitext_articles(lines, split), {split: None},
                        lambda d, h, s=split: s, "Salesforce/wikitext:wikitext-103-raw-v1", WIKITEXT_REVISION,
                        {"split_policy": "official validation/test partitions, reconstructed top-level articles",
                         "article_reconstruction": "join raw rows until next top-level '= Title =' heading; retain subsection rows"})
    builder.collect("c4", c4_documents(raw / "c4-en-train-00000.json.gz", "train"),
                    {"candidate": args.candidate_chunks}, lambda d, h: "candidate", "allenai/c4:en", C4_REVISION,
                    {"shards": ["en/c4-train.00000-of-01024.json.gz"],
                     "split_policy": "official train, remove all already-retained normalized exact document hashes",
                     "selection_bias": "source-order bounded subset of one of 1024 official train shards"})
    builder.audit_and_finish()


if __name__ == "__main__":
    main()
