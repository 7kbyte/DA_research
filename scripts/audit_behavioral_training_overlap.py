#!/usr/bin/env python3
"""Token-only overlap of frozen behavioral inputs with actual primary training.

This reads no checkpoint, loss, answer prediction, or evaluation result. The
optional extended-assets scope is explicitly a prepared-assets diagnostic.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from replay_study.tasks import load_jsonl, prepare_choice
from audit_data_ngrams import matching_ngram_mask, ngram_hashes


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def encode_queries(tokenizer, examples):
    """Reconstruct context+choice including its final predicted token."""
    queries, hashes, offsets = [], [], [0]
    fingerprint = hashlib.sha256()
    for row in examples:
        for choice_index, choice in enumerate(row["choices"]):
            prepared = prepare_choice(tokenizer, row["context"], choice, max_length=2048)
            ids = np.asarray(prepared["input_ids"] + prepared["labels"][-1:], dtype="<i8")
            grams = ngram_hashes(ids[None, :])[0] if len(ids) >= 13 else np.empty(0, dtype=np.uint64)
            query = {"task": row["task"], "item_id": row["item_id"], "choice_index": choice_index,
                     "tokens": len(ids), "ngram_occurrences": len(grams), "ids": ids,
                     "token_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                     "context_tokens_removed": prepared["context_tokens_removed"],
                     "boundary_prefix_matches": prepared["boundary_prefix_matches"]}
            fingerprint.update((row["item_id"] + ":" + str(choice_index) + ":" + query["token_sha256"] + "\n").encode())
            queries.append(query)
            hashes.append(grams)
            offsets.append(offsets[-1] + len(grams))
    return queries, np.concatenate(hashes), np.asarray(offsets), fingerprint.hexdigest()


def audit_reference(queries, flat_hashes, offsets, source_ids, source_doc_ids, *, matched_masks=None):
    """Every threshold hash flag is confirmed using actual 13-token tuples."""
    source_ids = np.asarray(source_ids)
    source_hashes = ngram_hashes(source_ids)
    reference = np.unique(source_hashes)
    matched = matching_ngram_mask(flat_hashes, reference)
    if matched_masks is not None:
        matched_masks.append(matched)
    cumulative = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(matched, dtype=np.int64)])
    counts = np.diff(cumulative[offsets])
    denominators = np.diff(offsets)
    # Exact 1/5 threshold, with sequences shorter than 13 inapplicable.
    flags = np.flatnonzero((denominators > 0) & (5 * counts >= denominators))
    wanted_parts = [flat_hashes[offsets[i]:offsets[i + 1]][matched[offsets[i]:offsets[i + 1]]] for i in flags]
    wanted = np.unique(np.concatenate(wanted_parts)) if wanted_parts else np.empty(0, dtype=np.uint64)
    by_tokens = {}
    rows, columns = np.nonzero(matching_ngram_mask(source_hashes, wanted))
    for row, column in zip(rows.tolist(), columns.tolist()):
        key = tuple(map(int, source_ids[row, column:column + 13]))
        by_tokens.setdefault(key, []).append((row, column))
    confirmed, rejected = [], []
    witness_ids, witnesses = {}, []
    for index in flags.tolist():
        query = queries[index]
        ids = query["ids"]
        positions, evidence = [], []
        for start in range(len(ids) - 12):
            key = tuple(map(int, ids[start:start + 13]))
            if key not in by_tokens:
                continue
            positions.append(start)
            if key not in witness_ids:
                row, column = by_tokens[key][0]
                witness_ids[key] = len(witnesses)
                witnesses.append({"witness_id": len(witnesses), "tokens": list(key),
                                  "source_chunk_index": row, "source_doc_id": str(source_doc_ids[row]),
                                  "source_token_start": column})
            evidence.append(witness_ids[key])
        if 5 * len(positions) < len(ids) - 12:
            rejected.append({"query_index": index, "confirmed_matches": len(positions),
                             "hashed_matches": int(counts[index])})
            continue
        entire = []
        for row, column in by_tokens.get(tuple(map(int, ids[:13])), []):
            if np.array_equal(source_ids[row, column:column + len(ids)], ids):
                entire.append({"source_chunk_index": row, "source_doc_id": str(source_doc_ids[row]),
                               "source_token_start": column})
        confirmed.append({k: v for k, v in query.items() if k != "ids"} | {
            "query_index": index, "hashed_matching_occurrences": int(counts[index]),
            "confirmed_matching_occurrences": len(positions),
            "confirmed_fraction": len(positions) / (len(ids) - 12),
            "matching_query_token_starts": positions, "source_witness_ids": evidence,
            "entire_sequence_occurrences": len(entire), "entire_sequence_locations": entire[:20]})
    tasks = sorted({q["task"] for q in queries})
    by_task = {}
    for task in tasks:
        indices = [i for i, q in enumerate(queries) if q["task"] == task]
        task_flags = [q for q in confirmed if q["task"] == task]
        by_task[task] = {"items": len({queries[i]["item_id"] for i in indices}), "choices": len(indices),
            "choices_shorter_than_13_inapplicable": int(sum(denominators[i] == 0 for i in indices)),
            "confirmed_flagged_choices": len(task_flags),
            "confirmed_flagged_items": len({q["item_id"] for q in task_flags}),
            "flagged_item_ids": sorted({q["item_id"] for q in task_flags}),
            "hashed_fraction_max": max((counts[i] / denominators[i] for i in indices if denominators[i]), default=None)}
    return {"source_chunks": len(source_ids), "source_unique_13grams": len(reference),
            "by_task": by_task, "confirmed_flagged_choices": confirmed,
            "confirmed_13gram_source_witnesses": witnesses, "hash_collision_rejections": rejected}


def references(scope):
    if scope == "main":
        registry_path = ROOT / "configs/formal_deterministic_registry.json"
        registry = json.loads(registry_path.read_text())
        if len(registry) != 36 or any(r.get("active") is not True or r.get("phase") != "formal" for r in registry):
            raise ValueError("main scope requires the strict active 36-run registry")
        refs = {}
        for row in registry:
            cfg = json.loads(Path(row["config_path"]).read_text())
            for field, role in (("target_path", "actual_target_training"), ("replay_path", "actual_selected_replay_bank")):
                if cfg.get(field):
                    path = Path(cfg[field]).resolve()
                    item = refs.setdefault(path, {"path": str(path), "role": role, "used_by_run_ids": []})
                    item["used_by_run_ids"].append(row["id"])
        return refs, {str(registry_path): sha(registry_path)}, "actual active strict main36 training assets"
    refs = {}
    paths = [ROOT / "artifacts/data_extended/finemath_train.pt"]
    paths += sorted((ROOT / "artifacts/extended").glob("*/banks/*.pt"))
    for path in paths:
        if path.exists():
            refs[path.resolve()] = {"path": str(path.resolve()), "role": "prepared_extended_target" if path.name == "finemath_train.pt" else "prepared_extended_selected_bank",
                                    "used_by_run_ids": [], "actual_training_use_not_established": True}
    return refs, {}, "prepared extended assets; not an assertion of actual training use"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("main", "extended-assets"), default="main")
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.monotonic()
    output = ROOT / f"artifacts/assets/behavioral_{args.scope}_training_overlap_audit.json"
    tokenizer_path = Path("/tmp/grad-research-20261106/models/pythia-410m-final")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    manifest_path = ROOT / "artifacts/behavioral/data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inputs = {str(manifest_path): sha(manifest_path)}
    examples = []
    for task in ("piqa", "hellaswag"):
        item = manifest["tasks"][task]
        path = ROOT / item["artifact"]
        actual = sha(path)
        if actual != item["artifact_sha256"]:
            raise ValueError("frozen task artifact SHA changed")
        inputs[str(path)] = actual
        rows = load_jsonl(path)
        if len(rows) != item["items"]:
            raise ValueError("frozen task item count changed")
        examples.extend(rows)
    queries, flat, offsets, token_sha = encode_queries(tokenizer, examples)
    refs, registry_hashes, scope_description = references(args.scope)
    inputs.update(registry_hashes)
    reports, masks, datasets = [], {}, {}
    for path, source in refs.items():
        dataset = torch.load(path, map_location="cpu", weights_only=False)
        ids = dataset["input_ids"].numpy()
        if ids.ndim != 2 or ids.shape[1] != 512 or ("attention_mask" in dataset and not dataset["attention_mask"].bool().all()):
            raise ValueError("source must contain full unpadded 512-token chunks")
        inputs[str(path)] = sha(path)
        source_doc_ids = dataset.get("doc_ids", [f"chunk{i}" for i in range(len(ids))])
        collected = []
        result = audit_reference(queries, flat, offsets, ids, source_doc_ids, matched_masks=collected)
        if args.scope == "main":
            masks[str(path)] = collected[0]
            datasets[str(path)] = (ids, source_doc_ids)
        reports.append(source | result | {"sha256": inputs[str(path)]})
        write(output.with_suffix(".progress.json"), {"completed_sources": len(reports), "total_sources": len(refs), "last_source": str(path),
            "flagged_item_counts": {k: v["confirmed_flagged_items"] for k, v in result["by_task"].items()}, "seconds": time.monotonic() - started})
        print(json.dumps({"source": str(path), "flagged_items": {k: v["confirmed_flagged_items"] for k, v in result["by_task"].items()}}), flush=True)
    # Target and replay matches can jointly reach 20% even when neither does.
    unions = []
    if args.scope == "main":
        run_sources = {}
        for source in refs.values():
            for run_id in source["used_by_run_ids"]:
                run_sources.setdefault(run_id, []).append(source["path"])
        groups = {}
        for run_id, paths in run_sources.items():
            groups.setdefault(tuple(sorted(paths)), []).append(run_id)
        for paths, run_ids in groups.items():
            union_mask = np.logical_or.reduce([masks[path] for path in paths])
            cumulative = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(union_mask, dtype=np.int64)])
            counts = np.diff(cumulative[offsets])
            denominators = np.diff(offsets)
            flags = np.flatnonzero((denominators > 0) & (5 * counts >= denominators))
            if len(flags):
                ids = np.concatenate([datasets[path][0] for path in paths])
                doc_ids = [path + "::" + str(doc) for path in paths for doc in datasets[path][1]]
                confirmed = audit_reference(queries, flat, offsets, ids, doc_ids)
            else:
                confirmed = {"confirmed_flagged_choices": [], "confirmed_13gram_source_witnesses": [], "hash_collision_rejections": []}
            unions.append({"source_paths": list(paths), "actual_run_ids": sorted(run_ids),
                "hashed_fraction_max": float(max((counts[i] / denominators[i] for i in range(len(queries)) if denominators[i]), default=0.)),
                "hashed_flagged_choices": len(flags), **confirmed})
    all_reports = reports + unions
    by_task = {task: sorted({q["item_id"] for r in all_reports for q in r["confirmed_flagged_choices"] if q["task"] == task})
               for task in ("piqa", "hellaswag")}
    result = {"status": "overlap_detected" if any(by_task.values()) else "passed", "scope": args.scope,
        "scope_description": scope_description, "prepared_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ngram_tokens": 13, "fixed_threshold": .2, "threshold_rule": "5 * exact matching query ngram occurrences >= query ngram occurrences; length<13 inapplicable",
        "query_policy": "prepare_choice(max_length=2048), full context+choice sequence = input_ids followed by final label; includes final predicted token; no gold-choice selection",
        "source_boundary_policy": "within each actual unpadded 512-token training chunk only, never cross chunk/document boundaries",
        "item_counting": "a task item is counted once across its choices and once across source artifacts; per-source choice details retained",
        "all_choice_token_fingerprint_sha256": token_sha, "task_items": {task: sum(x["task"] == task for x in examples) for task in by_task},
        "choice_sequences": len(queries), "choices_shorter_than_13_inapplicable": [
            {k: q[k] for k in ("task", "item_id", "choice_index", "tokens")} for q in queries if q["tokens"] < 13],
        "choices_with_context_truncation": sum(q["context_tokens_removed"] > 0 for q in queries),
        "choices_with_tokenization_boundary_mismatch": sum(not q["boundary_prefix_matches"] for q in queries),
        "unique_flagged_item_ids": by_task, "reports": reports, "actual_condition_target_plus_replay_unions": unions, "input_files_sha256": inputs,
        "tokenizer_files_sha256": {name: sha(tokenizer_path / name) for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")},
        "source_files_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (Path(__file__), ROOT / "scripts/audit_data_ngrams.py", ROOT / "src/replay_study/tasks.py")},
        "seconds": time.monotonic() - started, "gpu_used": False, "model_scores_read": False, "frozen_inputs_modified": False,
        "interpretation": "Lexical overlap diagnostic only. No automatic task exclusion. Candidate pools not selected into training are not inputs here. Common contexts can trigger multiple choices but are counted once per item. Zero flags cannot exclude semantic paraphrases, source-boundary overlaps or historical Pile pretraining contamination."}
    write(output, result)
    print(json.dumps({"status": result["status"], "unique_flagged_items": {k: len(v) for k, v in by_task.items()}, "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
