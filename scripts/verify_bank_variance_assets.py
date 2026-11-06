#!/usr/bin/env python3
"""Independently compare prepared B tensors with frozen pool rows; no models."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chunk_hashes(ids, mask):
    ids = ids.numpy().astype("<i8", copy=False)
    mask = mask.numpy().astype(np.uint8, copy=False)
    return [hashlib.sha256(x.tobytes() + m.tobytes()).hexdigest() for x, m in zip(ids, mask)]


def verify(registry_path, output):
    torch.set_num_threads(2)
    registry_path = Path(registry_path)
    rows = read(registry_path)
    preparation_path = registry_path.parent / "preparation_manifest.json"
    prep = read(preparation_path)
    if prep["training_started"] or prep["model_or_test_inference_performed"]:
        raise ValueError("expected prepared-only asset provenance")
    if sha(registry_path) != prep["registry_sha256"]:
        raise ValueError("this pre-activation verifier requires the original prepared registry bytes")
    matrix = {(g, k, s) for g in ("B4", "M", "C") for k in (20, 196) for s in range(1001, 1011)}
    if len(rows) != 60 or {(r["group"], r["bank_chunks"], r["selection_seed"]) for r in rows} != matrix:
        raise ValueError("registered 60-bank matrix is incomplete or duplicated")
    for key in ("fixed_pool", "statistics", "parent_registry"):
        if sha(prep[key]) != prep[key + "_sha256"]:
            raise ValueError(f"source SHA changed: {key}")
    for parent in prep["parent_registry_snapshot"]:
        proof = prep["parent_proof"][parent["id"]]
        if sha(parent["config_path"]) != proof["config_sha256"] or sha(parent["summary_path"]) != proof["summary_sha256"]:
            raise ValueError("completed-parent freeze bytes changed")
    pool = torch.load(prep["fixed_pool"], map_location="cpu", weights_only=False)
    stats = torch.load(prep["statistics"], map_location="cpu", weights_only=False)
    ids = pool["input_ids"]
    mask = pool.get("attention_mask", torch.ones_like(ids, dtype=torch.bool)).bool()
    if ids.shape != (1957, 512) or not bool(mask.all()):
        raise ValueError("expected the frozen full-length 1957-chunk pool11")
    source_hashes = chunk_hashes(ids, mask)
    if len(set(source_hashes)) != len(ids):
        raise ValueError("fixed candidate pool is not exact-unique")
    nll = np.asarray(stats["teacher_nll"], dtype=np.float64)
    strata = np.empty(len(ids), dtype=np.int64)
    bins = np.array_split(np.argsort(nll, kind="stable"), 5)
    sizes = np.array([len(b) for b in bins])
    for h, indices in enumerate(bins):
        strata[indices] = h
    proof_by_id = {r["id"]: r for r in prep["banks"]}
    methods = {"B4": "random", "M": "nll_stratified", "C": "cluster"}
    report = []
    parent_cfg = prep["parent_b4_config_snapshot"]
    allowed_changes = {"replay_path", "output_dir", "run_label", "resume_from", "max_steps"}
    for row in rows:
        if (row["phase"] != "conditional_bank_variance" or "seed" in row or row["training_seed"] != 11 or
                row["generation_seed"] != 11 or row["active"] is not False or row["status"] != "prepared_only"):
            raise ValueError("conditional prepared-only seed/phase identity changed")
        p = Path(row["bank_path"])
        metadata = read(p.with_suffix(".json"))
        config = read(row["config_path"])
        proof = proof_by_id[row["id"]]
        if (sha(p) != proof["bank_sha256"] or sha(p.with_suffix(".json")) != proof["bank_manifest_sha256"] or
                sha(row["config_path"]) != proof["config_sha256"]):
            raise ValueError("prepared artifact contents changed")
        if {k: v for k, v in config.items() if k not in allowed_changes} != {k: v for k, v in parent_cfg.items() if k not in allowed_changes}:
            raise ValueError("child changed parent scientific configuration")
        if metadata["method"] != methods[row["group"]] or metadata["selection_seed"] != row["selection_seed"]:
            raise ValueError("selection method/draw differs from registry")
        indices = np.asarray(metadata["selected_indices"], dtype=np.int64)
        if len(indices) != row["bank_chunks"] or len(set(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(ids)):
            raise ValueError("invalid or duplicate selected source indices")
        bank = torch.load(p, map_location="cpu", weights_only=False)
        index_tensor = torch.as_tensor(indices)
        actual_mask = bank.get("attention_mask", torch.ones_like(bank["input_ids"], dtype=torch.bool)).bool()
        if (not torch.equal(bank["selection_indices"], index_tensor) or not torch.equal(bank["input_ids"], ids[index_tensor]) or
                not torch.equal(actual_mask, mask[index_tensor])):
            raise ValueError("bank tensor differs from the recorded frozen pool rows")
        if "doc_ids" in pool and bank.get("doc_ids") != [pool["doc_ids"][i] for i in indices]:
            raise ValueError("bank document identities differ from source")
        actual_hashes = chunk_hashes(bank["input_ids"], actual_mask)
        if actual_hashes != [source_hashes[i] for i in indices] or actual_hashes != metadata["selected_chunk_hashes"]:
            raise ValueError("canonical token/mask row hashes differ")
        counts = np.bincount(strata[indices], minlength=5)
        composition = metadata["nll_composition"]
        values = nll[indices]
        for key, actual in (("teacher_nll_mean", values.mean()), ("teacher_nll_std_ddof1", values.std(ddof=1)),
                            ("teacher_nll_min", values.min()), ("teacher_nll_max", values.max())):
            if not np.isclose(composition[key], actual, rtol=0, atol=1e-14):
                raise ValueError("reported NLL composition differs from source statistics")
        if composition["pool_nll_stratum_counts"] != counts.tolist():
            raise ValueError("NLL stratum composition differs from selected indices")
        if row["group"] == "M":
            quotas = sizes * len(indices) / sizes.sum()
            allocation = np.floor(quotas).astype(int)
            allocation[np.argsort(-(quotas - allocation), kind="stable")[:len(indices) - allocation.sum()]] += 1
            if not np.array_equal(counts, allocation) or metadata["stratum_bank_sizes"] != allocation.tolist():
                raise ValueError("stratified bank violates the frozen integer quota")
        if (metadata["unique_prediction_tokens"] != 511 * len(indices) or metadata["bank_unique_chunks"] != len(indices) or
                metadata["source_pool_sha256"] != prep["fixed_pool_sha256"] or
                metadata["teacher_statistics_sha256"] != prep["statistics_sha256"] or
                metadata.get("clustering_wall_seconds", 0) > metadata["wall_seconds"]):
            raise ValueError("capacity, lineage or nested cost disagrees")
        report.append({"id": row["id"], "group": row["group"], "selection_seed": row["selection_seed"],
                       "bank_chunks": len(indices), "teacher_nll_mean": float(values.mean()),
                       "teacher_nll_std_ddof1": float(values.std(ddof=1)), "stratum_counts": counts.tolist(),
                       "bank_sha256": proof["bank_sha256"], "source_rows_exact_match": True})
    queue = read(registry_path.parent / "queue_template.json")
    if queue["launch_authorized"] is not False or queue["prepared_only"] is not True or len(queue["jobs"]) != 60:
        raise ValueError("prepared template must not be a launch-authorized queue")
    payload = {"verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "passed",
               "verified_banks": len(report), "training_started_by_verifier": False, "model_or_test_inference_performed": False,
               "registry_sha256": sha(registry_path), "preparation_manifest_sha256": sha(preparation_path),
               "verifier_sha256": sha(__file__), "source_token_rows_and_masks_checked": True,
               "scientific_parent_config_checked": True, "records": report}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify(args.registry, args.output)
    print(json.dumps({"status": result["status"], "verified_banks": result["verified_banks"]}))
