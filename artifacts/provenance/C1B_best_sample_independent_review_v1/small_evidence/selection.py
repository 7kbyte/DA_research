"""Auditable equal-budget replay-bank selection from a shared candidate pool."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .evaluate import load_tensor_data, prediction_mask, write_json


METHODS = ("random", "dedup_random", "nll_stratified", "cluster")


def chunk_hashes(data: str | Path | Mapping[str, Any]) -> list[str]:
    """Hash canonical token IDs and validity masks, independent of tensor dtype."""
    data = load_tensor_data(data)
    ids = data["input_ids"].cpu().numpy().astype("<i8", copy=False)
    mask = data.get("attention_mask")
    mask = np.ones(ids.shape, dtype=np.uint8) if mask is None else mask.bool().cpu().numpy().astype(np.uint8)
    return [hashlib.sha256(row.tobytes() + valid.tobytes()).hexdigest() for row, valid in zip(ids, mask)]


def unique_indices(hashes: list[str]) -> list[int]:
    seen = set()
    unique = []
    for index, value in enumerate(hashes):
        if value not in seen:
            seen.add(value)
            unique.append(index)
    return unique


def _proportional_allocation(sizes: np.ndarray, total: int) -> np.ndarray:
    """Largest-remainder allocation, stable tie-breaking by stratum index."""
    expected = sizes.astype(np.float64) * total / sizes.sum()
    counts = np.floor(expected).astype(np.int64)
    remainder = total - counts.sum()
    order = np.argsort(-(expected - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    if (counts > sizes).any() or counts.sum() != total:
        raise ValueError("invalid proportional allocation")
    return counts


def select_bank(data: str | Path | Mapping[str, Any], method: str, seed: int,
                bank_size: int | None = None, prediction_token_budget: int | None = None,
                statistics: str | Path | Mapping[str, Any] | None = None,
                n_strata: int = 5, kmeans_batch_size: int = 1024,
                kmeans_max_iter: int = 100, kmeans_n_init: int = 3) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select a bank and return (shared-format data, complete selection manifest).

    Budget matching requires equal prediction-token counts in every candidate;
    no token truncation silently changes the scientific unit. Random and dedup
    random use identical RNG draws when deduplication removes zero candidates.
    Cluster and stratified selection both operate on exact-deduplicated chunks.
    """
    begin = time.perf_counter()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    pool = load_tensor_data(data)
    ids = pool["input_ids"]
    if not len(ids):
        raise ValueError("cannot select from an empty pool")
    mask = pool.get("attention_mask", torch.ones_like(ids, dtype=torch.bool))
    counts = prediction_mask(mask).sum(dim=1)
    if counts.min().item() < 1 or not torch.equal(counts, counts[0].expand_as(counts)):
        raise ValueError("equal-budget selection requires equal positive prediction-token counts per chunk")
    per_chunk = counts[0].item()
    if prediction_token_budget is not None:
        if prediction_token_budget <= 0 or prediction_token_budget % per_chunk:
            raise ValueError(f"prediction_token_budget must be a positive multiple of {per_chunk}")
        from_budget = prediction_token_budget // per_chunk
        if bank_size is not None and bank_size != from_budget:
            raise ValueError("bank_size conflicts with prediction_token_budget")
        bank_size = from_budget
    if bank_size is None or bank_size < 1:
        raise ValueError("provide a positive bank_size or prediction_token_budget")
    hashes = chunk_hashes(pool)
    unique = unique_indices(hashes)
    candidates = np.arange(len(ids)) if method == "random" else np.asarray(unique, dtype=np.int64)
    if bank_size > len(candidates):
        raise ValueError("requested bank is larger than the available selection candidates")
    rng = np.random.default_rng(seed)
    extra: dict[str, Any] = {}
    stats = None
    if statistics is not None:
        stats = (torch.load(statistics, map_location="cpu", weights_only=False)
                 if isinstance(statistics, (str, Path)) else dict(statistics))
        for name in ("teacher_nll", "features", "prediction_token_counts"):
            if name in stats and len(stats[name]) != len(ids):
                raise ValueError(f"statistics {name} do not match the candidate pool length")
        if "prediction_token_counts" in stats and not torch.equal(torch.as_tensor(stats["prediction_token_counts"]), counts):
            raise ValueError("statistics token counts do not match the candidate pool")
        if isinstance(data, (str, Path)) and "source_pool_sha256" in stats:
            if hashlib.sha256(Path(data).read_bytes()).hexdigest() != stats["source_pool_sha256"]:
                raise ValueError("statistics were extracted from a different candidate pool hash")
    if method in ("random", "dedup_random"):
        selected = rng.choice(candidates, size=bank_size, replace=False)
    elif method == "nll_stratified":
        if stats is None or "teacher_nll" not in stats:
            raise ValueError("nll_stratified requires teacher_nll statistics")
        if n_strata < 1:
            raise ValueError("n_strata must be positive")
        nll = np.asarray(stats["teacher_nll"], dtype=np.float64)
        if not np.isfinite(nll).all():
            raise ValueError("teacher_nll must be finite")
        # Equal-frequency bins with stable original-index tie-breaking.
        order = candidates[np.argsort(nll[candidates], kind="stable")]
        strata = np.array_split(order, min(n_strata, len(order)))
        sizes = np.asarray([len(x) for x in strata])
        allocation = _proportional_allocation(sizes, bank_size)
        selected = np.concatenate([rng.choice(group, size=int(k), replace=False)
                                   for group, k in zip(strata, allocation)])
        rng.shuffle(selected)
        extra.update(n_strata=len(strata), stratum_sizes=sizes.tolist(),
                     stratum_bank_sizes=allocation.tolist(),
                     strata_nll_ranges=[[float(nll[g].min()), float(nll[g].max())] for g in strata],
                     quantile_tie_policy="stable_original_index", allocation="proportional_largest_remainder")
    else:
        if stats is None or "features" not in stats:
            raise ValueError("cluster requires frozen-teacher features")
        from sklearn.cluster import MiniBatchKMeans
        features = np.asarray(stats["features"], dtype=np.float32)[candidates]
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError("features must be a finite [chunks,dimensions] matrix")
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-12)
        cluster_begin = time.perf_counter()
        kmeans = MiniBatchKMeans(n_clusters=bank_size, random_state=seed, n_init=kmeans_n_init,
                                batch_size=max(bank_size, min(kmeans_batch_size, len(features))),
                                max_iter=kmeans_max_iter, reassignment_ratio=0.0)
        labels = kmeans.fit_predict(features)
        selected_local, used, empty = [], set(), []
        # Nonempty clusters are disjoint, so their closest representatives cannot collide.
        for cluster in range(bank_size):
            members = np.flatnonzero(labels == cluster)
            if not len(members):
                empty.append(cluster)
                continue
            distances = ((features[members] - kmeans.cluster_centers_[cluster]) ** 2).sum(axis=1)
            chosen = int(members[np.argmin(distances)])
            selected_local.append(chosen)
            used.add(chosen)
        # Deterministic nearest unused candidate to each empty center.
        for cluster in empty:
            distance = ((features - kmeans.cluster_centers_[cluster]) ** 2).sum(axis=1)
            order = np.argsort(distance, kind="stable")
            chosen = next(int(i) for i in order if int(i) not in used)
            selected_local.append(chosen)
            used.add(chosen)
        selected = candidates[np.asarray(selected_local)]
        extra.update(feature_normalization="L2", cluster_method="MiniBatchKMeans",
                     cluster_seed=seed, cluster_count=bank_size, cluster_n_init=kmeans_n_init,
                     cluster_max_iter=kmeans_max_iter, cluster_batch_size=kmeans.batch_size,
                     cluster_inertia=float(kmeans.inertia_), cluster_iterations=int(kmeans.n_iter_),
                     empty_clusters=empty, empty_cluster_fill="nearest_unused_candidate_stable_ties",
                     clustering_wall_seconds=time.perf_counter() - cluster_begin,
                     sklearn_version=importlib.metadata.version("scikit-learn"))
    selected = np.asarray(selected, dtype=np.int64)
    selected_tensor = torch.from_numpy(selected.copy())
    bank = {"input_ids": ids[selected_tensor].contiguous(), "selection_indices": selected_tensor}
    if "attention_mask" in pool:
        bank["attention_mask"] = pool["attention_mask"][selected_tensor].contiguous()
    if "doc_ids" in pool:
        bank["doc_ids"] = [pool["doc_ids"][i] for i in selected]
    selected_hashes = [hashes[i] for i in selected]
    unique_bank = len(set(selected_hashes))
    manifest = {"method": method, "selection_seed": seed, "pool_chunks": len(ids),
                "pool_unique_chunks": len(unique), "pool_duplicate_rate": 1 - len(unique) / len(ids),
                "deduplicate_before_selection": method != "random", "deduplication": "exact_tokens_and_mask_sha256",
                "candidate_chunks_after_dedup": len(candidates), "bank_chunks": bank_size,
                "bank_unique_chunks": unique_bank, "bank_duplicate_rate": 1 - unique_bank / bank_size,
                "prediction_tokens": int(bank_size * per_chunk), "valid_prediction_tokens": int(bank_size * per_chunk),
                "unique_prediction_tokens": int(unique_bank * per_chunk),
                "raw_input_tokens": int(mask[selected_tensor].sum().item()),
                "sequence_length": ids.shape[1], "selected_indices": selected.tolist(),
                "selected_chunk_hashes": selected_hashes, "wall_seconds": time.perf_counter() - begin,
                "rng": "numpy.random.default_rng(PCG64)", "numpy_version": np.__version__, **extra}
    if stats is not None and "teacher_nll" in stats:
        values = np.asarray(stats["teacher_nll"], dtype=np.float64)[selected]
        manifest["bank_teacher_nll"] = {"mean": float(values.mean()), "std": float(values.std()),
                                        "min": float(values.min()), "max": float(values.max())}
    if stats is not None:
        manifest["teacher_statistics"] = {key: value for key, value in stats.items()
                                          if not isinstance(value, torch.Tensor)}
        manifest["teacher_statistics_artifact"] = str(statistics) if isinstance(statistics, (str, Path)) else "in_memory"
        if isinstance(statistics, (str, Path)):
            manifest["teacher_statistics_sha256"] = hashlib.sha256(Path(statistics).read_bytes()).hexdigest()
    return bank, manifest


def save_bank(data: str | Path | Mapping[str, Any], output: str | Path, **kwargs: Any) -> dict[str, Any]:
    source_metadata = None
    source_manifest = Path(data).with_suffix(".json") if isinstance(data, (str, Path)) else None
    if source_manifest is not None and source_manifest.exists():
        source_metadata = json.loads(source_manifest.read_text())
        split = str(source_metadata.get("split", "")).lower()
        if any(word in split for word in ("valid", "test", "eval")):
            raise ValueError("replay selection may not use a held-out evaluation split")
    bank, manifest = select_bank(data, **kwargs)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output)
    manifest["artifact_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest["source_pool"] = str(data) if isinstance(data, (str, Path)) else "in_memory"
    if isinstance(data, (str, Path)):
        manifest["source_pool_sha256"] = hashlib.sha256(Path(data).read_bytes()).hexdigest()
    if source_metadata is not None:
        manifest.update(source=source_metadata.get("source"), revision=source_metadata.get("revision"),
                        tokenizer=source_metadata.get("tokenizer"), split="training_replay_bank",
                        source_manifest=str(source_manifest),
                        source_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
                        source_preprocessing=source_metadata.get("preprocessing"),
                        preprocessing={"selection": manifest["method"],
                                       "deduplication": manifest["deduplication"],
                                       "deduplicate_before_selection": manifest["deduplicate_before_selection"]})
    write_json(output.with_suffix(".json"), manifest)
    return manifest
