#!/usr/bin/env python3
"""Describe frozen bank NLL distributions and teacher-feature coverage (CPU)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

METHODS = ("random", "dedup_random", "nll_stratified", "cluster")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cdf_distances(pool, bank):
    a, b = np.sort(pool), np.sort(bank)
    grid = np.unique(np.concatenate([a, b]))
    delta = np.searchsorted(a, grid, side="right") / len(a) - np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(delta).max()), float((np.diff(grid) * np.abs(delta[:-1])).sum())


def analyze(root, seeds):
    rows, plots, sources = [], {}, {}
    for seed in seeds:
        stats_path = root / f"generated/pool_seed{seed}_stats.pt"
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        nll = stats["teacher_nll"].numpy().astype(np.float64)
        features = stats["features"].numpy().astype(np.float64)
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        sources[str(stats_path)] = sha(stats_path)
        plots[seed] = {"pool": nll}
        for method in METHODS:
            manifest_path = root / f"banks/generated_{method}_seed{seed}.json"
            manifest = json.loads(manifest_path.read_text())
            if manifest["source_pool_sha256"] != stats["source_pool_sha256"]:
                raise ValueError("Bank and teacher statistics belong to different pools")
            if manifest.get("teacher_statistics_sha256") != sources[str(stats_path)]:
                raise ValueError("Teacher statistics content differs from the selected-bank manifest")
            bank_path = manifest_path.with_suffix(".pt")
            sources[str(bank_path)] = sha(bank_path)
            if sources[str(bank_path)] != manifest["artifact_sha256"]:
                raise ValueError("Bank tensor content differs from its manifest")
            indices = np.asarray(manifest["selected_indices"], dtype=int)
            selected = nll[indices]
            norm2 = (features ** 2).sum(axis=1)
            distance = np.maximum(0, norm2[:, None] + norm2[indices][None, :]
                                  - 2 * features @ features[indices].T).min(axis=1)
            ks, wasserstein = cdf_distances(nll, selected)
            rows.append({"seed": seed, "method": method, "pool_chunks": len(nll), "bank_chunks": len(indices),
                         "teacher_nll_pool_mean": float(nll.mean()), "teacher_nll_bank_mean": float(selected.mean()),
                         "teacher_nll_bank_sd": float(selected.std(ddof=1)),
                         "empirical_cdf_max_distance": ks, "nll_wasserstein1": wasserstein,
                         "mean_nearest_normalized_feature_squared_distance": float(distance.mean()),
                         "p95_nearest_normalized_feature_squared_distance": float(np.quantile(distance, .95)),
                         "selection_seconds": manifest["wall_seconds"]})
            plots[seed][method] = selected
            sources[str(manifest_path)] = sha(manifest_path)
    return rows, plots, sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    p.add_argument("--output", type=Path, default=Path("results/bank_diagnostics"))
    args = p.parse_args()
    rows, curves, sources = analyze(args.artifact_root, args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "bank_geometry.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest = {"scope": "Frozen generated candidate pools and selected banks; no model test scores",
                "feature_metric": "Nearest-bank squared Euclidean distance after L2-normalizing teacher features",
                "interpretation": "Descriptive coverage in the chosen teacher representation, not semantic or held-out distribution coverage; CDF distance has no p-value interpretation here",
                "sources_sha256": sources, "records": rows}
    (args.output / "bank_geometry.json").write_text(json.dumps(manifest, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(args.seeds), figsize=(4 * len(args.seeds), 3.5), squeeze=False)
    labels = {"pool": "Candidate pool", "random": "Random", "nll_stratified": "NLL strata", "cluster": "Cluster"}
    for ax, seed in zip(axes[0], args.seeds):
        for method, label in labels.items():
            values = np.sort(curves[seed][method])
            ax.step(values, np.arange(1, len(values)+1)/len(values), where="post", label=label,
                    lw=1.5 if method == "pool" else 1, linestyle="--" if method == "pool" else "-")
        ax.set(title=f"Generation / selection seed {seed}", xlabel="Teacher mean NLL (nats)", ylabel="Empirical CDF")
        ax.grid(alpha=.2)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.output / f"teacher_nll_cdf.{extension}", dpi=160)
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "banks": len(rows), "model_inference": False}))


if __name__ == "__main__":
    main()
