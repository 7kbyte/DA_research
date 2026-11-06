#!/usr/bin/env python3
"""Prepare exploratory bank/complement diagnostic inputs, never run inference."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replay_study.analysis import _check_summary, selected_checkpoint_exposure


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def partition(pool, bank):
    ids, selected = pool["input_ids"], bank["selection_indices"].long()
    if ids.ndim != 2 or ids.shape[1] != 512 or bank["input_ids"].ndim != 2:
        raise ValueError("complete 512-token chunks required")
    if selected.ndim != 1 or len(selected) != len(bank["input_ids"]) or len(selected.unique()) != len(selected):
        raise ValueError("bank selection must consist of unique pool indices")
    if selected.numel() == 0 or selected.min() < 0 or selected.max() >= len(ids):
        raise ValueError("bank pool indices out of bounds")
    if not torch.equal(ids[selected], bank["input_ids"]):
        raise ValueError("bank tokens do not match selected pool rows")
    if [pool["doc_ids"][i] for i in selected.tolist()] != bank["doc_ids"]:
        raise ValueError("bank doc IDs do not match selected pool rows")
    for payload in (pool, bank):
        if "attention_mask" in payload and not bool(payload["attention_mask"].bool().all()):
            raise ValueError("unpadded full chunks required")
    chosen = set(selected.tolist())
    complement = torch.tensor([i for i in range(len(ids)) if i not in chosen], dtype=torch.long)
    if chosen.intersection(complement.tolist()) or sorted(selected.tolist() + complement.tolist()) != list(range(len(ids))):
        raise ValueError("index partition is not disjoint and exhaustive")
    if len(set(pool["doc_ids"])) != len(ids):
        raise ValueError("generated pool doc IDs must uniquely identify chunks")
    bank_hashes = {hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in ids[selected].cpu().contiguous()}
    complement_hashes = {hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in ids[complement].cpu().contiguous()}
    if bank_hashes.intersection(complement_hashes):
        raise ValueError("exact token duplicate crosses bank/complement index partition")
    copied_bank = {k: v.clone() if isinstance(v, torch.Tensor) else list(v) if isinstance(v, list) else v for k, v in bank.items()}
    copied_bank["pool_indices"] = selected.clone()
    other = {"input_ids": ids[complement].clone(), "doc_ids": [pool["doc_ids"][i] for i in complement.tolist()],
             "pool_indices": complement}
    if "attention_mask" in pool:
        other["attention_mask"] = pool["attention_mask"][complement].clone()
    return copied_bank, other


def main():
    torch.set_num_threads(2)
    output = ROOT / "artifacts/replay_fit_diagnostic_v1"
    if (output / "manifest.json").exists():
        raise FileExistsError("refusing to replace a frozen exploratory diagnostic")
    registry_path = ROOT / "configs/formal_deterministic_registry.json"
    audit_path = ROOT / "results/formal_run_audit.json"
    registry, audit = read(registry_path), read(audit_path)
    if audit["status"] != "passed" or audit["completed_runs"] != 36:
        raise ValueError("main36 audit must pass before preparing this post-validation diagnostic")
    rows = [r for r in registry if r["group"] in ("B3", "B4") and r["ratio"] == .05]
    if len(rows) != 6 or any(r.get("phase") != "formal" or r.get("active") is not True for r in rows):
        raise ValueError("strict shared-bank B3/B4 three-seed conditions required")
    # Reuse root's already frozen checkpoint identity bindings, without opening
    # or reading any of the associated test-score output files.
    checkpoint_manifest_path = ROOT / "docs/primary_evaluation_v2_nll_manifest.json"
    checkpoint_manifest = read(checkpoint_manifest_path)
    bindings = {(r["run_id"], r["kind"]): r for r in checkpoint_manifest["records"] if r["role"] == "primary"}
    partitions, evaluations = [], []
    for seed in (11, 22, 33):
        pool_path = ROOT / f"artifacts/generated/pool_seed{seed}.pt"
        bank_path = ROOT / f"artifacts/banks/generated_random_seed{seed}.pt"
        pool_meta, bank_meta = read(pool_path.with_suffix(".json")), read(bank_path.with_suffix(".json"))
        pool_sha, bank_sha = sha(pool_path), sha(bank_path)
        if pool_sha != pool_meta["artifact_sha256"] or bank_sha != bank_meta["artifact_sha256"] or bank_meta["source_pool_sha256"] != pool_sha:
            raise ValueError("frozen pool/bank SHA binding changed")
        pool = torch.load(pool_path, map_location="cpu", weights_only=False)
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        copied, complement = partition(pool, bank)
        if (len(pool["input_ids"]), len(copied["input_ids"]), len(complement["input_ids"])) != (1957, 196, 1761):
            raise ValueError("primary diagnostic requires 1957=196+1761 chunks per seed")
        corpora = {}
        for name, dataset in (("bank", copied), ("complement", complement)):
            path = output / f"seed{seed}_{name}.pt"
            if path.exists():
                raise FileExistsError("partial diagnostic asset exists; inspect before replacing")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dataset, path)
            reloaded = torch.load(path, map_location="cpu", weights_only=False)
            if not torch.equal(reloaded["input_ids"], pool["input_ids"][dataset["pool_indices"]]) or reloaded["doc_ids"] != dataset["doc_ids"]:
                raise ValueError("saved diagnostic tokens or doc IDs changed")
            corpora[name] = {"path": str(path), "sha256": sha(path), "chunks": len(dataset["input_ids"]),
                "prediction_tokens": len(dataset["input_ids"]) * 511, "pool_indices": dataset["pool_indices"].tolist(),
                "ordered_doc_ids_sha256": hashlib.sha256(json.dumps(dataset["doc_ids"], separators=(",", ":")).encode()).hexdigest(),
                "raw_token_bytes_sha256": hashlib.sha256(dataset["input_ids"].contiguous().numpy().tobytes()).hexdigest()}
        partitions.append({"seed": seed, "source_pool": str(pool_path), "source_pool_sha256": pool_sha,
            "source_pool_manifest_sha256": sha(pool_path.with_suffix(".json")), "source_bank": str(bank_path), "source_bank_sha256": bank_sha,
            "source_bank_manifest_sha256": sha(bank_path.with_suffix(".json")), "corpora": corpora,
            "disjoint_pool_indices": True, "exhaustive_pool_cover": True, "cross_partition_exact_token_duplicates": 0,
            "ordering": "bank preserves original selected order; complement follows ascending original pool index"})
        evaluations.append({"id": f"replay_fit_seed{seed}_initial", "kind": "initial", "seed": seed,
            "model": pool_meta["teacher"], "model_revision": pool_meta["revision"], "checkpoint": None,
            "checkpoint_target_prediction_tokens": 0, "checkpoint_replay_prediction_tokens": 0,
            "corpora": corpora, "active": False})
        for row in [r for r in rows if r["seed"] == seed]:
            summary = read(row["summary_path"])
            _check_summary(summary, row)
            if Path(summary["config"]["replay_path"]).resolve() != bank_path.resolve() or summary["status"] != "completed":
                raise ValueError("B3/B4 must actually share this frozen random bank")
            for endpoint in ("final", "best"):
                exposure = selected_checkpoint_exposure(summary, endpoint)
                binding = bindings[row["id"], endpoint]
                path = Path(binding["checkpoint"])
                stat = path.stat()
                if stat.st_size != binding["checkpoint_bytes"] or stat.st_mtime_ns != binding["checkpoint_mtime_ns"]:
                    raise ValueError("frozen checkpoint binding changed; require root revalidation")
                if binding["target_prediction_tokens_at_checkpoint"] != exposure["target_prediction_tokens"]:
                    raise ValueError("frozen endpoint exposure disagrees with actual validation selection")
                evaluations.append({"id": f"replay_fit_{row['id']}_{endpoint}", "kind": endpoint,
                    "run_id": row["id"], "group": row["group"], "seed": seed, "model": summary["config"]["model_name_or_path"],
                    "model_revision": summary["config"]["model_revision"], "checkpoint": str(path),
                    "checkpoint_sha256": binding["checkpoint_sha256"], "checkpoint_sha_source": str(checkpoint_manifest_path),
                    "checkpoint_sha_rechecked_during_preparation": False, "executor_must_rehash_checkpoint_before_inference": True,
                    "source_summary_sha256": sha(row["summary_path"]), "source_config_sha256": sha(row["config_path"]),
                    "exposure": exposure, "corpora": corpora, "active": False})
    protocol = {"phase": "post_validation_exploratory_replay_fit", "state": "prepared_only_not_scheduled",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "motivation": "Observed primary validation CE/KL contrast; no actual heldout NLL or behavioral result was inspected for this diagnostic",
        "main_fixed_exposure_endpoint": "final: target=10000000, replay=500000", "secondary_endpoint": "validation-selected best, actual exposure retained separately; no equal-budget claim",
        "primary_contrast": "gap_NLL = (NLL_adapted_complement - NLL_initial_complement) - (NLL_adapted_bank - NLL_initial_bank); compare gap_B3 - gap_B4 within seed",
        "secondary_metrics": "forward KL(initial teacher || adapted) separately for bank/complement and its analogous initial-subtracted gap; do not assume computed baseline KL is numerically zero",
        "metric_weighting": "prediction-token-weighted per split; every full512 chunk has511 predictions; report all per-seed and mean/SD(ddof1,n=3)",
        "initial_baseline_reuse": "Initial weights are shared across seeds but corpus identities differ. Cache by pinned base file SHA/revision + exact ordered bank/complement tensor SHA + scorer/execution policy. The same seed's initial result is shared by B3/B4 and final/best; never reuse an aggregate from another seed or a different heldout corpus merely because the base weights match.",
        "direction": "positive NLL gap means relative bank fit exceeds complement fit; it is not by itself proof of harmful forgetting or a complete causal mechanism",
        "independence_limits": "Complement excludes bank exactly but belongs to the same frozen teacher generation pool, which participated in statistics/selection. It is not a new independent heldout test. Common generation/training seed differences persist across the three paired seeds.",
        "selection_policy": "No tuning, bank changes, main-model selection or official test exclusion based on this exploratory outcome; final and best are separate strata",
        "execution_requirements": "Use the frozen main baseline/adapted execution policy and same tokenizer; include per-chunk doc_id/identity, NLL and forward KL; root must approve scheduling and bind scorer SHA before GPU execution",
        "planned_model_cases": 15, "initial_cases": 3, "adapted_cases": 12, "corpora_per_model_case": 2,
        "checkpoint_sha_binding_source": {"path": str(checkpoint_manifest_path), "sha256": sha(checkpoint_manifest_path)},
        "formal_registry_sha256": sha(registry_path), "formal_audit_sha256": sha(audit_path),
        "cpu_only": True, "real_model_inference_performed": False, "real_model_test_or_behavioral_scores_read": False}
    write(output / "protocol.json", protocol)
    manifest = {"status": "prepared_only", "phase": protocol["phase"], "active": False,
        "protocol_sha256": sha(output / "protocol.json"), "source_script_sha256": sha(__file__),
        "partitions": partitions, "evaluation_cases": evaluations, "gpu_used": False}
    write(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "partitions": len(partitions), "cases": len(evaluations), "output": str(output)}))


if __name__ == "__main__":
    main()
