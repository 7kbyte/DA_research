#!/usr/bin/env python3
"""Generate shared frozen-teacher pools, extract statistics, and build all banks."""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.evaluate import extract_teacher_statistics, load_causal_model, write_json
from replay_study.generate import GenerationConfig, generate_pool
from replay_study.selection import METHODS, save_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 11, 22, 33])
    parser.add_argument("--pilot-seeds", type=int, nargs="*", default=[101])
    parser.add_argument("--n-chunks", type=int, default=1957)
    parser.add_argument("--bank-size", type=int, default=196)
    parser.add_argument("--generation-batch-size", type=int, default=256)
    parser.add_argument("--statistics-batch-size", type=int, default=16)
    parser.add_argument("--c4-pool", default="artifacts/data/c4_candidate.pt")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--config-root", type=Path, default=Path("configs"))
    parser.add_argument("--summary", help="Required with custom output roots; default results/generation_summary.json")
    args = parser.parse_args()
    if args.summary is None:
        if args.artifact_root != Path("artifacts") or args.config_root != Path("configs"):
            parser.error("custom artifact/config roots require an explicit --summary path to protect the main study")
        args.summary = "results/generation_summary.json"
    from transformers import AutoTokenizer
    begin = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    c4_manifest = json.loads(Path(args.c4_pool).with_suffix(".json").read_text())
    c4_tokenizer = c4_manifest["tokenizer"]
    if not isinstance(c4_tokenizer, dict) or not c4_tokenizer.get("revision"):
        raise ValueError("C4 manifest must identify a pinned tokenizer repository and revision")
    reference_path = c4_tokenizer["repo"]
    provenance_path = Path("artifacts/assets/model_provenance.json")
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if (provenance.get("repo") == reference_path and provenance.get("revision") == c4_tokenizer["revision"]
                and Path(provenance["path"]).exists()):
            reference_path = provenance["path"]
    reference_tokenizer = AutoTokenizer.from_pretrained(reference_path, revision=c4_tokenizer["revision"])
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    reference_backend = json.loads(reference_tokenizer.backend_tokenizer.to_str())
    if backend != reference_backend or tokenizer.special_tokens_map != reference_tokenizer.special_tokens_map:
        raise ValueError("new model tokenizer algorithm/IDs/special tokens do not match the C4 candidate tokenizer")
    model = load_causal_model(args.model, args.revision, args.device)
    summary = {"teacher": args.model, "revision": args.revision,
               "precision": "FP32_weights_BF16_CUDA_autocast", "device": args.device,
               "gpu": torch.cuda.get_device_name(args.device), "model_load_wall_seconds": time.perf_counter() - begin,
               "artifact_root": str(args.artifact_root), "config_root": str(args.config_root),
               "c4_tokenizer_compatibility": "exact_backend_JSON_and_special_tokens_match",
               "tokenizer_backend_sha256": hashlib.sha256(json.dumps(backend, sort_keys=True).encode()).hexdigest(),
               "candidate_pool_prediction_tokens": args.n_chunks * 511,
               "bank_prediction_tokens": args.bank_size * 511,
               "pilot_seeds": args.pilot_seeds, "pools": {}}
    for seed in args.seeds:
        role = "pilot" if seed in args.pilot_seeds else "formal"
        pool_path = args.artifact_root / f"generated/pool_seed{seed}.pt"
        stats_path = args.artifact_root / f"generated/pool_seed{seed}_stats.pt"
        config = GenerationConfig(model=args.model, revision=args.revision, output=str(pool_path), seed=seed,
                                  n_chunks=args.n_chunks, batch_size=args.generation_batch_size,
                                  device=args.device, role=role)
        write_json(args.config_root / f"generation/seed{seed}.json", asdict(config))
        print(json.dumps({"stage": "generation_start", "seed": seed, "role": role}), flush=True)
        if pool_path.exists():
            generation = json.loads(pool_path.with_suffix(".json").read_text())
            saved_protocol, requested_protocol = dict(generation["config"]), asdict(config)
            saved_protocol.pop("resume", None)
            requested_protocol.pop("resume", None)
            if saved_protocol != requested_protocol:
                raise ValueError(f"existing pool protocol conflicts with requested generation: seed {seed}")
        else:
            config.resume = pool_path.with_suffix(".partial.pt").exists()
            generation = generate_pool(config, model, tokenizer, progress=lambda event: print(
                json.dumps({"stage": "generation_progress", "seed": seed, **event}), flush=True))
        summary["pools"][str(seed)] = {"role": role, "generation": generation}
        write_json(args.summary, summary)
        torch.cuda.reset_peak_memory_stats(args.device)
        stats = extract_teacher_statistics(model, pool_path, batch_size=args.statistics_batch_size, device=args.device)
        stats.update(teacher=args.model, revision=args.revision, source_pool=str(pool_path),
                     source_pool_sha256=hashlib.sha256(pool_path.read_bytes()).hexdigest())
        torch.save(stats, stats_path)
        metadata = {key: value for key, value in stats.items() if not isinstance(value, torch.Tensor)}
        metadata.update(chunks=len(stats["teacher_nll"]), prediction_tokens=int(stats["prediction_token_counts"].sum()),
                        feature_dimensions=stats["features"].shape[1],
                        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(args.device),
                        artifact_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(),
                        teacher_nll_mean=stats["teacher_nll"].mean().item(),
                        teacher_nll_std=stats["teacher_nll"].std().item())
        write_json(stats_path.with_suffix(".json"), metadata)
        summary["pools"][str(seed)]["teacher_statistics"] = metadata
        banks = {}
        for method in METHODS:
            output = str(args.artifact_root / f"banks/generated_{method}_seed{seed}.pt")
            manifest = save_bank(pool_path, output, method=method, seed=seed, bank_size=args.bank_size,
                                 statistics=stats_path)
            banks[method] = {"artifact": output, **manifest}
        output = str(args.artifact_root / f"banks/c4_random_seed{seed}.pt")
        banks["c4_random"] = {"artifact": output, **save_bank(args.c4_pool, output, method="random", seed=seed,
                                                               bank_size=args.bank_size)}
        if generation["diagnostics"]["exact_duplicate_chunk_rate"] == 0:
            assert banks["random"]["selected_indices"] == banks["dedup_random"]["selected_indices"]
        summary["pools"][str(seed)]["banks"] = banks
        summary["orchestration_wall_seconds"] = time.perf_counter() - begin
        write_json(args.summary, summary)
        print(json.dumps({"stage": "seed_complete", "seed": seed, "role": role,
                          "generation_wall_seconds": generation["generation_wall_seconds"],
                          "statistics_wall_seconds": stats["wall_seconds"],
                          "bank_paths": {name: value["artifact"] for name, value in banks.items()}}), flush=True)
    summary["complete"] = True
    summary["orchestration_wall_seconds"] = time.perf_counter() - begin
    write_json(args.summary, summary)


if __name__ == "__main__":
    main()
