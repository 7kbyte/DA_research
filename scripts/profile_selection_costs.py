#!/usr/bin/env python3
"""Independent NLL-only / feature-only / shared-joint preparation cost diagnostic.

No training, selection, generation, or frozen evaluation implementation is changed.
The diagnostic uses the currently frozen strict training/evaluation execution policy;
comparison errors against earlier frozen statistics are reported without hiding them.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.evaluate import (extract_teacher_statistics, load_causal_model,
                                  load_tensor_data, precision_context, prediction_mask, write_json)
from replay_study.execution import configure_execution, execution_provenance


def feature_only_statistics(model, data, batch_size=16, device="cpu", precision="bf16"):
    """Pythia's base transformer only: final normalized hidden means, no LM head."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not hasattr(model, "gpt_neox"):
        raise TypeError("feature-only diagnostic requires the Pythia/GPTNeoX gpt_neox base transformer")
    dataset = load_tensor_data(data)
    ids, attention = dataset["input_ids"], dataset.get("attention_mask")
    device = torch.device(device)
    prior = model.training
    model.eval()
    features, counts_all = [], []
    try:
        with torch.inference_mode():
            for start in range(0, len(ids), batch_size):
                x = ids[start:start + batch_size].to(device=device, dtype=torch.long)
                mask = (attention[start:start + batch_size].to(device=device, dtype=torch.bool)
                        if attention is not None else torch.ones_like(x, dtype=torch.bool))
                valid = prediction_mask(mask)
                counts = valid.sum(dim=1)
                if (counts == 0).any():
                    raise ValueError("feature extraction requires a valid prediction in every chunk")
                with precision_context(device, precision):
                    out = model.gpt_neox(input_ids=x, attention_mask=mask, use_cache=False,
                                         output_hidden_states=False, return_dict=True)
                hidden = out.last_hidden_state[:, :-1].float()
                features.append(((hidden * valid.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)).cpu())
                counts_all.append(counts.cpu())
    finally:
        model.train(prior)
    if not features:
        raise ValueError("cannot extract statistics from an empty pool")
    return {"features": torch.cat(features), "prediction_token_counts": torch.cat(counts_all)}


def compare_tensors(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError("recomputed tensor shape differs from frozen statistics")
    a, b = actual.double(), reference.double()
    delta = (a - b).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {"shape": list(actual.shape), "actual_dtype": str(actual.dtype),
            "reference_dtype": str(reference.dtype), "all_finite": finite,
            "exact_equal": bool(torch.equal(actual, reference)),
            "max_absolute_error": delta.max().item(), "mean_absolute_error": delta.mean().item(),
            "relative_l2_error": (torch.linalg.vector_norm(a - b) /
                                   torch.linalg.vector_norm(b).clamp_min(1e-30)).item()}


def measure_pass(function, device):
    """Whole-pass elapsed CUDA stream span; includes host gaps, not GPU active time."""
    device = torch.device(device)
    pair = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        pair[0].record(torch.cuda.current_stream(device))
    begin = time.perf_counter()
    result = function()
    if pair:
        pair[1].record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
    record = {"wall_seconds": time.perf_counter() - begin,
              "cuda_event_elapsed_seconds": pair[0].elapsed_time(pair[1]) / 1000 if pair else None,
              "timing_mode": "whole_pass_cuda_stream_elapsed_including_host_gaps" if pair else "cpu_wall_perf_counter",
              "peak_gpu_allocated_bytes_including_model": torch.cuda.max_memory_allocated(device) if pair else None,
              "peak_gpu_reserved_bytes_including_model": torch.cuda.max_memory_reserved(device) if pair else None}
    return result, record


def configure_frozen_strict_policy(device):
    """Use frozen execution helper unchanged; no flags affect other processes."""
    settings = SimpleNamespace(device=device, precision="bf16", deterministic_algorithms=True,
        attention_backend="auto", allow_tf32_matmul=True, allow_tf32_cudnn=True,
        cudnn_deterministic=True, cudnn_benchmark=False)
    configure_execution(settings)
    return execution_provenance(settings)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=Path("artifacts/generated/pool_seed11.pt"))
    parser.add_argument("--statistics", type=Path, default=Path("artifacts/generated/pool_seed11_stats.pt"))
    parser.add_argument("--device", required=True, help="Explicit GPU allocation required; no default GPU")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.repetitions < 1 or args.warmup_batches < 0:
        parser.error("batch-size/repetitions must be positive and warmup-batches nonnegative")
    if torch.device(args.device).type != "cuda":
        parser.error("deployed-cost CLI requires an explicitly allocated CUDA GPU; CPU invariants are tested separately")
    stats = torch.load(args.statistics, map_location="cpu", weights_only=False)
    pool_manifest = json.loads(args.pool.with_suffix(".json").read_text())
    stats_manifest = json.loads(args.statistics.with_suffix(".json").read_text())
    pool_sha, stats_sha = sha256(args.pool), sha256(args.statistics)
    if pool_sha != pool_manifest["artifact_sha256"] or pool_sha != stats["source_pool_sha256"]:
        raise ValueError("pool bytes differ from the frozen pool/statistics provenance")
    if stats_sha != stats_manifest["artifact_sha256"]:
        raise ValueError("statistics bytes differ from their frozen manifest")
    if stats["revision"] != pool_manifest["revision"] or stats["precision"] != "bf16":
        raise ValueError("teacher revision or precision differs from the frozen pool/statistics")
    if torch.__version__ != pool_manifest["torch_version"]:
        raise ValueError("Torch build differs from frozen generation; use the original environment")
    import transformers
    if transformers.__version__ != pool_manifest["transformers_version"]:
        raise ValueError("Transformers version differs from frozen generation")
    dataset = load_tensor_data(args.pool)
    policy = configure_frozen_strict_policy(args.device)
    begin = time.perf_counter()
    model = load_causal_model(stats["teacher"], stats["revision"], args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    load_seconds = time.perf_counter() - begin
    modes = {
        "nll_only": lambda data: extract_teacher_statistics(model, data, args.batch_size, args.device, include_features=False),
        "feature_only": lambda data: feature_only_statistics(model, data, args.batch_size, args.device),
        "joint_frozen_pipeline": lambda data: extract_teacher_statistics(model, data, args.batch_size, args.device, include_features=True)}
    result = {"scope": "Independent deployment-cost diagnostic; never add joint cost separately to every method",
              "model": stats["teacher"], "revision": stats["revision"], "device": args.device,
              "gpu": torch.cuda.get_device_name(args.device), "torch_version": torch.__version__,
              "transformers_version": transformers.__version__, "cuda_version": torch.version.cuda,
              "execution_policy": policy, "precision": "FP32 weights with BF16 CUDA autocast",
              "batch_size": args.batch_size, "repetitions": args.repetitions,
              "pool": str(args.pool), "pool_sha256": pool_sha, "statistics": str(args.statistics),
              "statistics_sha256": stats_sha, "script_sha256": sha256(__file__),
              "chunks": len(dataset["input_ids"]), "prediction_tokens": int(stats["prediction_token_counts"].sum()),
              "model_load_wall_seconds": load_seconds, "frozen_joint_pipeline_wall_seconds": stats["wall_seconds"],
              "random_and_exact_dedup_random_require_teacher_statistics": False,
              "method_deployment_accounting": {"M": "NLL-only plus selection", "C": "feature-only plus clustering/selection",
                                                "random": "no teacher statistics", "shared_study_pipeline": "one shared joint pass, count once"},
              "mode_order": list(modes), "modes": {}, "complete": False}
    for mode, function in modes.items():
        torch.cuda.synchronize(args.device)
        torch.cuda.empty_cache()  # isolate allocator caches between deployment modes
        warmup = {"batches": args.warmup_batches, "wall_seconds": 0.}
        if args.warmup_batches:
            n = args.batch_size * args.warmup_batches
            subset = {k: (v[:n] if isinstance(v, (torch.Tensor, list)) else v) for k,v in dataset.items()}
            warmed, timing = measure_pass(lambda: function(subset), args.device)
            warmup.update(timing)
            del warmed
        trials = []
        for repetition in range(args.repetitions):
            recomputed, timing = measure_pass(lambda: function(dataset), args.device)
            comparisons = {key: compare_tensors(value, stats[key]) for key,value in recomputed.items()
                           if key in ("teacher_nll", "features", "prediction_token_counts")}
            if not all(row["all_finite"] for row in comparisons.values()):
                raise ValueError("nonfinite diagnostic output")
            trials.append({"repetition": repetition, **timing, "frozen_stats_comparisons": comparisons})
            del recomputed
        result["modes"][mode] = {"warmup": warmup, "trials": trials}
        write_json(args.output, result)
    result["complete"] = True
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
