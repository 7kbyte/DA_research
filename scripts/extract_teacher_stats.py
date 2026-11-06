#!/usr/bin/env python3
"""Extract teacher NLL and mean last-layer features for one immutable pool."""
import argparse
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.evaluate import extract_teacher_statistics, load_causal_model, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--no-features", action="store_true")
    args = parser.parse_args()
    model = load_causal_model(args.model, args.revision, args.device)
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    result = extract_teacher_statistics(model, args.pool, args.batch_size, args.device,
                                        not args.no_features, precision=args.precision)
    result.update(teacher=args.model, revision=args.revision, source_pool=args.pool,
                  source_pool_sha256=hashlib.sha256(Path(args.pool).read_bytes()).hexdigest())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    metadata = {key: value for key, value in result.items() if not isinstance(value, torch.Tensor)}
    metadata.update(chunks=len(result["teacher_nll"]), prediction_tokens=int(result["prediction_token_counts"].sum()),
                    feature_dimensions=result["features"].shape[1] if "features" in result else None,
                    peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(args.device) if torch.device(args.device).type == "cuda" else 0,
                    artifact_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    write_json(output.with_suffix(".json"), metadata)
    print(output)


if __name__ == "__main__":
    main()
