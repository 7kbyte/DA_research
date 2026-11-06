#!/usr/bin/env python3
"""Evaluate one checkpoint on separately named corpora."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.evaluate import evaluate_corpora, load_causal_model, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Pinned base HF repository or local model directory")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint", help="Training checkpoint with model_state, optional for initial teacher")
    parser.add_argument("--corpus", action="append", required=True, help="NAME=/path/to/chunks.pt (repeatable)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--teacher-kl", action="store_true")
    parser.add_argument("--include-chunks", action="store_true")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    corpora = {}
    for item in args.corpus:
        name, path = item.split("=", 1)
        if name in corpora:
            parser.error(f"duplicate corpus name: {name}")
        corpora[name] = path
    model = load_causal_model(args.model, args.revision, args.device, checkpoint=args.checkpoint)
    teacher = (load_causal_model(args.model, args.revision, args.device) if args.checkpoint else model) if args.teacher_kl else None
    result = {"model": args.model, "revision": args.revision, "checkpoint": args.checkpoint,
              "precision": args.precision, "corpora": evaluate_corpora(
                  model, corpora, batch_size=args.batch_size, device=args.device, teacher=teacher,
                  include_chunks=args.include_chunks, max_batches=args.max_batches, precision=args.precision)}
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
