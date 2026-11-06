#!/usr/bin/env python3
"""Run frozen zero-shot task likelihoods and paired initial/adapted transitions."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.evaluate import load_causal_model, write_json
from replay_study.tasks import HARNESS_REVISION, evaluate_task, load_jsonl, transition_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint", help="Optional training checkpoint containing model_state")
    parser.add_argument("--data-root", type=Path, default=Path("artifacts/behavioral"))
    parser.add_argument("--tasks", nargs="+", choices=("piqa", "hellaswag"), default=["piqa", "hellaswag"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="Initial model summary JSON; pairs both official metrics")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true", help="Only first <=8 questions; excluded from formal results")
    parser.add_argument("--limit", type=int, default=8, help="Used only by --smoke")
    args = parser.parse_args()
    run_begin = time.perf_counter()
    if args.smoke and not 1 <= args.limit <= 8:
        parser.error("smoke must use 1..8 questions per task")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = load_causal_model(args.model, args.revision, args.device, checkpoint=args.checkpoint)
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    if baseline and baseline["harness_revision"] != HARNESS_REVISION:
        raise ValueError("baseline harness revision does not match frozen scorer")
    if baseline and (baseline.get("checkpoint") is not None or baseline["revision"] != args.revision or baseline["model"] != args.model):
        raise ValueError("baseline must be the identical initial model/revision without an adapted checkpoint")
    if baseline and baseline["precision"] != args.precision:
        raise ValueError("baseline and adapted evaluation precision must match")
    if baseline and baseline["evaluation_role"] != ("smoke" if args.smoke else "formal"):
        raise ValueError("cannot pair a smoke result with a formal full-split result")
    summary = {"model": args.model, "revision": args.revision, "checkpoint": args.checkpoint,
               "harness_revision": HARNESS_REVISION, "precision": args.precision,
               "evaluation_role": "smoke" if args.smoke else "formal", "tasks": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        path = args.data_root / f"{task}_validation.jsonl"
        manifest = json.loads(path.with_suffix(".json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["artifact_sha256"]:
            raise ValueError("task artifact hash differs from its frozen manifest")
        if manifest["harness_revision"] != HARNESS_REVISION:
            raise ValueError("task preparation harness revision differs from scorer")
        examples = load_jsonl(path)
        if args.smoke:
            examples = examples[:args.limit]
        metrics, predictions = evaluate_task(model, tokenizer, examples, args.batch_size, args.device,
                                              args.precision, args.max_length)
        prediction_path = args.output.with_name(args.output.stem + f"_{task}.jsonl")
        if baseline:
            prior = baseline["tasks"][task]
            if prior["max_length"] != args.max_length or prior["data_sha256"] != manifest["artifact_sha256"]:
                raise ValueError("baseline/adapted context limit or task data hash differs")
            if hashlib.sha256(Path(prior["predictions_path"]).read_bytes()).hexdigest() != prior["predictions_sha256"]:
                raise ValueError("baseline prediction file hash differs from its summary")
            original = load_jsonl(prior["predictions_path"])
            metrics["transitions"] = transition_summary(original, predictions)
        with prediction_path.open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics.update(predictions_path=str(prediction_path), predictions_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                       data_manifest=str(path.with_suffix(".json")), data_sha256=manifest["artifact_sha256"],
                       dataset_revision=manifest["revision"], complete_split=not args.smoke and len(predictions) == manifest["items"],
                       evaluation_role=summary["evaluation_role"])
        summary["tasks"][task] = metrics
        write_json(args.output, summary)
    summary["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(args.device) if torch.device(args.device).type == "cuda" else 0
    summary["wall_seconds"] = time.perf_counter() - run_begin
    write_json(args.output, summary)
    print(args.output)


if __name__ == "__main__":
    main()
