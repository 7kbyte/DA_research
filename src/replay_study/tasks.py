"""Frozen zero-shot PiQA/HellaSwag formats and paired behavioral evaluation.

Formatting/encoding follows EleutherAI lm-evaluation-harness commit
d6de81643928d653435c431bae19945d41d32520 (MIT, Copyright 2020 EleutherAI).
The upstream license and exact sources are retained in artifacts/behavioral/sources.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .evaluate import _device_of, _synchronize, _token_nll, precision_context

HARNESS_REVISION = "d6de81643928d653435c431bae19945d41d32520"
TASK_SOURCES = {
    "piqa": {"repo": "baber/piqa", "revision": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
             "file": "piqa_validation.parquet", "expected_items": 1838, "license": "AFL-3.0"},
    "hellaswag": {"repo": "Rowan/hellaswag", "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
                  "file": "data/validation-00000-of-00001.parquet", "expected_items": 10042, "license": "MIT"},
}


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def hellaswag_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def format_example(task: str, document: Mapping[str, Any], index: int) -> dict[str, Any]:
    if task == "piqa":
        context = f"Question: {document['goal']}\nAnswer:"
        choices = [document["sol1"], document["sol2"]]
    elif task == "hellaswag":
        context = hellaswag_preprocess(document["activity_label"] + ": " + document["ctx_a"] + " " + document["ctx_b"].capitalize())
        choices = [hellaswag_preprocess(ending) for ending in document["endings"]]
    else:
        raise ValueError(f"unsupported task: {task}")
    gold = int(document["label"])
    if not 0 <= gold < len(choices) or any(not choice for choice in choices):
        raise ValueError("invalid gold label or empty candidate")
    row = {"task": task, "item_id": f"{task}:validation:{index}", "source_row_index": index,
           "context": context, "choices": choices, "gold": gold}
    row["fingerprint"] = _fingerprint({key: row[key] for key in ("task", "item_id", "context", "choices", "gold")})
    return row


def encode_pair(tokenizer: Any, context: str, continuation: str, prefix_token_id: int = 0) -> tuple[list[int], list[int], bool]:
    """Match harness causal joint tokenization and trailing-whitespace transfer."""
    if not context:
        return [prefix_token_id], tokenizer.encode(continuation, add_special_tokens=False), True
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tokenizer.encode(context + continuation, add_special_tokens=False)
    context_tokens = tokenizer.encode(context, add_special_tokens=False)
    continuation_tokens = whole[len(context_tokens):]
    return context_tokens, continuation_tokens, whole[:len(context_tokens)] == context_tokens


def prepare_choice(tokenizer: Any, context: str, choice: str, max_length: int = 2048,
                   prefix_token_id: int = 0) -> dict[str, Any]:
    context_ids, continuation_ids, boundary_matches = encode_pair(tokenizer, context, " " + choice, prefix_token_id)
    if not context_ids or not continuation_ids:
        raise ValueError("choice scoring requires nonempty context and continuation token sequences")
    if len(continuation_ids) > max_length:
        raise ValueError("continuation exceeds max_length; no silent answer truncation is allowed")
    # Harness feeds (context+continuation)[-(max_length+1):][:-1], allowing
    # max_length input positions to predict the final continuation token.
    keep_context = max_length + 1 - len(continuation_ids)
    removed = max(0, len(context_ids) - keep_context)
    context_ids = context_ids[-keep_context:]
    full = context_ids + continuation_ids
    return {"input_ids": full[:-1], "labels": full[1:],
            "continuation_start": len(context_ids) - 1,
            "continuation_tokens": len(continuation_ids), "context_tokens_removed": removed,
            "boundary_prefix_matches": boundary_matches, "character_length": len(choice)}


def choice_predictions(logprobs: Sequence[float], choices: Sequence[str], gold: int) -> dict[str, Any]:
    if len(logprobs) != len(choices) or not logprobs or not all(math.isfinite(float(x)) for x in logprobs):
        raise ValueError("finite logprobs must match the nonempty choice list")
    lengths = [len(choice) for choice in choices]
    if min(lengths) < 1 or not 0 <= gold < len(choices):
        raise ValueError("invalid choices or gold label")
    normalized = [float(score) / length for score, length in zip(logprobs, lengths)]
    # Python max preserves the first index on exact ties, matching numpy.argmax.
    pred = max(range(len(choices)), key=lambda i: logprobs[i])
    pred_norm = max(range(len(choices)), key=lambda i: normalized[i])
    return {"choice_logprobs": [float(x) for x in logprobs], "choice_character_lengths": lengths,
            "choice_character_normalized_logprobs": normalized,
            "prediction_acc": pred, "prediction_acc_norm": pred_norm,
            "correct_acc": int(pred == gold), "correct_acc_norm": int(pred_norm == gold)}


def evaluate_task(model: torch.nn.Module, tokenizer: Any, examples: Sequence[Mapping[str, Any]],
                   batch_size: int = 16, device: str | torch.device | None = None,
                   precision: str = "bf16", max_length: int = 2048) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Teacher-force every full answer; score answer positions only, without sampling."""
    if not examples or batch_size < 1 or max_length < 1:
        raise ValueError("need examples, positive batch_size and max_length")
    if len({row["task"] for row in examples}) != 1:
        raise ValueError("evaluate_task accepts one task at a time")
    run_begin = time.perf_counter()
    device = torch.device(device) if device is not None else _device_of(model)
    encoded = []
    for row_index, row in enumerate(examples):
        for choice_index, choice in enumerate(row["choices"]):
            item = prepare_choice(tokenizer, row["context"], choice, max_length)
            item.update(row_index=row_index, choice_index=choice_index)
            encoded.append(item)
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]["input_ids"]), reverse=True)
    encoding_seconds = time.perf_counter() - run_begin
    logprobs = [[0.] * len(row["choices"]) for row in examples]
    was_training = model.training
    model.eval()
    _synchronize(device)
    begin = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                items = [encoded[index] for index in order[start:start + batch_size]]
                length = max(len(item["input_ids"]) for item in items)
                ids = torch.zeros(len(items), length, dtype=torch.long, device=device)
                labels, valid = torch.zeros_like(ids), torch.zeros_like(ids, dtype=torch.bool)
                answer_mask = torch.zeros_like(valid)
                for i, item in enumerate(items):
                    width = len(item["input_ids"])
                    ids[i, :width] = torch.tensor(item["input_ids"], device=device)
                    labels[i, :width] = torch.tensor(item["labels"], device=device)
                    valid[i, :width] = True
                    answer_mask[i, item["continuation_start"]:width] = True
                with precision_context(device, precision):
                    logits = model(input_ids=ids, attention_mask=valid, use_cache=False).logits
                token_nll = _token_nll(logits, labels, position_chunk_size=64)
                scores = -(token_nll * answer_mask).sum(dim=1)
                for item, score in zip(items, scores.cpu().tolist()):
                    logprobs[item["row_index"]][item["choice_index"]] = score
    finally:
        model.train(was_training)
    _synchronize(device)
    predictions = []
    offset = 0
    for index, example in enumerate(examples):
        entries = encoded[offset:offset + len(example["choices"])]
        offset += len(entries)
        predictions.append({"task": example["task"], "item_id": example["item_id"],
                            "fingerprint": example["fingerprint"], "gold": example["gold"],
                            "context": example["context"], "choices": list(example["choices"]),
                            **choice_predictions(logprobs[index], example["choices"], example["gold"]),
                            "choice_token_counts": [x["continuation_tokens"] for x in entries],
                            "context_tokens_removed": [x["context_tokens_removed"] for x in entries],
                            "boundary_prefix_matches": [x["boundary_prefix_matches"] for x in entries]})
    n = len(predictions)
    result = {"task": examples[0]["task"], "items": n,
              "acc": sum(x["correct_acc"] for x in predictions) / n,
              "acc_norm": sum(x["correct_acc_norm"] for x in predictions) / n,
              "official_metrics": ["acc", "acc_norm"], "official_single_primary_metric": None,
              "project_primary_metric": "acc_norm", "project_auxiliary_metric": "acc",
              "acc_norm_denominator": "Python character length of choice without added target delimiter",
              "wall_seconds": time.perf_counter() - run_begin,
              "encoding_wall_seconds": encoding_seconds,
              "inference_and_aggregation_wall_seconds": time.perf_counter() - begin, "precision": precision,
              "model_weight_dtype": str(next(model.parameters()).dtype), "max_length": max_length,
              "scored_continuation_tokens": sum(x["continuation_tokens"] for x in encoded),
              "choices_with_context_truncation": sum(x["context_tokens_removed"] > 0 for x in encoded),
              "boundary_prefix_mismatch_choices": sum(not x["boundary_prefix_matches"] for x in encoded),
              "batch_size": batch_size, "batching": "stable_descending_input_length_right_padding"}
    return result, predictions


def transition_summary(initial: Sequence[Mapping[str, Any]], current: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Paired answer transitions; reject missing items or different task versions."""
    def indexed(rows):
        result = {(row["task"], row["item_id"]): row for row in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate item IDs in prediction file")
        return result
    before, after = indexed(initial), indexed(current)
    if not before or before.keys() != after.keys():
        raise ValueError("initial/current item sets must match exactly")
    for key in before:
        if any(before[key][field] != after[key][field] for field in ("fingerprint", "gold")):
            raise ValueError("initial/current task fingerprint or gold mismatch")
    result = {"items": len(before), "metrics": {}}
    for metric in ("acc", "acc_norm"):
        groups = {name: [] for name in ("CC", "CW", "WC", "WW")}
        for key in before:
            left = bool(before[key]["correct_" + metric])
            right = bool(after[key]["correct_" + metric])
            groups[("C" if left else "W") + ("C" if right else "W")].append(key[1])
        counts = {name: len(ids) for name, ids in groups.items()}
        initial_correct, initial_wrong = counts["CC"] + counts["CW"], counts["WC"] + counts["WW"]
        result["metrics"][metric] = {"counts": counts, "item_ids": groups,
            "initial_correct": initial_correct, "initial_wrong": initial_wrong,
            "conditional_forgetting_rate": counts["CW"] / initial_correct if initial_correct else None,
            "conditional_acquisition_rate": counts["WC"] / initial_wrong if initial_wrong else None,
            "initial_accuracy": initial_correct / len(before),
            "current_accuracy": (counts["CC"] + counts["WC"]) / len(before),
            "accuracy_delta": (counts["WC"] - counts["CW"]) / len(before)}
    return result


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
