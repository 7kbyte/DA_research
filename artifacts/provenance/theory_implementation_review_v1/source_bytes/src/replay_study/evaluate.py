"""Token-exact causal likelihood and frozen-teacher drift evaluation."""
from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


def load_tensor_data(data: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load a trusted local torch artifact, or validate an in-memory mapping."""
    if isinstance(data, (str, Path)):
        data = torch.load(Path(data), map_location="cpu", weights_only=False)
    result = dict(data)
    ids = result.get("input_ids")
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[1] < 2:
        raise ValueError("input_ids must be a [chunks, sequence_length>=2] tensor")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must contain int32 or int64 token IDs")
    mask = result.get("attention_mask")
    if mask is not None and (not isinstance(mask, torch.Tensor) or mask.shape != ids.shape):
        raise ValueError("attention_mask must have the same shape as input_ids")
    if "doc_ids" in result and len(result["doc_ids"]) != len(ids):
        raise ValueError("doc_ids must have one entry per chunk")
    return result


def prediction_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.bool()
    return mask[:, :-1] & mask[:, 1:]


def _device_of(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def precision_context(device: str | torch.device, precision: str = "bf16"):
    """CUDA BF16 autocast over FP32 weights; CPU always uses float32."""
    device = torch.device(device)
    if precision not in ("bf16", "fp32"):
        raise ValueError("precision must be bf16 or fp32")
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and precision == "bf16" else nullcontext()


def _token_nll(logits: torch.Tensor, labels: torch.Tensor, position_chunk_size: int) -> torch.Tensor:
    """Float32 CE with bounded temporary logits memory; result is [batch,time]."""
    pieces = []
    for start in range(0, logits.shape[1], position_chunk_size):
        end = start + position_chunk_size
        block = logits[:, start:end].float()
        pieces.append(F.cross_entropy(block.transpose(1, 2), labels[:, start:end], reduction="none"))
    return torch.cat(pieces, dim=1)


def exact_forward_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                     position_chunk_size: int = 64, vocabulary_chunk_size: int = 4096,
                     clamp_roundoff: bool = True) -> torch.Tensor:
    """Full-vocabulary KL(teacher || student) per position, in float32.

    Both normalizers include the entire vocabulary; chunks only bound temporary
    memory. This is neither a top-k approximation nor sampled KL.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher and student logits must have identical shapes")
    if position_chunk_size < 1 or vocabulary_chunk_size < 1:
        raise ValueError("KL chunk sizes must be positive")
    pieces = []
    for start in range(0, student_logits.shape[1], position_chunk_size):
        end = start + position_chunk_size
        s, t = student_logits[:, start:end].float(), teacher_logits[:, start:end].float()
        slogz, tlogz = torch.logsumexp(s, dim=-1), torch.logsumexp(t, dim=-1)
        kl = torch.zeros_like(slogz)
        for vstart in range(0, s.shape[-1], vocabulary_chunk_size):
            vend = vstart + vocabulary_chunk_size
            logp = t[..., vstart:vend] - tlogz.unsqueeze(-1)
            logq = s[..., vstart:vend] - slogz.unsqueeze(-1)
            kl += (logp.exp() * (logp - logq)).sum(dim=-1)
        pieces.append(kl.clamp_min_(0) if clamp_roundoff else kl)
    return torch.cat(pieces, dim=1)


def aggregate_document_metrics(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sum independent chunk contributions per document, preserving token weights."""
    documents = {}
    for chunk in chunks:
        if "doc_id" not in chunk:
            raise ValueError("document aggregation requires doc_id on every chunk")
        doc_id = chunk["doc_id"]
        row = documents.setdefault(doc_id, {"doc_id": doc_id, "sum_nll": 0.0,
                                             "valid_prediction_tokens": 0, "chunk_count": 0})
        row["sum_nll"] += chunk["sum_nll"]
        row["valid_prediction_tokens"] += chunk["valid_prediction_tokens"]
        row["chunk_count"] += 1
        if "sum_kl" in chunk:
            row["sum_kl"] = row.get("sum_kl", 0.0) + chunk["sum_kl"]
    for row in documents.values():
        count = row["valid_prediction_tokens"]
        row["nll"] = row["sum_nll"] / count if count else None
        if "sum_kl" in row:
            row["kl"] = row["sum_kl"] / count if count else None
    return list(documents.values())


def evaluate_model(model: torch.nn.Module, data: str | Path | Mapping[str, Any],
                   batch_size: int = 4, device: str | torch.device | None = None,
                   teacher: torch.nn.Module | None = None, include_chunks: bool = False,
                   max_batches: int | None = None, position_chunk_size: int = 64,
                   vocabulary_chunk_size: int = 4096, precision: str = "bf16") -> dict[str, Any]:
    """Evaluate independent chunks, without gradients, restoring training modes.

    Models must already reside on ``device``. ``max_batches`` is for explicitly
    labelled smoke tests only; the return value marks incomplete corpus results.
    Valid predictions require both attention_mask[t] and attention_mask[t+1].
    """
    if batch_size < 1 or position_chunk_size < 1:
        raise ValueError("batch_size and position_chunk_size must be positive")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive or None")
    dataset = load_tensor_data(data)
    ids, attention = dataset["input_ids"], dataset.get("attention_mask")
    device = torch.device(device) if device is not None else _device_of(model)
    modes = [(model, model.training)]
    if teacher is not None and teacher is not model:
        modes.append((teacher, teacher.training))
    for module, _ in modes:
        module.eval()
    _synchronize(device)
    begin = time.perf_counter()
    nll_sum, kl_sum, count, raw_input_tokens, processed_chunks = 0.0, 0.0, 0, 0, 0
    kl_raw_min = float("inf")
    teacher_forward_seconds, kl_arithmetic_seconds = 0.0, 0.0
    teacher_events, kl_events = [], []
    chunks = []
    try:
        with torch.inference_mode():
            for batch_index, start in enumerate(range(0, len(ids), batch_size)):
                if max_batches is not None and batch_index >= max_batches:
                    break
                end = min(start + batch_size, len(ids))
                x = ids[start:end].to(device=device, dtype=torch.long)
                mask = (attention[start:end].to(device=device, dtype=torch.bool)
                        if attention is not None else torch.ones_like(x, dtype=torch.bool))
                valid = prediction_mask(mask)
                with precision_context(device, precision):
                    logits = model(input_ids=x, attention_mask=mask, use_cache=False).logits[:, :-1]
                nll = _token_nll(logits, x[:, 1:], position_chunk_size)
                row_counts = valid.sum(dim=1)
                row_nll = (nll * valid).sum(dim=1, dtype=torch.float64)
                row_kl = None
                if teacher is not None:
                    teacher_begin = time.perf_counter()
                    teacher_pair = None
                    if device.type == "cuda" and teacher is not model:
                        teacher_pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                        teacher_pair[0].record(torch.cuda.current_stream(device))
                    with precision_context(device, precision):
                        teacher_logits = (logits if teacher is model else
                                          teacher(input_ids=x, attention_mask=mask, use_cache=False).logits[:, :-1])
                    if teacher_pair is not None:
                        teacher_pair[1].record(torch.cuda.current_stream(device))
                        teacher_events.append(teacher_pair)
                    elif teacher is not model:
                        teacher_forward_seconds += time.perf_counter() - teacher_begin
                    kl_begin = time.perf_counter()
                    kl_pair = None
                    if device.type == "cuda":
                        kl_pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                        kl_pair[0].record(torch.cuda.current_stream(device))
                    token_kl = exact_forward_kl(logits, teacher_logits, position_chunk_size,
                                                vocabulary_chunk_size, clamp_roundoff=False)
                    if valid.any():
                        kl_raw_min = min(kl_raw_min, token_kl[valid].min().item())
                    token_kl = token_kl.clamp_min(0)
                    row_kl = (token_kl * valid).sum(dim=1, dtype=torch.float64)
                    if kl_pair is not None:
                        kl_pair[1].record(torch.cuda.current_stream(device))
                        kl_events.append(kl_pair)
                    else:
                        kl_arithmetic_seconds += time.perf_counter() - kl_begin
                    kl_sum += row_kl.sum().item()
                nll_sum += row_nll.sum().item()
                count += row_counts.sum().item()
                raw_input_tokens += mask.sum().item()
                processed_chunks += end - start
                if include_chunks:
                    for offset in range(end - start):
                        row_count = row_counts[offset].item()
                        chunk = {"chunk_index": start + offset, "sum_nll": row_nll[offset].item(),
                                 "valid_prediction_tokens": row_count,
                                 "nll": row_nll[offset].item() / row_count if row_count else None}
                        if "doc_ids" in dataset:
                            chunk["doc_id"] = dataset["doc_ids"][start + offset]
                        if row_kl is not None:
                            chunk.update(sum_kl=row_kl[offset].item(),
                                         kl=row_kl[offset].item() / row_count if row_count else None)
                        chunks.append(chunk)
    finally:
        for module, was_training in modes:
            module.train(was_training)
    _synchronize(device)
    teacher_forward_seconds += sum(start.elapsed_time(end) / 1000 for start, end in teacher_events)
    kl_arithmetic_seconds += sum(start.elapsed_time(end) / 1000 for start, end in kl_events)
    if count == 0:
        raise ValueError("evaluation data contain no valid next-token predictions")
    mean_nll = nll_sum / count
    result = {"sum_nll": nll_sum, "valid_prediction_tokens": count, "prediction_tokens": count,
              "raw_input_tokens": raw_input_tokens, "nll": mean_nll,
              "ppl": math.exp(mean_nll) if mean_nll < 709 else float("inf"),
              "chunks_evaluated": processed_chunks, "chunks_available": len(ids),
              "complete_corpus": processed_chunks == len(ids), "wall_seconds": time.perf_counter() - begin,
              "mask_policy": "attention_mask[t] AND attention_mask[t+1]; causal shift",
              "precision": precision, "model_weight_dtype": str(next(model.parameters()).dtype)}
    if teacher is not None:
        result.update(sum_kl=kl_sum, kl=kl_sum / count, kl_direction="teacher||student",
                      kl_vocabulary="exact_full_vocabulary", kl_arithmetic="float32",
                      kl_min_before_roundoff_clamp=kl_raw_min,
                      teacher_forward_seconds=teacher_forward_seconds,
                      kl_arithmetic_seconds=kl_arithmetic_seconds,
                      teacher_logits_reused_from_student=teacher is model,
                      teacher_timing_mode="cuda_event_elapsed" if device.type == "cuda" else "cpu_wall_perf_counter")
    if include_chunks:
        result["chunks"] = chunks
        if "doc_ids" in dataset:
            result["documents"] = aggregate_document_metrics(chunks)
    return result


def evaluate_corpora(model: torch.nn.Module, corpora: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Independent corpus results, intentionally without a pooled score."""
    return {name: evaluate_model(model, data, **kwargs) for name, data in corpora.items()}


def extract_teacher_statistics(model: torch.nn.Module, data: str | Path | Mapping[str, Any],
                               batch_size: int = 8, device: str | torch.device | None = None,
                               include_features: bool = True, position_chunk_size: int = 64,
                               precision: str = "bf16") -> dict[str, Any]:
    """Chunk NLL and mean last-layer features over valid prediction contexts.

    Context positions are 0..L-2, using exactly the CE mask. Clustering applies
    L2 normalization later; saved features here are unnormalized float32 means.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dataset = load_tensor_data(data)
    ids, attention = dataset["input_ids"], dataset.get("attention_mask")
    device = torch.device(device) if device is not None else _device_of(model)
    was_training = model.training
    model.eval()
    _synchronize(device)
    begin = time.perf_counter()
    all_nll, all_counts, all_features = [], [], []
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
                    out = model(input_ids=x, attention_mask=mask, use_cache=False, output_hidden_states=include_features)
                nll = _token_nll(out.logits[:, :-1], x[:, 1:], position_chunk_size)
                all_nll.append(((nll * valid).sum(dim=1) / counts).cpu())
                all_counts.append(counts.cpu())
                if include_features:
                    hidden = out.hidden_states[-1][:, :-1].float()
                    all_features.append(((hidden * valid.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)).cpu())
    finally:
        model.train(was_training)
    _synchronize(device)
    if not all_nll:
        raise ValueError("cannot extract statistics from an empty pool")
    result = {"teacher_nll": torch.cat(all_nll), "prediction_token_counts": torch.cat(all_counts),
              "wall_seconds": time.perf_counter() - begin,
              "precision": precision, "model_weight_dtype": str(next(model.parameters()).dtype),
              "feature_policy": "last_layer_mean_of_valid_prediction_contexts" if include_features else None}
    if include_features:
        result["features"] = torch.cat(all_features)
    return result


def load_causal_model(model_path: str, revision: str | None = None, device: str = "cuda:0",
                      dtype: str = "float32", checkpoint: str | None = None) -> torch.nn.Module:
    """Load a pinned HF model, optionally restoring training's model_state dict."""
    from transformers import AutoModelForCausalLM
    dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, torch_dtype=dtypes[dtype],
                                                attn_implementation="sdpa")
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"], strict=True)
    return model.to(device).eval()


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
