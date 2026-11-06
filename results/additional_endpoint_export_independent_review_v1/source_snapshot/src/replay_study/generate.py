"""Frozen initial-teacher candidate generation with complete sampling provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .evaluate import _device_of, _synchronize, load_causal_model, load_tensor_data, precision_context, write_json
from .selection import chunk_hashes


@dataclass
class GenerationConfig:
    model: str
    revision: str
    output: str
    seed: int
    n_chunks: int = 1957
    sequence_length: int = 512
    batch_size: int = 64
    device: str = "cuda:0"
    dtype: str = "float32"
    precision: str = "bf16"
    prompt_mode: str = "bos"
    prompt_data: str | None = None
    prompt_length: int = 32
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    save_every_batches: int = 5
    resume: bool = False
    role: str = "formal"


def _rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def generation_diagnostics(input_ids: torch.Tensor, eos_token_id: int | None = None) -> dict[str, Any]:
    """Descriptive diagnostics only: never filter or rewrite generated chunks."""
    rows = input_ids.cpu().long()
    hashes = chunk_hashes({"input_ids": rows})
    adjacent_rates, ngram_repeat_rates, most_common_rates = [], [], []
    for row in rows[:, 1:].tolist():
        adjacent_rates.append(sum(a == b for a, b in zip(row[:-1], row[1:])) / max(1, len(row) - 1))
        grams = [tuple(row[i:i + 4]) for i in range(max(0, len(row) - 3))]
        ngram_repeat_rates.append(1 - len(set(grams)) / len(grams) if grams else 0.0)
        _, counts = np.unique(row, return_counts=True)
        most_common_rates.append(float(counts.max()) / len(row))
    result = {"exact_duplicate_chunk_rate": 1 - len(set(hashes)) / len(hashes),
              "mean_adjacent_repeated_token_rate": float(np.mean(adjacent_rates)),
              "mean_within_chunk_repeated_4gram_rate": float(np.mean(ngram_repeat_rates)),
              "mean_most_common_token_fraction": float(np.mean(most_common_rates)),
              "chunks_with_repeated_4gram_rate_above_0_5": int(sum(x > 0.5 for x in ngram_repeat_rates)),
              "diagnostic_policy": "descriptive_only_no_filtering"}
    if eos_token_id is not None:
        eos = rows[:, 1:] == eos_token_id
        result.update(sampled_eos_tokens=int(eos.sum()), sampled_eos_rate=float(eos.float().mean()),
                      chunks_containing_sampled_eos=int(eos.any(dim=1).sum()))
    return result


def generate_pool(config: GenerationConfig, model: torch.nn.Module | None = None,
                  tokenizer: Any = None, progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Generate fixed-length independent sequences; EOS never terminates sampling.

    The frozen teacher is always the pinned initial model. Batch size is part of
    the reproducibility protocol because multinomial RNG depends on batching.
    Resume files preserve RNG and accumulated GPU time between batch boundaries.
    """
    if config.n_chunks < 1 or config.sequence_length < 2 or config.batch_size < 1:
        raise ValueError("n_chunks, batch_size and sequence_length must be positive (length>=2)")
    if not config.revision:
        raise ValueError("a pinned teacher revision is required")
    if config.prompt_mode not in ("bos", "data"):
        raise ValueError("prompt_mode must be bos or data")
    if config.temperature <= 0 or not 0 < config.top_p <= 1 or config.top_k < 0:
        raise ValueError("invalid sampling temperature, top_p or top_k")
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.pt")
    if output.exists() and not config.resume:
        raise FileExistsError(f"refusing to replace an existing candidate pool: {output}")
    from transformers import AutoTokenizer, GenerationConfig as HFGenerationConfig
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
    bos = config.bos_token_id if config.bos_token_id is not None else tokenizer.bos_token_id
    eos = config.eos_token_id if config.eos_token_id is not None else tokenizer.eos_token_id
    if bos is None:
        raise ValueError("BOS-only generation requires an explicit or tokenizer BOS token")
    prompt_metadata = None
    prompt_indices = None
    if config.prompt_mode == "bos":
        prompts = torch.full((config.n_chunks, 1), bos, dtype=torch.long)
    else:
        if not config.prompt_data or not 1 <= config.prompt_length < config.sequence_length:
            raise ValueError("data prompts need prompt_data and 1<=prompt_length<sequence_length")
        prompt_path = Path(config.prompt_data)
        prompt_manifest = prompt_path.with_suffix(".json")
        if not prompt_manifest.exists():
            raise ValueError("prompt data require a provenance manifest proving a training-only split")
        prompt_metadata = json.loads(prompt_manifest.read_text())
        split = str(prompt_metadata.get("split", "")).lower()
        if "train" not in split or any(word in split for word in ("valid", "test", "eval")):
            raise ValueError("generation prompt manifest split must explicitly identify training data")
        prompt_pool = load_tensor_data(prompt_path)
        if len(prompt_pool["input_ids"]) < config.n_chunks:
            raise ValueError("need at least n_chunks unique source prompt rows")
        if prompt_pool["input_ids"].shape[1] < config.prompt_length:
            raise ValueError("prompt artifact sequences are shorter than requested prefixes")
        generator = torch.Generator().manual_seed(config.seed)
        prompt_indices = torch.randperm(len(prompt_pool["input_ids"]), generator=generator)[:config.n_chunks]
        prompts = prompt_pool["input_ids"][prompt_indices, :config.prompt_length].long()
        if "attention_mask" in prompt_pool and not prompt_pool["attention_mask"][prompt_indices, :config.prompt_length].all():
            raise ValueError("prompt prefixes must be entirely valid")
    if model is None:
        model = load_causal_model(config.model, config.revision, config.device, config.dtype)
    device = _device_of(model)
    model.eval()
    model.requires_grad_(False)
    hf_config = HFGenerationConfig.from_model_config(model.config)
    hf_config.do_sample = True
    hf_config.temperature, hf_config.top_p, hf_config.top_k = config.temperature, config.top_p, config.top_k
    hf_config.max_new_tokens = config.sequence_length - prompts.shape[1]
    hf_config.min_new_tokens = 0
    hf_config.eos_token_id = None
    hf_config.forced_eos_token_id = None
    hf_config.pad_token_id = None  # no padding is needed or permitted in fixed-length sampling
    hf_config.bos_token_id = bos
    hf_config.use_cache = True
    hf_config.repetition_penalty = 1.0
    hf_config.no_repeat_ngram_size = 0
    hf_config.num_beams = 1
    hf_config.num_return_sequences = 1
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    batches, completed, previous_seconds = [], 0, 0.0
    protocol = asdict(config)
    protocol.pop("resume")
    if config.resume and partial.exists():
        saved = torch.load(partial, map_location="cpu", weights_only=False)
        if saved["protocol"] != protocol:
            raise ValueError("resume protocol does not match saved generation configuration")
        batches = [saved["input_ids"]]
        completed = len(saved["input_ids"])
        previous_seconds = saved["generation_wall_seconds"]
        _restore_rng(saved["rng_state"])
    elif output.exists():
        raise FileExistsError("completed pool already exists; resume only applies to .partial.pt")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    begin = time.perf_counter()
    with torch.inference_mode():
        for batch_index, start in enumerate(range(completed, config.n_chunks, config.batch_size)):
            batch_begin = time.perf_counter()
            x = prompts[start:start + config.batch_size].to(device)
            # Transformers 4.46 falls back from explicit eos=None to the model's
            # default generation config. Override both, restoring the model even
            # if generation fails; otherwise post-EOS PAD can masquerade as data.
            original_generation_config = model.generation_config
            try:
                model.generation_config = hf_config
                with precision_context(device, config.precision):
                    generated = model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=hf_config)
            finally:
                model.generation_config = original_generation_config
            if generated.shape != (len(x), config.sequence_length):
                raise RuntimeError(f"fixed-length generation failed: shape={tuple(generated.shape)}")
            batches.append(generated.to(device="cpu", dtype=torch.int32))
            completed += len(x)
            _synchronize(device)
            elapsed = previous_seconds + time.perf_counter() - begin
            event = {"completed_chunks": completed, "total_chunks": config.n_chunks,
                     "batch_wall_seconds": time.perf_counter() - batch_begin,
                     "generation_wall_seconds": elapsed,
                     "sampled_tokens_per_second": completed * hf_config.max_new_tokens / elapsed}
            if progress is not None:
                progress(event)
            if config.save_every_batches > 0 and (batch_index + 1) % config.save_every_batches == 0:
                torch.save({"protocol": protocol, "input_ids": torch.cat(batches), "rng_state": _rng_state(),
                            "generation_wall_seconds": elapsed}, partial)
    _synchronize(device)
    elapsed = previous_seconds + time.perf_counter() - begin
    ids = torch.cat(batches).contiguous()
    artifact = {"input_ids": ids,
                "doc_ids": [f"generated:seed{config.seed}:chunk{i}" for i in range(config.n_chunks)]}
    torch.save(artifact, output)
    manifest = {"source": "initial_frozen_teacher_self_generation", "split": "generated_training_candidates",
                "experiment_role": config.role,
                "teacher": config.model, "revision": config.revision, "tokenizer": config.model,
                "tokenizer_revision": config.revision, "config": asdict(config), "seed": config.seed,
                "chunks": config.n_chunks, "sequence_length": config.sequence_length,
                "raw_input_tokens": ids.numel(), "prediction_tokens": config.n_chunks * (config.sequence_length - 1),
                "prompt_tokens": prompts.numel(), "sampled_continuation_tokens": config.n_chunks * hf_config.max_new_tokens,
                "prompt_policy": "BOS_only" if config.prompt_mode == "bos" else "training_only_prefix",
                "prompt_in_replay_policy": "entire_chunk_included; first_token_excluded_by_causal_shift",
                "prompt_indices": prompt_indices.tolist() if prompt_indices is not None else None,
                "prompt_manifest": prompt_metadata, "bos_token_id": bos, "eos_token_id": eos,
                "eos_policy": "continue_sampling_to_fixed_length; EOS_not_a_stop_token",
                "eos_stopping_disabled_in_both_explicit_and_model_generation_config": True,
                "pad_token_id": None,
                "quality_filtering": "none", "document_boundary_policy": "one_independent_generation_per_chunk",
                "generation_wall_seconds": elapsed,
                "sampled_tokens_per_second": config.n_chunks * hf_config.max_new_tokens / elapsed,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                "peak_gpu_memory_measurement": "torch.cuda.max_memory_allocated",
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "torch_version": torch.__version__, "python_version": platform.python_version(),
                "transformers_version": importlib.metadata.version("transformers"),
                "numpy_version": np.__version__, "resolved_hf_generation_config": hf_config.to_dict(),
                "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "diagnostics": generation_diagnostics(ids, eos),
                "text_samples": [tokenizer.decode(row.tolist()) for row in ids[:8]]}
    write_json(output.with_suffix(".json"), manifest)
    if partial.exists():
        partial.unlink()
    return manifest
