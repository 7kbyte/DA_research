"""Deterministic, prediction-token-budgeted single-GPU full-model training."""
from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from fractions import Fraction
from typing import Any, Callable, Iterator, Optional

import torch
from torch import Tensor

from .objectives import causal_cross_entropy, causal_forward_kl, valid_prediction_mask


@dataclasses.dataclass
class TrainConfig:
    model_name_or_path: str
    target_path: str
    output_dir: str
    model_revision: Optional[str] = None
    replay_path: Optional[str] = None
    seed: int = 11
    target_prediction_tokens: int = 10_000_000
    replay_ratio: float = 0.0
    replay_objective: str = "none"
    replay_weight: float = 1.0
    microbatch_size: int = 8
    gradient_accumulation_steps: int = 4
    target_tokens_per_step: Optional[int] = None
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    precision: str = "bf16"
    device: str = "cuda"
    attention_implementation: str = "sdpa"
    gradient_checkpointing: bool = False
    loss_chunk_size: int = 128
    fused_adamw: bool = True
    deterministic_algorithms: bool = False
    allow_tf32: bool = True
    save_every_steps: int = 100
    eval_every_steps: int = 100
    evaluate_at_start: bool = False
    evaluate_at_end: bool = True
    validation_paths: dict[str, str] = dataclasses.field(default_factory=dict)
    evaluation_batch_size: int = 8
    evaluate_teacher_kl: bool = False
    keep_step_checkpoints: bool = False
    max_steps: Optional[int] = None
    deadline_utc: Optional[str] = None
    deadline_margin_seconds: float = 90.0
    resume_from: Optional[str] = None
    run_label: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: dict) -> "TrainConfig":
        unknown = set(payload) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown training config fields: {sorted(unknown)}")
        return cls(**payload)

    def validate(self) -> None:
        if self.replay_objective not in {"none", "ce", "kl"}:
            raise ValueError("replay_objective must be none, ce, or kl")
        if self.target_prediction_tokens < 1 or self.microbatch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("token budget and batch sizes must be positive")
        if self.target_tokens_per_step is not None and self.target_tokens_per_step < 1:
            raise ValueError("target_tokens_per_step must be positive")
        if not 0 <= self.replay_ratio <= 1 or self.replay_weight < 0:
            raise ValueError("replay_ratio must be in [0,1] and replay_weight nonnegative")
        if self.replay_objective == "none" and self.replay_ratio != 0:
            raise ValueError("none objective requires replay_ratio=0")
        if self.replay_objective != "none" and (not self.replay_path or self.replay_ratio <= 0):
            raise ValueError("replay objective needs replay_path and positive ratio")
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError("precision must be bf16 or fp32")
        if not 0 <= self.warmup_ratio < 1 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("invalid LR schedule")
        if self.learning_rate <= 0 or self.loss_chunk_size < 1 or self.max_grad_norm < 0:
            raise ValueError("invalid learning rate, loss chunk size, or clipping")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.deadline_utc:
            parse_utc(self.deadline_utc)


def parse_utc(value: str) -> float:
    parsed = dt.datetime.fromisoformat(value.replace(" UTC", "+00:00").replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.timestamp()


def replay_tokens_due(target_tokens: int, replay_ratio: float | str, replay_tokens: int = 0) -> int:
    """Integer cumulative budget; error is always less than one prediction token."""
    ratio = Fraction(str(replay_ratio))
    return (target_tokens * ratio.numerator // ratio.denominator) - replay_tokens


def learning_rate_at_exposure(config: TrainConfig, exposure: int) -> float:
    """Schedule depends solely on target exposure and is shared across methods."""
    progress = min(max(exposure / config.target_prediction_tokens, 0.0), 1.0)
    if config.warmup_ratio and progress < config.warmup_ratio:
        scale = progress / config.warmup_ratio
    else:
        phase = (progress - config.warmup_ratio) / (1.0 - config.warmup_ratio)
        scale = config.min_lr_ratio + (1.0 - config.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * phase))
    return config.learning_rate * scale


def _torch_load(path: str | Path) -> Any:
    # Explicit trusted local tensor artifacts, compatible with PyTorch 2.3.
    return torch.load(path, map_location="cpu", weights_only=False)


def load_training_data(path: str | Path) -> dict:
    payload = _torch_load(path)
    ids = payload.get("input_ids")
    if not isinstance(ids, Tensor) or ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] < 2:
        raise ValueError(f"Invalid input_ids in {path}")
    ids = ids.to(dtype=torch.long, device="cpu").contiguous()
    mask = payload.get("attention_mask")
    mask = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask.bool().cpu().contiguous()
    if mask.shape != ids.shape:
        raise ValueError("attention_mask must match input_ids")
    valid = valid_prediction_mask(ids, mask)
    counts = valid.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Each training chunk must have at least one valid prediction token")
    seen = set()
    unique_prediction_tokens = 0
    for index in range(len(ids)):
        digest = hashlib.sha256(ids[index].numpy().tobytes() + mask[index].numpy().tobytes()).digest()
        if digest not in seen:
            seen.add(digest)
            unique_prediction_tokens += int(counts[index])
    manifest_path = Path(path).with_suffix(".json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if "unique_prediction_tokens" in manifest and manifest["unique_prediction_tokens"] != unique_prediction_tokens:
            raise ValueError(f"Manifest unique_prediction_tokens disagrees with tensor chunk hashes: {path}")
    return {"input_ids": ids, "attention_mask": mask, "prediction_mask": valid,
            "unique_chunks": len(seen), "unique_prediction_tokens": unique_prediction_tokens,
            "prediction_counts": counts, "doc_ids": payload.get("doc_ids"),
            "prediction_tokens": int(counts.sum()), "raw_input_tokens": int(mask.sum()),
            "rows": int(ids.shape[0]), "sequence_length": int(ids.shape[1])}


def data_fingerprint(path: str | Path, data: dict) -> dict:
    digest = hashlib.sha256()
    digest.update(data["input_ids"].numpy().tobytes())
    digest.update(data["attention_mask"].numpy().tobytes())
    return {"path": str(Path(path).resolve()), "sha256_tensors": digest.hexdigest(),
            **{k: data[k] for k in ("prediction_tokens", "raw_input_tokens", "rows", "sequence_length", "unique_chunks", "unique_prediction_tokens")}}


class TokenStream:
    """Independently seeded shuffled chunk stream with resumable partial chunks.

    Each valid causal prediction position is used exactly once per epoch. A
    partial chunk carries its full preceding context, while only unconsumed
    prediction positions contribute to loss and exposure accounting.
    """
    def __init__(self, data: dict, seed: int):
        self.data = data
        self.seed = seed
        self.epoch = 0
        self.row = 0
        self.offset = 0
        self.exposure = 0
        self.trace_hash = "0" * 64
        self._set_order()

    def _set_order(self) -> None:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.order = torch.randperm(self.data["rows"], generator=generator).tolist()

    def state_dict(self) -> dict:
        return {"seed": self.seed, "epoch": self.epoch, "row": self.row,
                "offset": self.offset, "exposure": self.exposure, "trace_hash": self.trace_hash}

    def load_state_dict(self, state: dict) -> None:
        if state["seed"] != self.seed:
            raise ValueError("Stream seed changed across resumption")
        for name in ("epoch", "row", "offset", "exposure", "trace_hash"):
            setattr(self, name, state[name])
        self._set_order()

    def take(self, prediction_tokens: int, batch_size: int) -> Iterator[dict[str, Tensor | int]]:
        remaining = prediction_tokens
        items = []
        while remaining:
            index = self.order[self.row]
            valid_positions = self.data["prediction_mask"][index].nonzero(as_tuple=False).flatten()
            count = min(remaining, len(valid_positions) - self.offset)
            selected = valid_positions[self.offset:self.offset + count]
            length = int(selected[-1]) + 2
            loss_mask = torch.zeros(length - 1, dtype=torch.bool)
            loss_mask[selected] = True
            items.append((self.data["input_ids"][index, :length],
                          self.data["attention_mask"][index, :length], loss_mask))
            self.trace_hash = hashlib.sha256(f"{self.trace_hash}:{self.epoch}:{index}:{self.offset}:{count}".encode()).hexdigest()
            self.offset += count
            self.exposure += count
            remaining -= count
            if self.offset == len(valid_positions):
                self.offset = 0
                self.row += 1
                if self.row == self.data["rows"]:
                    self.row = 0
                    self.epoch += 1
                    self._set_order()
            if len(items) == batch_size or remaining == 0:
                length = max(len(item[0]) for item in items)
                ids = torch.zeros(len(items), length, dtype=torch.long)
                attention = torch.zeros_like(ids, dtype=torch.bool)
                loss = torch.zeros(len(items), length - 1, dtype=torch.bool)
                for row, (item_ids, item_attention, item_loss) in enumerate(items):
                    ids[row, :len(item_ids)] = item_ids
                    attention[row, :len(item_ids)] = item_attention
                    loss[row, :len(item_loss)] = item_loss
                yield {"input_ids": ids, "attention_mask": attention, "prediction_mask": loss,
                       "prediction_tokens": int(loss.sum()), "raw_input_tokens": int(attention.sum()),
                       "padded_input_tokens": ids.numel()}
                items = []


def _json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _write_event(events_path: Path, event_type: str, **payload) -> None:
    event = {"event": event_type, "utc": dt.datetime.now(dt.timezone.utc).isoformat(), **payload}
    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _rng_state() -> dict:
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _model_logits(model, ids: Tensor, attention: Tensor) -> Tensor:
    kwargs = {"input_ids": ids, "use_cache": False}
    if not bool(attention.all()):
        kwargs["attention_mask"] = attention
    result = model(**kwargs)
    return result.logits if hasattr(result, "logits") else result[0]


def _model_load(config: TrainConfig, teacher: bool = False):
    from transformers import AutoModelForCausalLM
    kwargs = {"revision": config.model_revision,
              "torch_dtype": torch.float32,
              "attn_implementation": config.attention_implementation}
    return AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **kwargs)


def train(config: TrainConfig, student=None, teacher=None,
          validation_callback: Optional[Callable[[Any, dict], dict]] = None) -> dict:
    """Train one independent run. Supplied models enable dependency-light CPU tests.

    Callback ``callback(student, state) -> JSON-compatible metrics`` runs in eval
    mode under no_grad. State includes counters, config, and checkpoint_path.
    Checkpoints include full optimizer, RNG and partial-stream state.
    """
    invocation_started = time.monotonic()
    config.validate()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    events = output / "events.jsonl"
    if events.exists() and not config.resume_from:
        raise FileExistsError(f"Run already exists: {events}; use resume_from or a new output_dir")
    _json_atomic(output / "config.json", dataclasses.asdict(config))
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(config.deterministic_algorithms)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
    device = torch.device(config.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    target_data = load_training_data(config.target_path)
    replay_data = load_training_data(config.replay_path) if config.replay_path else None
    provenance = {"target": data_fingerprint(config.target_path, target_data),
                  "replay": data_fingerprint(config.replay_path, replay_data) if replay_data else None,
                  "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                  "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                    for name in ("train.py", "objectives.py")},
                  "device": str(device), "gpu": torch.cuda.get_device_name(device) if use_cuda else None}
    target_stream = TokenStream(target_data, config.seed)
    replay_stream = TokenStream(replay_data, config.seed + 1_000_003) if replay_data else None
    student = _model_load(config) if student is None else student
    student = student.to(device=device, dtype=torch.float32)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    student.config.use_cache = False
    if config.gradient_checkpointing:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    need_teacher = config.replay_objective == "kl" or config.evaluate_teacher_kl
    if need_teacher:
        teacher = _model_load(config, teacher=True) if teacher is None else teacher
        teacher = teacher.to(device=device, dtype=torch.float32)
        teacher.eval()
        teacher.requires_grad_(False)
    else:
        teacher = None
    decay, no_decay = [], []
    for name, parameter in student.named_parameters():
        (no_decay if parameter.ndim < 2 or name.endswith("bias") else decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0}], lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon,
        fused=config.fused_adamw and use_cuda)
    counters = {"step": 0, "target_prediction_tokens": 0, "replay_prediction_tokens": 0,
                "target_raw_input_tokens": 0, "replay_raw_input_tokens": 0,
                "target_padded_input_tokens": 0, "replay_padded_input_tokens": 0,
                "training_seconds": 0.0, "validation_seconds": 0.0, "checkpoint_seconds": 0.0,
                "setup_seconds": 0.0, "validation_teacher_forward_seconds": 0.0,
                "validation_kl_arithmetic_seconds": 0.0,
                "prior_elapsed_seconds": 0.0}
    trajectory = []
    best_target_validation = None
    if config.resume_from:
        saved = _torch_load(config.resume_from)
        if "optimizer_state" not in saved:
            raise ValueError("Resumption requires a full training checkpoint, not a model-only selection checkpoint")
        # Preserve stale post-checkpoint events before retrying uncommitted steps.
        if events.exists():
            saved_step = saved["counters"]["step"]
            original_lines = events.read_text().splitlines()
            active_lines, stale_count = [], 0
            for line in original_lines:
                event = json.loads(line)
                event_step = event.get("step", event.get("counters", {}).get("step", 0))
                if event_step <= saved_step:
                    active_lines.append(line)
                else:
                    stale_count += 1
            if stale_count:
                archive = output / f"events.pre_resume_{time.time_ns()}.jsonl"
                archive.write_text("\n".join(original_lines) + "\n")
                temporary_events = events.with_suffix(".jsonl.tmp")
                temporary_events.write_text("\n".join(active_lines) + "\n")
                os.replace(temporary_events, events)
                _write_event(events, "rollback_after_checkpoint", step=saved_step,
                             archived_events=str(archive), stale_events=stale_count)
        # Operational controls may change; the scientific protocol must not.
        operational = {"resume_from", "device", "deadline_utc", "deadline_margin_seconds", "max_steps",
                       "save_every_steps", "eval_every_steps", "evaluate_at_start", "evaluate_at_end",
                       "keep_step_checkpoints", "output_dir"}
        previous = saved["config"]
        changes = {k: (previous.get(k), v) for k, v in dataclasses.asdict(config).items()
                   if k not in operational and previous.get(k) != v}
        if changes:
            raise ValueError(f"Scientific config changed during resumption: {changes}")
        if saved["provenance"]["target"]["sha256_tensors"] != provenance["target"]["sha256_tensors"]:
            raise ValueError("Target dataset changed across resumption")
        if saved["provenance"]["replay"] != provenance["replay"]:
            raise ValueError("Replay dataset changed across resumption")
        student.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        counters.update(saved["counters"])
        target_stream.load_state_dict(saved["target_stream"])
        if replay_stream:
            replay_stream.load_state_dict(saved["replay_stream"])
        trajectory = saved.get("validation_trajectory", [])
        best_target_validation = saved.get("best_target_validation")
        _restore_rng(saved["rng_state"])
    counters["setup_seconds"] += time.monotonic() - invocation_started
    start_wall = time.monotonic()
    elapsed_before = counters["prior_elapsed_seconds"]
    stop_requested = []
    previous_handlers = {}
    def request_stop(signum, frame):
        stop_requested.append(signum)
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[signum] = signal.signal(signum, request_stop)
        except ValueError:  # Tests can invoke train from a worker thread.
            pass
    latest_path = checkpoints / "latest.pt"
    def elapsed():
        return elapsed_before + time.monotonic() - start_wall
    def save_checkpoint():
        started = time.monotonic()
        counters["prior_elapsed_seconds"] = elapsed()
        payload = {"format_version": 1, "config": dataclasses.asdict(config), "provenance": provenance,
                   "model_state": student.state_dict(), "optimizer_state": optimizer.state_dict(),
                   "counters": dict(counters), "target_stream": target_stream.state_dict(),
                   "replay_stream": replay_stream.state_dict() if replay_stream else None,
                   "rng_state": _rng_state(), "validation_trajectory": trajectory,
                   "best_target_validation": best_target_validation}
        temporary = latest_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, latest_path)
        if config.keep_step_checkpoints:
            destination = checkpoints / f"step_{counters['step']:06d}.pt"
            if destination.exists():
                destination.unlink()
            os.link(latest_path, destination)
        counters["checkpoint_seconds"] += time.monotonic() - started
        _write_event(events, "checkpoint", step=counters["step"], path=str(latest_path),
                     checkpoint_seconds=counters["checkpoint_seconds"])
    def validate():
        nonlocal best_target_validation
        if validation_callback is None and not config.validation_paths:
            return
        started = time.monotonic()
        student.eval()
        state = {**counters, "config": dataclasses.asdict(config), "elapsed_seconds": elapsed(),
                 "checkpoint_path": str(latest_path)}
        with torch.no_grad():
            if validation_callback:
                metrics = validation_callback(student, state)
            else:
                from .evaluate import evaluate_model
                metrics = {name: evaluate_model(student, path, batch_size=config.evaluation_batch_size,
                            device=str(device), teacher=teacher, precision=config.precision)
                           for name, path in config.validation_paths.items()}
        evaluation_elapsed = time.monotonic() - started
        result = {"step": counters["step"], "target_prediction_tokens": counters["target_prediction_tokens"],
                  "replay_prediction_tokens": counters["replay_prediction_tokens"], "metrics": metrics}
        trajectory.append(result)
        target_nll = metrics.get("target", {}).get("nll")
        if target_nll is not None and math.isfinite(target_nll) and (
                best_target_validation is None or target_nll < best_target_validation["nll"]):
            best_path = checkpoints / "checkpoint_best.pt"
            best_target_validation = {"nll": target_nll, "step": counters["step"],
                                      "target_prediction_tokens": counters["target_prediction_tokens"],
                                      "checkpoint_path": str(best_path)}
            best_save_started = time.monotonic()
            temporary = best_path.with_suffix(".pt.tmp")
            torch.save({"format_version": 1, "checkpoint_kind": "model_only_best_target_validation",
                        "config": dataclasses.asdict(config), "provenance": provenance,
                        "model_state": student.state_dict(), "counters": dict(counters),
                        "selection": best_target_validation}, temporary)
            os.replace(temporary, best_path)
            counters["checkpoint_seconds"] += time.monotonic() - best_save_started
            _write_event(events, "best_target_validation", **best_target_validation)
        counters["validation_seconds"] += evaluation_elapsed
        for corpus_metrics in metrics.values():
            if isinstance(corpus_metrics, dict):
                counters["validation_teacher_forward_seconds"] += corpus_metrics.get("teacher_forward_seconds", 0.0)
                counters["validation_kl_arithmetic_seconds"] += corpus_metrics.get("kl_arithmetic_seconds", 0.0)
        _write_event(events, "validation", **result)
        student.train()
    def autocast():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if config.precision == "bf16" else contextlib.nullcontext()
    target_per_step = config.target_tokens_per_step or (
        config.microbatch_size * config.gradient_accumulation_steps * (target_data["sequence_length"] - 1))
    _write_event(events, "resume" if config.resume_from else "start", config=dataclasses.asdict(config),
                 provenance=provenance, counters=counters, target_tokens_per_step=target_per_step,
                 trainable_parameters=sum(p.numel() for p in student.parameters()),
                 weight_dtype=str(next(student.parameters()).dtype),
                 replay_bank_unique_prediction_tokens=replay_data["unique_prediction_tokens"] if replay_data else 0,
                 replay_bank_capacity_prediction_tokens=replay_data["prediction_tokens"] if replay_data else 0,
                 replay_bank_unique_chunks=replay_data["unique_chunks"] if replay_data else 0)
    status = "completed"
    student.train()
    try:
        if config.evaluate_at_start and counters["step"] == 0:
            validate()
        while counters["target_prediction_tokens"] < config.target_prediction_tokens:
            if stop_requested:
                status = "interrupted"
                break
            if config.max_steps and counters["step"] >= config.max_steps:
                status = "max_steps"
                break
            if config.deadline_utc and time.time() >= parse_utc(config.deadline_utc) - config.deadline_margin_seconds:
                status = "deadline"
                break
            step_start = time.monotonic()
            if use_cuda:
                torch.cuda.synchronize(device)
            wanted_target = min(target_per_step, config.target_prediction_tokens - counters["target_prediction_tokens"])
            target_after = counters["target_prediction_tokens"] + wanted_target
            wanted_replay = replay_tokens_due(target_after, config.replay_ratio, counters["replay_prediction_tokens"])
            # Use midpoint exposure so all methods have an identical nonzero warmup schedule.
            lr = learning_rate_at_exposure(config, counters["target_prediction_tokens"] + wanted_target // 2)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            target_sum, replay_sum = 0.0, 0.0
            step_counts = {"target_prediction_tokens": 0, "replay_prediction_tokens": 0,
                           "target_raw_input_tokens": 0, "replay_raw_input_tokens": 0,
                           "target_padded_input_tokens": 0, "replay_padded_input_tokens": 0}
            for kind, stream, wanted in (("target", target_stream, wanted_target),
                                         ("replay", replay_stream, wanted_replay)):
                if not wanted:
                    continue
                for batch in stream.take(wanted, config.microbatch_size):
                    ids = batch["input_ids"].to(device)
                    attention = batch["attention_mask"].to(device)
                    mask = batch["prediction_mask"].to(device)
                    with autocast():
                        student_logits = _model_logits(student, ids, attention)
                        if kind == "replay" and config.replay_objective == "kl":
                            with torch.no_grad():
                                teacher_logits = _model_logits(teacher, ids, attention)
                            stats = causal_forward_kl(student_logits, teacher_logits, ids, attention, mask,
                                                      chunk_size=config.loss_chunk_size)
                        else:
                            stats = causal_cross_entropy(student_logits, ids, attention, mask,
                                                         chunk_size=config.loss_chunk_size)
                        loss = stats.loss_sum / wanted
                        if kind == "replay":
                            loss = loss * config.replay_weight
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite {kind} loss at step {counters['step'] + 1}")
                    loss.backward()
                    value = float(stats.loss_sum.detach())
                    if kind == "target":
                        target_sum += value
                    else:
                        replay_sum += value
                    for suffix in ("prediction_tokens", "raw_input_tokens", "padded_input_tokens"):
                        step_counts[f"{kind}_{suffix}"] += int(batch[suffix])
                    # Release logits before the next forward, including teacher output.
                    del student_logits, stats, loss
                    if kind == "replay" and config.replay_objective == "kl":
                        del teacher_logits
            if step_counts["target_prediction_tokens"] != wanted_target or step_counts["replay_prediction_tokens"] != wanted_replay:
                raise AssertionError("Prediction-token accounting mismatch")
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.max_grad_norm if config.max_grad_norm else float("inf"))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradients at step {counters['step'] + 1}")
            optimizer.step()
            if use_cuda:
                torch.cuda.synchronize(device)
            counters["step"] += 1
            for name, value in step_counts.items():
                counters[name] += value
            step_seconds = time.monotonic() - step_start
            counters["training_seconds"] += step_seconds
            _write_event(events, "train_step", step=counters["step"], learning_rate=lr,
                         target_ce=target_sum / wanted_target,
                         replay_loss=replay_sum / wanted_replay if wanted_replay else None,
                         replay_weight=config.replay_weight, replay_ratio=config.replay_ratio,
                         realized_replay_ratio=counters["replay_prediction_tokens"] / counters["target_prediction_tokens"],
                         grad_norm=float(grad_norm), step_seconds=step_seconds,
                         target_prediction_tokens_per_second=wanted_target / step_seconds,
                         step_counts=step_counts, counters=counters,
                         target_stream_hash=target_stream.trace_hash,
                         replay_stream_hash=replay_stream.trace_hash if replay_stream else None,
                         peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if use_cuda else 0)
            if config.eval_every_steps and counters["step"] % config.eval_every_steps == 0:
                validate()
            if config.save_every_steps and counters["step"] % config.save_every_steps == 0:
                save_checkpoint()
        if config.evaluate_at_end and (not trajectory or trajectory[-1]["step"] != counters["step"]):
            validate()
        save_checkpoint()
        summary = {"status": status, "config": dataclasses.asdict(config), "provenance": provenance,
                   "counters": {**counters, "elapsed_seconds": elapsed(),
                                "total_accounted_wall_seconds": elapsed() + counters["setup_seconds"]},
                   "target_tokens_per_step": target_per_step,
                   "replay_bank_unique_prediction_tokens": replay_data["unique_prediction_tokens"] if replay_data else 0,
                   "replay_bank_capacity_prediction_tokens": replay_data["prediction_tokens"] if replay_data else 0,
                   "replay_bank_unique_chunks": replay_data["unique_chunks"] if replay_data else 0,
                   "replay_bank_raw_input_tokens": replay_data["raw_input_tokens"] if replay_data else 0,
                   "replay_budget_planned": replay_tokens_due(config.target_prediction_tokens, config.replay_ratio),
                   "replay_ratio_realized": counters["replay_prediction_tokens"] / max(counters["target_prediction_tokens"], 1),
                   "target_stream": target_stream.state_dict(),
                   "replay_stream": replay_stream.state_dict() if replay_stream else None,
                   "validation_trajectory": trajectory, "best_target_validation": best_target_validation,
                   "checkpoint_path": str(latest_path),
                   "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if use_cuda else 0}
        _json_atomic(output / "summary.json", summary)
        _write_event(events, "finish", status=status, counters=summary["counters"])
        return summary
    except BaseException as error:
        _write_event(events, "error", exception=type(error).__name__, message=str(error), counters=counters)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
