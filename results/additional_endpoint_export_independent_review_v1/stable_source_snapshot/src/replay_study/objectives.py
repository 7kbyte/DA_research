"""Exact causal CE and teacher-to-student KL with bounded FP32 workspace.

Losses are sums. Callers normalize over prediction tokens, requiring both
adjacent attention-mask positions to be valid. Backward recomputes probabilities
in token chunks instead of retaining full-vocabulary FP32 tensors.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor


def valid_prediction_mask(input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("input_ids must have shape [batch, sequence>=2]")
    if attention_mask is None:
        return torch.ones_like(input_ids[:, 1:], dtype=torch.bool)
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    mask = attention_mask.bool()
    return mask[:, :-1] & mask[:, 1:]


def _checked_mask(input_ids: Tensor, attention_mask: Optional[Tensor], prediction_mask: Optional[Tensor]) -> Tensor:
    result = valid_prediction_mask(input_ids, attention_mask)
    if prediction_mask is not None:
        if prediction_mask.shape != result.shape:
            raise ValueError("prediction_mask must have shape [batch, sequence-1]")
        result = result & prediction_mask.bool()
    return result


class _ChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, mask, chunk_size):
        ctx.save_for_backward(logits, labels, mask)
        ctx.chunk_size = chunk_size
        value = torch.zeros((), dtype=torch.float32, device=logits.device)
        positions = mask.nonzero(as_tuple=False)
        for start in range(0, len(positions), chunk_size):
            p = positions[start:start + chunk_size]
            z = logits[p[:, 0], p[:, 1]].float()
            y = labels[p[:, 0], p[:, 1]]
            value += (torch.logsumexp(z, dim=-1) - z.gather(1, y[:, None]).squeeze(1)).sum()
        return value

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, mask = ctx.saved_tensors
        gradient = torch.zeros_like(logits)
        positions = mask.nonzero(as_tuple=False)
        for start in range(0, len(positions), ctx.chunk_size):
            p = positions[start:start + ctx.chunk_size]
            probabilities = torch.softmax(logits[p[:, 0], p[:, 1]].float(), dim=-1)
            y = labels[p[:, 0], p[:, 1]]
            probabilities.scatter_add_(1, y[:, None], -torch.ones_like(y[:, None], dtype=probabilities.dtype))
            gradient[p[:, 0], p[:, 1]] = (probabilities * grad_output).to(logits.dtype)
        return gradient, None, None, None


class _ChunkedForwardKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_logits, teacher_logits, mask, chunk_size):
        ctx.save_for_backward(student_logits, teacher_logits, mask)
        ctx.chunk_size = chunk_size
        value = torch.zeros((), dtype=torch.float32, device=student_logits.device)
        positions = mask.nonzero(as_tuple=False)
        for start in range(0, len(positions), chunk_size):
            p = positions[start:start + chunk_size]
            teacher_logp = torch.log_softmax(teacher_logits[p[:, 0], p[:, 1]].float(), dim=-1)
            student_logp = torch.log_softmax(student_logits[p[:, 0], p[:, 1]].float(), dim=-1)
            value += (teacher_logp.exp() * (teacher_logp - student_logp)).sum()
        return value

    @staticmethod
    def backward(ctx, grad_output):
        student_logits, teacher_logits, mask = ctx.saved_tensors
        gradient = torch.zeros_like(student_logits)
        positions = mask.nonzero(as_tuple=False)
        for start in range(0, len(positions), ctx.chunk_size):
            p = positions[start:start + ctx.chunk_size]
            student_p = torch.softmax(student_logits[p[:, 0], p[:, 1]].float(), dim=-1)
            teacher_p = torch.softmax(teacher_logits[p[:, 0], p[:, 1]].float(), dim=-1)
            gradient[p[:, 0], p[:, 1]] = ((student_p - teacher_p) * grad_output).to(student_logits.dtype)
        return gradient, None, None, None


@dataclass
class LossStats:
    loss_sum: Tensor
    token_count: int

    @property
    def mean(self) -> Tensor:
        return self.loss_sum / max(self.token_count, 1)


def causal_cross_entropy(logits: Tensor, input_ids: Tensor, attention_mask: Optional[Tensor] = None,
                         prediction_mask: Optional[Tensor] = None, chunk_size: int = 128) -> LossStats:
    """Exact full-vocabulary causal CE; last-position logits never contribute."""
    if logits.shape[:2] != input_ids.shape or chunk_size < 1:
        raise ValueError("logits/input shape mismatch or invalid chunk_size")
    mask = _checked_mask(input_ids, attention_mask, prediction_mask)
    value = _ChunkedCE.apply(logits, input_ids[:, 1:], mask, chunk_size)
    return LossStats(value, int(mask.sum().item()))


def causal_forward_kl(student_logits: Tensor, teacher_logits: Tensor, input_ids: Tensor,
                      attention_mask: Optional[Tensor] = None, prediction_mask: Optional[Tensor] = None,
                      chunk_size: int = 128) -> LossStats:
    """Sum D_KL(teacher || student), all vocabulary items, temperature=1.

    Teacher logits are detached even if accidentally supplied with gradients.
    """
    if student_logits.shape != teacher_logits.shape or student_logits.shape[:2] != input_ids.shape:
        raise ValueError("student, teacher and input shapes must agree")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    mask = _checked_mask(input_ids, attention_mask, prediction_mask)
    value = _ChunkedForwardKL.apply(student_logits, teacher_logits.detach(), mask, chunk_size)
    return LossStats(value, int(mask.sum().item()))
