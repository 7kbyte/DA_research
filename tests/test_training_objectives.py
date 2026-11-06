import torch
import torch.nn.functional as F
from replay_study.objectives import causal_cross_entropy, causal_forward_kl, valid_prediction_mask


def test_ce_matches_reference_value_and_gradient_with_causal_padding_and_partial_mask():
    torch.manual_seed(2)
    logits = torch.randn(2, 6, 13, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    ids = torch.randint(13, (2, 6))
    attention = torch.tensor([[1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1]])
    partial = torch.tensor([[1, 0, 1, 1, 1], [1, 1, 0, 1, 0]], dtype=torch.bool)
    valid = attention[:, :-1].bool() & attention[:, 1:].bool() & partial
    expected = F.cross_entropy(reference[:, :-1][valid], ids[:, 1:][valid], reduction="sum")
    actual = causal_cross_entropy(logits, ids, attention, partial, chunk_size=2)
    assert actual.token_count == 4
    torch.testing.assert_close(actual.loss_sum, expected)
    expected.backward()
    actual.loss_sum.backward()
    torch.testing.assert_close(logits.grad, reference.grad)
    assert logits.grad[:, -1].abs().sum() == 0


def test_forward_kl_matches_teacher_to_student_reference_and_gradient():
    torch.manual_seed(3)
    student = torch.randn(2, 5, 11, requires_grad=True)
    reference = student.detach().clone().requires_grad_()
    teacher = (torch.randn(2, 5, 11) * 2).requires_grad_()
    ids = torch.zeros(2, 5, dtype=torch.long)
    attention = torch.tensor([[1, 1, 1, 0, 0], [0, 1, 1, 1, 1]])
    valid = valid_prediction_mask(ids, attention)
    tlogp = teacher.detach()[:, :-1][valid].log_softmax(-1)
    slogp = reference[:, :-1][valid].log_softmax(-1)
    expected = (tlogp.exp() * (tlogp - slogp)).sum()
    reverse = (slogp.exp() * (slogp - tlogp)).sum()
    actual = causal_forward_kl(student, teacher, ids, attention, chunk_size=2)
    assert actual.token_count == 5
    torch.testing.assert_close(actual.loss_sum, expected)
    assert abs(float(actual.loss_sum - reverse)) > 0.1
    expected.backward()
    actual.loss_sum.backward()
    torch.testing.assert_close(student.grad, reference.grad, atol=2e-7, rtol=2e-6)
    assert teacher.grad is None
    assert student.grad[:, -1].abs().sum() == 0


def test_identical_teacher_student_has_zero_kl_and_gradient():
    torch.manual_seed(4)
    student = torch.randn(2, 7, 17, requires_grad=True)
    ids = torch.zeros(2, 7, dtype=torch.long)
    result = causal_forward_kl(student, student.detach().clone(), ids, chunk_size=3)
    assert result.token_count == 12
    assert float(result.loss_sum) == 0.0
    result.mean.backward()
    assert torch.count_nonzero(student.grad) == 0


def test_empty_prediction_mask_produces_zero_differentiable_loss():
    logits = torch.randn(1, 3, 5, requires_grad=True)
    ids = torch.zeros(1, 3, dtype=torch.long)
    result = causal_cross_entropy(logits, ids, prediction_mask=torch.zeros(1, 2, dtype=torch.bool))
    result.mean.backward()
    assert result.token_count == 0
    assert torch.count_nonzero(logits.grad) == 0
