import torch

from tinyllava.train.trag import (
    aggregate_visual_attention,
    trag_loss,
    update_entropy_bounds,
    intermediate_layers,
)


def test_layer_aggregation_is_normalized():
    attention = torch.tensor(
        [
            [[1.0, 1.0], [3.0, 1.0]],
            [[3.0, 1.0], [1.0, 1.0]],
        ]
    )
    result = aggregate_visual_attention(attention)
    torch.testing.assert_close(result.sum(dim=-1), torch.ones(2))


def test_identical_distributions_have_zero_loss():
    attention = torch.tensor([[0.1, 0.2, 0.7], [0.25, 0.25, 0.5]])
    output = trag_loss(attention, attention, torch.tensor(0.0), torch.tensor(2.0))
    torch.testing.assert_close(output.loss, torch.tensor(0.0))


def test_diffuse_teacher_gets_more_forward_kl_weight():
    teacher = torch.tensor([[0.97, 0.01, 0.01, 0.01], [0.25, 0.25, 0.25, 0.25]])
    student = torch.full_like(teacher, 0.25)
    entropy = -(teacher * teacher.log()).sum(dim=-1)
    output = trag_loss(teacher, student, entropy.amin(), entropy.amax())
    assert output.forward_weight[0].item() == 0.0
    torch.testing.assert_close(output.forward_weight[1], torch.tensor(1.0))


def test_entropy_bounds_use_min_max_ema():
    first = update_entropy_bounds(torch.tensor([1.0, 3.0]), momentum=0.5)
    second = update_entropy_bounds(torch.tensor([2.0, 5.0]), previous=first, momentum=0.5)
    torch.testing.assert_close(second[0], torch.tensor(1.5))
    torch.testing.assert_close(second[1], torch.tensor(4.0))


def test_average_before_normalizing_unequal_visual_mass():
    # The reversed order incorrectly produces [0.5, 0.5].
    raw = torch.tensor([[[0.9, 0.0]], [[0.0, 0.1]]])
    actual = aggregate_visual_attention(raw)
    torch.testing.assert_close(actual, torch.tensor([[0.9, 0.1]]))


def test_intermediate_layers_use_zero_based_relative_depth():
    assert intermediate_layers(28) == list(range(9, 17))
    assert intermediate_layers(32) == list(range(10, 20))
    assert intermediate_layers(24) == list(range(8, 15))
    assert intermediate_layers(36) == list(range(11, 22))
    assert intermediate_layers(40) == list(range(12, 25))
    # Exact boundaries are included and free of float rounding (0.3 * 10 > 3).
    assert intermediate_layers(10) == [3, 4, 5, 6]


def test_kl_mixture_matches_independent_formula_and_has_student_gradient():
    teacher = torch.tensor([[0.8, 0.2], [0.4, 0.6]], requires_grad=True)
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]], requires_grad=True)
    student = logits.softmax(-1)
    lo, hi = torch.tensor(0.1), torch.tensor(1.0)
    entropy = -(teacher.detach() * teacher.detach().log()).sum(-1)
    weight = ((entropy - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    fwd = (teacher.detach() * (teacher.detach().log() - student.log())).sum(-1)
    rev = (student * (student.log() - teacher.detach().log())).sum(-1)
    result = trag_loss(teacher, student, lo, hi)
    torch.testing.assert_close(result.loss, (weight * fwd + (1 - weight) * rev).mean())
    result.loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_default_reduction_is_token_mean():
    teacher = torch.tensor([[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]])
    student = torch.tensor([[0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])
    lo, hi = torch.tensor(0.0), torch.tensor(1.0)
    output = trag_loss(teacher, student, lo, hi)
    torch.testing.assert_close(output.loss, output.token_loss.mean())
    torch.testing.assert_close(trag_loss(teacher, student, lo, hi, reduction="sum").loss,
                               output.token_loss.sum())
