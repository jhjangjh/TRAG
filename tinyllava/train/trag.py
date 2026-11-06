"""Core TRAG objective, separated from the training loop for testing and reuse."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def intermediate_layers(num_layers: int):
    """Zero-based layer indices l with relative depth 0.3 <= l/M <= 0.6.

    For example, M=24 -> 8..14 and M=28 -> 9..16. Integer arithmetic avoids
    float rounding at the boundaries.
    """
    if num_layers < 2:
        raise ValueError("at least two decoder layers are required")
    first = -(-3 * num_layers // 10)  # ceil(0.3 * M)
    last = 6 * num_layers // 10       # floor(0.6 * M)
    return list(range(first, last + 1))


@dataclass(frozen=True)
class TRAGOutput:
    loss: torch.Tensor
    token_loss: torch.Tensor
    entropy: torch.Tensor
    forward_kl: torch.Tensor
    reverse_kl: torch.Tensor
    forward_weight: torch.Tensor


def aggregate_visual_attention(attention: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Average layers and renormalize on the visual-key span (Eqs. 8-9).

    Args:
        attention: Tensor shaped ``[layers, ..., visual_tokens]``.
        eps: Numerical-stability constant.
    """
    if attention.ndim < 2:
        raise ValueError("attention must contain layer and visual-token dimensions")
    if attention.shape[0] == 0 or attention.shape[-1] == 0:
        raise ValueError("attention dimensions must be non-empty")

    aggregated = attention.float().mean(dim=0)
    return aggregated / (aggregated.sum(dim=-1, keepdim=True) + eps)


def update_entropy_bounds(
    entropy: torch.Tensor,
    previous: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    momentum: float = 0.995,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update running mini-batch entropy min/max bounds from Eq. 12."""
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    if entropy.numel() == 0:
        raise ValueError("entropy must be non-empty")

    batch_min = entropy.detach().float().amin()
    batch_max = entropy.detach().float().amax()
    if previous is None:
        return batch_min, batch_max

    previous_min, previous_max = previous
    updated_min = momentum * previous_min.to(batch_min) + (1.0 - momentum) * batch_min
    updated_max = momentum * previous_max.to(batch_max) + (1.0 - momentum) * batch_max
    return updated_min, updated_max


def trag_loss(
    teacher_attention: torch.Tensor,
    student_attention: torch.Tensor,
    entropy_min: torch.Tensor,
    entropy_max: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> TRAGOutput:
    """Compute entropy-weighted token-level forward/reverse KL (Eqs. 11-14).

    Both attention tensors must be normalized distributions shaped
    ``[..., response_tokens, visual_tokens]`` or ``[response_tokens, visual_tokens]``.
    """
    if teacher_attention.shape != student_attention.shape:
        raise ValueError(
            f"teacher/student shapes differ: {teacher_attention.shape} != {student_attention.shape}"
        )
    if teacher_attention.ndim < 2:
        raise ValueError("attention tensors must include response and visual-token dimensions")

    teacher = teacher_attention.detach().float()
    student = student_attention.float()

    teacher_log = teacher.clamp_min(eps).log()
    student_log = student.clamp_min(eps).log()
    entropy = -(teacher * teacher_log).sum(dim=-1)
    denominator = entropy_max.to(entropy) - entropy_min.to(entropy) + eps
    forward_weight = ((entropy - entropy_min.to(entropy)) / denominator).clamp(0.0, 1.0)

    forward_kl = (teacher * (teacher_log - student_log)).sum(dim=-1)
    reverse_kl = (student * (student_log - teacher_log)).sum(dim=-1)
    token_loss = forward_weight * forward_kl + (1.0 - forward_weight) * reverse_kl

    if reduction == "mean":
        loss = token_loss.mean()
    elif reduction == "sum":
        loss = token_loss.sum()
    elif reduction == "none":
        loss = token_loss
    else:
        raise ValueError(f"unsupported reduction: {reduction}")

    return TRAGOutput(loss, token_loss, entropy, forward_kl, reverse_kl, forward_weight)
