from __future__ import annotations

import math

import torch

from .language_growing_organism import ACTIVITY_BUDGET, ACTIVITY_RATE


def stable_gated_replicator_activity(
    previous: torch.Tensor | None,
    reaction: torch.Tensor,
    availability: torch.Tensor,
) -> torch.Tensor:
    """Forward-equivalent gated replicator with finite derivatives at zero availability.

    The original implementation used::

        variance.sqrt().clamp_min(1e-4)
        torch.exp(ACTIVITY_RATE * fitness).clamp_max(20.0)

    Those expressions are correct in the forward pass but can create NaN
    gradients at a closed newborn gate. ``sqrt(0)`` has an infinite derivative,
    and ``exp(large)`` can overflow to inf before the outer clamp masks the
    forward value. Autograd can then encounter 0 * inf in the backward pass.

    For non-negative variance and finite fitness these forms are mathematically
    forward-equivalent:

        sqrt(max(variance, 1e-8))
        exp(min(ACTIVITY_RATE * fitness, log(20)))

    The clamp is moved *before* the singular/overflowing operation so the
    derivative path remains finite without changing the intended dynamics.
    """
    drive = reaction.float().square().mean(dim=-1).add(1e-8).sqrt()
    mass = availability.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    mean = (availability * drive).sum(dim=-1, keepdim=True) / mass
    variance = (availability * (drive - mean).square()).sum(dim=-1, keepdim=True) / mass
    std = variance.clamp_min(1e-8).sqrt()
    fitness = (drive - mean) / std
    log_growth = (ACTIVITY_RATE * fitness).clamp_max(math.log(20.0))
    growth = torch.exp(log_growth)
    prior = availability if previous is None else previous.float() * availability
    updated = prior * growth
    return ACTIVITY_BUDGET * updated / updated.sum(dim=-1, keepdim=True).clamp_min(1e-12)
