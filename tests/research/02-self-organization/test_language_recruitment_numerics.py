from __future__ import annotations

import torch

from minicells.language_growing_organism import ACTIVITY_BUDGET, ACTIVITY_RATE
from minicells.language_recruitment_numerics import stable_gated_replicator_activity


def _reference_forward(previous, reaction, availability):
    drive = reaction.float().square().mean(dim=-1).add(1e-8).sqrt()
    mass = availability.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    mean = (availability * drive).sum(dim=-1, keepdim=True) / mass
    variance = (availability * (drive - mean).square()).sum(dim=-1, keepdim=True) / mass
    fitness = (drive - mean) / variance.sqrt().clamp_min(1e-4)
    growth = torch.exp(ACTIVITY_RATE * fitness).clamp_max(20.0)
    prior = availability if previous is None else previous.float() * availability
    updated = prior * growth
    return ACTIVITY_BUDGET * updated / updated.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def test_stable_replicator_is_forward_equivalent_on_regular_inputs() -> None:
    torch.manual_seed(19019)
    reaction = torch.randn(2, 5, 4, 8)
    availability = torch.rand(2, 5, 4).mul(0.8).add(0.2)
    previous = torch.rand(2, 5, 4)
    previous = ACTIVITY_BUDGET * previous / previous.sum(dim=-1, keepdim=True)
    expected = _reference_forward(previous, reaction, availability)
    actual = stable_gated_replicator_activity(previous, reaction, availability)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_zero_gate_gradient_is_finite_when_old_drive_variance_is_tiny() -> None:
    # Four old cells have almost identical reaction RMS while the closed newborn
    # has a much larger counterfactual drive. The old implementation overflows
    # exp(fitness) before clamp_max and produces NaN d(activity)/d(gate).
    gate = torch.tensor(0.0, requires_grad=True)
    availability = torch.stack((
        torch.tensor(1.0),
        torch.tensor(1.0),
        torch.tensor(1.0),
        torch.tensor(1.0),
        gate,
    )).view(1, 1, 5)
    drive = torch.tensor([1.0, 1.00001, 1.0, 1.0, 10.0]).view(1, 1, 5, 1)
    reaction = drive.expand(-1, -1, -1, 8).clone()
    activity = stable_gated_replicator_activity(None, reaction, availability)
    loss = activity[..., 0].sum()
    gradient = torch.autograd.grad(loss, gate)[0]
    assert torch.isfinite(activity).all()
    assert torch.isfinite(gradient)


def test_zero_gate_gradient_is_finite_at_exact_zero_variance() -> None:
    gate = torch.tensor(0.0, requires_grad=True)
    availability = torch.stack((torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0), gate)).view(1, 1, 4)
    reaction = torch.ones(1, 1, 4, 8)
    activity = stable_gated_replicator_activity(None, reaction, availability)
    gradient = torch.autograd.grad(activity[..., 0].sum(), gate)[0]
    assert torch.isfinite(activity).all()
    assert torch.isfinite(gradient)
