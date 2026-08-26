from __future__ import annotations

import copy

import torch

from minicells.language_conditional_recruitment import (
    HomeostaticProfile,
    calibrate_homeostasis,
    forward_with_recruitment,
    newborn_recruitment_mean,
)
from minicells.language_growing_organism import build_cellular_model
from minicells.language_localized_learning import LocalizedLearningState, conservative_fork


def tiny_model():
    torch.manual_seed(18018)
    return build_cellular_model(
        32,
        "G",
        max_context=12,
        dim=32,
        heads=4,
        ffn_dim=64,
        iterations=2,
        attention_window=6,
        initial_cells=3,
        max_cells=6,
    )


def profile_with_threshold(model, threshold: float) -> HomeostaticProfile:
    return HomeostaticProfile(
        mean=torch.zeros(model.iterations, model.max_cells, model.dim),
        scale=torch.ones(model.iterations, model.max_cells, model.dim),
        threshold=torch.full((model.iterations, model.max_cells), threshold),
    )


def test_homeostatic_profile_is_recurrent_step_specific() -> None:
    model = tiny_model()
    inputs = [torch.randint(0, 32, (4, 10)), torch.randint(0, 32, (4, 10))]
    profile = calibrate_homeostasis(model, inputs, quantile=0.9)
    assert profile.mean.shape == (model.iterations, model.max_cells, model.dim)
    assert profile.scale.shape == profile.mean.shape
    assert profile.threshold.shape == (model.iterations, model.max_cells)
    assert torch.isfinite(profile.threshold[:, :3]).all()
    assert torch.isinf(profile.threshold[:, 3:]).all()
    assert (profile.scale[:, :3] > 0).all()


def test_force_zero_recruitment_restores_exact_phase1_computation() -> None:
    base = tiny_model()
    phase1_state = copy.deepcopy(base.state_dict())
    inputs = torch.randint(0, 32, (3, 10))
    expected = base.forward_variable(inputs).output.logits.detach()
    profile = calibrate_homeostasis(base, [inputs])

    adapted = tiny_model()
    adapted.load_state_dict(phase1_state)
    localized = LocalizedLearningState.capture(adapted)
    child = conservative_fork(adapted, 1, step=0, direction=torch.ones(adapted.dim))
    assert child == 3
    with torch.no_grad():
        adapted.cell_memory[child].add_(3.0)
        adapted.connect(0, child)
    actual = forward_with_recruitment(
        adapted,
        inputs,
        localized,
        profile,
        force_recruitment=0.0,
    ).output.logits.detach()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_force_one_recruitment_matches_static_adapted_dynamics() -> None:
    model = tiny_model()
    inputs = torch.randint(0, 32, (3, 10))
    profile = calibrate_homeostasis(model, [inputs])
    localized = LocalizedLearningState.capture(model)
    child = conservative_fork(model, 1, step=0, direction=torch.ones(model.dim))
    assert child == 3
    model.connect(0, child)
    with torch.no_grad():
        model.cell_memory[child].add_(0.5)
    expected = model.forward_variable(inputs).output.logits.detach()
    actual = forward_with_recruitment(
        model,
        inputs,
        localized,
        profile,
        force_recruitment=1.0,
    ).output.logits.detach()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_gate_extremes_change_newborn_recruitment_not_old_phenotype() -> None:
    model = tiny_model()
    inputs = torch.randint(0, 32, (2, 10))
    localized = LocalizedLearningState.capture(model)
    child = conservative_fork(model, 1, step=0, direction=torch.ones(model.dim))
    assert child == 3
    old_before = model.cell_memory[:3].detach().clone()

    low = forward_with_recruitment(model, inputs, localized, profile_with_threshold(model, 1e6))
    high = forward_with_recruitment(model, inputs, localized, profile_with_threshold(model, -1e6))
    low_gate = float(newborn_recruitment_mean(low, model, localized))
    high_gate = float(newborn_recruitment_mean(high, model, localized))
    assert low_gate < 0.03
    assert high_gate > 0.99
    assert torch.equal(model.cell_memory[:3].detach(), old_before)


def test_forced_sleep_blocks_newborn_metabolic_and_structural_effect() -> None:
    model = tiny_model()
    inputs = torch.randint(0, 32, (2, 10))
    base_logits = model.forward_variable(inputs).output.logits.detach()
    profile = calibrate_homeostasis(model, [inputs])
    localized = LocalizedLearningState.capture(model)
    child = conservative_fork(model, 2, step=0, direction=torch.ones(model.dim))
    assert child == 3
    model.connect(0, child)
    with torch.no_grad():
        model.cell_memory[child].mul_(20.0)
    result = forward_with_recruitment(model, inputs, localized, profile, force_recruitment=0.0)
    assert float(result.recruitment_trace[..., -1].abs().max()) == 0.0
    assert torch.allclose(result.output.logits.detach(), base_logits, atol=1e-6, rtol=1e-6)
