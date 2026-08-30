from __future__ import annotations

import copy

import torch

from minicells.language_growing_organism import build_cellular_model
from minicells.language_localized_learning import LocalizedLearningState, conservative_fork
from minicells.language_pressure_recruitment import (
    PressureProfile,
    calibrate_pressure_homeostasis,
    forward_with_pressure_recruitment,
    newborn_recruitment_mean,
    shadow_pressure_trace,
)


def tiny_model():
    torch.manual_seed(18118)
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


def profile_with_threshold(model, threshold: float) -> PressureProfile:
    return PressureProfile(
        mean=torch.zeros(model.iterations, model.max_cells),
        scale=torch.ones(model.iterations, model.max_cells),
        threshold=torch.full((model.iterations, model.max_cells), threshold),
    )


def test_pressure_profile_is_recurrent_step_and_cell_specific() -> None:
    model = tiny_model()
    localized = LocalizedLearningState.capture(model)
    inputs = [torch.randint(0, 32, (4, 10)), torch.randint(0, 32, (4, 10))]
    profile = calibrate_pressure_homeostasis(model, localized, inputs, quantile=0.9)
    assert profile.mean.shape == (model.iterations, model.max_cells)
    assert profile.scale.shape == profile.mean.shape
    assert profile.threshold.shape == profile.mean.shape
    assert torch.isfinite(profile.threshold[:, :3]).all()
    assert torch.isinf(profile.threshold[:, 3:]).all()
    assert (profile.scale[:, :3] > 0).all()


def test_shadow_pressure_is_exactly_feedback_isolated_from_newborn() -> None:
    model = tiny_model()
    inputs = torch.randint(0, 32, (3, 10))
    localized = LocalizedLearningState.capture(model)
    before = shadow_pressure_trace(model, inputs, localized)
    child = conservative_fork(model, 1, step=0, direction=torch.ones(model.dim))
    assert child == 3
    model.connect(0, child)
    model.connect(child, 2)
    with torch.no_grad():
        model.cell_memory[child].mul_(100.0).add_(50.0)
    after = shadow_pressure_trace(model, inputs, localized)
    assert torch.equal(before, after)


def test_force_zero_recruitment_restores_exact_phase1_computation() -> None:
    base = tiny_model()
    phase1_state = copy.deepcopy(base.state_dict())
    inputs = torch.randint(0, 32, (3, 10))
    expected = base.forward_variable(inputs).output.logits.detach()
    base_localized = LocalizedLearningState.capture(base)
    profile = calibrate_pressure_homeostasis(base, base_localized, [inputs])

    adapted = tiny_model()
    adapted.load_state_dict(phase1_state)
    localized = LocalizedLearningState.capture(adapted)
    child = conservative_fork(adapted, 1, step=0, direction=torch.ones(adapted.dim))
    assert child == 3
    adapted.connect(0, child)
    with torch.no_grad():
        adapted.cell_memory[child].add_(3.0)
    actual = forward_with_pressure_recruitment(
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
    localized = LocalizedLearningState.capture(model)
    profile = calibrate_pressure_homeostasis(model, localized, [inputs])
    child = conservative_fork(model, 1, step=0, direction=torch.ones(model.dim))
    assert child == 3
    model.connect(0, child)
    with torch.no_grad():
        model.cell_memory[child].add_(0.5)
    expected = model.forward_variable(inputs).output.logits.detach()
    actual = forward_with_pressure_recruitment(
        model,
        inputs,
        localized,
        profile,
        force_recruitment=1.0,
    ).output.logits.detach()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_pressure_threshold_extremes_control_newborn_without_old_drift() -> None:
    model = tiny_model()
    inputs = torch.randint(0, 32, (2, 10))
    localized = LocalizedLearningState.capture(model)
    child = conservative_fork(model, 1, step=0, direction=torch.ones(model.dim))
    assert child == 3
    old_before = model.cell_memory[:3].detach().clone()
    low = forward_with_pressure_recruitment(model, inputs, localized, profile_with_threshold(model, 1e6))
    high = forward_with_pressure_recruitment(model, inputs, localized, profile_with_threshold(model, -1e6))
    assert float(newborn_recruitment_mean(low, model, localized)) < 0.03
    assert float(newborn_recruitment_mean(high, model, localized)) > 0.99
    assert torch.equal(model.cell_memory[:3].detach(), old_before)


def test_shadow_pressure_does_not_change_when_recruitment_is_forced() -> None:
    model = tiny_model()
    inputs = torch.randint(0, 32, (2, 10))
    localized = LocalizedLearningState.capture(model)
    profile = calibrate_pressure_homeostasis(model, localized, [inputs])
    child = conservative_fork(model, 2, step=0, direction=torch.ones(model.dim))
    assert child == 3
    model.connect(0, child)
    off = forward_with_pressure_recruitment(model, inputs, localized, profile, force_recruitment=0.0)
    on = forward_with_pressure_recruitment(model, inputs, localized, profile, force_recruitment=1.0)
    assert torch.equal(off.shadow_pressure_trace, on.shadow_pressure_trace)
