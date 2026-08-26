from __future__ import annotations

import numpy as np
import pytest
import torch

from minicells.language_models import count_parameters
from minicells.language_plastic_metrics import paired_bootstrap_delta
from minicells.language_plastic_tissue import (
    ACTIVITY_BUDGET,
    SYNAPTIC_BUDGET,
    TISSUE_HEIGHT,
    PlasticReactionDiffusionStage,
    _activity_from_state,
    _initial_connectome,
    build_plastic_reaction_diffusion_model,
)
from minicells.language_skill_data import MODEL_LENGTH
from minicells.language_sparse_topology import build_sparse_topology_model


def tiny_model():
    torch.manual_seed(123)
    return build_plastic_reaction_diffusion_model(
        32,
        tissue_height=TISSUE_HEIGHT,
        max_context=MODEL_LENGTH,
        dim=16,
        heads=2,
        ffn_dim=32,
        windows=(4, 8, 16),
        iterations=(2, 2, 2),
    )


def tiny_stage():
    torch.manual_seed(123)
    return PlasticReactionDiffusionStage(
        dim=8,
        heads=2,
        ffn_dim=16,
        window=4,
        iterations=2,
        carry_bias=2.0,
    )


def test_activity_budget_is_continuous_and_conserved() -> None:
    state = torch.randn(2, 5, TISSUE_HEIGHT)
    activity = _activity_from_state(state)
    assert torch.all(activity > 0)
    expected = torch.full((2, 5), ACTIVITY_BUDGET)
    assert torch.allclose(activity.sum(dim=-1), expected, atol=1e-6)


def test_connectome_has_zero_self_edges_and_homeostatic_budget() -> None:
    connectome = _initial_connectome(
        2,
        5,
        TISSUE_HEIGHT,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    diagonal = torch.diagonal(connectome, dim1=-2, dim2=-1)
    assert torch.all(diagonal == 0)
    expected = torch.full((2, 5, TISSUE_HEIGHT), SYNAPTIC_BUDGET)
    assert torch.allclose(connectome.sum(dim=-1), expected, atol=1e-6)


def test_hebbian_plasticity_preserves_budget_and_changes_connectome() -> None:
    stage = tiny_stage()
    connectome = _initial_connectome(
        1,
        2,
        TISSUE_HEIGHT,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    reaction = torch.randn(1, 2, TISSUE_HEIGHT, 8)
    reaction[:, :, 1] = reaction[:, :, 6]
    activity = torch.full((1, 2, TISSUE_HEIGHT), ACTIVITY_BUDGET / TISSUE_HEIGHT)
    updated = stage._update_connectome(connectome, reaction, activity)
    assert not torch.allclose(updated, connectome)
    assert torch.allclose(
        updated.sum(dim=-1),
        torch.full((1, 2, TISSUE_HEIGHT), SYNAPTIC_BUDGET),
        atol=1e-6,
    )
    assert torch.all(torch.diagonal(updated, dim1=-2, dim2=-1) == 0)


def test_diffusion_is_zero_when_all_cells_have_same_state() -> None:
    stage = tiny_stage()
    base = torch.randn(1, 3, 1, 8)
    state = base.expand(1, 3, TISSUE_HEIGHT, 8).clone()
    connectome = _initial_connectome(
        1,
        3,
        TISSUE_HEIGHT,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    diffusion = stage._diffusion(state, connectome, intervention="normal")
    assert torch.allclose(diffusion, torch.zeros_like(diffusion), atol=1e-6)


def test_plastic_model_has_no_router_gate_or_trainable_connectome() -> None:
    model = tiny_model()
    names = {name for name, _ in model.named_parameters()}
    assert not any("gate_head" in name for name in names)
    assert not any("norm_gate" in name for name in names)
    assert not any("connectome" in name for name in names)
    assert not any("router" in name for name in names)


def test_plastic_model_uses_fewer_parameters_than_015_sparse_control() -> None:
    torch.manual_seed(123)
    baseline = build_sparse_topology_model(
        32,
        variant="B",
        tissue_height=TISSUE_HEIGHT,
        active_latent=2,
        max_context=MODEL_LENGTH,
        dim=16,
        heads=2,
        ffn_dim=32,
        windows=(4, 8, 16),
        iterations=(2, 2, 2),
    )
    plastic = tiny_model()
    assert count_parameters(plastic) < count_parameters(baseline)


def test_observability_trace_matches_evolution_steps_and_budgets() -> None:
    model = tiny_model()
    input_ids = torch.randint(0, 32, (2, MODEL_LENGTH))
    result = model.forward_variable(
        input_ids,
        stage_depths=(1, 2, 1),
        collect_observability=True,
    )
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.activity_trace.shape[0] == 4
    assert diagnostics.connectome_trace.shape[0] == 4
    assert diagnostics.activity.shape == (2, MODEL_LENGTH, TISSUE_HEIGHT)
    assert diagnostics.connectome.shape == (
        2,
        MODEL_LENGTH,
        TISSUE_HEIGHT,
        TISSUE_HEIGHT,
    )
    assert torch.allclose(
        diagnostics.activity.sum(dim=-1),
        torch.full((2, MODEL_LENGTH), ACTIVITY_BUDGET),
        atol=1e-5,
    )
    assert torch.allclose(
        diagnostics.connectome.sum(dim=-1),
        torch.full((2, MODEL_LENGTH, TISSUE_HEIGHT), SYNAPTIC_BUDGET),
        atol=1e-5,
    )


def test_diffusion_off_intervention_records_zero_diffusion() -> None:
    model = tiny_model()
    input_ids = torch.randint(0, 32, (2, MODEL_LENGTH))
    result = model.forward_variable(
        input_ids,
        stage_depths=(1, 1, 1),
        intervention="diffusion_off",
        collect_observability=True,
    )
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert torch.allclose(
        diagnostics.diffusion_rms_trace,
        torch.zeros_like(diagnostics.diffusion_rms_trace),
        atol=1e-8,
    )
    assert torch.all(diagnostics.reaction_rms_trace > 0)


def test_paired_bootstrap_detects_causal_intervention_cost() -> None:
    normal = np.linspace(0.7, 1.0, 200)
    intervention = normal + 0.08
    stats = paired_bootstrap_delta(normal, intervention, seed=7, samples=500)
    assert stats["mean_delta_nll"] == pytest.approx(0.08)
    assert stats["bootstrap_95_lower"] > 0
