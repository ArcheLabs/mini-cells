from __future__ import annotations

import torch

from minicells.language_emergent_topology import (
    ACTIVITY_BUDGET,
    LOCAL_COUPLING,
    LONG_RANGE_MAX_COUPLING,
    EmergentTopologyStage,
    _initial_activity,
    _local_weights,
    _long_range_mask,
    _permute_plastic_distribution,
    _uniform_plastic_distribution,
    build_emergent_topology_model,
)
from minicells.language_models import count_parameters


def tiny_model(variant: str):
    return build_emergent_topology_model(
        vocab_size=32,
        variant_code=variant,
        tissue_height=8,
        max_context=8,
        dim=16,
        heads=2,
        ffn_dim=32,
        windows=(2, 2, 2),
        iterations=(1, 1, 1),
    )


def test_variants_have_identical_parameters_and_initialization() -> None:
    torch.manual_seed(123)
    local = tiny_model("L")
    torch.manual_seed(123)
    emergent = tiny_model("E")
    assert count_parameters(local) == count_parameters(emergent)
    for key, value in local.state_dict().items():
        assert torch.equal(value, emergent.state_dict()[key]), key


def test_no_trainable_router_or_connectome_parameters() -> None:
    model = tiny_model("E")
    names = [name.lower() for name, _ in model.named_parameters()]
    assert not any("router" in name for name in names)
    assert not any("connectome" in name for name in names)
    assert not any("plastic_distribution" in name for name in names)
    assert not any("gate_head" in name for name in names)


def test_local_substrate_contains_only_immediate_neighbors() -> None:
    weights = _local_weights(8, device=torch.device("cpu"), dtype=torch.float32)
    row = torch.arange(8)
    allowed = (row[:, None] - row[None, :]).abs() == 1
    assert torch.all(weights.masked_select(~allowed) == 0)
    assert torch.allclose(weights.sum(dim=-1), torch.full((8,), LOCAL_COUPLING))


def test_uniform_long_range_distribution_has_zero_emergent_strength() -> None:
    distribution = _uniform_plastic_distribution(2, 3, 8, device=torch.device("cpu"), dtype=torch.float32)
    weights, strength, tv = EmergentTopologyStage._plastic_weights(distribution)
    assert torch.allclose(strength, torch.zeros_like(strength), atol=1e-6)
    assert torch.allclose(weights, torch.zeros_like(weights), atol=1e-6)
    assert torch.allclose(tv, torch.zeros_like(tv), atol=1e-6)


def test_plastic_distribution_never_uses_local_or_self_edges() -> None:
    distribution = _uniform_plastic_distribution(1, 1, 8, device=torch.device("cpu"), dtype=torch.float32)
    allowed = _long_range_mask(8, device=torch.device("cpu")).view(1, 1, 8, 8)
    assert torch.all(distribution.masked_select(~allowed) == 0)
    assert torch.allclose(distribution.sum(dim=-1), torch.ones(1, 1, 8))


def test_replicator_preserves_budget_and_concentrates_activity() -> None:
    activity = _initial_activity(1, 1, 8, device=torch.device("cpu"))
    reaction = torch.zeros(1, 1, 8, 4)
    reaction[..., 3, :] = 4.0
    updated = EmergentTopologyStage._replicator_activity(activity, reaction)
    assert torch.allclose(updated.sum(dim=-1), torch.full((1, 1), ACTIVITY_BUDGET), atol=1e-6)
    assert updated[..., 3].item() > activity[..., 3].item()
    before_participation = activity.sum(dim=-1).square() / activity.square().sum(dim=-1)
    after_participation = updated.sum(dim=-1).square() / updated.square().sum(dim=-1)
    assert after_participation.item() < before_participation.item()


def test_plasticity_can_create_nonzero_long_range_topology() -> None:
    distribution = _uniform_plastic_distribution(1, 1, 8, device=torch.device("cpu"), dtype=torch.float32)
    reaction = torch.zeros(1, 1, 8, 4)
    reaction[..., 0, 0] = 1.0
    reaction[..., 3, 0] = 1.0
    reaction[..., 0, 1] = 0.2
    reaction[..., 4, 1] = 1.0
    activity = _initial_activity(1, 1, 8, device=torch.device("cpu"))
    for _ in range(4):
        distribution = EmergentTopologyStage._update_plastic_distribution(distribution, reaction, activity)
    weights, strength, tv = EmergentTopologyStage._plastic_weights(distribution)
    assert strength.max().item() > 0.0
    assert strength.max().item() <= LONG_RANGE_MAX_COUPLING + 1e-6
    assert tv.max().item() > 0.0
    assert weights[..., 0, 3].item() > 0.0


def test_topology_shuffle_changes_identity_but_preserves_row_strength_and_entropy() -> None:
    distribution = _uniform_plastic_distribution(1, 1, 8, device=torch.device("cpu"), dtype=torch.float32)
    reaction = torch.randn(1, 1, 8, 6)
    activity = _initial_activity(1, 1, 8, device=torch.device("cpu"))
    for _ in range(5):
        distribution = EmergentTopologyStage._update_plastic_distribution(distribution, reaction, activity)
    shuffled = _permute_plastic_distribution(distribution)
    weights, strength, _ = EmergentTopologyStage._plastic_weights(distribution)
    shuffled_weights, shuffled_strength, _ = EmergentTopologyStage._plastic_weights(shuffled)
    assert torch.allclose(distribution.sum(dim=-1), shuffled.sum(dim=-1), atol=1e-6)
    assert torch.allclose(strength, shuffled_strength, atol=1e-6)
    assert torch.allclose(weights.sum(dim=-1), shuffled_weights.sum(dim=-1), atol=1e-6)
    assert not torch.allclose(distribution, shuffled)


def test_diffusion_is_zero_when_all_cell_states_agree() -> None:
    state = torch.ones(2, 3, 8, 5)
    local = _local_weights(8, device=torch.device("cpu"), dtype=torch.float32).view(1, 1, 8, 8).expand(2, 3, -1, -1)
    assert torch.allclose(EmergentTopologyStage._diffusion(state, local), torch.zeros_like(state), atol=1e-6)


def test_forward_observability_exposes_separate_local_and_plastic_dynamics() -> None:
    torch.manual_seed(7)
    model = tiny_model("E")
    inputs = torch.randint(0, 32, (2, 6))
    result = model.forward_variable(inputs, stage_depths=(1, 1, 1), collect_observability=True)
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.activity.shape == (2, 6, 8)
    assert diagnostics.plastic_weights.shape == (2, 6, 8, 8)
    assert diagnostics.activity_trace.shape[0] == 3
    assert diagnostics.plastic_strength_trace.shape[0] == 3
    assert diagnostics.reaction_rms_trace.shape == (3,)
    assert diagnostics.local_diffusion_rms_trace.shape == (3,)
    assert diagnostics.plastic_diffusion_rms_trace.shape == (3,)
    assert torch.allclose(diagnostics.activity.sum(dim=-1), torch.full((2, 6), ACTIVITY_BUDGET), atol=1e-5)


def test_plasticity_off_keeps_long_range_effect_at_zero() -> None:
    torch.manual_seed(9)
    model = tiny_model("E")
    inputs = torch.randint(0, 32, (2, 6))
    result = model.forward_variable(
        inputs,
        stage_depths=(1, 1, 1),
        intervention="plasticity_off",
        collect_observability=True,
    )
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert torch.allclose(diagnostics.plastic_diffusion_rms_trace, torch.zeros_like(diagnostics.plastic_diffusion_rms_trace), atol=1e-7)


def test_tissue_topology_does_not_break_sequence_causality() -> None:
    torch.manual_seed(11)
    model = tiny_model("E").eval()
    prefix = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = prefix.clone()
    changed[:, 4:] = torch.tensor([[7, 8]])
    with torch.no_grad():
        a = model.forward_variable(prefix, stage_depths=(1, 1, 1)).output.logits
        b = model.forward_variable(changed, stage_depths=(1, 1, 1)).output.logits
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-5, rtol=1e-5)
