from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from minicells.hybrid_clm import (
    HybridCellOverlay,
    HybridCLMError,
    HybridManifest,
    mask_address_gradients_,
    mask_transform_gradients_,
)


class _ToyLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.proj(hidden),)


class _ToyBackbone(nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ToyLayer(width) for _ in range(layers)])


class _ToyModel(nn.Module):
    def __init__(self, width: int = 8, layers: int = 5) -> None:
        super().__init__()
        self.model = _ToyBackbone(width, layers)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return hidden


def _overlay(seed: int = 7) -> HybridCellOverlay:
    return HybridCellOverlay(
        hidden_size=8,
        read_layer_index=1,
        write_layer_indices=(2, 4),
        max_cells=6,
        rank=2,
        gate_threshold=0.6,
        seed=seed,
    )


def _prepare_committed_cell(overlay: HybridCellOverlay, slot: int) -> None:
    with torch.no_grad():
        overlay.gate_bias[slot] = 8.0
        overlay.up[:, slot].normal_(mean=0.0, std=0.05)
    overlay.freeze_address_(slot)
    overlay.commit_cell_(slot)


def test_allocation_is_exactly_invisible_to_production() -> None:
    torch.manual_seed(1)
    model = _ToyModel()
    overlay = _overlay()
    hidden = torch.randn(2, 4, 8)
    expected = model(hidden)
    slot = overlay.allocate_cell()
    with torch.no_grad():
        overlay.gate_bias[slot] = 10.0
        overlay.up[:, slot].normal_(mean=0.0, std=0.3)
    with overlay.installed(model):
        observed = model(hidden)
    assert torch.equal(observed, expected)


def test_independent_gate_does_not_renormalize_existing_cell() -> None:
    torch.manual_seed(2)
    overlay = _overlay()
    first = overlay.allocate_cell()
    features = torch.randn(5, 8)
    before = overlay.address_probability_for_features(features, first).detach().clone()
    second = overlay.allocate_cell()
    with torch.no_grad():
        overlay.gate_weight[second].normal_()
        overlay.gate_bias[second] = 10.0
    after = overlay.address_probability_for_features(features, first).detach().clone()
    assert torch.equal(before, after)


def test_shadow_can_exercise_uncommitted_cell_without_production_change() -> None:
    torch.manual_seed(3)
    model = _ToyModel()
    overlay = _overlay()
    hidden = torch.randn(1, 4, 8)
    slot = overlay.allocate_cell()
    with torch.no_grad():
        overlay.gate_bias[slot] = 10.0
        overlay.up[:, slot].normal_(mean=0.0, std=0.2)
    production_before = model(hidden)
    with overlay.installed(model):
        production = model(hidden)
    with overlay.shadow([slot]), overlay.installed(model):
        shadow = model(hidden)
    assert torch.equal(production, production_before)
    assert not torch.equal(shadow, production)


def test_commit_and_uncommit_are_exact_lifecycle_switches() -> None:
    torch.manual_seed(4)
    model = _ToyModel()
    overlay = _overlay()
    hidden = torch.randn(1, 3, 8)
    slot = overlay.allocate_cell()
    _prepare_committed_cell(overlay, slot)
    with overlay.installed(model):
        committed = model(hidden).detach().clone()
    overlay.uncommit_cell_(slot)
    with overlay.installed(model):
        rolled_back = model(hidden).detach().clone()
    assert torch.equal(rolled_back, model(hidden))
    overlay.commit_cell_(slot)
    with overlay.installed(model):
        reapplied = model(hidden).detach().clone()
    assert torch.equal(reapplied, committed)


def test_child_allocation_is_function_preserving_even_with_inherited_state() -> None:
    torch.manual_seed(5)
    model = _ToyModel()
    overlay = _overlay()
    hidden = torch.randn(1, 4, 8)
    parent = overlay.allocate_cell()
    _prepare_committed_cell(overlay, parent)
    with overlay.installed(model):
        before = model(hidden).detach().clone()
    child = overlay.allocate_cell(
        parent_slot=parent,
        inherit_address=True,
        inherit_transform=True,
    )
    assert int(overlay.parent_slot[child].item()) == parent
    with overlay.installed(model):
        after = model(hidden).detach().clone()
    assert torch.equal(after, before)


def test_gradient_masks_separate_address_and_transform_training() -> None:
    torch.manual_seed(6)
    overlay = _overlay()
    slot = overlay.allocate_cell()
    features = torch.randn(4, 8)
    loss = overlay.address_probability_for_features(features, slot).sum()
    loss.backward()
    mask_address_gradients_(overlay, slot)
    assert overlay.gate_weight.grad is not None
    assert torch.count_nonzero(overlay.gate_weight.grad[slot]).item() > 0
    assert torch.count_nonzero(overlay.gate_weight.grad[:slot]).item() == 0

    overlay.zero_grad(set_to_none=True)
    hidden = torch.randn(1, 2, 8)
    gate = torch.ones(1, 2, overlay.max_cells, dtype=torch.bool)
    overlay._cached_active = gate
    overlay._cached_probabilities = gate.float()
    transformed = overlay._transform(hidden, 0).sum()
    transformed.backward()
    mask_transform_gradients_(overlay, slot)
    assert overlay.up.grad is not None
    assert torch.count_nonzero(overlay.up.grad[:, slot]).item() > 0
    assert torch.count_nonzero(overlay.up.grad[:, slot + 1 :]).item() == 0


def test_artifacts_reapply_into_new_slots_and_manifest_merge_is_set_union() -> None:
    overlay = _overlay()
    first = overlay.allocate_cell()
    with torch.no_grad():
        overlay.gate_bias[first] = 9.0
        overlay.up[:, first].fill_(0.25)
    overlay.freeze_address_(first)
    artifact_a = overlay.export_artifact(first, cell_id="cell-A")

    second = overlay.allocate_cell()
    with torch.no_grad():
        overlay.gate_bias[second] = 7.0
        overlay.up[:, second].fill_(-0.5)
    overlay.freeze_address_(second)
    artifact_b = overlay.export_artifact(second, cell_id="cell-B")

    fresh = _overlay(seed=99)
    loaded_slot = fresh.apply_artifact_(artifact_a)
    assert loaded_slot == 0
    assert torch.equal(fresh.up[:, loaded_slot].cpu(), artifact_a.state["up"])
    assert bool(fresh.committed_mask[loaded_slot])

    base = HybridManifest("foundation", "rev")
    left = base.add(artifact_a)
    right = base.add(artifact_b)
    merged = left.merge(right)
    assert dict(merged.cells) == {
        "cell-A": artifact_a.digest(),
        "cell-B": artifact_b.digest(),
    }
    assert dict(merged.remove("cell-A").cells) == {"cell-B": artifact_b.digest()}

    conflicting = copy.deepcopy(artifact_a)
    conflicting.state["up"][0, 0, 0] += 1.0
    with pytest.raises(HybridCLMError):
        left.add(conflicting)


def test_commit_requires_frozen_address() -> None:
    overlay = _overlay()
    slot = overlay.allocate_cell()
    with pytest.raises(HybridCLMError):
        overlay.commit_cell_(slot)
