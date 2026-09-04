from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from minicells.functional_cellization import (
    FunctionalCellizationError,
    FunctionalCellOverlay,
    disjoint_mutations,
    mask_cell_gradients_,
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
    def __init__(self, width: int = 8, layers: int = 4) -> None:
        super().__init__()
        self.model = _ToyBackbone(width, layers)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return hidden


def _overlay(seed: int = 7) -> FunctionalCellOverlay:
    return FunctionalCellOverlay(
        hidden_size=8,
        layer_indices=(1, 3),
        max_cells=4,
        initial_active_cells=2,
        rank=2,
        top_k=1,
        seed=seed,
    )


def test_zero_init_overlay_is_exact_compatibility_shell() -> None:
    torch.manual_seed(1)
    model = _ToyModel()
    hidden = torch.randn(2, 5, 8)
    expected = model(hidden)
    overlay = _overlay()
    assert overlay.zero_output_is_exact()
    with overlay.installed(model):
        observed = model(hidden)
    assert torch.equal(observed, expected)


def test_cross_layer_route_is_sparse_and_reused() -> None:
    torch.manual_seed(2)
    model = _ToyModel()
    hidden = torch.randn(2, 5, 8)
    overlay = _overlay()
    with torch.no_grad():
        overlay.up[0, 0, 0, 0] = 0.5
        overlay.up[1, 0, 0, 0] = 0.5
    with overlay.installed(model):
        _ = model(hidden)
        positions = torch.tensor([1, 3])
        summary = overlay.prompt_routes(positions)
        assert summary.probabilities.shape == (2, 4)
        assert torch.allclose(summary.probabilities.sum(dim=-1), torch.ones(2))
        assert set(summary.primary_cell.tolist()).issubset({0, 1})


def test_spawn_child_is_function_preserving_until_child_is_trained() -> None:
    torch.manual_seed(3)
    model = _ToyModel()
    hidden = torch.randn(1, 4, 8)
    overlay = _overlay()
    with torch.no_grad():
        overlay.up[:, 0].normal_(mean=0.0, std=0.05)
    with overlay.installed(model):
        before = model(hidden).detach().clone()
    child = overlay.spawn_child(0)
    assert child == 2
    assert torch.equal(overlay.up[:, child], overlay.up[:, 0])
    with overlay.installed(model):
        after = model(hidden).detach().clone()
    assert torch.allclose(before, after, atol=1e-7, rtol=0.0)


def test_cell_mutations_compose_and_rollback_exactly() -> None:
    overlay = _overlay()
    base = overlay.snapshot()

    with torch.no_grad():
        overlay.up[:, 0].add_(0.25)
    mutation_a = overlay.export_mutation(0)
    overlay.restore_(base)

    with torch.no_grad():
        overlay.up[:, 1].sub_(0.5)
    mutation_b = overlay.export_mutation(1)
    overlay.restore_(base)

    disjoint_mutations(mutation_a, mutation_b)
    overlay.apply_mutation_(mutation_a)
    overlay.apply_mutation_(mutation_b)
    assert torch.equal(overlay.up[:, 0].cpu(), mutation_a.state["up"])
    assert torch.equal(overlay.up[:, 1].cpu(), mutation_b.state["up"])

    overlay.restore_(base)
    for name, value in overlay.state_dict().items():
        assert torch.equal(value.cpu(), base[name])

    with pytest.raises(FunctionalCellizationError):
        disjoint_mutations(mutation_a, copy.deepcopy(mutation_a))


def test_gradient_mask_limits_local_write_to_selected_cell() -> None:
    torch.manual_seed(5)
    overlay = _overlay()
    hidden = torch.randn(2, 3, 8)
    out = overlay._transform(
        hidden,
        torch.full((2, 3, 4), 0.25),
        site=0,
    ).sum()
    out.backward()
    mask_cell_gradients_(overlay, [1], include_keys=False)
    assert overlay.up.grad is not None
    assert torch.count_nonzero(overlay.up.grad[:, 0]).item() == 0
    assert torch.count_nonzero(overlay.up.grad[:, 1]).item() > 0
    assert torch.count_nonzero(overlay.up.grad[:, 2:]).item() == 0
