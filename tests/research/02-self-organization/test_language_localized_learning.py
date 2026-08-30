from __future__ import annotations

import torch

from minicells.language_growing_organism import StructuralProbe, build_cellular_model
from minicells.language_localized_learning import (
    LocalizedGrowthController,
    LocalizedLearningState,
    conservative_fork,
    graft_localized_tissue,
    mask_to_newborn_gradients,
    restore_structure,
    set_newborn_tissue_active,
)


def tiny_model(*, initial_cells: int = 3, max_cells: int = 6):
    torch.manual_seed(17017)
    return build_cellular_model(
        32,
        "G",
        max_context=12,
        dim=32,
        heads=4,
        ffn_dim=64,
        iterations=2,
        attention_window=6,
        initial_cells=initial_cells,
        max_cells=max_cells,
    )


def probe_for(model, pressure: torch.Tensor, *, utility: torch.Tensor | None = None) -> StructuralProbe:
    if utility is None:
        utility = torch.zeros(model.max_cells, model.max_cells)
    conflict = torch.zeros(model.max_cells)
    direction = torch.zeros(model.max_cells, model.dim)
    for cell in range(model.max_cells):
        direction[cell, cell % model.dim] = 1.0
    return StructuralProbe(utility, pressure, conflict, direction, 1.0)


def test_conservative_fork_keeps_parent_bit_identical() -> None:
    model = tiny_model()
    parent_before = model.cell_memory[1].detach().clone()
    direction = torch.zeros(model.dim)
    direction[0] = 1.0
    child = conservative_fork(model, 1, step=7, direction=direction)
    assert child == 3
    assert torch.equal(model.cell_memory[1].detach(), parent_before)
    assert not torch.equal(model.cell_memory[child].detach(), parent_before)
    assert bool(model.adjacency[1, child]) and bool(model.adjacency[child, 1])
    assert int(model.parent[child]) == 1


def test_gradient_diversion_updates_newborn_only() -> None:
    model = tiny_model()
    state = LocalizedLearningState.capture(model)
    child = conservative_fork(model, 1, step=0, direction=torch.ones(model.dim))
    assert child == 3
    before = model.cell_memory.detach().clone()
    optimizer = torch.optim.SGD([model.cell_memory], lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    model.cell_memory.grad = torch.ones_like(model.cell_memory)
    mask_to_newborn_gradients(model, state)
    assert torch.equal(model.cell_memory.grad[:3], torch.zeros_like(model.cell_memory.grad[:3]))
    assert torch.equal(model.cell_memory.grad[child], torch.ones_like(model.cell_memory.grad[child]))
    optimizer.step()
    assert torch.equal(model.cell_memory[:3].detach(), before[:3])
    assert not torch.equal(model.cell_memory[child].detach(), before[child])


def test_localized_controller_cannot_rewire_old_old_edges() -> None:
    model = tiny_model()
    state = LocalizedLearningState.capture(model)
    pressure = torch.tensor([0.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    controller = LocalizedGrowthController(state, max_newborns=1, connect_score=0.5, persistence=1)
    controller.allocate_initial(model, probe_for(model, pressure), step=0)
    old_old_before = model.adjacency[:3, :3].clone()
    utility = torch.zeros(6, 6)
    utility[0, 2] = 100.0
    utility[2, 0] = 90.0
    controller.apply(model, probe_for(model, pressure, utility=utility), step=50)
    assert torch.equal(model.adjacency[:3, :3], old_old_before)


def test_localized_graft_copies_newborn_not_old_phenotype() -> None:
    donor = tiny_model()
    recipient = tiny_model()
    old_recipient = recipient.cell_memory.detach().clone()
    child = conservative_fork(donor, 1, step=20, direction=torch.ones(donor.dim))
    assert child == 3
    donor.connect(0, child)
    with torch.no_grad():
        donor.cell_memory[1].add_(5.0)
        donor.cell_memory[child].add_(0.75)
    graft_localized_tissue(recipient, donor, [child])
    assert torch.equal(recipient.cell_memory[0], old_recipient[0])
    assert torch.equal(recipient.cell_memory[1], old_recipient[1])
    assert torch.equal(recipient.cell_memory[2], old_recipient[2])
    assert torch.equal(recipient.cell_memory[child], donor.cell_memory[child])
    assert bool(recipient.adjacency[child, 1]) == bool(donor.adjacency[child, 1])
    assert bool(recipient.adjacency[0, child]) == bool(donor.adjacency[0, child])


def test_newborn_ablation_and_restore_are_exact() -> None:
    model = tiny_model()
    state = LocalizedLearningState.capture(model)
    child = conservative_fork(model, 1, step=10, direction=torch.ones(model.dim))
    assert child == 3
    alive = model.alive_mask.clone()
    adjacency = model.adjacency.clone()
    saved_alive, saved_adjacency = set_newborn_tissue_active(model, state, False)
    assert not bool(model.alive_mask[child])
    assert not bool(model.adjacency[child].any())
    assert not bool(model.adjacency[:, child].any())
    restore_structure(model, saved_alive, saved_adjacency)
    assert torch.equal(model.alive_mask, alive)
    assert torch.equal(model.adjacency, adjacency)


def test_base_memory_drift_is_zero_when_only_newborn_changes() -> None:
    model = tiny_model()
    state = LocalizedLearningState.capture(model)
    child = conservative_fork(model, 2, step=1, direction=torch.ones(model.dim))
    assert child == 3
    with torch.no_grad():
        model.cell_memory[child].add_(1.0)
    assert float(state.base_memory_drift(model)) == 0.0
