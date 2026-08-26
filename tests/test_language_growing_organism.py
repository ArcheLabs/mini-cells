from __future__ import annotations

import torch
from torch.nn import functional as F

from minicells.language_growing_organism import (
    ACTIVITY_BUDGET,
    StructuralController,
    build_cellular_model,
    build_parameter_matched_small_transformer,
    make_structural_probe,
)
from minicells.language_models import count_parameters


def tiny_model(variant: str, *, max_cells: int = 6):
    torch.manual_seed(123)
    return build_cellular_model(
        32,
        variant,
        max_context=12,
        dim=32,
        heads=4,
        ffn_dim=64,
        iterations=2,
        attention_window=6,
        initial_cells=3,
        max_cells=max_cells,
    )


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def test_fixed_and_growing_have_identical_parameterization_and_initialization() -> None:
    fixed = tiny_model("F")
    growing = tiny_model("G")
    assert count_parameters(fixed) == count_parameters(growing)
    for key, value in fixed.state_dict().items():
        assert torch.equal(value, growing.state_dict()[key]), key


def test_initial_graph_is_bidirectional_chain_and_protected() -> None:
    model = tiny_model("G")
    expected = torch.zeros(6, 6, dtype=torch.bool)
    expected[0, 1] = expected[1, 0] = True
    expected[1, 2] = expected[2, 1] = True
    assert torch.equal(model.adjacency.cpu(), expected)
    assert torch.equal(model.protected_edges.cpu(), expected)
    assert model.alive_count == 3


def test_forward_computes_only_alive_cells_and_preserves_activity_budget() -> None:
    model = tiny_model("G")
    inputs = torch.randint(0, 32, (2, 8))
    result = model.forward_variable(inputs, collect_observability=True)
    d = result.diagnostics
    assert d is not None
    assert d.cell_states.shape == (2, 8, 3, 32)
    assert d.activity.shape == (2, 8, 3)
    assert torch.allclose(d.activity.sum(dim=-1), torch.full((2, 8), ACTIVITY_BUDGET), atol=1e-5)


def test_sequence_boundary_remains_causal() -> None:
    model = tiny_model("G").eval()
    left = torch.tensor([[1, 2, 3, 4, 5, 6]])
    right = left.clone()
    right[:, 4:] = torch.tensor([[7, 8]])
    with torch.no_grad():
        a = model.forward_variable(left, collect_observability=True).output.logits
        b = model.forward_variable(right, collect_observability=True).output.logits
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-5, rtol=1e-5)


def test_connect_and_prune_only_remove_unprotected_edges() -> None:
    model = tiny_model("G")
    assert model.connect(0, 2)
    assert bool(model.adjacency[0, 2])
    assert model.prune(0, 2)
    assert not bool(model.adjacency[0, 2])
    assert not model.prune(0, 1)
    assert bool(model.adjacency[0, 1])


def test_fork_activates_slot_splits_memory_and_creates_lineage_edge() -> None:
    model = tiny_model("G")
    before = model.cell_memory[1].detach().clone()
    direction = torch.zeros(32)
    direction[0] = 1.0
    child = model.fork_cell(1, step=100, direction=direction)
    assert child == 3
    assert model.alive_count == 4
    assert int(model.parent[child]) == 1
    assert int(model.birth_step[child]) == 100
    assert bool(model.adjacency[1, child]) and bool(model.adjacency[child, 1])
    midpoint = 0.5 * (model.cell_memory[1].detach() + model.cell_memory[child].detach())
    assert torch.allclose(midpoint, before, atol=1e-6)
    assert not torch.equal(model.cell_memory[1].detach(), model.cell_memory[child].detach())


def test_fixed_variant_cannot_fork() -> None:
    model = tiny_model("F")
    assert model.fork_cell(1, step=100, direction=torch.ones(32)) is None
    assert model.alive_count == 3


def test_structural_probe_measures_edges_pressure_conflict_and_split_direction() -> None:
    model = tiny_model("G")
    inputs = torch.randint(0, 32, (4, 8))
    targets = torch.randint(0, 32, (4, 8))
    microbatches = [(inputs[:2], targets[:2]), (inputs[2:], targets[2:])]
    probe = make_structural_probe(model, microbatches, loss_fn=loss_fn)
    assert probe.edge_utility.shape == (6, 6)
    assert probe.pressure.shape == (6,)
    assert probe.conflict.shape == (6,)
    assert probe.split_direction.shape == (6, 32)
    assert torch.isfinite(probe.edge_utility).all()
    assert torch.isfinite(probe.pressure).all()
    assert ((probe.conflict >= 0) & (probe.conflict <= 1)).all()


def test_controller_never_changes_fixed_variant() -> None:
    model = tiny_model("F")
    inputs = torch.randint(0, 32, (4, 8))
    targets = torch.randint(0, 32, (4, 8))
    probe = make_structural_probe(model, [(inputs[:2], targets[:2]), (inputs[2:], targets[2:])], loss_fn=loss_fn)
    controller = StructuralController(max_cells=6, persistence=1)
    before = model.adjacency.clone()
    assert controller.apply(model, probe, step=100) == []
    assert torch.equal(before, model.adjacency)


def test_freeze_genome_leaves_only_cell_memory_trainable() -> None:
    model = tiny_model("G")
    model.freeze_genome()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable == ["cell_memory"]


def test_tissue_copy_transfers_selected_memory_and_subgraph_but_not_interface_memory() -> None:
    donor = tiny_model("G")
    recipient = tiny_model("G")
    direction = torch.zeros(32)
    direction[2] = 1.0
    child = donor.fork_cell(1, step=50, direction=direction)
    assert child == 3
    donor.connect(0, child)
    interface_before = recipient.cell_memory[0].detach().clone()
    with torch.no_grad():
        donor.cell_memory[0].add_(1.0)
        donor.cell_memory[child].add_(0.5)
    recipient.copy_tissue_from(donor, [1, child])
    assert bool(recipient.alive_mask[child])
    assert torch.equal(recipient.cell_memory[child], donor.cell_memory[child])
    assert torch.equal(recipient.adjacency[1, child], donor.adjacency[1, child])
    assert torch.equal(recipient.adjacency[0, child], donor.adjacency[0, child])
    assert torch.equal(recipient.cell_memory[0], interface_before)
    assert not torch.equal(recipient.cell_memory[0], donor.cell_memory[0])


def test_parameter_matched_transformer_is_close_to_cellular_model() -> None:
    model = build_cellular_model(256, "G", max_context=32, dim=64, heads=4, ffn_dim=128, iterations=2)
    target = count_parameters(model)
    transformer, metadata = build_parameter_matched_small_transformer(256, target, max_context=32)
    assert count_parameters(transformer) == metadata["parameters"]
    assert metadata["relative_parameter_error"] <= 0.15
