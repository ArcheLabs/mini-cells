import numpy as np
import torch

from minicells.language_conflict_differentiation import (
    DOMAINS,
    ForkableTextNCA,
    conflict_gate,
    learn_conflict_geometry,
    route_gradient,
    summarize_identity,
)


def test_opposed_gradient_population_has_conflict_geometry() -> None:
    gradients = torch.tensor([
        [1.0, 0.1, 0.0],
        [0.9, -0.1, 0.0],
        [1.1, 0.0, 0.1],
        [-1.0, -0.1, 0.0],
        [-0.9, 0.1, 0.0],
        [-1.1, 0.0, -0.1],
    ])
    geometry = learn_conflict_geometry(gradients)
    assert geometry.directional_cancellation > 0.9
    assert geometry.pc1_variance_ratio > 0.9
    assert geometry.split_balance >= 0.49
    assert conflict_gate(geometry)


def test_gradient_projection_routes_opposite_directions_to_different_children() -> None:
    gradients = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],
        [-1.0, 0.0],
        [-0.9, -0.1],
    ])
    geometry = learn_conflict_geometry(gradients)
    left, left_score = route_gradient(torch.tensor([1.0, 0.0]), geometry)
    right, right_score = route_gradient(torch.tensor([-1.0, 0.0]), geometry)
    assert left != right
    assert left_score * right_score < 0.0


def test_identity_summary_is_permutation_invariant() -> None:
    parent = {DOMAINS[0]: 2.0, DOMAINS[1]: 4.0}
    direct = summarize_identity(
        {DOMAINS[0]: (1.0, 1.6), DOMAINS[1]: (3.8, 2.0)},
        parent,
    )
    swapped = summarize_identity(
        {DOMAINS[0]: (1.6, 1.0), DOMAINS[1]: (2.0, 3.8)},
        parent,
    )
    assert direct.passes
    assert swapped.passes
    assert np.isclose(direct.normalized_identity_margin, swapped.normalized_identity_margin)
    assert direct.assignment == (0, 1)
    assert swapped.assignment == (1, 0)


def test_generic_children_do_not_count_as_identity() -> None:
    parent = {DOMAINS[0]: 2.0, DOMAINS[1]: 2.0}
    summary = summarize_identity(
        {DOMAINS[0]: (1.0, 1.0), DOMAINS[1]: (1.0, 1.0)},
        parent,
    )
    assert not summary.opposite_preference
    assert not summary.passes


def test_fork_changes_only_population_phenotype_not_shared_genome() -> None:
    torch.manual_seed(21021)
    model = ForkableTextNCA(128)
    base_before = {key: value.detach().clone() for key, value in model.base.state_dict().items()}
    parent = model.parent_trait.detach().clone()
    axis = torch.zeros(model.dim)
    axis[0] = 1.0
    model.initialize_children(axis, symmetry_break=True)
    for key, value in model.base.state_dict().items():
        assert torch.equal(value, base_before[key])
    midpoint = 0.5 * (model.child_traits[0].detach() + model.child_traits[1].detach())
    assert torch.allclose(midpoint, parent)
    assert not torch.equal(model.child_traits[0].detach(), model.child_traits[1].detach())


def test_shared_genome_is_frozen_after_fork() -> None:
    model = ForkableTextNCA(128)
    model.freeze_genome_for_fork()
    assert all(not parameter.requires_grad for parameter in model.base.parameters())
    assert not model.parent_trait.requires_grad
    assert model.child_traits.requires_grad
