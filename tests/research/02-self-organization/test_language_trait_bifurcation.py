import torch

from minicells.language_conflict_differentiation import DOMAINS
from minicells.language_trait_bifurcation import (
    BIFURCATION_GAIN_MIN,
    BIFURCATION_SPLIT_BALANCE_MIN,
    GEOMETRY_MARGIN_ADVANTAGE_MIN,
    axis_stability,
    bifurcation_window_pass,
    fit_two_mode_gradient_field,
    geometry_advantage,
    route_gradient_to_mode,
    routing_purity_from_branches,
    stratified_capacity_branch,
    summarize_identity,
    summarize_persistent_bifurcation,
)


def _two_mode_gradients() -> torch.Tensor:
    left = torch.tensor([[1.0, 0.0, 0.0]]).repeat(8, 1)
    right = torch.tensor([[0.0, 1.0, 0.0]]).repeat(8, 1)
    return torch.cat([left, right], dim=0)


def test_k2_fit_detects_clear_unlabeled_bimodality() -> None:
    gradients = _two_mode_gradients()
    geometry = fit_two_mode_gradient_field(gradients)
    assert geometry.bifurcation_gain > BIFURCATION_GAIN_MIN
    assert geometry.split_balance >= BIFURCATION_SPLIT_BALANCE_MIN
    assert geometry.residual_k2 < geometry.residual_k1
    assert bifurcation_window_pass(geometry)

    branches = [route_gradient_to_mode(row, geometry)[0] for row in gradients]
    labels = [DOMAINS[0]] * 8 + [DOMAINS[1]] * 8
    assert routing_purity_from_branches(branches, labels) == 1.0


def test_identical_gradient_field_does_not_bifurcate() -> None:
    gradients = torch.tensor([[1.0, 0.0, 0.0]]).repeat(16, 1)
    geometry = fit_two_mode_gradient_field(gradients)
    assert abs(geometry.bifurcation_gain) < 1e-8
    assert not bifurcation_window_pass(geometry)


def test_axis_stability_is_cluster_permutation_invariant() -> None:
    geometry = fit_two_mode_gradient_field(_two_mode_gradients())
    flipped = type(geometry)(
        one_mode_centroid=geometry.one_mode_centroid,
        centroids=geometry.centroids.flip(0),
        axis=-geometry.axis,
        residual_k1=geometry.residual_k1,
        residual_k2=geometry.residual_k2,
        bifurcation_gain=geometry.bifurcation_gain,
        split_balance=geometry.split_balance,
        centroid_separation=geometry.centroid_separation,
    )
    assert abs(axis_stability([geometry, flipped]) - 1.0) < 1e-6


def test_persistent_bifurcation_requires_stable_repeated_windows() -> None:
    geometry = fit_two_mode_gradient_field(_two_mode_gradients())
    summary = summarize_persistent_bifurcation([geometry, geometry, geometry], geometry)
    assert summary.windows_passed == 3
    assert summary.axis_stability > 0.99
    assert summary.persistent


def test_stratified_capacity_control_exactly_balances_both_domains() -> None:
    for replicate in range(3):
        for domain in DOMAINS:
            branches = [stratified_capacity_branch(domain, occurrence, replicate) for occurrence in range(200)]
            assert branches.count(0) == 100
            assert branches.count(1) == 100


def test_functional_identity_is_permutation_invariant() -> None:
    parent = {DOMAINS[0]: 5.0, DOMAINS[1]: 7.0}
    losses = {
        DOMAINS[0]: (4.0, 5.5),
        DOMAINS[1]: (7.5, 5.5),
    }
    identity = summarize_identity(losses, parent)
    assert identity.passes
    assert identity.assignment == (0, 1)

    swapped = {
        DOMAINS[0]: (5.5, 4.0),
        DOMAINS[1]: (5.5, 7.5),
    }
    swapped_identity = summarize_identity(swapped, parent)
    assert swapped_identity.passes
    assert swapped_identity.assignment == (1, 0)
    assert abs(swapped_identity.normalized_identity_margin - identity.normalized_identity_margin) < 1e-8


def test_geometry_advantage_threshold_is_explicit() -> None:
    assert geometry_advantage(0.16, 0.02) > GEOMETRY_MARGIN_ADVANTAGE_MIN
    assert geometry_advantage(0.04, 0.02) < GEOMETRY_MARGIN_ADVANTAGE_MIN
