from __future__ import annotations

from collections import Counter

import torch

from research.minicells.language_online_trait_genesis import (
    GrowthEvidence,
    align_growth_centroids,
    developmental_curriculum,
    mode_set_stability,
    select_model_order,
    summarize_multi_identity,
    update_growth_evidence,
)


def _cloud(center: torch.Tensor, count: int, noise: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return center[None, :] + noise * torch.randn(count, len(center), generator=generator)


def test_structural_objective_keeps_unimodal_field_at_one() -> None:
    center = torch.tensor([1.0, 0.0, 0.0, 0.0])
    gradients = _cloud(center, 96, 0.05, 1)
    selection = select_model_order(gradients)
    assert selection.selected_k == 1


def test_structural_objective_discovers_two_and_three_modes_without_labels() -> None:
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    c = torch.tensor([0.0, 0.0, 1.0, 0.0])
    two = torch.cat([_cloud(a, 48, 0.04, 2), _cloud(b, 48, 0.04, 3)])
    three = torch.cat([
        _cloud(a, 32, 0.04, 4),
        _cloud(b, 32, 0.04, 5),
        _cloud(c, 32, 0.04, 6),
    ])
    assert select_model_order(two).selected_k == 2
    assert select_model_order(three).selected_k == 3


def test_growth_requires_three_stable_online_evaluations() -> None:
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    gradients = torch.cat([_cloud(a, 48, 0.03, 7), _cloud(b, 48, 0.03, 8)])
    selection = select_model_order(gradients)
    evidence = GrowthEvidence()
    evidence, ready1 = update_growth_evidence(evidence, active_k=1, selection=selection)
    evidence, ready2 = update_growth_evidence(evidence, active_k=1, selection=selection)
    evidence, ready3 = update_growth_evidence(evidence, active_k=1, selection=selection)
    assert not ready1
    assert not ready2
    assert ready3
    assert evidence.stable_evaluations == 3


def test_mode_stability_is_permutation_invariant() -> None:
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    swapped = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert mode_set_stability(reference, swapped) > 0.999


def test_growth_alignment_preserves_existing_mode_order_and_adds_one() -> None:
    old = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    new = torch.tensor([[0.0, 0.0, 1.0], [0.02, 0.98, 0.0], [0.99, 0.01, 0.0]])
    ordered, newborn, parent = align_growth_centroids(old, new)
    assert newborn == 2
    assert parent in (0, 1)
    assert torch.linalg.vector_norm(ordered[0] - old[0]) < 0.05
    assert torch.linalg.vector_norm(ordered[1] - old[1]) < 0.05


def test_curriculum_has_exact_negative_control_and_three_mode_counts() -> None:
    rows = developmental_curriculum(0)
    counts = Counter((str(row["stage"]), str(row["stream_key"])) for row in rows)
    assert counts[("A_STORY_ONLY", "STORY")] == 192
    assert counts[("B_EMERGING_MATH", "STORY")] == 269
    assert counts[("B_EMERGING_MATH", "ARITH_A")] == 115
    assert counts[("C_DUPLICATE_CONTROL", "STORY")] == 64
    assert counts[("C_DUPLICATE_CONTROL", "ARITH_A")] == 64
    assert counts[("C_DUPLICATE_CONTROL", "ARITH_B")] == 64
    assert counts[("D_THIRD_MODE", "STORY")] == 128
    assert counts[("D_THIRD_MODE", "ARITH_A")] == 128
    assert counts[("D_THIRD_MODE", "TRANSFORM")] == 128
    duplicate_families = {
        str(row["family"])
        for row in rows
        if row["stage"] == "C_DUPLICATE_CONTROL" and str(row["stream_key"]).startswith("ARITH")
    }
    assert duplicate_families == {"ARITHMETIC"}


def test_multi_identity_is_permutation_invariant() -> None:
    baselines = {"STORY": 5.0, "ARITHMETIC": 7.0, "TRANSFORM": 8.0}
    losses = {
        "STORY": (6.0, 4.0, 6.2),
        "ARITHMETIC": (5.5, 7.2, 7.1),
        "TRANSFORM": (8.4, 8.2, 6.0),
    }
    summary = summarize_multi_identity(losses, baselines, ("STORY", "ARITHMETIC", "TRANSFORM"))
    assert summary.assignment == (1, 0, 2)
    assert summary.passes
    swapped = {
        family: (values[2], values[0], values[1])
        for family, values in losses.items()
    }
    swapped_summary = summarize_multi_identity(swapped, baselines, ("STORY", "ARITHMETIC", "TRANSFORM"))
    assert swapped_summary.passes
    assert abs(swapped_summary.normalized_identity_margin - summary.normalized_identity_margin) < 1e-9
