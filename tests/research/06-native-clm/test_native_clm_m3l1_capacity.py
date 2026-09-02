from __future__ import annotations

import torch

from minicells.native_clm_m3l1_capacity import (
    M3L1CapacityConfig,
    _evaluate_gate,
    _fit_rank0_sketch,
    aggregate_m3l1_capacity,
    derive_full_covariance_gate,
    fit_full_gaussian_state,
)


def test_rank0_sketch_is_diagonal_only() -> None:
    torch.manual_seed(1)
    queries = torch.randn(128, 12)
    sketch = _fit_rank0_sketch(
        queries,
        diagonal_regularization=1e-4,
        device=torch.device("cpu"),
    )
    assert sketch.rank == 0
    assert sketch.basis.shape == (12, 0)
    assert sketch.eigenvalues.numel() == 0
    assert sketch.residual_variance.shape == (12,)
    assert sketch.storage_bytes == 8 + 4 * (12 + 12)


def test_full_covariance_gate_separates_synthetic_shift() -> None:
    torch.manual_seed(2)
    old_train = torch.randn(1024, 16) * 0.15
    current_train = torch.randn(1024, 16) * 0.15
    old_train[:, 0] -= 0.7
    current_train[:, 0] += 0.7
    old_test = torch.randn(512, 16) * 0.15
    current_test = torch.randn(512, 16) * 0.15
    old_test[:, 0] -= 0.7
    current_test[:, 0] += 0.7

    old = fit_full_gaussian_state(
        old_train,
        diagonal_regularization=1e-4,
        device=torch.device("cpu"),
    )
    current = fit_full_gaussian_state(
        current_train,
        diagonal_regularization=1e-4,
        device=torch.device("cpu"),
    )
    gate = derive_full_covariance_gate(
        old,
        current,
        diagonal_regularization=1e-4,
        target_old_fpr=0.1,
    )
    metrics = _evaluate_gate(
        old_test,
        current_test,
        gate,
        oracle_auc=0.999,
        device=torch.device("cpu"),
    )
    assert metrics["auc"] > 0.99
    assert metrics["current_tpr"] > 0.95
    assert old.storage_bytes == 8 + 4 * (16 + 16 * 16)


def _candidate(label: str, auc: float, *, rank: int | None) -> dict:
    return {
        "candidate": label,
        "family": (
            "full_covariance_gaussian" if rank is None else "low_rank_gaussian"
        ),
        "rank": rank,
        "historical_address_state_bytes": 600000 if rank is None else 3000 + 1000 * rank,
        "auc": auc,
        "old_fpr": 0.10,
        "current_tpr": 0.82,
        "normalized_oracle_excess_recovery": 0.92,
    }


def _seed_summary(*, passing_rank: int | None, full_auc: float) -> dict:
    edges = []
    for edge_index in range(8):
        candidates = []
        for rank in (0, 8, 16, 32, 64, 128):
            auc = 0.88
            if passing_rank is not None and rank >= passing_rank:
                auc = 0.91
            candidates.append(_candidate(f"rank-{rank}", auc, rank=rank))
        candidates.append(_candidate("full-covariance", full_auc, rank=None))
        edges.append(
            {
                "seed": 73611,
                "parent_id": edge_index,
                "child_id": edge_index + 8,
                "root_id": edge_index,
                "old_domains": ["A"],
                "current_domain": "B",
                "transition": "A->B",
                "valid": True,
                "offline_oracle_auc": 0.94,
                "current_cosine_auc": 0.52,
                "candidates": candidates,
            }
        )
    return {
        "seed": 73611,
        "edge_count": len(edges),
        "valid_edge_count": len(edges),
        "edges": edges,
    }


def test_aggregate_reports_minimum_passing_low_rank() -> None:
    result = aggregate_m3l1_capacity(
        [_seed_summary(passing_rank=32, full_auc=0.93)],
        config=M3L1CapacityConfig(),
        parent_m3l_commit="parent",
        parent_m3r_hf_revision="hf",
    )
    assert result["classification"] == "LOW_RANK_CAPACITY_SUFFICIENT"
    assert result["minimum_passing_low_rank"] == 32
    assert result["capacity_curve"]["rank-16"]["passes_m3l_feasibility_gates"] is False
    assert result["capacity_curve"]["rank-32"]["passes_m3l_feasibility_gates"] is True


def test_aggregate_reports_full_covariance_required() -> None:
    result = aggregate_m3l1_capacity(
        [_seed_summary(passing_rank=None, full_auc=0.92)],
        config=M3L1CapacityConfig(),
        parent_m3l_commit="parent",
        parent_m3r_hf_revision="hf",
    )
    assert result["classification"] == "FULL_COVARIANCE_REQUIRED"
    assert result["minimum_passing_low_rank"] is None
    assert result["full_covariance_passes"] is True


def test_aggregate_reports_gaussian_family_limited() -> None:
    result = aggregate_m3l1_capacity(
        [_seed_summary(passing_rank=None, full_auc=0.89)],
        config=M3L1CapacityConfig(),
        parent_m3l_commit="parent",
        parent_m3r_hf_revision="hf",
    )
    assert result["classification"] == "GAUSSIAN_FAMILY_LIMITED"
    assert result["minimum_passing_low_rank"] is None
    assert result["full_covariance_passes"] is False


def test_capacity_grid_validation_is_frozen_and_sorted() -> None:
    config = M3L1CapacityConfig()
    config.validate()
    assert config.ranks == (0, 8, 16, 32, 64, 128)
