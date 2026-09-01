from __future__ import annotations

import json
from pathlib import Path

import torch

from minicells.subspace_mitosis_005 import (
    CoreValidation005Config,
    LinearCell,
    constrained_update,
    extend_basis,
    make_transaction,
    run_primary_seed,
    smoke_config,
    summarize_experiment,
    wrong_basis,
)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "research" / "validations" / "core-005-subspace-certified-mitosis" / "protocol.json"


def test_protocol_is_frozen_without_calibration() -> None:
    config = CoreValidation005Config.from_protocol(PROTOCOL)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "PROTOCOL_FROZEN_UNRUN"
    assert protocol["replication"]["formal_seeds"] == [80501, 80502, 80503]
    assert protocol["replication"]["development_seed"] is None
    assert protocol["replication"]["calibration_allowed"] is False
    assert config.feature_dim == 12
    assert config.transactions_per_base == 12


def test_projected_write_exactly_preserves_registered_span() -> None:
    config = CoreValidation005Config.from_protocol(PROTOCOL)
    old = torch.zeros(4, config.feature_dim, dtype=torch.float64)
    old[:, 0] = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float64)
    old[:, 1] = torch.tensor([0.5, -1.0, 2.0, 1.0], dtype=torch.float64)
    basis = extend_basis(
        torch.zeros(config.feature_dim, 0, dtype=torch.float64),
        old,
        tolerance=config.numerical_rank_tolerance,
    )
    z = torch.zeros(3, config.feature_dim, dtype=torch.float64)
    z[:, 2] = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    residual = torch.tensor(
        [[1.0, -0.5, 0.25], [-2.0, 1.0, -0.5], [3.0, -1.5, 0.75]],
        dtype=torch.float64,
    )
    solved = constrained_update(
        z,
        residual,
        basis,
        numerical_rank_tolerance=config.numerical_rank_tolerance,
    )
    assert solved["fit_error"] <= config.feasibility_threshold
    delta = solved["delta_weight"]
    assert torch.max(torch.abs(delta @ basis)).item() <= 1e-12
    assert torch.max(torch.abs(old @ delta.T)).item() <= 1e-12


def test_forced_collision_is_infeasible_for_true_q_but_exposes_wrong_q() -> None:
    config = smoke_config(CoreValidation005Config.from_protocol(PROTOCOL))
    cell = LinearCell.empty(config)
    for local_index in (0, 1, 2):
        transaction = make_transaction(
            config,
            base_id=0,
            local_index=local_index,
            seed=9000 + local_index,
        )
        cell.basis = extend_basis(
            cell.basis,
            transaction.z,
            tolerance=config.numerical_rank_tolerance,
        )
    collision = make_transaction(config, base_id=0, local_index=3, seed=9010)
    true_solve = constrained_update(
        collision.z,
        collision.residual,
        cell.basis,
        numerical_rank_tolerance=config.numerical_rank_tolerance,
    )
    wrong_solve = constrained_update(
        collision.z,
        collision.residual,
        wrong_basis(cell.basis),
        numerical_rank_tolerance=config.numerical_rank_tolerance,
    )
    assert true_solve["fit_error"] > config.feasibility_threshold
    assert wrong_solve["fit_error"] <= config.feasibility_threshold


def test_smoke_closes_replay_free_growth_loop() -> None:
    config = smoke_config(CoreValidation005Config.from_protocol(PROTOCOL))
    run = run_primary_seed(config, seed=80501)
    assert set(run["variants"]) == {
        "unsafe_always",
        "certificate_no_growth",
        "certificate_growth",
        "wrong_certificate",
    }
    growth = run["variants"]["certificate_growth"]["summary"]
    no_growth = run["variants"]["certificate_no_growth"]["summary"]
    wrong = run["variants"]["wrong_certificate"]["summary"]
    assert growth["learner_old_sample_accesses"] == 0
    assert growth["learner_old_label_accesses"] == 0
    assert growth["learner_replay_items_retained"] == 0
    assert growth["false_safe_count"] == 0
    assert growth["decision_mismatch_count"] == 0
    assert growth["growth_commits"] == 3
    assert growth["child_reuse_commits"] == 4
    assert growth["cumulative_committed_new_gain"] > no_growth["cumulative_committed_new_gain"]
    assert wrong["false_safe_count"] >= 1


def test_formal_decision_requires_all_seeds() -> None:
    good = {
        "gate_summary": {
            "pass": True,
            "gates": {
                "no_replay": True,
                "certificate_matches_full_history": True,
                "zero_false_safe": True,
                "stability": True,
                "plasticity": True,
                "growth_rescue": True,
                "child_reuse": True,
                "bounded_growth": True,
                "wrong_certificate_causal_failure": True,
            },
        }
    }
    bad = {
        "gate_summary": {
            "pass": False,
            "gates": {
                "no_replay": True,
                "certificate_matches_full_history": False,
                "zero_false_safe": True,
                "stability": True,
                "plasticity": True,
                "growth_rescue": True,
                "child_reuse": True,
                "bounded_growth": True,
                "wrong_certificate_causal_failure": True,
            },
        }
    }
    decision = summarize_experiment(
        [good, good, bad],
        positive_status="YES",
        negative_status="NO",
    )
    assert decision["status"] == "NO"
    assert decision["pass"] is False
    assert decision["passed_seeds"] == 2
