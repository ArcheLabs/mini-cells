from __future__ import annotations

import json
from pathlib import Path

from minicells.constructive_clm_003 import protection_only_smoke


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = (
    REPO_ROOT
    / "research/validations/constructive-clm-003-protected-growing-cells/protocol.json"
)


def test_constructive_clm_003_seed_discipline_is_frozen() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "PROTOCOL_FROZEN_UNRUN"
    development = set(protocol["development_only_seeds_observed_and_excluded"])
    formal = set(protocol["formal_seeds_frozen_after_development"])
    assert development == {401, 402, 403}
    assert formal == {90511, 90512, 90513}
    assert development.isdisjoint(formal)
    assert protocol["learner"]["hard_cell_cap"] is None


def test_protection_only_smoke_locks_replay_free_bounded_mitosis() -> None:
    result = protection_only_smoke(401)
    unsafe = result["variants"]["unsafe"]
    no_growth = result["variants"]["certificate_no_growth"]
    certificate = result["variants"]["certificate_growth"]
    replay = result["variants"]["replay_growth_oracle"]

    assert result["roots"] == 6
    assert result["true_functional_cells"] == 18

    assert unsafe["final_historical_regression_mse"] >= 1e-4
    assert no_growth["final_historical_regression_mse"] <= 1e-10
    assert no_growth["acquisition_gain"] <= 0.5 * replay["acquisition_gain"]

    assert certificate["replay_accesses"] == 0
    assert certificate["old_sample_accesses"] == 0
    assert certificate["old_label_accesses"] == 0
    assert certificate["final_historical_regression_mse"] <= 1e-10
    assert certificate["final_cells"] == 18
    assert certificate["child_count"] == 12
    assert certificate["tail_spawns"] == 0
    assert all(size == 3 for size in certificate["lineage_sizes"])
    assert certificate["route"]["exact_mode_accuracy"] >= 0.99
    assert certificate["tail_child_route_accuracy"] >= 0.99
    assert certificate["final_behavior"]["mse"] <= 1e-8

    assert replay["replay_accesses"] > 0
    assert replay["final_cells"] == 18
    assert replay["final_behavior"]["mse"] <= 1e-8
    assert certificate["acquisition_gain"] >= 0.98 * replay["acquisition_gain"]
