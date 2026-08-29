from __future__ import annotations

import json
from pathlib import Path

import torch

from minicells.knowledge_subsumption import KnowledgeSubsumptionConfig
from minicells.residual_memorization import (
    ResidualMemorizationConfig,
    coupling_metrics,
    rank_all_frequency_pairs,
    summarize_experiment,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research" / "core-validation-001b-protocol.json"
NOTEBOOK = ROOT / "research" / "kaggle" / "core-validation-001b-residual-memorization.ipynb"


def _sweep(left: list[float], right: list[float]) -> list[dict[str, object]]:
    return [
        {
            "k": index,
            "excluded": {
                "old": {"accuracy": a, "nll": 0.0},
                "heldout": {"accuracy": b, "nll": 0.0},
            },
            "restricted": {},
        }
        for index, (a, b) in enumerate(zip(left, right))
    ]


def test_protocol_is_frozen_diagnostic_extension() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "core-validation-001b"
    assert payload["parent_experiment"] == "core-validation-001"
    assert payload["source"]["fresh_rerun_required"] is True
    assert payload["source"]["reuse_parent_runner"] == "scripts/run_core_validation_001.py"
    assert payload["scope"]["training_changes_from_parent"] == "none"
    assert payload["scope"]["growth"] is False
    assert payload["scope"]["replay_old_examples"] is False
    assert payload["gates"]["replication"]["require_oracle_valid"] is True


def test_rank_all_frequency_pairs_covers_non_dc_pairs() -> None:
    config = KnowledgeSubsumptionConfig(
        modulus=7,
        curriculum_fractions=(0.2, 0.2),
        phase_steps=(2, 2),
        embedding_dim=8,
        num_cells=4,
        neurons_per_cell=2,
        batch_size=16,
        key_frequency_pairs=1,
    )
    embedding = torch.randn(config.modulus, config.embedding_dim)
    ranking = rank_all_frequency_pairs(embedding)
    assert len(ranking) == 3
    assert set(ranking) == {1, 2, 3}


def test_coupling_metrics_accepts_synchronized_decay() -> None:
    metrics = coupling_metrics(
        _sweep([1.0, 0.8, 0.5, 0.1], [0.98, 0.79, 0.49, 0.09]),
        "old",
        "heldout",
    )
    assert metrics["exclusion_accuracy_correlation"] > 0.99
    assert metrics["mean_absolute_gap"] < 0.02
    assert metrics["maximum_positive_gap"] < 0.03


def test_coupling_metrics_detects_membership_advantage() -> None:
    metrics = coupling_metrics(
        _sweep([1.0, 0.9, 0.75, 0.1], [1.0, 0.55, 0.2, 0.1]),
        "old",
        "heldout",
    )
    assert metrics["maximum_positive_gap"] > 0.5
    assert metrics["mean_absolute_gap"] > 0.2


def test_residual_config_matches_protocol() -> None:
    config = ResidualMemorizationConfig.from_protocol(PROTOCOL)
    assert config.minimum_early_gap == 0.50
    assert config.minimum_correlation == 0.95
    assert config.maximum_mean_absolute_gap == 0.05
    assert config.maximum_positive_gap == 0.10
    assert config.maximum_dc_only_accuracy == 0.15


def test_summary_requires_primary_control_and_oracle() -> None:
    primary = {
        "task": "modular_addition",
        "gates": {"pass": True},
    }
    control = {
        "task": "balanced_random_labels",
        "gates": {"pass": False},
        "source_control_valid": True,
    }
    oracle = {"gates": {"valid": True}}
    decision = summarize_experiment(
        [primary, primary | {"seed": 2}, primary | {"seed": 3}, control],
        oracle,
        positive_status="YES",
        negative_status="NO",
    )
    assert decision["status"] == "YES"
    assert decision["supported"] is True

    bad_oracle = summarize_experiment(
        [primary, control],
        {"gates": {"valid": False}},
        positive_status="YES",
        negative_status="NO",
    )
    assert bad_oracle["status"] == "NO"


def test_notebook_reuses_parent_runner_and_publishes_001b() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        line
        for cell in payload["cells"]
        for line in cell.get("source", [])
    )
    assert "run_core_validation_001.py" in source
    assert "--skip-oracle" in source
    assert "analyze_core_validation_001b.py" in source
    assert "publish_core_validation_001b.py" in source
    assert "kaggle/core-validation-001b-residual-memorization-results" in source
