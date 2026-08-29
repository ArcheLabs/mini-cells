from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from minicells.knowledge_subsumption import (
    CellularModularNet,
    KnowledgeSubsumptionConfig,
    fourier_filter_embedding,
    make_curriculum,
    mean_pairwise_jaccard,
    path_fingerprints,
    responsibility_matrix,
    select_key_frequency_pairs,
    summarize_experiment,
    train_sequential_run,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research" / "core-validation-001-protocol.json"


def smoke_config() -> KnowledgeSubsumptionConfig:
    return KnowledgeSubsumptionConfig(
        modulus=7,
        curriculum_fractions=(0.20, 0.20),
        phase_steps=(2, 2),
        eval_interval_steps=1,
        embedding_dim=8,
        num_cells=4,
        neurons_per_cell=2,
        batch_size=16,
        probe_examples_per_partition=8,
        path_cells=2,
        key_frequency_pairs=1,
        early_minimum_seen_accuracy=0.0,
        early_maximum_unseen_accuracy=1.0,
        late_minimum_old_accuracy=0.0,
        late_minimum_current_accuracy=0.0,
        late_minimum_heldout_accuracy=0.0,
        restricted_minimum_old_accuracy=0.0,
        restricted_minimum_heldout_accuracy=0.0,
        early_excluded_minimum_seen_accuracy=0.0,
        late_excluded_maximum_old_accuracy=1.0,
        late_excluded_maximum_heldout_accuracy=1.0,
    )


def test_frozen_protocol_matches_config() -> None:
    payload = json.loads(PROTOCOL.read_text())
    config = KnowledgeSubsumptionConfig.from_protocol(PROTOCOL)
    assert payload["experiment_id"] == "core-validation-001"
    assert config.modulus == 31
    assert config.num_cells == 16
    assert config.hidden_dim == 128
    assert sum(config.curriculum_fractions) < 1.0
    assert payload["scope"]["growth"] is False
    assert payload["task"]["replay_old_examples"] is False


def test_curriculum_is_disjoint_and_control_is_balanced() -> None:
    config = smoke_config()
    addition = make_curriculum(config, seed=11, task="modular_addition")
    control = make_curriculum(config, seed=11, task="balanced_random_labels")
    groups = [*addition.phases, addition.heldout]
    flattened = torch.cat(groups)
    assert len(torch.unique(flattened)) == len(flattened) == config.modulus**2
    expected = torch.bincount(addition.labels, minlength=config.modulus)
    observed = torch.bincount(control.labels, minlength=config.modulus)
    assert torch.equal(expected, observed)
    assert not torch.equal(addition.labels, control.labels)


def test_hidden_cell_ablation_and_responsibility_are_explicit() -> None:
    config = smoke_config()
    curriculum = make_curriculum(config, seed=12, task="modular_addition")
    model = CellularModularNet(config)
    indices = curriculum.phases[0][:5]
    pairs = curriculum.pairs[indices]
    baseline = model(pairs)
    ablated = model(pairs, ablate_cells=(0,))
    assert baseline.shape == ablated.shape == (len(indices), config.modulus)
    responsibility = responsibility_matrix(
        model, curriculum, indices, device=torch.device("cpu")
    )
    assert responsibility.shape == (config.num_cells, len(indices))
    assert torch.all(responsibility >= 0)


def test_fourier_restricted_and_excluded_embeddings_partition_key_pairs() -> None:
    config = smoke_config()
    model = CellularModularNet(config)
    keys = select_key_frequency_pairs(model.embedding.weight, 1)
    restricted = fourier_filter_embedding(model.embedding.weight, keys, keep_keys=True)
    excluded = fourier_filter_embedding(model.embedding.weight, keys, keep_keys=False)
    assert restricted.shape == excluded.shape == model.embedding.weight.shape
    assert len(keys) == 1
    assert torch.isfinite(restricted).all()
    assert torch.isfinite(excluded).all()


def test_path_reuse_metric_is_well_formed() -> None:
    responsibility = torch.tensor(
        [
            [4.0, 4.0, 4.0],
            [3.0, 0.1, 0.1],
            [0.1, 3.0, 0.1],
            [0.1, 0.1, 3.0],
        ]
    )
    paths = path_fingerprints(responsibility, 2)
    reuse = mean_pairwise_jaccard(paths)
    assert 0 < reuse < 1


def test_decision_requires_every_primary_seed_and_zero_control_false_positives() -> None:
    passing = {"gates": {"pass": True}}
    failing = {"gates": {"pass": False}}
    runs = [
        {"task": "modular_addition", **passing},
        {"task": "modular_addition", **passing},
        {"task": "balanced_random_labels", **failing},
    ]
    assert summarize_experiment(runs)["supported"] is True
    runs[1] = {"task": "modular_addition", **failing}
    assert summarize_experiment(runs)["supported"] is False
    runs[1] = {"task": "modular_addition", **passing}
    runs[2] = {"task": "balanced_random_labels", **passing}
    assert summarize_experiment(runs)["supported"] is False


def test_tiny_sequential_run_produces_all_core_measurements() -> None:
    config = replace(smoke_config(), phase_steps=(1, 1), eval_interval_steps=1)
    run = train_sequential_run(
        config,
        seed=13,
        task="modular_addition",
        device=torch.device("cpu"),
    )
    assert run["task"] == "modular_addition"
    assert set(run["gates"]) == {
        "early_memorization",
        "late_generalization",
        "generalizing_circuit",
        "memorization_cleanup",
        "pass",
    }
    assert "key_frequency_pairs" in run["mechanistic"]
    assert "path_reuse_gain" in run["mechanistic"]
