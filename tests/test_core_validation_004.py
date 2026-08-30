from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from minicells.growth_plasticity_004_config import CoreValidation004Config
from minicells.growth_plasticity_004_world import GrowthPlasticityWorld
from minicells.growth_plasticity_004_model import GrowingRoutedModel
from minicells.growth_plasticity_004_ops import (
    evaluate_candidate, pretrain_model, smoke_config, train_private_cell_candidate,
)
from minicells.growth_plasticity_004_experiment import (
    run_primary_seed,
    summarize_experiment,
)

PROTOCOL = Path(__file__).resolve().parents[1] / "research" / "core-validation-004-protocol.json"


def test_protocol_parses_and_freezes_g8() -> None:
    cfg = CoreValidation004Config.from_protocol(PROTOCOL)
    assert cfg.granularity == 8
    import json
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["replication"]["seeds"] == [80411, 80412, 80413]
    assert 80401 not in protocol["replication"]["seeds"]
    assert cfg.maximum_active_growth_cells_per_input == 1
    assert cfg.minimum_committed_gain_ratio_vs_local_always == 0.80


def test_zero_output_spawn_is_functionally_identity_before_training() -> None:
    cfg = smoke_config(CoreValidation004Config.from_protocol(PROTOCOL))
    world = GrowthPlasticityWorld(cfg, seed=11)
    model = GrowingRoutedModel(cfg, world, seed=12)
    x = torch.randn(24, cfg.content_dim)
    c = torch.arange(24) % cfg.num_contexts
    before = model(x, c).detach().clone()
    model.add_growth_cell(target_context=5, seed=13)
    after = model(x, c).detach()
    assert torch.equal(before, after)


def test_private_growth_route_has_zero_structural_escape() -> None:
    cfg = smoke_config(CoreValidation004Config.from_protocol(PROTOCOL))
    cfg = replace(cfg, growth_steps=12, growth_learning_rate=0.08)
    world = GrowthPlasticityWorld(cfg, seed=21)
    model, _ = pretrain_model(cfg, world, seed=22, device=torch.device("cpu"))
    old_amp = world.initial_amplitudes()
    new_amp = old_amp.clone()
    target = cfg.anchor_contexts
    new_amp[target] += cfg.update_amplitude
    candidate = train_private_cell_candidate(
        model,
        cfg,
        world,
        amplitudes=new_amp,
        target_context=target,
        seed=23,
        device=torch.device("cpu"),
        spawn=True,
    )
    hx, hc = world.fixed_historical_inputs(examples_per_context=4, seed=24)
    metrics = evaluate_candidate(
        model,
        candidate,
        cfg,
        world,
        old_amplitudes=old_amp,
        new_amplitudes=new_amp,
        target_context=target,
        historical_x=hx,
        historical_contexts=hc,
        new_validation_seed=25,
        device=torch.device("cpu"),
        candidate_kind="spawn",
    )
    assert metrics["dependency_coverage"] == 0.0
    assert metrics["structural_escape_rate"] == 0.0
    assert metrics["global_regression"] == 0.0


def test_smoke_seed_executes_all_three_variants() -> None:
    cfg = smoke_config(CoreValidation004Config.from_protocol(PROTOCOL))
    run = run_primary_seed(cfg, seed=80401, device=torch.device("cpu"))
    assert set(run["variants"]) == {"local_always", "local_tx", "local_tx_growth"}
    growth = run["variants"]["local_tx_growth"]["summary"]
    assert 0.0 <= growth["effective_acceptance_rate"] <= 1.0
    assert growth["spawned_cells"] <= cfg.num_contexts - cfg.anchor_contexts
    assert growth["maximum_active_growth_cells_per_input"] <= 1


def test_experiment_requires_all_seeds() -> None:
    good = {
        "gate_summary": {
            "pass": True,
            "gates": {
                "plasticity_recovery": True,
                "scope_safety": True,
                "structural_locality": True,
                "bounded_growth": True,
                "private_cell_reuse": True,
            },
        }
    }
    bad = {
        "gate_summary": {
            "pass": False,
            "gates": {
                "plasticity_recovery": False,
                "scope_safety": True,
                "structural_locality": True,
                "bounded_growth": True,
                "private_cell_reuse": True,
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
