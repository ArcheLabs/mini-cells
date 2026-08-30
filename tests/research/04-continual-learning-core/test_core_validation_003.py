from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from minicells.dependency_scoped_experiment import run_primary_seed, summarize_experiment
from minicells.dependency_scoped_transactional import (
    CoreValidation003Config,
    RoutedCellModel,
    RoutedContinualWorld,
    evaluate_candidate,
    smoke_config,
    train_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research" / "validations" / "core-003-dependency-scoped-transactional-learning" / "protocol.json"


def _smoke() -> CoreValidation003Config:
    return smoke_config(CoreValidation003Config.from_protocol(PROTOCOL))


def test_granularity_keeps_expert_budget_approximately_constant_and_shrinks_block() -> None:
    config = _smoke()
    world = RoutedContinualWorld(config, seed=11)
    coarse = RoutedCellModel(config, world, granularity=1, seed=22)
    fine = RoutedCellModel(config, world, granularity=4, seed=23)

    coarse_expert_budget = coarse.num_experts * coarse.expert_block_parameter_count()
    fine_expert_budget = fine.num_experts * fine.expert_block_parameter_count()
    ratio = fine_expert_budget / coarse_expert_budget
    assert 0.9 <= ratio <= 1.1
    assert fine.local_candidate_parameter_fraction(2) < coarse.local_candidate_parameter_fraction(2)


def test_frozen_local_candidate_cannot_escape_preupdate_dependency_scope() -> None:
    config = replace(_smoke(), update_steps=2, update_train_examples=8)
    world = RoutedContinualWorld(config, seed=31)
    model = RoutedCellModel(config, world, granularity=4, seed=32)
    old_amplitudes = world.initial_amplitudes()
    target = int(world.mutable_context_ids[0].item())
    new_amplitudes = old_amplitudes.clone()
    new_amplitudes[target] += config.update_amplitude

    candidate = train_candidate(
        model,
        config,
        world,
        amplitudes=new_amplitudes,
        target_context=target,
        seed=33,
        device=torch.device("cpu"),
        update_shared=False,
    )
    historical_x, historical_contexts = world.fixed_historical_inputs(
        examples_per_context=4, seed=34
    )
    metrics = evaluate_candidate(
        model,
        candidate,
        config,
        world,
        old_amplitudes=old_amplitudes,
        new_amplitudes=new_amplitudes,
        target_context=target,
        historical_x=historical_x,
        historical_contexts=historical_contexts,
        new_validation_seed=35,
        device=torch.device("cpu"),
    )
    assert metrics["structural_escape_rate"] == 0.0
    assert metrics["routing_drift_rate"] == 0.0
    assert 0.0 <= metrics["dependency_coverage"] < 1.0


def test_router_drift_stress_breaks_structural_invariant() -> None:
    config = replace(_smoke(), update_steps=1, update_train_examples=8)
    world = RoutedContinualWorld(config, seed=41)
    model = RoutedCellModel(config, world, granularity=4, seed=42)
    old_amplitudes = world.initial_amplitudes()
    target = int(world.mutable_context_ids[0].item())
    new_amplitudes = old_amplitudes.clone()
    new_amplitudes[target] += config.update_amplitude

    candidate = train_candidate(
        model,
        config,
        world,
        amplitudes=new_amplitudes,
        target_context=target,
        seed=43,
        device=torch.device("cpu"),
        update_shared=False,
    )
    candidate.perturb_router(seed=44, noise_scale=1.0)
    historical_x, historical_contexts = world.fixed_historical_inputs(
        examples_per_context=4, seed=45
    )
    metrics = evaluate_candidate(
        model,
        candidate,
        config,
        world,
        old_amplitudes=old_amplitudes,
        new_amplitudes=new_amplitudes,
        target_context=target,
        historical_x=historical_x,
        historical_contexts=historical_contexts,
        new_validation_seed=46,
        device=torch.device("cpu"),
    )
    assert metrics["routing_drift_rate"] > 0.0
    assert metrics["structural_escape_rate"] > 0.0


def test_experiment_decision_requires_one_shared_granularity() -> None:
    runs = [
        {
            "supported_granularities": [4],
            "hypothesis_diagnostics": {
                "h1_structural_locality": True,
                "h5_router_drift_increases_escape": True,
                "h5_router_drift_increases_false_safe": True,
            },
        },
        {
            "supported_granularities": [8],
            "hypothesis_diagnostics": {
                "h1_structural_locality": True,
                "h5_router_drift_increases_escape": True,
                "h5_router_drift_increases_false_safe": True,
            },
        },
    ]
    decision = summarize_experiment(
        runs,
        granularities=(1, 4, 8),
        positive_status="YES",
        negative_status="NO",
    )
    assert decision["pass"] is False
    assert decision["status"] == "NO"

    runs[1]["supported_granularities"] = [4, 8]
    decision = summarize_experiment(
        runs,
        granularities=(1, 4, 8),
        positive_status="YES",
        negative_status="NO",
    )
    assert decision["pass"] is True
    assert decision["shared_supported_granularities"] == [4]


def test_cpu_smoke_pipeline_executes() -> None:
    config = _smoke()
    run = run_primary_seed(config, seed=51, device=torch.device("cpu"))
    assert len(run["granularities"]) == 2
    for granularity_run in run["granularities"]:
        assert set(granularity_run["variants"]) == {
            "standard_moe_always",
            "local_always",
            "local_tx_frozen",
            "local_tx_router_drift",
        }
        assert len(granularity_run["variants"]["local_tx_frozen"]["records"]) == config.transactions
