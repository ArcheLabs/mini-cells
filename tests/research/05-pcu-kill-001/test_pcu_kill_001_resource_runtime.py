"""Regression tests for the single-foundation PCU Granite execution layout."""

from __future__ import annotations

import torch

from minicells.pcu_kill_001 import execution, experiment
from minicells.pcu_kill_001.backends import make_toy_model
from minicells.pcu_kill_001.cellular import GraniteArchitectureInspector
from minicells.pcu_kill_001.model import target_module
from minicells.pcu_kill_001.resource_runtime import (
    cellularize_in_place,
    full_moe_overlay_equivalence,
    g0_preflight,
    inference_logits,
    run_granite_engineering,
)


def test_single_foundation_cellularization_does_not_clone_backbone() -> None:
    model = make_toy_model()
    inspector = GraniteArchitectureInspector.inspect(model, require_granite=False)
    embedding = model.model.embed_tokens.weight
    embedding_ptr = embedding.data_ptr()
    parent_experts = target_module(model, inspector.target_path).experts

    cellular, exact_parent = cellularize_in_place(model, inspector)

    assert cellular is model
    assert exact_parent is parent_experts
    assert model.model.embed_tokens.weight.data_ptr() == embedding_ptr
    assert target_module(model, inspector.target_path).experts is not parent_experts
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in exact_parent.parameters())


def test_full_moe_overlay_restores_resident_cellular_experts() -> None:
    model = make_toy_model()
    inspector = GraniteArchitectureInspector.inspect(model, require_granite=False)
    cellular, parent_experts = cellularize_in_place(model, inspector)
    block = target_module(cellular, inspector.target_path)
    resident = block.experts
    probe = torch.randn(24, inspector.hidden_size)

    metrics = full_moe_overlay_equivalence(block, parent_experts, resident, probe)

    assert metrics.passed
    assert block.experts is resident


def test_inference_logits_never_retains_a_backward_graph() -> None:
    model = make_toy_model()
    inputs = {"input_ids": torch.randint(0, model.vocab_size, (8, 8))}
    logits = inference_logits(model, inputs)
    assert logits.requires_grad is False
    assert logits.grad_fn is None


def test_package_installs_resource_bounded_engineering_and_preflight() -> None:
    assert experiment._run_granite_engineering is run_granite_engineering
    assert experiment._logits is inference_logits
    assert execution._g0_preflight is g0_preflight
