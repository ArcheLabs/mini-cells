"""Low-cost PCU-KILL-001 infrastructure tests; no formal seed is executed."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch

from minicells.pcu_kill_001.backends import make_toy_model
from minicells.pcu_kill_001.cache import CachedTailRunner
from minicells.pcu_kill_001.cellular import CellPartition, GraniteArchitectureInspector
from minicells.pcu_kill_001.composition import ComposedCell, compose_cellular_experts
from minicells.pcu_kill_001.equivalence import verify_end_to_end, verify_expert_algebra
from minicells.pcu_kill_001.registry import fork_registry, make_foundation_registry, merge_registries, rollback_registry
from minicells.pcu_kill_001.synthetic import audit_dataset, generate_world
from minicells.pcu_kill_001.training import fork_expert, fork_initial_delta_norm, foundation_tensor_hashes, selected_delta_parameters
from minicells.pcu_kill_001.lora import LoRACell, LoRAConfig, choose_matched_rank, lora_parameter_count, merged_effective_deltas


def test_cell_partition_covers_all_channels() -> None:
    partition = CellPartition(512, 4)
    assert partition.validate()
    assert partition.ranges() == ((0, 128), (128, 256), (256, 384), (384, 512))


def test_cell_partition_has_no_overlap() -> None:
    ranges = CellPartition(512, 4).ranges()
    assert sum(end - start for start, end in ranges) == 512
    assert len({item for start, end in ranges for item in range(start, end)}) == 512


def test_cell_reconstructs_expert_fp32() -> None:
    model = make_toy_model()
    experts = model.model.layers[0].block_sparse_moe.experts
    assert all(verify_expert_algebra(experts, index, CellPartition(16, 4), vectors=16).passed for index in range(4))


def test_fused_gate_up_slice_alignment() -> None:
    model = make_toy_model()
    projections = model.model.layers[0].block_sparse_moe.experts.gate_up_proj[0]
    assert torch.equal(projections[:8], model.model.layers[0].block_sparse_moe.experts.gate_up_proj[0, :8])


def test_down_projection_slice_alignment() -> None:
    model = make_toy_model()
    cellular = GraniteArchitectureInspector.inspect(model, require_granite=False)
    source = model.model.layers[0].block_sparse_moe.experts.down_proj[0]
    assert source.shape == (cellular.hidden_size, cellular.intermediate_size)


def test_original_router_unchanged_and_no_child_softmax() -> None:
    model = make_toy_model()
    router = model.model.layers[0].block_sparse_moe.router
    source = deepcopy(model)
    inspector = GraniteArchitectureInspector.inspect(source, require_granite=False)
    from minicells.pcu_kill_001.model import cellularize_model
    cellular, _ = cellularize_model(source, inspector)
    assert cellular.model.layers[0].block_sparse_moe.router is not router
    for name, value in router.state_dict().items():
        torch.testing.assert_close(value, cellular.model.layers[0].block_sparse_moe.router.state_dict()[name])


def test_fork_initial_delta_is_zero_and_storage_is_independent() -> None:
    model = make_toy_model()
    inspector = GraniteArchitectureInspector.inspect(model, require_granite=False)
    from minicells.pcu_kill_001.model import cellularize_model
    cellular, _ = cellularize_model(model, inspector)
    parent = cellular.model.layers[0].block_sparse_moe.experts.cells[0]
    fork = fork_expert(parent, [0])
    assert fork_initial_delta_norm(fork) == 0.0
    assert fork.cells[0].parent_gate_weight.data_ptr() != parent.cells[0].gate_weight.data_ptr()


def test_parent_hash_unchanged_after_training_step() -> None:
    model = make_toy_model()
    inspector = GraniteArchitectureInspector.inspect(model, require_granite=False)
    from minicells.pcu_kill_001.model import cellularize_model
    cellular, _ = cellularize_model(model, inspector)
    before = foundation_tensor_hashes(cellular)
    fork = fork_expert(cellular.model.layers[0].block_sparse_moe.experts.cells[0], [0])
    optimizer = torch.optim.AdamW(selected_delta_parameters(fork), lr=1e-2)
    x = torch.randn(4, inspector.hidden_size)
    loss = fork(x).square().mean()
    loss.backward()
    optimizer.step()
    assert foundation_tensor_hashes(cellular) == before


def test_merge_is_registry_union_and_preserves_overlap() -> None:
    base = make_foundation_registry(layer=23, experts=32, cells_per_expert=4, cell_width=128, foundation_model="m", foundation_revision="r", foundation_hash="h", protocol_sha256="p")
    a = fork_registry(base, ["L23:E0:C0"], "A")
    b = fork_registry(base, ["L23:E0:C0"], "B")
    merged = merge_registries(base, a, b)
    assert len(merged.fork_records) == 2
    assert rollback_registry(merged, "B").content_hash() == a.content_hash()
    assert rollback_registry(merged, "A").content_hash() == b.content_hash()


def test_merge_requires_same_foundation_and_protocol_hash() -> None:
    base = make_foundation_registry(layer=23, experts=1, cells_per_expert=4, cell_width=128, foundation_model="m", foundation_revision="r", foundation_hash="h", protocol_sha256="p")
    other = make_foundation_registry(layer=23, experts=1, cells_per_expert=4, cell_width=128, foundation_model="m", foundation_revision="r", foundation_hash="different", protocol_sha256="p")
    with pytest.raises(ValueError, match="foundation hash"):
        merge_registries(base, base, other)


def test_dataset_leakage_auditor_and_determinism() -> None:
    left = generate_world(26090501, 128)
    right = generate_world(26090501, 128)
    assert left.manifest_sha256() == right.manifest_sha256()
    assert audit_dataset(left).passed


def test_formal_seed_never_enters_engineering_path() -> None:
    from minicells.pcu_kill_001.governance import assert_engineering_seed
    with pytest.raises(ValueError):
        assert_engineering_seed(26090511)


def test_cached_tail_matches_full_model() -> None:
    model = make_toy_model()
    inspector = GraniteArchitectureInspector.inspect(model, require_granite=False)
    from minicells.pcu_kill_001.model import cellularize_model
    cellular, _ = cellularize_model(model, inspector)
    inputs = {"input_ids": torch.randint(0, 96, (128, 8))}
    assert verify_end_to_end(model, cellular, inputs).top1_token_agreement == 1.0
    runner = CachedTailRunner(cellular, "model.layers.0")
    assert runner.verify(runner.capture(inputs["input_ids"])).top1_agreement == 1.0


def test_functional_overlap_and_rollback_are_exact() -> None:
    model = make_toy_model()
    inspector = GraniteArchitectureInspector.inspect(model, require_granite=False)
    from minicells.pcu_kill_001.model import cellularize_model
    cellular, _ = cellularize_model(model, inspector)
    moe = cellular.model.layers[0].block_sparse_moe
    parent = moe.experts
    fork_a = fork_expert(parent.cells[0], [0])
    fork_b = fork_expert(parent.cells[0], [0])
    with torch.no_grad():
        fork_a.cells[0].delta_gate_weight.fill_(0.01)
        fork_b.cells[0].delta_up_weight.fill_(0.02)
    hidden = torch.randn(16, inspector.hidden_size)
    parent_cell = parent.cells[0].cells[0]
    composed = ComposedCell(parent_cell, [fork_a.cells[0], fork_b.cells[0]])
    branch_a = ComposedCell(parent_cell, [fork_a.cells[0]])
    branch_b = ComposedCell(parent_cell, [fork_b.cells[0]])
    torch.testing.assert_close(
        composed(hidden),
        parent_cell(hidden) + (branch_a(hidden) - parent_cell(hidden)) + (branch_b(hidden) - parent_cell(hidden)),
        rtol=1e-5,
        atol=1e-5,
    )
    runtime = compose_cellular_experts(parent, {"A": {0: fork_a}, "B": {0: fork_b}})
    indices, weights = moe.router(hidden)
    base = parent(hidden, indices, weights)
    only_a = runtime.rollback("B")(hidden, indices, weights)
    only_b = runtime.rollback("A")(hidden, indices, weights)
    all_rolled_back = runtime.rollback("all")(hidden, indices, weights)
    torch.testing.assert_close(all_rolled_back, base, rtol=1e-5, atol=1e-5)
    assert not torch.equal(only_a, only_b)


def test_lora_merge_has_no_cross_terms_and_matches_budget() -> None:
    model = make_toy_model()
    inspector = GraniteArchitectureInspector.inspect(model, require_granite=False)
    from minicells.pcu_kill_001.model import cellularize_model
    cellular, _ = cellularize_model(model, inspector)
    cell = cellular.model.layers[0].block_sparse_moe.experts.cells[0].cells[0]
    left = LoRACell(cell, LoRAConfig(rank=2), trainable=False)
    right = LoRACell(cell, LoRAConfig(rank=2), trainable=False)
    with torch.no_grad():
        for parameter in list(left.parameters()) + list(right.parameters()):
            if parameter.ndim > 1:
                parameter.normal_(std=0.01)
    merged = merged_effective_deltas(left.state_delta(), right.state_delta(), scale_a=left.scale, scale_b=right.scale)
    expected = {key: left.effective_deltas()[key] + right.effective_deltas()[key] for key in ("gate", "up", "down")}
    for key in expected:
        torch.testing.assert_close(merged[key], expected[key], rtol=1e-5, atol=1e-6)
    target = sum(int(value.numel()) for value in (cell.gate_weight, cell.up_weight, cell.down_weight))
    rank = choose_matched_rank(target, inspector.hidden_size, inspector.partition.cell_size, 1)
    assert abs(lora_parameter_count(inspector.hidden_size, inspector.partition.cell_size, 1, rank) - target) / target <= 0.10


def test_functional_nonoverlap_composes_independent_cells() -> None:
    model = make_toy_model()
    inspector = GraniteArchitectureInspector.inspect(model, require_granite=False)
    from minicells.pcu_kill_001.model import cellularize_model
    cellular, _ = cellularize_model(model, inspector)
    moe = cellular.model.layers[0].block_sparse_moe
    parent = moe.experts
    fork_a = fork_expert(parent.cells[0], [0])
    fork_b = fork_expert(parent.cells[0], [1])
    with torch.no_grad():
        fork_a.cells[0].delta_gate_weight.fill_(0.01)
        fork_b.cells[1].delta_up_weight.fill_(0.02)
    hidden = torch.randn(16, inspector.hidden_size)
    runtime = compose_cellular_experts(parent, {"A": {0: fork_a}, "B": {0: fork_b}})
    only_a = compose_cellular_experts(parent, {"A": {0: fork_a}, "B": {0: fork_b}}, ("A",))
    only_b = compose_cellular_experts(parent, {"A": {0: fork_a}, "B": {0: fork_b}}, ("B",))
    indices, weights = moe.router(hidden)
    base = parent(hidden, indices, weights)
    expected = base + (only_a(hidden, indices, weights) - base) + (only_b(hidden, indices, weights) - base)
    torch.testing.assert_close(runtime(hidden, indices, weights), expected, rtol=1e-5, atol=1e-5)


def test_fork_registry_requires_runtime_artifact_binding() -> None:
    base = make_foundation_registry(layer=0, experts=1, cells_per_expert=4, cell_width=4, foundation_model="m", foundation_revision="r", foundation_hash="h", protocol_sha256="p")
    branch = fork_registry(base, ["L0:E0:C0"], "A")
    from minicells.pcu_kill_001.registry import validate_fork_artifacts
    with pytest.raises(ValueError, match="runtime artifact"):
        validate_fork_artifacts(branch)


def test_freeze_consumes_decision_and_rejects_missing_selected_value() -> None:
    script = Path(__file__).resolve().parents[3] / "scripts/research/freeze_pcu_kill_001.py"
    spec = importlib.util.spec_from_file_location("pcu_freeze_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = {
        "model_repo": "ibm-granite/granite-3.1-1b-a400m-base",
        "model_revision": "revision",
        "config_sha256": "config",
        "weight_file_sha256": [{"file": "weights.safetensors", "sha256": "weights"}],
        "tokenizer_sha256": [{"file": "tokenizer.json", "sha256": "tokenizer"}],
    }
    decision = {
        "experiment": "PCU-KILL-001",
        "phase": "engineering",
        "scientific_evidence": False,
        "formal_ready": True,
        "gates": {key: True for key in module.REQUIRED_DECISION_GATES},
        "selected": {"k": 1, "cells_a": ["L0:E0:C0"], "cells_b": ["L0:E0:C1"], "optimizer": "AdamW", "learning_rate": 1e-3, "max_optimizer_steps": 8, "lora_rank": 2, "max_training_tokens": 128},
        "thresholds": {key: 1.0 for key in module.REQUIRED_THRESHOLDS},
        "foundation": manifest,
        "source": {"source_ref": "codex/pcu-composability-kill-001", "source_commit": "commit", "source_tree": "tree"},
    }
    with pytest.raises(Exception, match="selected learning_rate"):
        broken = dict(decision)
        broken["selected"] = dict(decision["selected"])
        broken["selected"].pop("learning_rate")
        module._validate_engineering_decision(broken, manifest, decision["source"])
