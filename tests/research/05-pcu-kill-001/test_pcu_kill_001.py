"""Low-cost PCU-KILL-001 infrastructure tests; no formal seed is executed."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from minicells.pcu_kill_001.backends import make_toy_model
from minicells.pcu_kill_001.cache import CachedTailRunner
from minicells.pcu_kill_001.cellular import CellPartition, GraniteArchitectureInspector
from minicells.pcu_kill_001.equivalence import verify_end_to_end, verify_expert_algebra
from minicells.pcu_kill_001.registry import fork_registry, make_foundation_registry, merge_registries, rollback_registry
from minicells.pcu_kill_001.synthetic import audit_dataset, generate_world
from minicells.pcu_kill_001.training import fork_expert, fork_initial_delta_norm, foundation_tensor_hashes, selected_delta_parameters


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
