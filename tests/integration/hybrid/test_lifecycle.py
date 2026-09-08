from __future__ import annotations

import torch
import pytest

from minicells import CellMutation, CellPlacement, HybridCLM
from minicells.pcu_kill_001.backends import make_toy_model


def test_granite_compatible_zero_state_and_roundtrip() -> None:
    hybrid = HybridCLM(
        make_toy_model(seed=26090501),
        base_model="toy://pcu-kill-001",
        base_revision="toy-rev",
    )
    hybrid.cellularize(CellPlacement(layer=0, experts="all"))
    inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)}
    report = hybrid.verify_zero_state(inputs)
    assert report.matched_graph_pass

    target = "model.layers.0.block_sparse_moe.experts.cells.0.cells.0.gate_weight"
    parameter = dict(hybrid.model.named_parameters())[target]
    mutation = CellMutation(
        {"delta": torch.full_like(parameter, 1e-4)},
        targets=[{"key": "delta", "name": target}],
        base_model=hybrid.base_model,
        base_revision=hybrid.base_revision,
        architecture_hash=hybrid.inspect().architecture_signature,
    )
    hybrid.attach(mutation)
    with hybrid.mutation_enabled(mutation, alpha=0.0):
        assert hybrid.verify_zero_state(inputs).matched_graph_pass
    hybrid.detach(mutation)
    assert hybrid.verify_restoration().passed


def test_serialization_roundtrip(tmp_path) -> None:
    pytest.importorskip("safetensors")
    hybrid = HybridCLM(make_toy_model(), base_model="toy://pcu-kill-001", base_revision="toy-rev")
    hybrid.cellularize(CellPlacement(layer=0, experts=[0]))
    target = "model.layers.0.block_sparse_moe.experts.cells.0.cells.0.gate_weight"
    parameter = dict(hybrid.model.named_parameters())[target]
    mutation = CellMutation(
        {"delta": torch.zeros_like(parameter)},
        targets=[{"key": "delta", "name": target}],
        base_model=hybrid.base_model,
        base_revision=hybrid.base_revision,
        architecture_hash=hybrid.inspect().architecture_signature,
    )
    path = mutation.save_pretrained(tmp_path / "mutation")
    loaded = CellMutation.from_pretrained(path)
    assert loaded.tensors["delta"].shape == parameter.shape
