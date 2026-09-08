from __future__ import annotations

import pytest
import torch

from minicells import CellMutation, CellPlacement, HybridCLM, ModelInspection
from minicells.hybrid.errors import ArtifactValidationError, PlacementError, UnsupportedArchitectureError
from minicells.pcu_kill_001.backends import make_toy_model


def make_hybrid() -> HybridCLM:
    model = make_toy_model(seed=26090501)
    return HybridCLM(model, base_model="toy://pcu-kill-001", base_revision="toy-rev")


def test_inspection_and_explicit_placement() -> None:
    hybrid = make_hybrid()
    inspection = hybrid.inspect()
    assert isinstance(inspection, ModelInspection)
    assert inspection.architecture == "granitemoe"
    assert inspection.support_level == "SUPPORTED"
    assert HybridCLM.inspect(hybrid.model).architecture == "granitemoe"
    hybrid.cellularize(CellPlacement(layer=0, experts=[0, 1]))
    assert hybrid.placements[0].expert_ids == (0, 1)


def test_placement_rejects_unknown_layer() -> None:
    with pytest.raises(PlacementError):
        make_hybrid().cellularize(CellPlacement(layer=4, experts=[0]))


def test_unknown_architecture_fails_closed() -> None:
    class Unknown:
        config = type("Config", (), {"model_type": "unknown"})()

    with pytest.raises(UnsupportedArchitectureError):
        HybridCLM(Unknown())


def test_alpha_and_detach_restore_exactly() -> None:
    hybrid = make_hybrid().cellularize(CellPlacement(layer=0, experts=[0]))
    target = "model.layers.0.block_sparse_moe.experts.cells.0.cells.0.gate_weight"
    parameter = dict(hybrid.model.named_parameters())[target]
    mutation = CellMutation(
        {"delta": torch.ones_like(parameter)},
        targets=[{"key": "delta", "name": target}],
        base_model=hybrid.base_model,
        base_revision=hybrid.base_revision,
        architecture_hash=hybrid.inspect().architecture_signature,
    )
    before = parameter.detach().clone()
    hybrid.attach(mutation)
    hybrid.set_alpha(mutation, 0.0)
    assert torch.equal(parameter, before)
    hybrid.set_alpha(mutation, 1.0)
    assert torch.equal(parameter, before + 1)
    hybrid.detach(mutation)
    assert torch.equal(parameter, before)
    assert hybrid.verify_restoration().passed


def test_manifest_mismatch_is_hard_failure() -> None:
    hybrid = make_hybrid().cellularize(CellPlacement(layer=0, experts=[0]))
    target = "model.layers.0.block_sparse_moe.experts.cells.0.cells.0.gate_weight"
    mutation = CellMutation(
        {"delta": torch.zeros_like(dict(hybrid.model.named_parameters())[target])},
        targets=[{"key": "delta", "name": target}],
        base_model=hybrid.base_model,
        base_revision="wrong-revision",
    )
    with pytest.raises(ArtifactValidationError):
        hybrid.attach(mutation)
