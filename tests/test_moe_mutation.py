from __future__ import annotations

from pathlib import Path

import torch

from minicells.moe_mutation import (
    apply_expert_slice_mutation_,
    capture_expert_slices,
    load_expert_slice_mutation,
    restore_expert_slices_,
    save_expert_slice_mutation,
)


class _DummyPackedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.packed = torch.nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(3, 2, 4))
        self.unrelated = torch.nn.Parameter(torch.tensor([7.0]))


def test_expert_slice_mutation_round_trip(tmp_path: Path) -> None:
    model = _DummyPackedModel()
    parameters = dict(model.named_parameters())
    original = model.packed.detach().clone()
    captured = capture_expert_slices(parameters, ["packed"], 1)
    delta = torch.full_like(captured["packed"], 0.25)

    mutation_dir = tmp_path / "mutation"
    manifest = save_expert_slice_mutation(
        mutation_dir,
        base_manifest_identity="base-sha",
        source_model_id="local/dummy",
        source_revision="revision-sha",
        layer_index=0,
        expert_index=1,
        deltas={"packed": delta},
        canonical_tensor_names={
            "packed": "model.layers.0.block_sparse_moe.input_linear.weight"
        },
        metadata={"seed": 1},
    )

    assert manifest["target"]["expert_is_cell"] is False
    assert manifest["deltas"][0]["address"].endswith("::expert[1]")
    loaded_manifest, loaded = load_expert_slice_mutation(mutation_dir)
    assert loaded_manifest["identity_sha256"] == manifest["identity_sha256"]
    assert torch.equal(loaded["delta_00"], delta)

    apply_expert_slice_mutation_(model, mutation_dir)
    assert torch.equal(model.packed[0], original[0])
    assert torch.equal(model.packed[2], original[2])
    assert torch.equal(model.packed[1], original[1] + delta)
    assert model.unrelated.item() == 7.0

    restore_expert_slices_(parameters, captured, 1)
    assert torch.equal(model.packed, original)


def test_capture_is_detached_copy() -> None:
    model = _DummyPackedModel()
    parameters = dict(model.named_parameters())
    captured = capture_expert_slices(parameters, ["packed"], 2)
    with torch.no_grad():
        model.packed[2].zero_()
    assert not torch.equal(captured["packed"], model.packed[2])
