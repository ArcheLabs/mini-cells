from __future__ import annotations

import json

import torch

from minicells.moe_subexpert import (
    MoeSubexpertError,
    apply_group_mutation_,
    capture_group,
    group_delta,
    load_group_mutation,
    restore_group_,
    save_group_mutation,
)


def _parameters() -> dict[str, torch.nn.Parameter]:
    torch.manual_seed(7)
    return {
        "gate_up": torch.nn.Parameter(torch.randn(2, 16, 5)),
        "down": torch.nn.Parameter(torch.randn(2, 5, 8)),
    }


def test_aligned_group_capture_restore_and_artifact(tmp_path):
    parameters = _parameters()
    original = capture_group(
        parameters,
        gate_up_name="gate_up",
        down_name="down",
        expert_index=1,
        group_index=2,
        group_size=2,
    )

    with torch.no_grad():
        parameters["gate_up"][1, 4:6].add_(0.25)
        parameters["gate_up"][1, 12:14].sub_(0.5)
        parameters["down"][1, :, 4:6].add_(0.125)

    current = capture_group(
        parameters,
        gate_up_name="gate_up",
        down_name="down",
        expert_index=1,
        group_index=2,
        group_size=2,
    )
    deltas = group_delta(current, original)
    manifest = save_group_mutation(
        tmp_path / "mutation",
        base_manifest_identity="abc123",
        source_model_id="test/model",
        source_revision="deadbeef",
        layer_index=3,
        expert_index=1,
        group_index=2,
        group_size=2,
        intermediate_size=8,
        gate_up_runtime_name="gate_up",
        down_runtime_name="down",
        gate_up_canonical_name="canonical.gate_up",
        down_canonical_name="canonical.down",
        deltas=deltas,
        metadata={"test": True},
    )
    assert manifest["target"]["expert_fraction"] == 0.25
    assert manifest["target"]["group_is_cell"] is False

    restore_group_(
        parameters,
        original,
        gate_up_name="gate_up",
        down_name="down",
        expert_index=1,
        group_index=2,
        group_size=2,
    )
    restored = capture_group(
        parameters,
        gate_up_name="gate_up",
        down_name="down",
        expert_index=1,
        group_index=2,
        group_size=2,
    )
    assert all(torch.equal(restored[key], original[key]) for key in original)

    loaded_manifest, tensors = load_group_mutation(tmp_path / "mutation")
    assert loaded_manifest["identity_sha256"] == manifest["identity_sha256"]
    assert set(tensors) == {"gate", "up", "down"}

    apply_group_mutation_(parameters, tmp_path / "mutation")
    reapplied = capture_group(
        parameters,
        gate_up_name="gate_up",
        down_name="down",
        expert_index=1,
        group_index=2,
        group_size=2,
    )
    assert all(torch.equal(reapplied[key], current[key]) for key in current)


def test_manifest_tamper_is_rejected(tmp_path):
    parameters = _parameters()
    original = capture_group(
        parameters,
        gate_up_name="gate_up",
        down_name="down",
        expert_index=0,
        group_index=0,
        group_size=2,
    )
    zero = {key: torch.zeros_like(value) for key, value in original.items()}
    root = tmp_path / "mutation"
    save_group_mutation(
        root,
        base_manifest_identity="abc123",
        source_model_id="test/model",
        source_revision="deadbeef",
        layer_index=0,
        expert_index=0,
        group_index=0,
        group_size=2,
        intermediate_size=8,
        gate_up_runtime_name="gate_up",
        down_runtime_name="down",
        gate_up_canonical_name="gate_up",
        down_canonical_name="down",
        deltas=zero,
    )
    path = root / "mutation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["target"]["group_index"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_group_mutation(root)
    except MoeSubexpertError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("tampered manifest was accepted")
