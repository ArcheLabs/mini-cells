from __future__ import annotations

import torch

from minicells.hybrid_clm import (
    HybridCellOverlay,
    HybridManifest,
    load_cell_artifact,
    save_cell_artifact,
)


def _overlay(seed: int = 17) -> HybridCellOverlay:
    return HybridCellOverlay(
        hidden_size=8,
        read_layer_index=1,
        write_layer_indices=(2, 3),
        max_cells=4,
        rank=2,
        gate_threshold=0.6,
        seed=seed,
    )


def test_cell_artifact_roundtrip_and_manifest_fork_merge(tmp_path) -> None:
    overlay = _overlay()
    slot_a = overlay.allocate_cell()
    with torch.no_grad():
        overlay.gate_bias[slot_a] = 9.0
        overlay.up[:, slot_a].fill_(0.125)
    overlay.freeze_address_(slot_a)
    artifact_a = overlay.export_artifact(slot_a, cell_id="cell-A")

    slot_b = overlay.allocate_cell()
    with torch.no_grad():
        overlay.gate_bias[slot_b] = 8.0
        overlay.up[:, slot_b].fill_(-0.25)
    overlay.freeze_address_(slot_b)
    artifact_b = overlay.export_artifact(slot_b, cell_id="cell-B")

    path_a = tmp_path / "cell-A.pt"
    path_b = tmp_path / "cell-B.pt"
    save_cell_artifact(path_a, artifact_a)
    save_cell_artifact(path_b, artifact_b)
    loaded_a = load_cell_artifact(path_a)
    loaded_b = load_cell_artifact(path_b)
    assert loaded_a.digest() == artifact_a.digest()
    assert loaded_b.digest() == artifact_b.digest()

    base = HybridManifest("foundation", "revision")
    branch_a = base.add(loaded_a)
    branch_b = base.add(loaded_b)
    merged = branch_a.merge(branch_b)
    assert dict(merged.cells) == {
        "cell-A": loaded_a.digest(),
        "cell-B": loaded_b.digest(),
    }
    assert merged.remove("cell-A") == branch_b

    restored = _overlay(seed=99)
    restored.apply_artifact_(loaded_a)
    restored.apply_artifact_(loaded_b)
    assert int(restored.committed_mask.sum().item()) == 2
    assert torch.equal(restored.up[:, 0].cpu(), loaded_a.state["up"])
    assert torch.equal(restored.up[:, 1].cpu(), loaded_b.state["up"])
