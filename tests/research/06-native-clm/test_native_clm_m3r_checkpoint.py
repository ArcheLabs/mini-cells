from __future__ import annotations

from pathlib import Path

import torch

from minicells.native_clm_m3 import GrowthWindow, NativeCLMM3GrowthConfig
from minicells.native_clm_m3r import LineageNativeCLM, maybe_spawn_lineage_from_pressure
from minicells.native_clm_v0 import NativeCLM, NativeCLMConfig


def _config() -> NativeCLMConfig:
    return NativeCLMConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        n_layers=2,
        n_heads=4,
        d_ff=32,
        initial_cells=2,
        active_cells=1,
        cellular_layer_index=0,
        certificate_max_rank=8,
    )


def test_m1_style_base_checkpoint_loads_into_lineage_model_without_function_drift(
    tmp_path: Path,
) -> None:
    torch.manual_seed(17)
    base = NativeCLM(_config())
    base.eval()
    tokens = torch.randint(0, 32, (2, 8))
    before = base(tokens, return_info=True)

    checkpoint = tmp_path / "m1-style.pt"
    base.save_checkpoint(checkpoint, extra={"milestone": "M1"})
    lineage, extra = LineageNativeCLM.load_checkpoint(checkpoint)
    lineage.eval()
    after = lineage(tokens, return_info=True)

    assert extra["milestone"] == "M1"
    assert lineage.cell_count == base.cell_count == 2
    assert lineage.lineage_root_count == 2
    assert torch.equal(after["logits"], before["logits"])
    assert torch.equal(after["cell_info"]["root_idx"], before["cell_info"]["top_idx"])
    assert torch.equal(after["cell_info"]["root_probs"], before["cell_info"]["top_probs"])


def test_lineage_checkpoint_round_trip_preserves_topology_and_function(tmp_path: Path) -> None:
    torch.manual_seed(23)
    model = LineageNativeCLM(_config(), lineage_root_count=2)
    optimizer = torch.optim.AdamW([cell.weight for cell in model.cellular.cells], lr=1e-3)
    tokens = torch.randint(0, 32, (2, 8))

    window = GrowthWindow(d_model=16, cell_count=2)
    window.steps = 10
    window.loss_sum = 20.0
    window.route_hits[0] = 1024
    window.ratio_weighted_sum[0] = 102.4
    window.query_sums[0] = torch.arange(1, 17, dtype=torch.float64)
    model.cellular.cells[0].certificate_rank.fill_(8)
    event = maybe_spawn_lineage_from_pressure(
        model,
        optimizer,
        window,
        NativeCLMM3GrowthConfig(max_new_cells=8, max_final_cells=10),
        global_step=100,
        last_growth_step=None,
        spawned_count=0,
        probe_tokens=tokens,
    )
    assert event is not None

    model.eval()
    before = model(tokens, return_info=True)
    checkpoint = tmp_path / "m3r-lineage.pt"
    model.save_checkpoint(checkpoint, extra={"milestone": "M3R"})
    restored, extra = LineageNativeCLM.load_checkpoint(checkpoint)
    restored.eval()
    after = restored(tokens, return_info=True)

    child_id = int(event["child_id"])
    assert extra["milestone"] == "M3R"
    assert restored.lineage_root_count == 2
    assert restored.cell_count == 3
    assert int(restored.cellular.cells[child_id].parent_id.item()) == 0
    assert restored._direct_children() == {0: child_id}
    assert torch.equal(after["cell_info"]["root_idx"], before["cell_info"]["root_idx"])
    assert torch.equal(after["cell_info"]["top_idx"], before["cell_info"]["top_idx"])
    assert torch.equal(after["logits"], before["logits"])
