from __future__ import annotations

import torch

from minicells.moe_multicoordinate import (
    capture_coordinate_set,
    restore_coordinate_set_,
)


def test_capture_and_restore_resolve_legacy_parent_target_without_runtime_names() -> None:
    gate_up_name = "model.layers.23.block_sparse_moe.input_linear.weight"
    down_name = "model.layers.23.block_sparse_moe.output_linear.weight"
    gate_up = torch.nn.Parameter(torch.arange(48, dtype=torch.float32).reshape(2, 8, 3))
    down = torch.nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))
    parameters = {
        gate_up_name: gate_up,
        down_name: down,
    }
    target = {
        "layer_index": 23,
        "expert_index": 1,
        "group_index": 1,
        "group_size": 2,
        "intermediate_size": 4,
        "channel_start": 2,
        "channel_end": 4,
    }

    original_gate_up = gate_up.detach().clone()
    original_down = down.detach().clone()
    captured = capture_coordinate_set(parameters, [target])

    with torch.no_grad():
        gate_up[1].add_(1000.0)
        down[1].sub_(1000.0)

    restore_coordinate_set_(parameters, [target], captured)

    intermediate = 4
    start, end = 2, 4
    assert torch.equal(
        gate_up[1, start:end],
        original_gate_up[1, start:end],
    )
    assert torch.equal(
        gate_up[1, intermediate + start : intermediate + end],
        original_gate_up[1, intermediate + start : intermediate + end],
    )
    assert torch.equal(
        down[1, :, start:end],
        original_down[1, :, start:end],
    )
