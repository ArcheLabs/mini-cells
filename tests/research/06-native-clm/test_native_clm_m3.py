from __future__ import annotations

import torch

from minicells.native_clm_m3 import (
    GrowthWindow,
    NativeCLMM3GrowthConfig,
    compare_arms,
    maybe_spawn_from_pressure,
)
from minicells.native_clm_v0 import NativeCLM, NativeCLMConfig


def _tiny_model() -> NativeCLM:
    return NativeCLM(
        NativeCLMConfig(
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
    )


def test_pressure_spawn_clones_parent_and_registers_optimizer_parameter() -> None:
    model = _tiny_model()
    optimizer = torch.optim.AdamW([cell.weight for cell in model.cellular.cells], lr=1e-3)
    window = GrowthWindow(d_model=16, cell_count=2)
    window.steps = 10
    window.loss_sum = 20.0
    window.route_hits[0] = 1024
    window.ratio_weighted_sum[0] = 102.4
    window.query_sums[0] = torch.arange(1, 17, dtype=torch.float64)
    model.cellular.cells[0].certificate_rank.fill_(8)
    parent_weight = model.cellular.cells[0].weight.detach().clone()

    event = maybe_spawn_from_pressure(
        model,
        optimizer,
        window,
        NativeCLMM3GrowthConfig(),
        global_step=100,
        last_growth_step=None,
        spawned_count=0,
    )

    assert event is not None
    assert event["parent_id"] == 0
    assert event["child_id"] == 2
    assert model.cell_count == 3
    child = model.cellular.cells[2]
    assert torch.equal(child.weight, parent_weight)
    assert child.rank == 0
    assert child.route_key.requires_grad is False
    assert any(child.weight is parameter for group in optimizer.param_groups for parameter in group["params"])


def _summary(*, arm: str, a_final_ratio: float, cells: int, spawned: int) -> dict:
    def metrics(loss: float, fraction: float = 0.25) -> dict:
        return {"loss": loss, "active_fraction_vs_dense": fraction}

    return {
        "arm": arm,
        "seed": 1,
        "parent_checkpoint_sha256": "abc",
        "learner_replay_bytes": 0,
        "shared_and_original_router_frozen": True,
        "growth_controller_uses_phase_or_eval_labels": False,
        "final_cell_count": cells,
        "spawned_cells": spawned,
        "child_post_birth_route_hits": {str(8 + idx): 4096 for idx in range(spawned)},
        "evaluation_matrix": {
            "initial": {"A": metrics(1.0), "B": metrics(2.0), "C": metrics(3.0), "D": metrics(2.5)},
            "after_B": {"A": metrics(1.0), "B": metrics(1.2), "C": metrics(3.0), "D": metrics(2.5)},
            "after_C": {"A": metrics(1.0), "B": metrics(1.2), "C": metrics(1.5), "D": metrics(2.5)},
            "after_D": {
                "A": metrics(a_final_ratio, 2 / cells),
                "B": metrics(1.25, 2 / cells),
                "C": metrics(1.55, 2 / cells),
                "D": metrics(1.8, 2 / cells),
            },
        },
    }


def test_registered_m3_comparison_can_distinguish_growth_restoration() -> None:
    fixed = _summary(arm="fixed_protected", a_final_ratio=1.45, cells=8, spawned=0)
    growth = _summary(arm="growth_protected", a_final_ratio=1.10, cells=12, spawned=4)
    thresholds = {
        "minimum_phase_gain_each_B_C_D": 0.05,
        "maximum_growth_A_regression": 0.20,
        "minimum_fixed_A_regression_to_expose_capacity_limit": 0.30,
        "minimum_A_retention_advantage_vs_fixed": 0.10,
        "maximum_growth_mean_forgetting": 0.15,
        "minimum_growth_to_fixed_plasticity_ratio": 0.80,
        "maximum_active_fraction_vs_dense": 0.30,
        "minimum_spawned_cells": 1,
        "maximum_spawned_cells": 8,
        "minimum_child_reuse_fraction": 0.75,
        "minimum_child_post_birth_route_hits": 512,
    }
    result = compare_arms(fixed, growth, thresholds=thresholds)
    assert result["growth_A_regression"] < result["fixed_A_regression"]
    assert result["child_reuse_fraction"] == 1.0
    assert result["pass"] is True
