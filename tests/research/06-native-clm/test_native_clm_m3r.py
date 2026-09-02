from __future__ import annotations

import torch

from minicells.native_clm_m3 import GrowthWindow, NativeCLMM3GrowthConfig
from minicells.native_clm_m3r import (
    LineageNativeCLM,
    compare_m3r_arms,
    maybe_spawn_lineage_from_pressure,
)
from minicells.native_clm_v0 import NativeCLMConfig


def _tiny_model() -> LineageNativeCLM:
    return LineageNativeCLM(
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
        ),
        lineage_root_count=2,
    )


def test_lineage_spawn_is_function_preserving_and_cannot_change_root_slot() -> None:
    torch.manual_seed(7)
    model = _tiny_model()
    model.eval()
    tokens = torch.randint(0, 32, (2, 8))
    before = model(tokens, return_info=True)
    root_idx_before = before["cell_info"]["root_idx"].clone()
    root_probs_before = before["cell_info"]["root_probs"].clone()
    logits_before = before["logits"].clone()

    optimizer = torch.optim.AdamW([cell.weight for cell in model.cellular.cells], lr=1e-3)
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
    assert model.cell_count == 3
    after = model(tokens, return_info=True)
    assert torch.equal(after["cell_info"]["root_idx"], root_idx_before)
    assert torch.equal(after["cell_info"]["root_probs"], root_probs_before)
    assert torch.allclose(after["logits"], logits_before, atol=1e-6, rtol=0.0)
    assert event["birth_root_topk_match"] == 1.0
    assert event["birth_root_prob_max_abs_drift"] == 0.0
    assert event["birth_logits_max_abs_drift"] <= 1e-6


def test_lineage_chain_only_splits_leaves() -> None:
    model = _tiny_model()
    optimizer = torch.optim.AdamW([cell.weight for cell in model.cellular.cells], lr=1e-3)
    tokens = torch.randint(0, 32, (2, 8))

    first = GrowthWindow(d_model=16, cell_count=2)
    first.steps = 10
    first.loss_sum = 20.0
    first.route_hits[0] = 1024
    first.ratio_weighted_sum[0] = 102.4
    first.query_sums[0] = torch.arange(1, 17, dtype=torch.float64)
    model.cellular.cells[0].certificate_rank.fill_(8)
    event = maybe_spawn_lineage_from_pressure(
        model,
        optimizer,
        first,
        NativeCLMM3GrowthConfig(max_new_cells=8, max_final_cells=10),
        global_step=100,
        last_growth_step=None,
        spawned_count=0,
        probe_tokens=tokens,
    )
    assert event is not None
    child_id = event["child_id"]

    second = GrowthWindow(d_model=16, cell_count=3)
    second.steps = 10
    second.loss_sum = 20.0
    second.route_hits[0] = 4096
    second.ratio_weighted_sum[0] = 0.0
    second.query_sums[0] = torch.ones(16, dtype=torch.float64) * 4096
    second.route_hits[child_id] = 1024
    second.ratio_weighted_sum[child_id] = 102.4
    second.query_sums[child_id] = torch.arange(1, 17, dtype=torch.float64)
    model.cellular.cells[child_id].certificate_rank.fill_(8)
    event2 = maybe_spawn_lineage_from_pressure(
        model,
        optimizer,
        second,
        NativeCLMM3GrowthConfig(max_new_cells=8, max_final_cells=10),
        global_step=200,
        last_growth_step=100,
        spawned_count=1,
        probe_tokens=tokens,
    )
    assert event2 is not None
    assert event2["parent_id"] == child_id


def _summary(*, arm: str, a_loss: float, cells: int, child_a: float, child_new: float) -> dict:
    def metrics(loss: float, usage: list[float]) -> dict:
        return {"loss": loss, "active_fraction_vs_dense": 2 / cells, "cell_usage_share": usage}

    roots = [max(0.0, (1.0 - child_a) / 8)] * 8
    child_count = max(0, cells - 8)
    child_usage_a = [child_a / max(1, child_count)] * child_count
    roots_new = [max(0.0, (1.0 - child_new) / 8)] * 8
    child_usage_new = [child_new / max(1, child_count)] * child_count
    initial_usage = [0.125] * 8
    initial = {
        "A": metrics(1.0, initial_usage),
        "B": metrics(2.0, initial_usage),
        "C": metrics(3.0, initial_usage),
        "D": metrics(2.5, initial_usage),
    }
    return {
        "arm": arm,
        "seed": 1,
        "parent_checkpoint_sha256": "abc",
        "learner_replay_bytes": 0,
        "shared_and_original_router_frozen": True,
        "growth_controller_uses_phase_or_eval_labels": False,
        "growth_config": {"same": True},
        "spawned_cells": child_count,
        "final_cell_count": cells,
        "child_post_birth_route_hits": {str(8 + idx): 4096 for idx in range(child_count)},
        "growth_events": [
            {
                "birth_logits_max_abs_drift": 0.0,
                "birth_logits_mse": 0.0,
                "birth_root_topk_match": 1.0,
                "birth_root_prob_max_abs_drift": 0.0,
            }
            for _ in range(child_count)
        ],
        "lineage_chain_valid": True,
        "root_route_probes": {
            stage: {domain: f"stable-{domain}" for domain in "ABCD"}
            for stage in ("initial", "after_B", "after_C", "after_D")
        },
        "evaluation_matrix": {
            "initial": initial,
            "after_B": {
                **initial,
                "B": metrics(1.2, roots_new + child_usage_new),
            },
            "after_C": {
                **initial,
                "B": metrics(1.2, roots_new + child_usage_new),
                "C": metrics(1.5, roots_new + child_usage_new),
            },
            "after_D": {
                "A": metrics(a_loss, roots + child_usage_a),
                "B": metrics(1.25, roots_new + child_usage_new),
                "C": metrics(1.55, roots_new + child_usage_new),
                "D": metrics(1.8, roots_new + child_usage_new),
            },
        },
    }


def test_registered_m3r_comparison_can_detect_read_preserving_restoration() -> None:
    global_control = _summary(
        arm="global_growth_control", a_loss=1.50, cells=12, child_a=0.50, child_new=0.55
    )
    lineage = _summary(
        arm="lineage_growth", a_loss=1.10, cells=12, child_a=0.15, child_new=0.40
    )
    thresholds = {
        "minimum_phase_gain_each_B_C_D": 0.05,
        "maximum_lineage_A_regression": 0.20,
        "minimum_global_A_regression": 0.30,
        "minimum_A_retention_advantage_vs_global": 0.10,
        "maximum_lineage_mean_forgetting": 0.15,
        "minimum_lineage_to_global_plasticity_ratio": 0.80,
        "maximum_active_fraction_vs_dense": 0.30,
        "minimum_spawned_cells": 1,
        "maximum_spawned_cells": 8,
        "minimum_child_reuse_fraction": 0.75,
        "minimum_child_post_birth_route_hits": 512,
        "maximum_birth_logits_max_abs_drift": 1e-5,
        "maximum_birth_logits_mse": 1e-10,
        "maximum_birth_root_prob_drift": 1e-7,
        "minimum_A_child_share_reduction_vs_global": 0.10,
        "minimum_child_selectivity_margin": 0.10,
    }
    result = compare_m3r_arms(global_control, lineage, thresholds=thresholds)
    assert result["A_retention_advantage"] >= 0.10
    assert result["A_child_share_reduction_vs_global"] >= 0.10
    assert result["child_selectivity_margin"] >= 0.10
    assert result["pass"] is True
