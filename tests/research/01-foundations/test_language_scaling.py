from __future__ import annotations

import pandas as pd
import torch
from torch import nn

from minicells.language_models import count_parameters
from minicells.language_scaling import (
    SCALING_CHECKPOINTS,
    TRAIN_STREAM_TOKENS,
    build_scaling_models,
    make_scaling_decision,
    summarize_scaling,
)


def _synthetic_checkpoints(candidate_ppl: list[float], transformer_ppl: list[float]) -> pd.DataFrame:
    rows = []
    for model, values, parameters in (
        ("minicells-v2", candidate_ppl, 1_170_816),
        ("transformer-s", transformer_ppl, 1_183_936),
    ):
        for index, (tokens, ppl) in enumerate(zip(SCALING_CHECKPOINTS, values), start=1):
            rows.append(
                {
                    "model": model,
                    "step": index,
                    "consumed_tokens": tokens,
                    "validation_nll": float(torch.log(torch.tensor(ppl)).item()),
                    "validation_ppl": ppl,
                    "elapsed_seconds": float(index * 10),
                    "tokens_per_second": 20_000.0,
                    "parameters": parameters,
                    "peak_vram_bytes": 123,
                }
            )
    return pd.DataFrame(rows)


def test_scaling_budget_and_stream_are_large_enough() -> None:
    assert SCALING_CHECKPOINTS == (500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000)
    assert TRAIN_STREAM_TOKENS >= 12_000_000
    assert TRAIN_STREAM_TOKENS > SCALING_CHECKPOINTS[-1]


def test_006_candidate_is_exact_005b_winner_structure() -> None:
    candidate, transformer, config = build_scaling_models(2048)
    assert candidate.stage_supervision is False
    assert [stage.attention.window for stage in candidate.stages] == [8, 32, 128]
    assert [stage.iterations for stage in candidate.stages] == [4, 4, 4]
    assert all(isinstance(stage.norm_attention, nn.LayerNorm) for stage in candidate.stages)
    hidden = candidate.stages[0].gru.hidden_size
    update_bias = candidate.stages[0].gru.bias_ih[hidden : 2 * hidden]
    assert torch.all(update_bias > 0)
    target = count_parameters(candidate)
    actual = count_parameters(transformer)
    assert actual == config["parameters"]
    assert abs(actual - target) / target <= 0.05


def test_scaling_summary_and_green_decision() -> None:
    frame = _synthetic_checkpoints(
        [130.0, 100.0, 78.0, 58.0, 48.0],
        [113.0, 90.0, 72.0, 54.0, 44.0],
    )
    summary, ratios = summarize_scaling(frame)
    decision = make_scaling_decision(summary, ratios, source_005b_ppl=129.2)
    assert list(ratios["consumed_tokens"]) == list(SCALING_CHECKPOINTS)
    assert decision["status"] == "GREEN"
    assert decision["comparison"]["ppl_ratio_10m"] <= 1.15


def test_scaling_decision_rejects_widening_gap() -> None:
    frame = _synthetic_checkpoints(
        [125.0, 105.0, 95.0, 90.0, 88.0],
        [112.0, 82.0, 62.0, 43.0, 35.0],
    )
    summary, ratios = summarize_scaling(frame)
    decision = make_scaling_decision(summary, ratios, source_005b_ppl=129.2)
    assert decision["status"] == "RED"
    assert decision["comparison"]["ratio_change_500k_to_10m"] > 0.10
