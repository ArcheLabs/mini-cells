from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from minicells.language_30m import (
    CHECKPOINT_TOKENS,
    MINICELLS_DIM,
    MODEL_NAME,
    RESUME_INTERVAL_TOKENS,
    TARGET_TOKENS,
    TOKENS_PER_STEP,
    TRANSFORMER_NAME,
    build_minicells_30m,
    build_transformer_30m,
    make_30m_decision,
    memmap_batch,
)
from minicells.language_models import count_parameters


def test_30m_models_are_parameter_matched() -> None:
    minicells = build_minicells_30m(2048)
    transformer, match = build_transformer_30m(2048)
    assert count_parameters(minicells) == 29_602_800
    assert count_parameters(transformer) == 29_458_432
    assert float(match["relative_parameter_error"]) < 0.01


def test_30m_candidate_keeps_carry_biased_gru() -> None:
    model = build_minicells_30m(2048)
    assert model.max_context == 128
    assert model.stage_supervision is False
    assert len(model.stages) == 3
    assert [stage.attention.window for stage in model.stages] == [8, 32, 128]
    assert [stage.iterations for stage in model.stages] == [4, 4, 4]
    hidden = MINICELLS_DIM
    for stage in model.stages:
        update_ih = stage.gru.bias_ih[hidden : 2 * hidden]
        update_hh = stage.gru.bias_hh[hidden : 2 * hidden]
        assert torch.allclose(update_ih, torch.ones_like(update_ih))
        assert torch.allclose(update_hh, torch.ones_like(update_hh))


def test_30m_budget_and_resume_grid_are_exact() -> None:
    assert TARGET_TOKENS % TOKENS_PER_STEP == 0
    assert all(tokens % TOKENS_PER_STEP == 0 for tokens in CHECKPOINT_TOKENS)
    assert TARGET_TOKENS % RESUME_INTERVAL_TOKENS == 0
    assert CHECKPOINT_TOKENS[-1] == TARGET_TOKENS


def test_memmap_batch_preserves_next_token_targets(tmp_path: Path) -> None:
    path = tmp_path / "tokens.u16"
    values = np.memmap(path, dtype=np.uint16, mode="w+", shape=(100,))
    values[:] = np.arange(100, dtype=np.uint16)
    values.flush()
    del values
    stream = np.memmap(path, dtype=np.uint16, mode="r")
    inputs, targets = memmap_batch(stream, (3, 20), 5, torch.device("cpu"))
    assert inputs.tolist() == [[3, 4, 5, 6, 7], [20, 21, 22, 23, 24]]
    assert targets.tolist() == [[4, 5, 6, 7, 8], [21, 22, 23, 24, 25]]


def test_30m_green_decision_requires_competitive_endpoint_and_slope() -> None:
    summary = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "parameters": 29_602_800,
                "ppl_100m": 10.5,
                "nll_100m": 2.35,
                "learning_slope_alpha": 0.12,
            },
            {
                "model": TRANSFORMER_NAME,
                "parameters": 29_458_432,
                "ppl_100m": 10.0,
                "nll_100m": 2.30,
                "learning_slope_alpha": 0.125,
            },
        ]
    ).set_index("model", drop=False).reset_index(drop=True)
    ratios = pd.DataFrame(
        {
            "consumed_tokens": list(CHECKPOINT_TOKENS),
            "ppl_ratio": [1.06, 1.055, 1.05, 1.05, 1.05],
        }
    )
    decision = make_30m_decision(summary, ratios, source_006_ratio_10m=1.0256)
    assert decision["status"] == "GREEN"
    assert decision["diagnosis"] == "MINICELLS_30M_PARAMETER_SCALING_COMPETITIVE"
