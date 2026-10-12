from __future__ import annotations

import json
from pathlib import Path

import torch

import native_clm_runtime as base
from native_clm_phase_c import (
    PARAMETER_TOLERANCE,
    PHASE_C_NAMES,
    X1_NAME,
    X2_NAME,
    X3_NAME,
    X4_NAME,
    build_phase_c,
    estimate_flops,
    parameter_summary,
    phase_c_config,
)
from native_clm_visualize import write_phase_c_visualizations


def test_phase_c_parameters_are_matched() -> None:
    summary = parameter_summary()
    expected = {
        X1_NAME: 1_893_912,
        X2_NAME: 1_893_578,
        X3_NAME: 1_893_536,
        X4_NAME: 1_893_912,
    }
    for name, count in expected.items():
        assert summary[name]["parameters"] == count
        assert float(summary[name]["relative_error"]) < PARAMETER_TOLERANCE


def test_phase_c_forward_is_finite() -> None:
    ids = torch.randint(0, base.VOCAB_SIZE, (2, 16))
    for name in PHASE_C_NAMES:
        model = build_phase_c(name)
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (2, 16, base.VOCAB_SIZE)
        assert torch.isfinite(logits).all()


def test_phase_c_is_winner_cross_over() -> None:
    assert phase_c_config(X1_NAME)["factor"] == "N1+SwiGLU"
    assert phase_c_config(X2_NAME)["factor"] == "N1+RoPE"
    assert phase_c_config(X3_NAME)["factor"] == "N1+SwiGLU+RoPE"
    assert phase_c_config(X4_NAME)["factor"] == "N1+SwiGLU+RoPE+RMSNorm"
    for name in PHASE_C_NAMES:
        config = phase_c_config(name)
        assert config["parent_native"] == "N1-residual-only"
        assert config["update"] == "fixed-scaled-residual"
        assert estimate_flops(name, tokens=10_000_000)["train_flops_estimate"] > 0


def test_phase_c_visualizations_are_dependency_free(tmp_path: Path) -> None:
    ids = ["T1", "C1", "M4", "N1", "X1", "X2", "X3", "X4"]
    leaderboard = []
    curves = {}
    for index, model_id in enumerate(ids):
        leaderboard.append({
            "id": model_id,
            "validation_ppl_10m": 12.0 + index * 0.2,
            "train_flops_estimate": 1.0e14 + index * 2.0e13,
        })
        curves[model_id] = [
            {"consumed_tokens": 1_000_000, "validation_nll": 4.0 + index * 0.01},
            {"consumed_tokens": 2_500_000, "validation_nll": 3.2 + index * 0.01},
            {"consumed_tokens": 5_000_000, "validation_nll": 2.8 + index * 0.01},
            {"consumed_tokens": 7_500_000, "validation_nll": 2.6 + index * 0.01},
            {"consumed_tokens": 10_000_000, "validation_nll": 2.5 + index * 0.01},
        ]
    manifest = write_phase_c_visualizations(leaderboard, curves, tmp_path)
    for filename in manifest["files"].values():
        text = (tmp_path / filename).read_text(encoding="utf-8")
        assert text.startswith("<svg")
        assert "Native CLM" in text
    stored = json.loads((tmp_path / "phase-c-visualizations.json").read_text(encoding="utf-8"))
    assert stored["format"] == "minicells.native-clm-phase-c-visualizations.v1"
