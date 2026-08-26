from __future__ import annotations

import torch

from minicells.language_2d import LatentTissueNCALM
from minicells.language_halting import adaptive_forward
from minicells.language_models import TextNCALM


def build_1d() -> TextNCALM:
    torch.manual_seed(1010)
    return TextNCALM(
        vocab_size=64,
        max_context=8,
        dim=16,
        heads=4,
        ffn_dim=32,
        windows=(2, 4, 8),
        iterations=(4, 4, 4),
        rms_norm=False,
        carry_bias=2.0,
        tie_embeddings=True,
        stage_supervision=False,
    )


def build_2d() -> LatentTissueNCALM:
    torch.manual_seed(1010)
    return LatentTissueNCALM(
        vocab_size=64,
        tissue_height=4,
        max_context=8,
        dim=16,
        heads=4,
        ffn_dim=32,
        windows=(2, 4, 8),
        iterations=(4, 4, 4),
        carry_bias=2.0,
        tie_embeddings=True,
        stage_supervision=False,
    )


def test_adaptive_none_replays_exact_fixed_1d() -> None:
    model = build_1d().eval()
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    expected = model(inputs).logits
    actual = adaptive_forward(model, inputs, threshold=None).output.logits
    assert torch.allclose(expected, actual, atol=0.0, rtol=0.0)


def test_adaptive_none_replays_exact_fixed_2d() -> None:
    model = build_2d().eval()
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    expected = model(inputs).logits
    actual = adaptive_forward(model, inputs, threshold=None).output.logits
    assert torch.allclose(expected, actual, atol=0.0, rtol=0.0)


def test_large_threshold_can_stop_each_1d_stage_after_one_iteration() -> None:
    model = build_1d().eval()
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    result = adaptive_forward(model, inputs, threshold=1e9, min_iterations=1)
    assert result.stage_steps == (1, 1, 1)
    assert result.total_steps == 3
    assert all(len(values) == 1 for values in result.stage_residuals)


def test_large_threshold_can_stop_each_2d_stage_after_one_iteration() -> None:
    model = build_2d().eval()
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    result = adaptive_forward(model, inputs, threshold=1e9, min_iterations=1)
    assert result.stage_steps == (1, 1, 1)
    assert result.total_steps == 3
    assert all(len(values) == 1 for values in result.stage_residuals)


def test_zero_threshold_preserves_full_trained_depth() -> None:
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    for model in (build_1d().eval(), build_2d().eval()):
        result = adaptive_forward(model, inputs, threshold=0.0, min_iterations=1)
        assert result.stage_steps == (4, 4, 4)
        assert result.total_steps == 12


def test_min_iterations_is_respected() -> None:
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    for model in (build_1d().eval(), build_2d().eval()):
        result = adaptive_forward(model, inputs, threshold=1e9, min_iterations=2)
        assert result.stage_steps == (2, 2, 2)
        assert result.total_steps == 6
