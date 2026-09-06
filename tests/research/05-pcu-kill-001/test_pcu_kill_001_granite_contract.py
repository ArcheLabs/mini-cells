"""Regression tests for the real Granite MoE block input contract."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from minicells.pcu_kill_001.equivalence import verify_full_moe


class _StrictThreeDimensionalMoE(nn.Module):
    """Minimal stand-in for GraniteMoeSparseMoeBlock's 3-D public contract."""

    def forward(self, layer_input: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = layer_input.size()
        assert batch > 0 and length > 0 and hidden > 0
        return layer_input * 1.25


def test_full_moe_promotes_flat_token_probe_to_single_sequence() -> None:
    probe = torch.randn(17, 8)
    metrics = verify_full_moe(
        _StrictThreeDimensionalMoE(),
        _StrictThreeDimensionalMoE(),
        probe,
    )
    assert metrics.passed
    assert metrics.max_abs_error == 0.0


def test_full_moe_preserves_native_sequence_probe() -> None:
    probe = torch.randn(3, 7, 8)
    metrics = verify_full_moe(
        _StrictThreeDimensionalMoE(),
        _StrictThreeDimensionalMoE(),
        probe,
    )
    assert metrics.passed


def test_full_moe_rejects_non_sequence_shapes() -> None:
    with pytest.raises(ValueError, match="full MoE verification requires"):
        verify_full_moe(
            _StrictThreeDimensionalMoE(),
            _StrictThreeDimensionalMoE(),
            torch.randn(8),
        )
