from __future__ import annotations

import torch

from minicells.language_2d import LatentTissueNCALM
from minicells.language_models import TextNCALM
from minicells.language_stabilization import (
    make_depth_schedule,
    scale_step_embeddings,
    stabilizing_forward,
)


def build_1d() -> TextNCALM:
    torch.manual_seed(11)
    return TextNCALM(
        vocab_size=64,
        max_context=16,
        dim=32,
        heads=4,
        ffn_dim=64,
        windows=(4, 8, 16),
        iterations=(4, 4, 4),
        carry_bias=2.0,
    )


def build_2d() -> LatentTissueNCALM:
    torch.manual_seed(12)
    return LatentTissueNCALM(
        vocab_size=64,
        tissue_height=3,
        max_context=16,
        dim=32,
        heads=4,
        ffn_dim=64,
        windows=(4, 8, 16),
        iterations=(4, 4, 4),
        carry_bias=2.0,
    )


def test_depth_schedule_is_deterministic_and_bounded() -> None:
    first = make_depth_schedule(100, seed=123)
    second = make_depth_schedule(100, seed=123)
    assert first == second
    assert len(first) == 100
    assert all(len(depths) == 3 for depths in first)
    assert all(2 <= depth <= 4 for depths in first for depth in depths)


def test_1d_full_depth_replays_standard_forward() -> None:
    model = build_1d().eval()
    inputs = torch.randint(0, 64, (2, 12))
    reference = model(inputs).logits
    candidate = stabilizing_forward(model, inputs, stage_depths=(4, 4, 4)).output.logits
    torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-6)


def test_2d_full_depth_replays_standard_forward() -> None:
    model = build_2d().eval()
    inputs = torch.randint(0, 64, (2, 12))
    reference = model(inputs).logits
    candidate = stabilizing_forward(model, inputs, stage_depths=(4, 4, 4)).output.logits
    torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-6)


def test_short_depth_has_trainable_stability_loss_1d() -> None:
    model = build_1d().train()
    inputs = torch.randint(0, 64, (2, 12))
    result = stabilizing_forward(model, inputs, stage_depths=(2, 3, 4))
    assert result.stage_depths == (2, 3, 4)
    assert len(result.stage_residual_rms) == 3
    assert torch.isfinite(result.stability_loss)
    result.stability_loss.backward()
    assert model.stages[0].gru.weight_hh.grad is not None


def test_short_depth_has_trainable_stability_loss_2d() -> None:
    model = build_2d().train()
    inputs = torch.randint(0, 64, (2, 12))
    result = stabilizing_forward(model, inputs, stage_depths=(3, 2, 4))
    assert len(result.stage_residual_rms) == 3
    assert torch.isfinite(result.stability_loss)
    result.stability_loss.backward()
    assert model.stages[1].vertical.conv.weight.grad is not None


def test_step_embedding_scaling_only_changes_step_embeddings() -> None:
    model = build_1d()
    before_step = model.stages[0].step_embedding.detach().clone()
    before_token = model.token_embedding.weight.detach().clone()
    scale_step_embeddings(model, 0.25)
    torch.testing.assert_close(model.stages[0].step_embedding, before_step * 0.25)
    torch.testing.assert_close(model.token_embedding.weight, before_token)
