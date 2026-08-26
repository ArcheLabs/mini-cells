from __future__ import annotations

import torch

from research.minicells.language_2d import LatentTissueNCALM
from research.minicells.language_models import TextNCALM
from research.minicells.language_settling import relaxation_forward, settling_forward


def build_1d() -> TextNCALM:
    torch.manual_seed(12)
    return TextNCALM(
        vocab_size=64,
        max_context=16,
        dim=32,
        heads=4,
        ffn_dim=64,
        windows=(4, 8, 16),
        iterations=(4, 4, 4),
        rms_norm=False,
        carry_bias=2.0,
        stage_supervision=False,
    )


def build_2d() -> LatentTissueNCALM:
    torch.manual_seed(13)
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
        stage_supervision=False,
    )


def test_relaxation_can_run_beyond_original_step_embedding_length() -> None:
    ids = torch.randint(0, 64, (2, 12))
    for model in (build_1d(), build_2d()):
        result = relaxation_forward(model, ids, stage_depths=(6, 6, 6))
        assert result.output.logits.shape == (2, 12, 64)
        assert len(result.stage_last_residuals) == 3
        assert all(torch.isfinite(value) for value in result.stage_last_residuals)


def test_shared_rule_does_not_depend_on_absolute_step_embeddings() -> None:
    model = build_1d().eval()
    ids = torch.randint(0, 64, (1, 10))
    before = relaxation_forward(model, ids, stage_depths=(3, 3, 3)).output.logits
    with torch.no_grad():
        for stage in model.stages:
            stage.step_embedding.normal_(mean=100.0, std=5.0)
    after = relaxation_forward(model, ids, stage_depths=(3, 3, 3)).output.logits
    torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)


def test_settling_losses_are_finite_and_differentiable_1d() -> None:
    model = build_1d()
    ids = torch.randint(0, 64, (2, 10))
    result = settling_forward(model, ids, stage_depths=(2, 3, 4))
    assert len(result.stage_probe_residuals) == 3
    assert torch.isfinite(result.state_stability_loss)
    assert torch.isfinite(result.logit_consistency_loss)
    loss = result.output.logits.float().square().mean()
    loss = loss + result.state_stability_loss + result.logit_consistency_loss
    loss.backward()
    assert model.stages[0].gru.weight_hh.grad is not None
    assert torch.isfinite(model.stages[0].gru.weight_hh.grad).all()


def test_settling_losses_are_finite_and_differentiable_2d() -> None:
    model = build_2d()
    ids = torch.randint(0, 64, (2, 10))
    result = settling_forward(model, ids, stage_depths=(2, 3, 4))
    loss = result.output.logits.float().square().mean()
    loss = loss + result.state_stability_loss + result.logit_consistency_loss
    loss.backward()
    assert model.stages[0].vertical.conv.weight.grad is not None
    assert torch.isfinite(model.stages[0].vertical.conv.weight.grad).all()


def test_relaxation_remains_causal_for_1d_and_2d() -> None:
    prefix = torch.randint(0, 64, (1, 11))
    changed = prefix.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 64
    for model in (build_1d().eval(), build_2d().eval()):
        first = relaxation_forward(model, prefix, stage_depths=(5, 5, 5)).output.logits
        second = relaxation_forward(model, changed, stage_depths=(5, 5, 5)).output.logits
        torch.testing.assert_close(first[:, :-1], second[:, :-1], rtol=0.0, atol=1e-6)


def test_probe_output_is_exactly_one_more_final_stage_update() -> None:
    model = build_1d().eval()
    ids = torch.randint(0, 64, (1, 9))
    settled = settling_forward(model, ids, stage_depths=(2, 2, 2))
    deeper = relaxation_forward(model, ids, stage_depths=(2, 2, 3))
    torch.testing.assert_close(
        settled.probe_output.logits,
        deeper.output.logits,
        rtol=1e-5,
        atol=1e-6,
    )
