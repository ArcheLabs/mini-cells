from __future__ import annotations

import torch

from minicells.language_data import make_training_schedule
from minicells.language_models import (
    TextNCALM,
    build_minitextnca_plus,
    build_parameter_matched_transformer,
    count_parameters,
)


def test_training_schedule_consumes_exact_budget() -> None:
    schedule = make_training_schedule(
        800_000,
        seed=1,
        budget_tokens=500_000,
        batch_size=8,
        sequence_length=125,
    )
    assert schedule.steps == 500
    assert schedule.tokens_per_step == 1000
    assert schedule.consumed_tokens == 500_000
    assert len(schedule.starts) == 500
    assert all(len(starts) == 8 for starts in schedule.starts)


def test_textnca_is_causal() -> None:
    torch.manual_seed(1)
    model = TextNCALM(
        vocab_size=32,
        max_context=8,
        dim=16,
        heads=4,
        ffn_dim=32,
        windows=(2, 4, 8),
        iterations=(1, 1, 1),
        rms_norm=True,
        carry_bias=2.0,
    ).eval()
    left = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    right = left.clone()
    right[0, -1] = 9
    with torch.no_grad():
        logits_left = model(left).logits
        logits_right = model(right).logits
    assert torch.allclose(logits_left[:, :-1], logits_right[:, :-1], atol=1e-6, rtol=1e-6)


def test_minitextnca_has_three_hierarchical_recurrent_stages() -> None:
    model = build_minitextnca_plus(vocab_size=128)
    assert len(model.stages) == 3
    assert [stage.attention.window for stage in model.stages] == [8, 32, 128]
    assert [stage.iterations for stage in model.stages] == [4, 4, 4]
    assert model.stage_supervision
    hidden = model.stages[0].gru.hidden_size
    update_bias = model.stages[0].gru.bias_ih[hidden : 2 * hidden]
    assert torch.all(update_bias > 0)


def test_transformer_is_parameter_matched_within_five_percent() -> None:
    target = count_parameters(build_minitextnca_plus(vocab_size=2048))
    transformer, config = build_parameter_matched_transformer(2048, target)
    actual = count_parameters(transformer)
    assert actual == config["parameters"]
    assert abs(actual - target) / target <= 0.05


def test_language_model_outputs_have_expected_shape() -> None:
    model = TextNCALM(
        vocab_size=64,
        max_context=16,
        dim=16,
        heads=4,
        ffn_dim=32,
        windows=(4, 8, 16),
        iterations=(1, 1, 1),
        rms_norm=True,
        carry_bias=2.0,
        stage_supervision=True,
    )
    output = model(torch.randint(0, 64, (2, 10)))
    assert output.logits.shape == (2, 10, 64)
    assert len(output.stage_logits) == 3
    assert all(stage.shape == (2, 10, 64) for stage in output.stage_logits)
