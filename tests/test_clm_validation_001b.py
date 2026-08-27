from __future__ import annotations

import torch
from torch.nn import functional as F

from minicells.language_clm_validation import (
    configure_hard_program_stage,
    make_validation_001b_decision,
    minimum_quality_safe_k,
    quality_gated_progression,
    reset_program_routing_logits,
    routing_variation_metrics,
    shuffled_masks,
    static_topk_masks,
)
from minicells.language_models import TextNCALM
from minicells.textnca_to_clm import convert_textnca_to_sparse_cellular


def _model():
    torch.manual_seed(11)
    source = TextNCALM(
        vocab_size=37, max_context=12, dim=16, heads=4, ffn_dim=16,
        windows=(2, 4, 8), iterations=(2, 1, 1),
    )
    return source, convert_textnca_to_sparse_cellular(source, num_programs=8)


def test_reset_program_logits_top8_is_dense_preserving() -> None:
    source, model = _model()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()
              if isinstance(value, torch.Tensor)}
    reset_program_routing_logits(model, seed=72001)
    after_reset = model.state_dict()
    for name, value in before.items():
        if not name.endswith("receptor.out_proj.bias"):
            torch.testing.assert_close(after_reset[name], value)
    model.set_routing_mode("hard_program")
    model.set_program_top_k(8)
    inputs = torch.randint(0, 37, (2, 9))
    torch.testing.assert_close(model(inputs).logits, source(inputs).logits, rtol=5e-5, atol=1e-6)
    for stage in model.stages:
        assert float(stage.receptor.out_proj.bias[0]) == 8.0
        assert abs(float(stage.receptor.out_proj.bias[1:].mean())) < 1e-7


def test_top8_dense_forward_gives_program_router_gradient() -> None:
    _, model = _model()
    reset_program_routing_logits(model, seed=72001)
    model.set_routing_mode("hard_program")
    model.set_program_top_k(8)
    inputs = torch.randint(0, 37, (2, 9))
    targets = torch.randint(0, 37, (2, 9))
    loss = F.cross_entropy(model(inputs).logits.flatten(0, 1), targets.flatten())
    loss.backward()
    gradient = torch.cat(
        [stage.receptor.out_proj.bias.grad[1:].reshape(-1) for stage in model.stages]
    )
    assert float(gradient.norm()) > 0


def test_stage_configuration_preserves_optimizer_identity_and_state() -> None:
    _, model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    returned = configure_hard_program_stage(model, optimizer, top_k=7)
    assert returned is optimizer
    assert model.routing_config.program_top_k == 7
    returned = configure_hard_program_stage(model, optimizer, top_k=6)
    assert returned is optimizer
    assert model.routing_config.program_top_k == 6


def test_variation_axes_are_explicit_and_identical_routing_is_zero() -> None:
    identical = [torch.ones(3, 4, 8) for _ in range(2)]
    assert routing_variation_metrics(identical) == {
        "sample": 0.0, "position": 0.0, "temporal": 0.0,
    }
    dynamic = [mask.clone() for mask in identical]
    dynamic[0][1, :, :4] = 0
    metrics = routing_variation_metrics(dynamic)
    assert metrics["sample"] > 0
    assert metrics["temporal"] > 0


def test_static_and_shuffled_controls_preserve_program_count() -> None:
    masks = [[torch.tensor([[[1, 1, 0, 0, 1, 0, 1, 0]],
                            [[0, 1, 1, 0, 1, 0, 0, 1]]], dtype=torch.float32)]]
    static = static_topk_masks(masks, 4)
    assert int(static.sum()) == 4
    original = masks[0]
    shuffled = shuffled_masks(original, torch.tensor([1, 0]))
    assert int(shuffled[0].sum()) == int(original[0].sum())
    torch.testing.assert_close(shuffled[0].mean((0, 1)), original[0].mean((0, 1)))
    assert not torch.equal(shuffled[0], original[0])


def test_quality_safe_k_stops_at_first_failure() -> None:
    progression = [(8, 1.00), (7, 1.01), (6, 1.02), (5, 1.08), (4, 1.01)]
    assert minimum_quality_safe_k(progression) == 6
    assert quality_gated_progression(progression) == [8, 7, 6, 5]


def _decision_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    arms = []
    progression = []
    for replicate in range(3):
        progression.append({"replicate": replicate, "quality_safe_k": 6, "selected": True})
        for arm, nll, ppl in (
            ("dense", 2.0, 7.39), ("dynamic", 2.01, 7.46),
            ("static", 2.02, 7.54), ("shuffled", 2.02, 7.54),
        ):
            arms.append({
                "replicate": replicate, "arm": arm, "validation_nll": nll,
                "validation_ppl": ppl, "structural_variation": 0.06,
                "temporal_variation": 0.1, "receptor_ratio": 0.021,
                "executor_ratio": 0.99,
            })
    return arms, progression


def test_program_only_success_does_not_depend_on_whole_executor_ratio() -> None:
    arms, progression = _decision_rows()
    decision = make_validation_001b_decision(arms, progression)
    assert decision["diagnosis"] == "CLM_PROGRAM_CONDITIONALITY_SIGNAL"
