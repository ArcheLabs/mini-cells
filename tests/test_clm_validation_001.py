from __future__ import annotations

import torch

from minicells.language_clm_validation import (
    CONDITIONALITY_THRESHOLD,
    RoutingRecorder,
    dense_equivalence_passes,
    expand_static_mask,
    make_validation_decision,
    replay_routing,
    routing_variation,
    shuffled_masks,
    static_topk_masks,
)
from minicells.language_models import TextNCALM
from minicells.textnca_to_clm import convert_textnca_to_sparse_cellular


def _model():
    source = TextNCALM(vocab_size=31, max_context=12, dim=16, heads=4, ffn_dim=16,
                       windows=(2, 4, 8), iterations=(2, 1, 1))
    model = convert_textnca_to_sparse_cellular(source, num_programs=8)
    model.set_routing_mode("hard_program")
    model.set_program_top_k(4)
    return model


def test_static_mask_is_global_topk_with_matched_compute() -> None:
    first = [torch.tensor([[[1, 1, 1, 1, 0, 0, 0, 0]]], dtype=torch.float32)]
    second = [torch.tensor([[[1, 1, 0, 0, 1, 1, 0, 0]]], dtype=torch.float32)]
    mask = static_topk_masks([first, second], 4)
    assert int(mask.sum()) == 4
    expanded = expand_static_mask(_model(), torch.zeros(3, 5, dtype=torch.long), mask)
    assert len(expanded) == 4
    assert all(item.shape == (3, 5, 8) and torch.equal(item[0], item[2]) for item in expanded)


def test_real_gpu_conversion_drift_is_inside_preregistered_tolerance() -> None:
    assert dense_equivalence_passes(
        ppl_ratio=0.999999940395357,
        max_logits_abs_diff=1.430511474609375e-05,
        max_recurrent_state_abs_diff=2.086162567138672e-07,
    )
    assert not dense_equivalence_passes(
        ppl_ratio=1.0,
        max_logits_abs_diff=5.1e-05,
        max_recurrent_state_abs_diff=2.086162567138672e-07,
    )


def test_shuffling_preserves_usage_and_changes_assignment() -> None:
    masks = [torch.eye(4).repeat(1, 2).reshape(4, 1, 8)]
    shuffled = shuffled_masks(masks, torch.tensor([1, 0, 3, 2]))
    torch.testing.assert_close(shuffled[0].mean((0, 1)), masks[0].mean((0, 1)))
    assert not torch.equal(shuffled[0], masks[0])


def test_recorder_and_replay_use_the_same_number_of_local_steps() -> None:
    model = _model()
    inputs = torch.randint(0, 31, (2, 7))
    with RoutingRecorder(model) as recorder:
        expected = model(inputs).logits
    with replay_routing(model, recorder.masks):
        actual = model(inputs).logits
    torch.testing.assert_close(actual, expected)
    assert len(recorder.masks) == 4


def test_routing_variation_distinguishes_static_from_conditional_masks() -> None:
    static = [torch.ones(3, 2, 8) for _ in range(3)]
    assert routing_variation(static) == (0.0, 0.0)
    conditional = [item.clone() for item in static]
    conditional[1][1] = 0
    structural, temporal = routing_variation(conditional)
    assert structural > 0
    assert temporal > 0


def _rows(*, conditional: bool) -> list[dict[str, object]]:
    rows = []
    for replicate in range(3):
        for arm, nll, ppl in (
            ("dense", 2.0, 7.39), ("dynamic", 2.01, 7.46),
            ("static", 2.02 if conditional else 2.011, 7.54),
            ("shuffled", 2.02 if conditional else 2.011, 7.54),
        ):
            rows.append({"replicate": replicate, "top_k": 4, "arm": arm,
                         "validation_nll": nll, "validation_ppl": ppl,
                         "executor_ratio": 0.5,
                         "structural_variation": CONDITIONALITY_THRESHOLD if conditional else 0.0})
    return rows


def test_decision_requires_quality_variation_and_both_causal_controls() -> None:
    assert make_validation_decision(_rows(conditional=True))["diagnosis"] == (
        "CLM_PROGRAM_CONDITIONALITY_SIGNAL"
    )
    assert make_validation_decision(_rows(conditional=False))["diagnosis"] == (
        "CLM_PROGRAM_SPARSITY_ONLY"
    )
