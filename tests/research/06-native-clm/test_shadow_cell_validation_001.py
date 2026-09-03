from __future__ import annotations

import torch

from minicells.native_clm_v0 import NativeCLM, NativeCLMConfig
from minicells.shadow_cell_validation_001 import (
    EncodedSplit,
    ShadowValidationConfig,
    _primary_point,
    answer_logits,
    build_seed_world,
    classify_shadow_validation,
    generate_rule_world_split,
    pareto_hypervolume,
    train_matched_direct_and_shadow,
    validate_standard_forward_equivalence,
)


def _tiny_model() -> NativeCLM:
    return NativeCLM(
        NativeCLMConfig(
            vocab_size=256,
            max_seq_len=13,
            d_model=24,
            n_layers=2,
            n_heads=3,
            d_ff=48,
            dropout=0.0,
            initial_cells=2,
            active_cells=1,
            cellular_layer_index=1,
            certificate_max_rank=0,
            tie_embeddings=True,
        )
    )


def test_rule_world_is_unique_and_uses_overlapping_surface() -> None:
    used_a: set[tuple[int, int, int, int]] = set()
    used_b: set[tuple[int, int, int, int]] = set()
    a = generate_rule_world_split(
        name="A",
        domain="A",
        count=256,
        seed=100,
        used=used_a,
    )
    b = generate_rule_world_split(
        name="B",
        domain="B",
        count=256,
        seed=101,
        used=used_b,
    )
    assert a.tokens.shape == (256, 13)
    assert b.tokens.shape == (256, 13)
    assert len(used_a) == 256
    assert len(used_b) == 256
    assert all(bytes(row.tolist()).startswith(b"p") for row in a.tokens)
    assert all(bytes(row.tolist()).startswith(b"p") for row in b.tokens)
    for split in (a, b):
        for row, answer in zip(split.tokens[:32], split.answers[:32], strict=True):
            text = bytes(row.tolist()).decode("ascii")
            p = int(text[1:3])
            q = int(text[4:6])
            x = int(text[7:9])
            y = int(text[10:12])
            if split.domain == "A":
                assert p < q
                expected = x % 10
            else:
                assert p > q
                expected = y % 10
            assert int(answer) == ord(str(expected))


def test_seed_world_splits_are_disjoint_within_domain() -> None:
    counts = {
        "A_train": 200,
        "A_calibration": 100,
        "A_eval": 100,
        "B_train": 200,
        "B_calibration": 100,
        "B_eval": 100,
    }
    world = build_seed_world(95101, counts)
    for domain in ("A", "B"):
        names = [name for name in world if name.startswith(domain)]
        rows = [
            {bytes(row.tolist()) for row in world[name].tokens}
            for name in names
        ]
        for i, left in enumerate(rows):
            for right in rows[i + 1 :]:
                assert left.isdisjoint(right)


def test_answer_helper_matches_native_forward_and_birth_is_exact() -> None:
    torch.manual_seed(9)
    model = _tiny_model().eval()
    tokens = torch.randint(0, 255, (8, 13))
    drift = validate_standard_forward_equivalence(model, tokens)
    assert drift < 1e-6

    with torch.no_grad():
        _, _, route = answer_logits(model, tokens)
        parent = int(torch.mode(route).values.item())
        parent_weight = model.cellular.cells[parent].weight.detach().clone()
        base, _, _ = answer_logits(model, tokens)
        shadow, _, _ = answer_logits(
            model,
            tokens,
            parent_id=parent,
            candidate_weight=parent_weight,
            expression=0.73,
        )
    assert float((base - shadow).abs().max()) < 1e-6


def test_matched_direct_and_shadow_candidates_are_identical() -> None:
    torch.manual_seed(10)
    model = _tiny_model().eval()
    tokens = torch.randint(0, 255, (64, 13))
    answers = torch.randint(ord("0"), ord("9") + 1, (64,))
    with torch.no_grad():
        _, hidden, route = answer_logits(model, tokens)
    parent = int(torch.mode(route).values.item())
    encoded = EncodedSplit(
        name="B_train",
        domain="B",
        hidden=hidden.detach().cpu(),
        route_idx=route.detach().cpu(),
        answers=answers,
    )
    direct, shadow, summary = train_matched_direct_and_shadow(
        model=model,
        parent_id=parent,
        encoded_b_train=encoded,
        seed=95101,
        device=torch.device("cpu"),
        config=ShadowValidationConfig(
            base_steps=1,
            base_batch_size=8,
            adapt_steps=4,
            adapt_batch_size=16,
            gate_steps=1,
            gate_batch_size=8,
            precision="fp32",
        ),
    )
    assert torch.equal(direct, shadow)
    assert summary["operator_relative_error"] == 0.0


def test_pareto_hypervolume_rewards_left_upper_frontier() -> None:
    weak = [
        {"A_regression": 0.0, "B_gain_fraction_of_direct": 0.0},
        {"A_regression": 0.5, "B_gain_fraction_of_direct": 1.0},
    ]
    strong = [
        {"A_regression": 0.0, "B_gain_fraction_of_direct": 0.0},
        {"A_regression": 0.2, "B_gain_fraction_of_direct": 1.0},
    ]
    assert pareto_hypervolume(strong) > pareto_hypervolume(weak)
    primary = _primary_point(
        [
            {"maturity": 0.25, "A_regression": 0.05, "B_gain_fraction_of_direct": 0.8},
            {"maturity": 0.50, "A_regression": 0.09, "B_gain_fraction_of_direct": 0.95},
            {"maturity": 0.75, "A_regression": 0.20, "B_gain_fraction_of_direct": 1.0},
        ],
        maximum_a_regression=0.10,
    )
    assert primary is not None
    assert primary["maturity"] == 0.50


def _synthetic_seed_result(*, immediate: bool = False, shuffle_advantage: float = 0.2) -> dict:
    return {
        "gates": {
            "base_training": True,
            "parent_conflict": True,
            "direct_plasticity": True,
            "gate_capacity": True,
            "identity_control": True,
            "conditional_primary": True,
            "immediate_primary": immediate,
        },
        "hypervolume": {
            "conditional_improvement_vs_direct_interp": 0.3,
            "conditional_improvement_vs_shadow_global": 0.3,
        },
        "causal_control": {
            "correct_vs_shuffled_A_regression_advantage": shuffle_advantage,
        },
    }


def test_registered_classification_requires_conditional_maturation() -> None:
    thresholds = {
        "minimum_conditional_hypervolume_improvement_vs_direct_interp": 0.20,
        "minimum_conditional_hypervolume_improvement_vs_shadow_global": 0.20,
        "minimum_correct_vs_shuffled_A_regression_advantage": 0.10,
    }
    supported = [_synthetic_seed_result() for _ in range(3)]
    assert (
        classify_shadow_validation(supported, thresholds=thresholds)
        == "SHADOW_CELL_CONTROLLED_MATURATION_SUPPORTED"
    )
    immediate = [_synthetic_seed_result(immediate=True) for _ in range(3)]
    assert (
        classify_shadow_validation(immediate, thresholds=thresholds)
        == "SHADOW_ISOLATION_SUPPORTED_MATURATION_NOT_NECESSARY"
    )
    shuffled = [_synthetic_seed_result(shuffle_advantage=0.01) for _ in range(3)]
    assert (
        classify_shadow_validation(shuffled, thresholds=thresholds)
        == "SHADOW_ISOLATION_SUPPORTED_MATURATION_NOT_NECESSARY"
    )
