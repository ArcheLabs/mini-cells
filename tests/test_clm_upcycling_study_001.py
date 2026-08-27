from __future__ import annotations

import torch

from minicells.clm_upcycling_validation import (
    cosine_kmeans,
    make_upcycling_decision,
    static_templates,
)
from minicells.language_models import TextNCALM
from minicells.upcycled_cellular_textnca import UpcyclingConfig, convert_textnca_to_upcycled


def _models():
    torch.manual_seed(41)
    source = TextNCALM(
        vocab_size=37,
        max_context=12,
        dim=16,
        heads=4,
        ffn_dim=32,
        windows=(2, 4, 8),
        iterations=(2, 1, 1),
        carry_bias=2.0,
        stage_supervision=True,
    )
    upcycled = convert_textnca_to_upcycled(
        source, config=UpcyclingConfig(num_experts=4, top_k=1)
    )
    return source, upcycled


def test_dense_copy_upcycling_is_function_preserving_for_random_router() -> None:
    source, model = _models()
    inputs = torch.randint(0, 37, (2, 9))
    expected = source(inputs)
    actual = model(inputs)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-5, atol=1e-6)
    for left, right in zip(actual.stage_logits, expected.stage_logits):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)


def test_conversion_inherits_values_not_frozen_teacher_flags() -> None:
    source, _ = _models()
    source.requires_grad_(False)
    model = convert_textnca_to_upcycled(source)
    assert not any(parameter.requires_grad for parameter in source.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_geometry_router_initialization_preserves_function() -> None:
    source, model = _models()
    inputs = torch.randint(0, 37, (2, 9))
    prototypes = [torch.randn(4, 16) for _ in model.stages]
    model.set_router_prototypes(prototypes)
    torch.testing.assert_close(
        model(inputs).logits, source(inputs).logits, rtol=1e-5, atol=1e-6
    )


def test_copied_experts_start_equal_but_are_parameter_independent() -> None:
    _, model = _models()
    bank = model.stages[0].program_bank
    first = bank.experts[0][0].weight
    second = bank.experts[1][0].weight
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, second)
    with torch.no_grad():
        first.add_(1)
    assert not torch.equal(first, second)


def test_prototype_router_is_strictly_pointwise_local() -> None:
    _, model = _models()
    router = model.stages[0].program_bank.router
    perception = torch.randn(2, 7, 16)
    expected = router(perception)
    changed = perception.clone()
    changed[1, 3] += 10
    actual = router(changed)
    keep = torch.ones(2, 7, dtype=torch.bool)
    keep[1, 3] = False
    torch.testing.assert_close(actual[keep], expected[keep])


def test_masked_dense_and_sparse_dispatch_match_after_expert_divergence() -> None:
    _, model = _models()
    with torch.no_grad():
        model.stages[0].program_bank.experts[0][2].bias.add_(0.1)
    inputs = torch.randint(0, 37, (2, 9))
    model.set_execution_backend("masked_dense")
    dense = model(inputs).logits
    model.set_execution_backend("sparse_dispatch")
    sparse = model(inputs).logits
    torch.testing.assert_close(sparse, dense, rtol=5e-5, atol=1e-6)


def test_cosine_kmeans_is_deterministic_and_nonempty() -> None:
    torch.manual_seed(9)
    samples = torch.randn(128, 16)
    first, first_diag = cosine_kmeans(samples, 4, seed=17)
    second, second_diag = cosine_kmeans(samples, 4, seed=17)
    torch.testing.assert_close(first, second)
    assert first_diag == second_diag
    assert len(first_diag["occupancy"]) == 4
    assert abs(sum(first_diag["occupancy"]) - 1.0) < 1e-6


def test_static_templates_are_route_slot_specific_and_input_independent() -> None:
    first = torch.tensor([[[1.0, 0, 0, 0]], [[1.0, 0, 0, 0]]])
    second = torch.tensor([[[0.0, 1, 0, 0]], [[0.0, 1, 0, 0]]])
    templates = static_templates([[first, second], [first, second]])
    assert len(templates) == 2
    assert int(templates[0].argmax()) == 0
    assert int(templates[1].argmax()) == 1


def test_upcycling_decision_separates_quality_from_causal_routing() -> None:
    replicates = [
        {
            "replicate": r,
            "dense_nll": 2.0,
            "dense_ppl": 7.4,
            "random_parity": "CLM_UPCYCLING_EQUIVALENCE",
            "geometry_parity": "CLM_UPCYCLING_EQUIVALENCE",
        }
        for r in range(3)
    ]
    controls = []
    for r in range(3):
        for method in ("copy_random", "copy_geometry"):
            for arm, nll in (("dynamic", 2.0), ("static", 2.02), ("shuffled", 2.02)):
                controls.append({
                    "replicate": r,
                    "method": method,
                    "arm": arm,
                    "nll": nll,
                    "ppl": 7.4,
                    "sample_variation": 0.08,
                    "usage_entropy": 0.95,
                })
    decision = make_upcycling_decision(replicates, controls)
    assert decision["diagnosis"] == "CLM_UPCYCLING_CONDITIONALITY_SIGNAL"

    for row in controls:
        if row["arm"] in ("static", "shuffled"):
            row["nll"] = 2.001
    decision = make_upcycling_decision(replicates, controls)
    assert decision["diagnosis"] == "CLM_UPCYCLING_QUALITY_SIGNAL"
