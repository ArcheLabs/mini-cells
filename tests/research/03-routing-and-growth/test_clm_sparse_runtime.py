from __future__ import annotations

import copy

import torch

from minicells.clm_growth import ProgressiveGrowthCLM
from minicells.clm_sparse_runtime import install_optimized_runtime, runtime_status
from minicells.language_models import TextNCALM
from minicells.upcycled_cellular_textnca import UpcyclingConfig, convert_textnca_to_upcycled


def _model() -> ProgressiveGrowthCLM:
    torch.manual_seed(601)
    source = TextNCALM(
        vocab_size=31,
        max_context=16,
        dim=16,
        heads=4,
        ffn_dim=32,
        windows=(4, 8, 16),
        iterations=(1, 1, 1),
        carry_bias=2.0,
    )
    upcycled = convert_textnca_to_upcycled(
        source,
        config=UpcyclingConfig(num_experts=4, top_k=1),
    )
    return ProgressiveGrowthCLM(upcycled)


def test_autotuned_sparse_matches_dense_and_reference_sparse() -> None:
    model = _model().eval()
    inputs = torch.randint(0, 31, (4, 12))
    with torch.no_grad():
        baseline = model(inputs, execution_backend="masked_dense").logits
    install_optimized_runtime(model)
    with torch.no_grad():
        reference = model(inputs, execution_backend="reference_sparse").logits
        optimized = model(inputs, execution_backend="sparse_dispatch").logits
    torch.testing.assert_close(reference, baseline, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(optimized, baseline, rtol=2e-5, atol=2e-6)
    statuses = runtime_status(model)
    assert len(statuses) == 3
    allowed = {
        "autotuned_grouped_mm",
        "autotuned_reference_sparse",
        "autotuned_packed_batched_dense",
    }
    assert all(row["backend"] in allowed for row in statuses)
    assert all(row["fast_model_forward"] is True for row in statuses)
    assert all(int(row["autotune_shapes"]) == 1 for row in statuses)


def test_autotuned_sparse_rebuilds_after_birth() -> None:
    model = install_optimized_runtime(_model().eval())
    inputs = torch.randint(0, 31, (4, 12))
    with torch.no_grad():
        model(inputs, execution_backend="sparse_dispatch")
    assert all(row["packed_inference_cache"] for row in runtime_status(model))

    model.birth(
        stage=1,
        parent_id="s1-e0",
        routed_perceptions=torch.randn(512, 16),
        token=100,
    )
    with torch.no_grad():
        dense = model(inputs, execution_backend="masked_dense").logits
        sparse = model(inputs, execution_backend="sparse_dispatch").logits
    torch.testing.assert_close(sparse, dense, rtol=2e-5, atol=2e-6)
    statuses = runtime_status(model)
    assert statuses[1]["expert_count"] == 5
    assert int(statuses[1]["autotune_shapes"]) == 1


def test_batched_dense_preserves_forward_and_training_gradients() -> None:
    baseline = _model().train()
    optimized = copy.deepcopy(baseline).train()
    install_optimized_runtime(optimized)
    inputs = torch.randint(0, 31, (3, 12))

    baseline.zero_grad(set_to_none=True)
    optimized.zero_grad(set_to_none=True)
    baseline_logits = baseline(inputs, execution_backend="masked_dense").logits
    optimized_logits = optimized(inputs, execution_backend="batched_dense").logits
    torch.testing.assert_close(optimized_logits, baseline_logits, rtol=2e-5, atol=2e-6)

    weights = torch.linspace(0.1, 1.0, baseline_logits.numel()).view_as(baseline_logits)
    (baseline_logits * weights).sum().backward()
    (optimized_logits * weights).sum().backward()

    baseline_parameters = dict(baseline.named_parameters())
    optimized_parameters = dict(optimized.named_parameters())
    assert baseline_parameters.keys() == optimized_parameters.keys()
    for name in baseline_parameters:
        expected = baseline_parameters[name].grad
        actual = optimized_parameters[name].grad
        assert (expected is None) == (actual is None), name
        if expected is not None:
            torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6, msg=name)


def test_sparse_dispatch_with_grad_keeps_reference_semantics() -> None:
    model = install_optimized_runtime(_model().train())
    inputs = torch.randint(0, 31, (2, 12))
    logits = model(inputs, execution_backend="sparse_dispatch").logits
    logits.square().mean().backward()
    statuses = runtime_status(model)
    assert all(row["backend"] == "reference_sparse_grad" for row in statuses)
