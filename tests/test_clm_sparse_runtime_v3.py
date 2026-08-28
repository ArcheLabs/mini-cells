from __future__ import annotations

import torch

from minicells import clm_sparse_runtime as runtime_v1
from minicells.clm_growth import ProgressiveGrowthCLM
from minicells.clm_sparse_runtime_v3 import (
    _fixed_padded_sparse,
    _tiered_padded_sparse,
    install_optimized_runtime,
    make_tiered_capacity_plans,
    precision_aware_parity,
    runtime_status,
)
from minicells.language_models import TextNCALM
from minicells.upcycled_cellular_textnca import UpcyclingConfig, convert_textnca_to_upcycled


def _model(*, iterations: tuple[int, int, int] = (1, 1, 1)) -> ProgressiveGrowthCLM:
    torch.manual_seed(811)
    source = TextNCALM(
        vocab_size=41,
        max_context=16,
        dim=16,
        heads=4,
        ffn_dim=32,
        windows=(4, 8, 16),
        iterations=iterations,
        carry_bias=2.0,
    )
    upcycled = convert_textnca_to_upcycled(
        source,
        config=UpcyclingConfig(num_experts=4, top_k=1),
    )
    return ProgressiveGrowthCLM(upcycled)


def test_precision_aware_parity_accepts_fp16_ulp_but_not_fp32_drift() -> None:
    reference = torch.tensor([[1.0, -0.5], [0.25, 2.0]], dtype=torch.float32)
    one_fp16_ulp = reference + torch.tensor(0.0009765625)
    fp16 = precision_aware_parity(
        reference,
        one_fp16_ulp,
        compute_dtype=torch.float16,
    )
    fp32 = precision_aware_parity(
        reference,
        one_fp16_ulp,
        compute_dtype=torch.float32,
    )
    assert fp16["ok"] is True
    assert fp32["ok"] is False
    assert float(fp16["max_abs_diff"]) >= 0.0009


def test_tiered_capacity_reduces_padding_for_skewed_routes() -> None:
    counts = (97, 189, 686, 52)
    plans = make_tiered_capacity_plans(counts, token_count=1024)
    assert plans
    best = plans[0]
    assert len(best.groups) == 2
    assert best.padded_pairs < 4 * max(counts)
    assert best.padded_pairs < 4096
    assert sorted(index for group in best.groups for index in group) == [0, 1, 2, 3]


def test_fixed_and_tiered_sparse_match_reference_with_overflow() -> None:
    model = _model().eval()
    bank = model.stages[0].program_bank
    perception = torch.randn(32, 16)
    assignments = torch.tensor([0] * 20 + [1] * 5 + [2] * 4 + [3] * 3)
    gates = torch.nn.functional.one_hot(assignments, num_classes=4).to(perception.dtype)
    expert_ids = tuple(bank.expert_ids)
    packed = runtime_v1._pack(bank, expert_ids, torch.float32)
    plans = make_tiered_capacity_plans((20, 5, 4, 3), token_count=32, alignment=4)
    assert plans
    with torch.no_grad():
        reference = runtime_v1._reference_sparse(bank, perception, gates, expert_ids)
        fixed = _fixed_padded_sparse(
            bank,
            perception,
            gates,
            expert_ids,
            packed,
            capacity=8,
        )
        tiered = _tiered_padded_sparse(
            bank,
            perception,
            gates,
            expert_ids,
            packed,
            plans[0],
        )
    torch.testing.assert_close(fixed, reference, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(tiered, reference, rtol=2e-5, atol=2e-6)


def test_runtime_v3_calibrates_per_recurrent_step_then_reuses_hot_profiles() -> None:
    model = install_optimized_runtime(_model(iterations=(2, 2, 2)).eval())
    inputs = torch.randint(0, 41, (4, 12))
    with torch.no_grad():
        first = model(inputs, execution_backend="sparse_dispatch").logits
    first_status = runtime_status(model)
    with torch.no_grad():
        second = model(inputs, execution_backend="sparse_dispatch").logits
    second_status = runtime_status(model)
    torch.testing.assert_close(second, first, rtol=3e-3, atol=3e-5)
    for before, after in zip(first_status, second_status):
        assert int(before["calibrated_profile_count"]) == 2
        assert int(after["calibrated_profile_count"]) == 2
        assert int(after["hot_profile_count"]) == 2
        steps = [int(profile["step"]) for profile in after["calibration_profiles"]]
        assert steps == [0, 1]


def test_runtime_v3_matches_masked_before_and_after_birth() -> None:
    model = install_optimized_runtime(_model().eval())
    inputs = torch.randint(0, 41, (4, 12))
    with torch.no_grad():
        dense_before = model(inputs, execution_backend="masked_dense").logits
        sparse_before = model(inputs, execution_backend="sparse_dispatch").logits
    torch.testing.assert_close(sparse_before, dense_before, rtol=3e-3, atol=3e-5)

    model.birth(
        stage=1,
        parent_id="s1-e0",
        routed_perceptions=torch.randn(512, 16),
        token=100,
    )
    with torch.no_grad():
        dense_after = model(inputs, execution_backend="masked_dense").logits
        sparse_after = model(inputs, execution_backend="sparse_dispatch").logits
    torch.testing.assert_close(sparse_after, dense_after, rtol=3e-3, atol=3e-5)
    status = runtime_status(model)
    assert int(status[1]["expert_count"]) == 5
    assert int(status[1]["runtime_generation"]) == 3
