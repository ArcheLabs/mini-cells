from __future__ import annotations

import torch

from minicells import clm_sparse_runtime as runtime_v1
from minicells.clm_growth import ProgressiveGrowthCLM
from minicells.clm_sparse_runtime_v2 import (
    _padded_sparse,
    _parity_metrics,
    install_optimized_runtime,
    runtime_status,
)
from minicells.language_models import TextNCALM
from minicells.upcycled_cellular_textnca import UpcyclingConfig, convert_textnca_to_upcycled


def _model() -> ProgressiveGrowthCLM:
    torch.manual_seed(701)
    source = TextNCALM(
        vocab_size=37,
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


def test_padded_sparse_matches_reference_with_overflow() -> None:
    model = _model().eval()
    bank = model.stages[0].program_bank
    perception = torch.randn(32, 16)
    # Deliberately overload expert 0 so capacity_factor=1.0 exercises the exact
    # overflow path rather than only the balanced fast path.
    assignments = torch.tensor([0] * 20 + [1] * 5 + [2] * 4 + [3] * 3)
    gates = torch.nn.functional.one_hot(assignments, num_classes=4).to(perception.dtype)
    expert_ids = tuple(bank.expert_ids)
    with torch.no_grad():
        reference = runtime_v1._reference_sparse(bank, perception, gates, expert_ids)
        padded = _padded_sparse(
            bank,
            perception,
            gates,
            expert_ids,
            capacity_factor=1.0,
        )
    torch.testing.assert_close(padded, reference, rtol=2e-5, atol=2e-6)


def test_runtime_v2_matches_masked_before_and_after_birth() -> None:
    model = install_optimized_runtime(_model().eval())
    inputs = torch.randint(0, 37, (4, 12))
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
    assert runtime_status(model)[1]["expert_count"] == 5


def test_internal_parity_gate_rejects_runtime_drift() -> None:
    reference = torch.tensor([[1.0, -2.0], [0.5, 0.25]])
    exact = _parity_metrics(reference, reference.clone())
    bad = _parity_metrics(reference, reference + 0.1)
    assert exact["ok"] is True
    assert bad["ok"] is False
    assert float(bad["max_abs_diff"]) > 0.09


def test_runtime_status_reports_route_load_and_compute_fraction() -> None:
    model = install_optimized_runtime(_model().eval())
    inputs = torch.randint(0, 37, (4, 12))
    with torch.no_grad():
        model(inputs, execution_backend="sparse_dispatch")
    rows = runtime_status(model)
    assert len(rows) == 3
    for row in rows:
        counts = row["route_counts"]
        assert sum(counts) == 48
        assert len(counts) == int(row["expert_count"])
        fraction = float(row["expert_token_pair_fraction_vs_dense"])
        assert 0.0 < fraction <= 1.0
        assert int(row["dense_expert_token_pairs"]) == 48 * int(row["expert_count"])
        assert "autotune_v2" in row
