from __future__ import annotations

import json
from pathlib import Path

import torch

from minicells.real_representation_008_postmortem import (
    _rank_approx,
    action_residual,
    factorized_dictionary_diagnostics,
    fro_residual,
    global_pca_diagnostics,
    per_write_svd,
)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "research" / "validations" / "core-008-postmortem-functional-capacity" / "protocol.json"


def _row(g: torch.Tensor, *, partition: str = "train") -> dict:
    dim = g.shape[0]
    return {
        "token_sha256": "x",
        "partition": partition,
        "g": g.to(torch.float64) / torch.linalg.norm(g),
        "z": torch.eye(dim, dtype=torch.float64),
    }


def test_rank_approx_is_exact_at_true_rank() -> None:
    a = torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float64)
    approx = _rank_approx(a, 1)
    assert fro_residual(a, approx) < 1e-12
    assert action_residual(torch.eye(2, dtype=torch.float64), a, approx) < 1e-12


def test_per_write_svd_residual_is_monotone() -> None:
    g = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64))
    rows = [_row(g), _row(g, partition="eval")]
    summary = per_write_svd(rows, [1, 2, 4])
    vals = [next(r for r in summary if r["partition"] == "eval" and r["rank"] == k)["median_frobenius_residual"] for k in (1, 2, 4)]
    assert vals[0] >= vals[1] >= vals[2]
    assert vals[2] < 1e-12


def test_global_pca_recovers_shared_one_dimensional_family() -> None:
    base = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    alt = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    rows = [
        _row(base),
        _row(alt),
        _row(base + alt),
        _row(base, partition="eval"),
    ]
    dense, sparse = global_pca_diagnostics(rows, [1, 2], [1, 2])
    eval2 = next(r for r in dense if r["partition"] == "eval" and r["dimension"] == 2)
    assert eval2["median_frobenius_residual"] < 1e-10
    eval_sparse = [r for r in sparse if r["partition"] == "eval"]
    by_k = {int(r["sparsity"]): float(r["median_frobenius_residual"]) for r in eval_sparse}
    assert by_k[2] <= by_k[1] + 1e-12


def test_factorized_dictionary_respects_rank_unit_budget() -> None:
    e0 = torch.zeros(4, 4, dtype=torch.float64); e0[0, 0] = 1
    e1 = torch.zeros(4, 4, dtype=torch.float64); e1[1, 1] = 1
    rows = [_row(e0), _row(e1), _row(e0 + e1), _row(e0, partition="eval")]
    out = factorized_dictionary_diagnostics(rows, [1, 2], total_rank_units=4, top_k=2, refinement_rounds=1)
    for row in out:
        assert int(row["total_rank_units"]) <= 4
        assert int(row["atom_count"]) <= 4


def test_protocol_is_diagnostic_and_preserves_core008() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["scientific_decision"] is False
    assert protocol["source_status_must_remain_unchanged"] is True
    assert protocol["data_identity"]["reuse_core008_formal_seeds"] == [80821, 80822, 80823]
    assert protocol["factor_budget"]["total_rank_units"] == 32
