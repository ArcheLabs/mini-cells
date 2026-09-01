from __future__ import annotations

import json
from pathlib import Path

import torch

from minicells.real_representation_009d_experiment import (
    FactorRow,
    OperatorRow,
    _project_dense,
    _rank1_core,
    _ridge_fit,
    _ridge_predict,
    _select_lambda,
    _sparse_core,
    rotated_factor_bases,
    select_discovery_lock,
)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "research" / "validations" / "core-009d-compositional-operator-geometry" / "protocol.json"


def _orthobasis(rows: int, cols: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(rows, rows, generator=gen, dtype=torch.float64))
    return q[:, :cols].contiguous()


def _row(g: torch.Tensor, token: str = "x") -> OperatorRow:
    return OperatorRow("train", "synthetic", token, torch.randn(5, 64, dtype=torch.float64), g)


def test_rotated_null_preserves_dense_factor_subspace_projection() -> None:
    left = _orthobasis(64, 56, 1)
    right = _orthobasis(64, 8, 2)
    g = torch.randn(64, 64, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    g /= torch.linalg.norm(g)
    row = _row(g)
    rotated_left, rotated_right = rotated_factor_bases(left, right, 56, 8, seed=81300)
    original = _project_dense(row, left, right, 56, 8)
    rotated = _project_dense(row, rotated_left, rotated_right, 56, 8)
    assert torch.allclose(original, rotated, atol=1e-10, rtol=1e-10)


def test_rank1_core_is_lossless_for_rank1_operator_inside_factor_subspace() -> None:
    left = _orthobasis(64, 56, 4)
    right = _orthobasis(64, 8, 5)
    gen = torch.Generator().manual_seed(6)
    alpha = torch.randn(56, generator=gen, dtype=torch.float64)
    beta = torch.randn(8, generator=gen, dtype=torch.float64)
    g = torch.outer(left @ alpha, right @ beta)
    g /= torch.linalg.norm(g)
    row = _row(g)
    dense = _project_dense(row, left, right, 56, 8)
    compressed = _rank1_core(row, left, right, 56, 8)
    assert torch.allclose(dense, g, atol=1e-10, rtol=1e-10)
    assert torch.allclose(compressed, dense, atol=1e-10, rtol=1e-10)


def test_sparse_tensor_core_recovers_exact_sparse_coordinate_operator() -> None:
    left = _orthobasis(64, 56, 7)
    right = _orthobasis(64, 8, 8)
    core = torch.zeros(56, 8, dtype=torch.float64)
    core[2, 1] = 0.7
    core[9, 5] = -0.4
    core[41, 3] = 0.2
    g = left @ core @ right.T
    g /= torch.linalg.norm(g)
    recovered = _sparse_core(_row(g), left, right, 56, 8, 3)
    assert torch.allclose(recovered, g, atol=1e-10, rtol=1e-10)


def test_train_only_ridge_recovers_synthetic_right_to_left_coupling() -> None:
    gen = torch.Generator().manual_seed(9)
    x = torch.randn(64, 8, generator=gen, dtype=torch.float64)
    w = torch.randn(8, 56, generator=gen, dtype=torch.float64) / 4.0
    y = x @ w
    rows = [
        FactorRow("train", "synthetic", f"{i:064x}", 1.0, torch.zeros(64), torch.zeros(64))
        for i in range(len(x))
    ]
    best, oof = _select_lambda(rows, x, y, [0.0001, 0.001, 0.01, 0.1, 1.0], 81300)
    state = _ridge_fit(x, y, best)
    pred = _ridge_predict(x, state)
    assert float(torch.linalg.norm(pred - y) / torch.linalg.norm(y)) < 1e-3
    assert float(torch.linalg.norm(oof - y) / torch.linalg.norm(y)) < 2e-2


def test_factor_compression_alone_never_opens_confirmation() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payloads = []
    for seed in protocol["discovery"]["seeds"]:
        payloads.append({
            "seed": seed,
            "rank1_core_guard": {"pass": True},
            "sparse_tensor_configs": [
                {"active_coordinates": 8, "viable": False},
                {"active_coordinates": 16, "viable": False},
            ],
            "right_conditioned": {"viable": False},
        })
    lock, summary = select_discovery_lock(payloads, protocol)
    assert lock is None
    assert summary["rank1_core_guard_all_completed_seeds"] is True
