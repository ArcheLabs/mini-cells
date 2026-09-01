from __future__ import annotations

import json
from pathlib import Path

import torch

from minicells.real_representation_009a_experiment import (
    _covariance_bases,
    _project_left,
    _project_right,
    _project_two_sided,
    summarize_discovery,
)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "research" / "validations" / "core-009a-factorized-functional-coordinates" / "protocol.json"


def _synthetic_rows() -> tuple[list[dict], torch.Tensor, torch.Tensor]:
    torch.manual_seed(9)
    dim = 64
    ql, _ = torch.linalg.qr(torch.randn(dim, 3, dtype=torch.float64))
    qr, _ = torch.linalg.qr(torch.randn(dim, 5, dtype=torch.float64))
    rows = []
    for i in range(24):
        c = torch.randn(3, 5, dtype=torch.float64)
        g = ql @ c @ qr.T
        g = g / torch.linalg.norm(g)
        z = torch.randn(12, dim, dtype=torch.float64)
        rows.append({"partition": "train" if i < 18 else "eval", "g": g, "z": z})
    return rows, ql, qr


def test_two_sided_basis_recovers_shared_tensor_product_space() -> None:
    rows, _, _ = _synthetic_rows()
    train = [r for r in rows if r["partition"] == "train"]
    left, right, _, _ = _covariance_bases(train)
    for row in rows:
        approx = _project_two_sided(row["g"], left, right, 3, 5)
        assert torch.allclose(approx, row["g"], atol=1e-10)


def test_one_sided_projection_needs_the_matching_factor_space() -> None:
    rows, _, _ = _synthetic_rows()
    train = [r for r in rows if r["partition"] == "train"]
    left, right, _, _ = _covariance_bases(train)
    g = rows[-1]["g"]
    assert torch.linalg.norm(g - _project_left(g, left, 3)) < 1e-10
    assert torch.linalg.norm(g - _project_right(g, right, 5)) < 1e-10
    assert torch.linalg.norm(g - _project_left(g, left, 1)) > 1e-3
    assert torch.linalg.norm(g - _project_right(g, right, 1)) > 1e-3


def test_protocol_keeps_discovery_and_confirmation_untouched() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    discovery = set(protocol["discovery"]["seeds"])
    confirmation = set(protocol["confirmation"]["seeds"])
    assert discovery == {80901, 80902}
    assert confirmation == {80911, 80912, 80913}
    assert discovery.isdisjoint(confirmation)
    assert all(int(m) + int(n) == 64 for m, n in protocol["write_geometry"]["budget_matched_splits"])
    assert all(64 * (int(m) + int(n)) == 4096 for m, n in protocol["write_geometry"]["budget_matched_splits"])


def _fake_run(seed: int, values: dict[tuple[int, int], float]) -> dict:
    rows = []
    for (m, n), value in values.items():
        rows.append({
            "partition": "eval",
            "left_dim": m,
            "right_dim": n,
            "basis_parameter_count": 4096,
            "median_local_action_residual": value,
            "median_frobenius_residual": value,
        })
        rows.append({
            "partition": "train",
            "left_dim": m,
            "right_dim": n,
            "basis_parameter_count": 4096,
            "median_local_action_residual": max(0.0, value - 0.02),
            "median_frobenius_residual": value,
        })
    return {"seed": seed, "budget_splits": rows}


def test_discovery_selection_uses_frozen_residual_then_balance_tiebreak() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    splits = [tuple(map(int, x)) for x in protocol["write_geometry"]["budget_matched_splits"]]
    base = {s: 0.60 for s in splits}
    base[(24, 40)] = 0.30
    base[(40, 24)] = 0.30
    runs = [_fake_run(80901, base), _fake_run(80902, base)]
    decision = summarize_discovery(runs, protocol)
    # Equal residual and equal imbalance -> lower m is the deterministic final tie-break.
    assert decision["provisional_winner"] == {"left_dim": 24, "right_dim": 40}
    assert decision["winner_meets_viability"] is True
    assert decision["confirmation_allowed"] is True


def test_discovery_blocks_confirmation_when_no_budget_split_is_viable() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    splits = [tuple(map(int, x)) for x in protocol["write_geometry"]["budget_matched_splits"]]
    vals = {s: 0.55 for s in splits}
    decision = summarize_discovery([_fake_run(80901, vals), _fake_run(80902, vals)], protocol)
    assert decision["winner_meets_viability"] is False
    assert decision["confirmation_allowed"] is False
