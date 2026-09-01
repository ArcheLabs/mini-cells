from __future__ import annotations

import json
from pathlib import Path

import torch

from minicells.real_representation_009b2_experiment import EffectSequence, confirmation_gate_row, fit_uncentered_basis, normalized_residual, online_incremental_basis, ordered_train_rows, select_discovery_dimension, summarize_basis

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "research" / "validations" / "core-009b2-persistent-effect-geometry" / "protocol.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _effect(vec: torch.Tensor, i: int, partition: str = "train") -> EffectSequence:
    vec = vec.to(dtype=torch.float64)
    return EffectSequence(partition=partition, source=f"s{i % 3}", token_sha256=f"{i:064x}", effect=vec, effect_norm=float(torch.linalg.norm(vec).item()))


def test_uncentered_basis_recovers_known_low_dimensional_effect_space() -> None:
    gen = torch.Generator().manual_seed(912)
    q, _ = torch.linalg.qr(torch.randn(64, 4, generator=gen, dtype=torch.float64))
    train = [_effect(q @ torch.randn(4, generator=gen, dtype=torch.float64), i, "train") for i in range(40)]
    eval_rows = [_effect(q @ torch.randn(4, generator=gen, dtype=torch.float64), 100 + i, "eval") for i in range(12)]
    basis, _ = fit_uncentered_basis(train)
    summary = summarize_basis(eval_rows, basis, 4)
    assert summary["median_normalized_residual"] < 1e-10
    assert summary["p90_normalized_residual"] < 1e-10


def test_normalized_residual_is_scale_invariant_but_effect_is_not_renormalized() -> None:
    e1 = torch.tensor([3.0, 4.0] + [0.0] * 62, dtype=torch.float64)
    e2 = 7.0 * e1
    basis = torch.zeros(64, 1, dtype=torch.float64); basis[0, 0] = 1.0
    assert abs(normalized_residual(e1, basis) - normalized_residual(e2, basis)) < 1e-12
    assert abs(_effect(e2, 1).effect_norm - 35.0) < 1e-12


def test_online_incremental_growth_stops_on_true_shared_subspace() -> None:
    gen = torch.Generator().manual_seed(913)
    q, _ = torch.linalg.qr(torch.randn(64, 4, generator=gen, dtype=torch.float64))
    train = [_effect(q @ torch.randn(4, generator=gen, dtype=torch.float64), i, "train") for i in range(80)]
    eval_rows = [_effect(q @ torch.randn(4, generator=gen, dtype=torch.float64), 100 + i, "eval") for i in range(20)]
    result = online_incremental_basis(train, eval_rows, seed=81111, ordering="canonical", threshold=0.25)
    assert result["final_dimension"] <= 4
    assert result["eval_median_normalized_residual"] < 1e-10
    assert result["late_growth_per_100_writes"] == 0.0
    assert result["independent_memory_compression_ratio"] >= 20.0


def test_sha_orderings_are_deterministic_and_distinct_from_canonical() -> None:
    rows = [_effect(torch.eye(64, dtype=torch.float64)[i], i) for i in range(8)]
    a = ordered_train_rows(rows, seed=81111, ordering="sha-0")
    b = ordered_train_rows(rows, seed=81111, ordering="sha-0")
    assert [r.token_sha256 for r in a] == [r.token_sha256 for r in b]
    assert [r.token_sha256 for r in a] != [r.token_sha256 for r in rows]


def _discovery_payload(seed: int, good_from: int) -> dict:
    dims = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]
    rows = []
    for d in dims:
        good = d >= good_from
        train_med = 0.18 if good else 0.40
        eval_med = 0.20 if good else 0.45
        eval_p90 = 0.40 if good else 0.70
        rows.append({"dimension": d, "train": {"median_normalized_residual": train_med}, "eval": {"median_normalized_residual": eval_med, "p90_normalized_residual": eval_p90}, "train_to_eval_median_residual_gap": eval_med - train_med})
    return {"seed": seed, "dimension_rows": rows}


def test_discovery_selects_smallest_compact_dimension_only() -> None:
    protocol = _protocol()
    locked, _ = select_discovery_dimension([_discovery_payload(81101, 12), _discovery_payload(81102, 16)], protocol)
    assert locked == 16
    locked, _ = select_discovery_dimension([_discovery_payload(81101, 40), _discovery_payload(81102, 40)], protocol)
    assert locked is None


def test_confirmation_gate_requires_compact_low_growth_and_generalization() -> None:
    protocol = _protocol()
    payload = {"seed": 81111, "locked_dimension": 16, "offline": {"train": {"median_normalized_residual": 0.16}, "eval": {"median_normalized_residual": 0.20, "p90_normalized_residual": 0.40}, "train_to_eval_median_residual_gap": 0.04}, "online": [{"final_dimension": 20, "eval_median_normalized_residual": 0.21, "eval_p90_normalized_residual": 0.42, "late_growth_per_100_writes": 2.0, "independent_memory_compression_ratio": 10.0} for _ in range(4)]}
    assert confirmation_gate_row(payload, protocol)["pass"] is True
    payload["online"][0]["late_growth_per_100_writes"] = 8.0
    assert confirmation_gate_row(payload, protocol)["pass"] is False


def test_protocol_freezes_disjoint_seed_sets_and_forbids_trivial_64d_positive() -> None:
    protocol = _protocol()
    discovery = protocol["discovery"]["seeds"]; confirmation = protocol["confirmation"]["seeds"]
    assert discovery == [81101, 81102]
    assert confirmation == [81111, 81112, 81113]
    assert set(discovery).isdisjoint(confirmation)
    assert protocol["offline_geometry"]["compact_dimension_limit"] == 32
    assert max(protocol["offline_geometry"]["compact_candidate_dimensions"]) == 32
    assert 64 in protocol["offline_geometry"]["dimension_grid"]
    assert protocol["online_growth"]["residual_threshold_tau"] == 0.25
