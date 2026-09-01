from __future__ import annotations

import json
from pathlib import Path

import torch

from minicells.real_representation_009b1_experiment import (
    CausalSequence,
    decompose_direction,
    discovery_scale_is_viable,
    eta_for_target_ratio,
    select_peers,
    summarize_confirmation,
    summarize_discovery,
)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "research" / "validations" / "core-009b1-carrier-causal-sufficiency" / "protocol.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _seq(name: str, source: str, partition: str = "eval") -> CausalSequence:
    dim = 64
    hidden = torch.ones(4, 768, dtype=torch.float32)
    labels = torch.zeros(4, dtype=torch.long)
    z = torch.zeros(4, dim, dtype=torch.float64)
    z[:, 0] = 2.0
    g = torch.zeros(dim, dim, dtype=torch.float64)
    g[0, 0] = 0.8
    g[0, 1] = 0.6
    return CausalSequence(
        partition=partition,
        source=source,
        token_sha256=name,
        hidden=hidden,
        labels=labels,
        z=z,
        ghat=g,
        raw_write_norm=1.0,
    )


def test_carrier_residual_reconstruct_full_without_renormalization() -> None:
    seq = _seq("a", "s")
    r = torch.zeros(64, dtype=torch.float64)
    r[0] = 1.0
    carrier, residual = decompose_direction(seq.ghat, r)
    assert torch.allclose(carrier + residual, seq.ghat)
    assert abs(float(torch.linalg.norm(carrier)) - 0.8) < 1e-12
    assert abs(float(torch.linalg.norm(residual)) - 0.6) < 1e-12


def test_eta_hits_requested_target_hidden_ratio_for_full_direction() -> None:
    seq = _seq("a", "s")
    rho = 0.003
    eta = eta_for_target_ratio(seq, seq.ghat, rho)
    coeff = seq.z @ seq.ghat.T
    achieved = eta * float(torch.linalg.norm(coeff)) / float(torch.linalg.norm(seq.hidden.to(torch.float64)))
    assert abs(achieved - rho) < 1e-12


def test_peer_selection_is_deterministic_and_respects_source_boundary() -> None:
    target = _seq("target", "s0")
    rows = [target, _seq("same1", "s0"), _seq("same2", "s0")]
    rows += [_seq(f"x{i}", f"s{i}") for i in range(1, 7)]
    m1, u1 = select_peers(target, rows, seed=81011, matched_count=1, unrelated_count=6)
    m2, u2 = select_peers(target, rows, seed=81011, matched_count=1, unrelated_count=6)
    assert [x.token_sha256 for x in m1] == [x.token_sha256 for x in m2]
    assert [x.token_sha256 for x in u1] == [x.token_sha256 for x in u2]
    assert len(m1) == 1 and m1[0].source == target.source
    assert len(u1) == 6 and all(x.source != target.source for x in u1)
    assert len({x.source for x in u1}) == 6


def _discovery_run(seed: int, values: dict[float, tuple[float, float, float, float]]) -> dict:
    rows = []
    for rho, (descent, gain, median_err, p90_err) in values.items():
        rows.append({
            "seed": seed,
            "rho": rho,
            "count": 56,
            "full_descent_fraction": descent,
            "median_full_nll_gain": gain,
            "median_full_normalized_nll_gain": gain,
            "median_half_step_linearity_error": median_err,
            "p90_half_step_linearity_error": p90_err,
        })
    return {"seed": seed, "scale_summary": rows}


def test_discovery_locks_largest_scale_that_is_full_write_stable_on_both_seeds() -> None:
    protocol = _protocol()
    vals = {
        0.0003: (1.0, 1e-4, 0.02, 0.05),
        0.001: (1.0, 2e-4, 0.05, 0.10),
        0.003: (0.98, 3e-4, 0.10, 0.20),
        0.01: (0.90, 5e-4, 0.10, 0.20),
    }
    decision = summarize_discovery([
        _discovery_run(81001, vals),
        _discovery_run(81002, vals),
    ], protocol)
    assert decision["confirmation_allowed"] is True
    assert decision["locked_rho"] == 0.003


def test_discovery_viability_does_not_contain_carrier_metrics() -> None:
    protocol = _protocol()
    row = {
        "full_descent_fraction": 1.0,
        "median_full_normalized_nll_gain": 1e-4,
        "median_half_step_linearity_error": 0.1,
        "p90_half_step_linearity_error": 0.2,
        "carrier_over_full_target_gain": -999.0,
    }
    assert discovery_scale_is_viable(row, protocol) is True
    assert protocol["discovery"]["carrier_or_residual_metrics_forbidden_for_scale_selection"] is True


def _confirmation_run(seed: int, *, carrier: float, residual: float, harm: float) -> dict:
    return {
        "seed": seed,
        "summary": {
            "seed": seed,
            "rho": 0.003,
            "count": 56,
            "full_descent_fraction": 1.0,
            "carrier_descent_fraction": 0.98,
            "median_carrier_over_full_target_gain": carrier,
            "median_residual_over_full_target_gain": residual,
            "median_carrier_excess_unrelated_harm_over_full_target_gain": harm,
            "median_full_target_gain": 0.01,
            "median_carrier_target_gain": 0.0095,
            "median_residual_target_gain": 0.0005,
            "median_full_unrelated_positive_harm": 0.0001,
            "median_carrier_unrelated_positive_harm": 0.0001,
            "median_carrier_frobenius_fraction": 0.97,
            "median_residual_frobenius_fraction": 0.15,
        },
    }


def test_confirmation_requires_all_three_frozen_seeds() -> None:
    protocol = _protocol()
    good = [_confirmation_run(s, carrier=0.95, residual=0.05, harm=0.01) for s in [81011, 81012, 81013]]
    decision = summarize_confirmation(good, protocol)
    assert decision["scientific_decision"] is True
    assert decision["supported"] is True
    assert decision["passed_seeds"] == 3

    bad = good[:-1] + [_confirmation_run(81013, carrier=0.70, residual=0.05, harm=0.01)]
    decision = summarize_confirmation(bad, protocol)
    assert decision["supported"] is False


def test_protocol_freezes_causal_order_and_untouched_confirmation() -> None:
    protocol = _protocol()
    assert protocol["discovery"]["seeds"] == [81001, 81002]
    assert protocol["confirmation"]["seeds"] == [81011, 81012, 81013]
    assert set(protocol["discovery"]["seeds"]).isdisjoint(protocol["confirmation"]["seeds"])
    assert protocol["geometry"]["same_step_rule"].startswith("full, carrier and residual always use the same eta")
    assert protocol["stop_rule"]["confirmation_success"].startswith("Proceed to Core 009B-2")
