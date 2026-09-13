from __future__ import annotations

import torch

import native_clm_runtime as base
from native_clm_phase_b import (
    M1_NAME,
    M2_NAME,
    M3_NAME,
    M4_NAME,
    M_NAMES,
    N1_ALPHA,
    N1_NAME,
    N2_NAME,
    N3_NAME,
    N4_NAME,
    N_NAMES,
    PARAMETER_TOLERANCE,
    PHASE_B_NAMES,
    build_phase_b,
    estimate_forward_flops,
    parameter_summary,
    phase_b_config,
)


def test_phase_b_parameter_match() -> None:
    summary = parameter_summary()
    for name in PHASE_B_NAMES:
        assert summary[name]["relative_error"] < PARAMETER_TOLERANCE, (name, summary[name])


def test_phase_b_forward_shapes_and_finite_logits() -> None:
    ids = torch.randint(0, base.VOCAB_SIZE, (2, 8))
    for name in PHASE_B_NAMES:
        model = build_phase_b(name)
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (2, 8, base.VOCAB_SIZE), name
        assert torch.isfinite(logits).all(), name


def test_modernization_track_is_one_factor_until_m4() -> None:
    assert tuple(phase_b_config(name)["track"] for name in M_NAMES) == ("modernization",) * 4
    assert phase_b_config(M1_NAME)["factor"] == "normalization"
    assert phase_b_config(M2_NAME)["factor"] == "activation"
    assert phase_b_config(M3_NAME)["factor"] == "position"
    assert phase_b_config(M4_NAME)["factor"] == "combined-modernization"

    assert build_phase_b(M1_NAME).pos is not None
    assert build_phase_b(M2_NAME).pos is not None
    assert build_phase_b(M3_NAME).pos is None
    assert build_phase_b(M4_NAME).pos is None


def test_native_track_semantics() -> None:
    assert tuple(phase_b_config(name)["track"] for name in N_NAMES) == ("native",) * 4
    assert 0.0 < N1_ALPHA < 1.0
    assert phase_b_config(N1_NAME)["factor"] == "gate-ablation"
    assert phase_b_config(N2_NAME)["factor"] == "dual-gate-aru"
    assert phase_b_config(N3_NAME)["factor"] == "recurrent-step-conditioned-write"
    assert phase_b_config(N4_NAME)["factor"] == "adaptive-communication"
    assert phase_b_config(N4_NAME)["realized_compute_reduction_claim"] is False


def test_n4_does_not_claim_sparse_compute() -> None:
    # N4 computes attention and only gates its contribution. The FLOP model must
    # therefore not treat it like Phase-A C4 sparse communication.
    c1_like = estimate_forward_flops(M1_NAME)
    n4 = estimate_forward_flops(N4_NAME)
    assert n4 >= c1_like
