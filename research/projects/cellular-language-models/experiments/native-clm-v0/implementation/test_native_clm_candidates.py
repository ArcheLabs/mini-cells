from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import native_clm_runtime as base
from native_clm_candidates import (
    C1_NAME,
    C2_NAME,
    C3_NAME,
    C4_NAME,
    CANDIDATE_NAMES,
    build_candidate,
    estimate_forward_flops,
    parameter_summary,
)


def test_phase_a_parameter_matching() -> None:
    summary = parameter_summary()
    for name in CANDIDATE_NAMES:
        assert summary[name]["relative_error"] < 0.01


def test_phase_a_forward_shapes() -> None:
    ids = torch.randint(0, base.VOCAB_SIZE, (1, 8))
    for name in CANDIDATE_NAMES:
        model = build_candidate(name).eval()
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (1, 8, base.VOCAB_SIZE)
        assert torch.isfinite(logits).all()


def test_controlled_ablation_parameter_identity() -> None:
    c0 = base.count_parameters(base.build_model(base.C0_NAME))
    assert base.count_parameters(build_candidate(C3_NAME)) == c0
    assert base.count_parameters(build_candidate(C4_NAME)) == c0


def test_compute_direction_is_explicit() -> None:
    c0 = base.estimate_forward_flops(base.C0_NAME, base.TRAIN_SEQUENCE_LENGTH)
    assert estimate_forward_flops(C4_NAME) < c0
    assert estimate_forward_flops(C2_NAME) > c0
    assert estimate_forward_flops(C1_NAME) > 0
