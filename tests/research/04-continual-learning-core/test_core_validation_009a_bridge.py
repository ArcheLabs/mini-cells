from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import torch

from minicells.real_representation_009a_bridge import (
    BridgeSequence,
    _token_right_covariances,
    apply_z_transform,
    fit_z_transform_state,
    run_bridge,
    spectrum_energy,
)
from minicells.real_representation_009a_experiment import normalize_write

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "research" / "validations" / "core-009a-right-collapse-bridge" / "protocol.json"
RUNNER = ROOT / "scripts" / "research" / "run_core_validation_009a_bridge_seed.py"


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _synthetic_mean_collapse() -> list[BridgeSequence]:
    gen = torch.Generator().manual_seed(9019)
    dim = 64
    out = []
    for i in range(18):
        tokens = 32
        q = torch.zeros(tokens, dim, dtype=torch.float64)
        q[:, 0] = 1.0
        z = torch.randn(tokens, dim, generator=gen, dtype=torch.float64)
        z[:, 0] += 12.0
        raw = normalize_write(torch.einsum("to,ti->oi", q, z) / tokens)
        out.append(
            BridgeSequence(
                partition="train" if i < 12 else "eval",
                token_sha256=f"synthetic-{i}",
                q=q,
                z=z,
                raw_write=raw,
            )
        )
    return out


def test_train_only_centering_and_whitening_remove_first_two_activation_moments() -> None:
    seqs = _synthetic_mean_collapse()
    state = fit_z_transform_state(seqs, whitening_floor_fraction=1e-6)
    train_centered = torch.cat(
        [apply_z_transform(s.z, "centered", state) for s in seqs if s.partition == "train"],
        dim=0,
    )
    assert torch.linalg.norm(train_centered.mean(dim=0)) < 1e-10

    train_white = torch.cat(
        [apply_z_transform(s.z, "whitened", state) for s in seqs if s.partition == "train"],
        dim=0,
    )
    cov = train_white.T @ train_white / train_white.shape[0]
    assert torch.max(torch.abs(cov - torch.eye(64, dtype=torch.float64))) < 1e-8


def test_mean_direction_removal_annihilates_the_fitted_mean_axis() -> None:
    seqs = _synthetic_mean_collapse()
    state = fit_z_transform_state(seqs, whitening_floor_fraction=1e-6)
    transformed = torch.cat(
        [apply_z_transform(s.z, "mean_direction_removed", state) for s in seqs], dim=0
    )
    assert torch.max(torch.abs(transformed @ state.mean_direction)) < 1e-10


def test_token_normalized_right_covariance_detects_mean_direction_anisotropy() -> None:
    seqs = _synthetic_mean_collapse()
    state = fit_z_transform_state(seqs, whitening_floor_fraction=1e-6)
    raw_cov, _, _ = _token_right_covariances(seqs, "raw", state)
    centered_cov, _, _ = _token_right_covariances(seqs, "centered", state)
    raw_vals = torch.linalg.eigvalsh(raw_cov).flip(0).clamp_min(0)
    centered_vals = torch.linalg.eigvalsh(centered_cov).flip(0).clamp_min(0)
    raw_top1 = float(raw_vals[0] / raw_vals.sum())
    centered_top1 = float(centered_vals[0] / centered_vals.sum())
    assert raw_top1 > 0.65
    assert raw_top1 - centered_top1 > 0.50


def test_full_bridge_can_distinguish_mean_induced_sequence_collapse() -> None:
    result = run_bridge(_synthetic_mean_collapse(), _protocol(), seed=80911)
    raw = spectrum_energy(result["conditions"]["raw"], "sequence_right_spectrum", 1)
    centered = spectrum_energy(result["conditions"]["centered"], "sequence_right_spectrum", 1)
    mean_removed = spectrum_energy(
        result["conditions"]["mean_direction_removed"], "sequence_right_spectrum", 1
    )
    assert raw > 0.95
    assert raw - centered > 0.20
    assert raw - mean_removed > 0.20
    assert result["scientific_decision"] is False
    assert result["source_009a_status_changed"] is False


def test_runner_filters_router_only_after_full_extraction() -> None:
    namespace = runpy.run_path(str(RUNNER))
    filter_fn = namespace["_analysis_sequences"]
    rows = [
        SimpleNamespace(partition="router", marker=0),
        SimpleNamespace(partition="train", marker=1),
        SimpleNamespace(partition="eval", marker=2),
        SimpleNamespace(partition="router", marker=3),
    ]
    filtered = filter_fn(rows)
    assert [row.marker for row in filtered] == [1, 2]
    assert {row.partition for row in filtered} == {"train", "eval"}


def test_protocol_is_diagnostic_and_pins_the_positive_009a_source() -> None:
    protocol = _protocol()
    assert protocol["scientific_decision"] is False
    assert protocol["scope"]["may_change_source_009a_decision"] is False
    assert protocol["source_009a"]["status"] == "FACTORIZED_FUNCTIONAL_COORDINATES_SUPPORTED"
    assert protocol["source_009a"]["scientific_decision"] is True
    assert protocol["source_009a"]["locked_split"] == {"left_dim": 56, "right_dim": 8}
    assert protocol["replication"]["diagnostic_seeds"] == [80911, 80912, 80913]
    assert protocol["replication"]["maximum_source_009a_raw_56x8_action_residual_delta"] == 1e-8
