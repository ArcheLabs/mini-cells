from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import native_clm_runtime as base
from native_clm_replication import (
    CONFIRMATION_SEEDS,
    MODEL_IDS,
    PARAMETER_TOLERANCE,
    REPLICATION_MODELS,
    REPLICATION_SEEDS,
    build_replication_model,
    parameter_summary,
    validate_replication_request,
)
from native_clm_replication_visualize import (
    write_cross_seed_ppl,
    write_paired_delta,
    write_seed_visualizations,
)


def test_replication_protocol_is_frozen() -> None:
    assert REPLICATION_SEEDS == (91002, 91003)
    assert set(CONFIRMATION_SEEDS) == {91101, 91102, 91103}
    assert tuple(MODEL_IDS[name] for name in REPLICATION_MODELS) == ("T1", "M4", "X3", "X4")
    validate_replication_request([91002, 91003])
    with pytest.raises(RuntimeError):
        validate_replication_request([91002])
    with pytest.raises(RuntimeError):
        validate_replication_request([91002, 91003, 91101])


def test_replication_models_are_parameter_matched_and_finite() -> None:
    params = parameter_summary()
    ids = torch.randint(0, base.VOCAB_SIZE, (1, 8))
    for name in REPLICATION_MODELS:
        assert float(params[name]["relative_error"]) < PARAMETER_TOLERANCE
        model = build_replication_model(name)
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (1, 8, base.VOCAB_SIZE)
        assert torch.isfinite(logits).all()


def _rows(seed: int) -> list[dict[str, object]]:
    base_ppl = 12.0 + (seed - 91002) * 0.1
    return [
        {"seed": seed, "id": "T1", "model": "T1-modern-transformer", "validation_ppl_10m": base_ppl, "train_flops_estimate": 1.0e14},
        {"seed": seed, "id": "M4", "model": "M4-c1-modernized", "validation_ppl_10m": base_ppl - 0.1, "train_flops_estimate": 3.2e14},
        {"seed": seed, "id": "X3", "model": "X3-n1-swiglu-rope", "validation_ppl_10m": base_ppl - 0.2, "train_flops_estimate": 3.3e14},
        {"seed": seed, "id": "X4", "model": "X4-n1-swiglu-rope-rmsnorm", "validation_ppl_10m": base_ppl - 0.25, "train_flops_estimate": 3.3e14},
    ]


def _curves(seed: int) -> dict[str, list[dict[str, object]]]:
    result = {}
    for index, model_id in enumerate(("T1", "M4", "X3", "X4")):
        result[model_id] = [
            {"consumed_tokens": 1_000_000, "validation_nll": 4.0 - 0.02 * index + 0.001 * (seed-91002)},
            {"consumed_tokens": 2_500_000, "validation_nll": 3.2 - 0.02 * index},
            {"consumed_tokens": 5_000_000, "validation_nll": 2.8 - 0.02 * index},
            {"consumed_tokens": 7_500_000, "validation_nll": 2.6 - 0.02 * index},
            {"consumed_tokens": 10_000_000, "validation_nll": 2.5 - 0.02 * index},
        ]
    return result


def test_replication_visualizations_are_dependency_free(tmp_path: Path) -> None:
    per_seed = {}
    svg_paths = []
    for seed in REPLICATION_SEEDS:
        seed_dir = tmp_path / f"seed-{seed}"
        rows = _rows(seed)
        per_seed[seed] = rows
        files = write_seed_visualizations(rows, _curves(seed), seed, seed_dir)
        svg_paths.extend(seed_dir / filename for filename in files.values())

    cross = tmp_path / "replication-cross-seed-ppl.svg"
    paired = tmp_path / "replication-paired-delta-vs-t1.svg"
    write_cross_seed_ppl(per_seed, cross)
    write_paired_delta(per_seed, paired)
    svg_paths.extend((cross, paired))

    for path in svg_paths:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("<svg")
        assert "Native CLM" in text
        assert "<script" not in text.lower()
