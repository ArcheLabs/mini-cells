"""Guards for ephemeral-Kaggle recovery of the final PCU objective diagnostic."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_recovery_reruns_only_k64_ce_control() -> None:
    source = _text("scripts/research/recover_pcu_objective_k64_control.py")
    assert "width=64" in source
    assert "EXPECTED_DIRECT_ACCURACY = 0.265625" in source
    assert '"control_objective": "answer-token-causal-cross-entropy"' in source
    assert '"recovery_mode": "paired_ce_k64_only"' in source
    assert '"historical_sweep_not_reconstructed": True' in source
    assert "width=16" not in source
    assert "width=32" not in source


def test_recovery_envelope_is_explicitly_not_historical_sweep_evidence() -> None:
    source = _text("scripts/research/recover_pcu_objective_k64_control.py")
    assert "local-only compatibility envelope" in source
    assert "historical_sweep_not_reconstructed" in source
    assert "PAIRED_CE_K64.json" in source


def test_final_publisher_accepts_paired_control_without_publishing_fake_sweep() -> None:
    source = _text("scripts/research/publish_pcu_objective_alignment_001.py")
    assert '"paired_ce_k64_recovery"' in source
    assert "PAIRED_CE_K64.json" in source
    assert "EXPECTED_CE_K64_ACCURACY = 0.265625" in source
    assert 'if baseline["mode"] == "published_locality_sweep"' in source
    assert "assert_locality_published(args.branch)" in source


def test_kaggle_notebook_recovers_when_full_locality_artifacts_are_missing() -> None:
    source = _text("research/notebooks/08-pretrained-model-lift/pcu-objective-alignment-001-kaggle.ipynb")
    assert "recover_pcu_objective_k64_control.py" in source
    assert "paired_ce_k64_recovery" in source
    assert "Historical locality artifacts unavailable" in source
    assert "PAIRED_CE_K64.json is allowed to exist" in source
