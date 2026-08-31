from __future__ import annotations

import json
from pathlib import Path

import pytest

from minicells.clm04mini.model import TinyCLMDecoder
from minicells.clm04mini.preview import PREVIEW_MIXTURE
from minicells.clm04mini.release import (
    READINESS_FORMAT,
    RELEASE_CLM_EXPECTED_PARAMETERS,
    RELEASE_DENSE_EXPECTED_PARAMETERS,
    RELEASE_PROFILES,
    DenseDecoder,
    expected_profile_tokens,
    release_dense_config,
    release_model_config,
    release_pipeline_identity,
    validate_release_assets,
    verify_smoke_readiness,
)


def test_release_profiles_freeze_1m_then_30m() -> None:
    assert RELEASE_PROFILES == {"smoke-1m": 1_000_000, "release-30m": 30_000_000}
    assert expected_profile_tokens("smoke-1m") == 1_000_000
    assert expected_profile_tokens("release-30m") == 30_000_000


def test_release_clm_and_dense_are_effectively_equal_parameter() -> None:
    clm = TinyCLMDecoder(release_model_config())
    dense = DenseDecoder(release_dense_config())
    clm_count = sum(parameter.numel() for parameter in clm.parameters())
    dense_count = sum(parameter.numel() for parameter in dense.parameters())
    assert clm_count == RELEASE_CLM_EXPECTED_PARAMETERS == 5_273_088
    assert dense_count == RELEASE_DENSE_EXPECTED_PARAMETERS == 5_273_120
    assert dense_count - clm_count == 32


def test_release_pipeline_preserves_successful_preview_recipe() -> None:
    identity = release_pipeline_identity()
    assert identity["base_mixture"] == PREVIEW_MIXTURE
    assert identity["clm_model"]["shared_cell_ff_hidden"] == 256
    assert identity["clm_model"]["base_cells"] == 32
    assert identity["clm_model"]["cell_hidden"] == 32
    assert identity["transactions"] == 192
    assert identity["dense_continual_variant"] == "dense_full_always"


def test_release_asset_profile_rejects_wrong_budget(tmp_path: Path) -> None:
    payload = {
        "format": "minicells.clm-0.4-preview.assets.v1",
        "target_tokens": 1_000_000,
        "base_tokens": 1_000_001,
        "mixture": PREVIEW_MIXTURE,
        "tokenizer_hash": "a" * 64,
    }
    (tmp_path / "asset-summary.json").write_text(json.dumps(payload), encoding="utf-8")
    assert validate_release_assets("smoke-1m", tmp_path)["base_tokens"] == 1_000_001
    with pytest.raises(RuntimeError, match="requires target_tokens"):
        validate_release_assets("release-30m", tmp_path)


def test_30m_readiness_rejects_source_or_tokenizer_drift(tmp_path: Path) -> None:
    readiness = tmp_path / "release-readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "format": READINESS_FORMAT,
                "status": "READY_FOR_30M",
                "pipeline_sha256": "pipeline",
                "source_fingerprint": {"sha256": "source"},
                "tokenizer_hash": "tokenizer",
                "transactions": 192,
            }
        ),
        encoding="utf-8",
    )
    result = verify_smoke_readiness(
        readiness,
        source_fingerprint={"sha256": "source"},
        pipeline_sha256="pipeline",
        tokenizer_hash="tokenizer",
    )
    assert result["status"] == "READY_FOR_30M"
    with pytest.raises(RuntimeError, match="source files changed"):
        verify_smoke_readiness(
            readiness,
            source_fingerprint={"sha256": "different"},
            pipeline_sha256="pipeline",
            tokenizer_hash="tokenizer",
        )
    with pytest.raises(RuntimeError, match="tokenizer identity"):
        verify_smoke_readiness(
            readiness,
            source_fingerprint={"sha256": "source"},
            pipeline_sha256="pipeline",
            tokenizer_hash="different",
        )
