from __future__ import annotations

from pathlib import Path

import torch

from minicells.write_addressability import WriteAddressabilityConfig
from minicells.write_addressability_002c import (
    CoreValidation002CConfig,
    classify_seed,
    dense_linear_decoder,
    oracle_omp_decoders,
    summarize_decoder_metrics,
)


def tiny_config() -> CoreValidation002CConfig:
    return CoreValidation002CConfig(
        base=WriteAddressabilityConfig(
            observation_dim=8,
            num_features=6,
            active_features=2,
            output_dim=4,
            latent_dim=12,
            latent_topk=4,
            pretrain_steps=2,
            pretrain_examples=32,
            pretrain_batch_size=8,
            validation_examples=16,
        ),
        widths=(1, 2, 4),
        train_probe_examples=128,
        test_probe_examples=64,
        probe_batch_size=32,
        omp_jitter=1e-8,
        dense_ridge_lambda=1e-4,
        maximum_sparse_base_normalized_mse=0.20,
        maximum_sparse_affected_fit_error=0.10,
        maximum_sparse_leakage=0.005,
        maximum_relative_fit_error_vs_width1=0.60,
        minimum_joint_feature_success_fraction=0.50,
        minimum_feature_improvement_fraction=0.50,
    )


def test_oracle_omp_recovers_exact_two_coordinate_signal() -> None:
    generator = torch.Generator().manual_seed(123)
    train = torch.randn(6000, 12, generator=generator, dtype=torch.float64)
    test = torch.randn(3000, 12, generator=generator, dtype=torch.float64)
    truth = torch.zeros(12, 6, dtype=torch.float64)
    for feature in range(6):
        truth[feature, feature] = 1.0
        truth[feature + 6, feature] = 0.5
    targets = train @ truth
    gram = train.transpose(0, 1) @ train / len(train)
    cross = train.transpose(0, 1) @ targets / len(train)
    decoders, supports = oracle_omp_decoders(
        gram,
        cross,
        widths=(1, 2, 4),
        jitter=1e-8,
    )
    target_test = test @ truth
    width1_error = ((test @ decoders[1] - target_test).square().mean() / target_test.square().mean()).item()
    width2_error = ((test @ decoders[2] - target_test).square().mean() / target_test.square().mean()).item()
    assert width1_error > 0.05
    assert width2_error < 1e-8
    assert set(supports[2][0]) == {0, 6}


def test_dense_linear_reference_recovers_linear_code() -> None:
    generator = torch.Generator().manual_seed(77)
    z = torch.randn(4000, 10, generator=generator, dtype=torch.float64)
    truth = torch.randn(10, 5, generator=generator, dtype=torch.float64)
    s = z @ truth
    gram = z.transpose(0, 1) @ z / len(z)
    cross = z.transpose(0, 1) @ s / len(z)
    decoder = dense_linear_decoder(gram, cross, ridge_lambda=1e-8)
    relative = ((z @ decoder - s).square().mean() / s.square().mean()).item()
    assert relative < 1e-12


def test_summary_uses_joint_feature_success() -> None:
    metrics = {
        "affected_fit_error": [0.02, 0.04, 0.2, 0.3],
        "off_support_leakage": [0.001, 0.003, 0.001, 0.02],
        "unconditional_normalized_mse": [0.1, 0.1, 0.2, 0.4],
        "context_ratio_variance": [0.01, 0.02, 0.1, 0.2],
        "positive_examples": [10, 10, 10, 10],
    }
    summary = summarize_decoder_metrics(metrics, maximum_fit_error=0.10, maximum_leakage=0.005)
    assert summary["joint_feature_success_fraction"] == 0.5


def test_seed_classification_distinguishes_sparse_and_dense_only() -> None:
    config = tiny_config()
    width1 = [0.25] * 6
    width2 = [0.05] * 6
    width4 = [0.04] * 6
    low_leakage = [0.001] * 6
    context = [0.01] * 6
    positives = [10.0] * 6

    def values(fit: list[float], leakage: list[float]) -> dict[str, list[float]]:
        return {
            "affected_fit_error": fit,
            "off_support_leakage": leakage,
            "unconditional_normalized_mse": fit,
            "context_ratio_variance": context,
            "positive_examples": positives,
        }

    metrics = {
        "sparse_r1": values(width1, low_leakage),
        "sparse_r2": values(width2, low_leakage),
        "sparse_r4": values(width4, low_leakage),
        "dense_linear": values([0.01] * 6, low_leakage),
    }
    summaries = {
        name: summarize_decoder_metrics(
            value,
            maximum_fit_error=config.maximum_sparse_affected_fit_error,
            maximum_leakage=config.maximum_sparse_leakage,
        )
        for name, value in metrics.items()
    }
    result = classify_seed(
        config,
        base_normalized_mse=0.05,
        metrics=metrics,
        summaries=summaries,
    )
    assert result["pass"] is True
    assert result["representation_regime"] == "SPARSE_LINEAR"

    bad_metrics = dict(metrics)
    bad_metrics["sparse_r2"] = values([0.22] * 6, low_leakage)
    bad_metrics["sparse_r4"] = values([0.21] * 6, low_leakage)
    bad_summaries = {
        name: summarize_decoder_metrics(
            value,
            maximum_fit_error=config.maximum_sparse_affected_fit_error,
            maximum_leakage=config.maximum_sparse_leakage,
        )
        for name, value in bad_metrics.items()
    }
    result = classify_seed(
        config,
        base_normalized_mse=0.05,
        metrics=bad_metrics,
        summaries=bad_summaries,
    )
    assert result["pass"] is False
    assert result["representation_regime"] == "DENSE_LINEAR_ONLY"


def test_frozen_protocol_parses() -> None:
    root = Path(__file__).resolve().parents[1]
    config = CoreValidation002CConfig.from_protocol(root / "research" / "validations" / "core-002c-oracle-tomography" / "protocol.json")
    assert config.widths == (1, 2, 4, 8, 16)
    assert config.train_probe_examples == 32768
    assert config.test_probe_examples == 16384
