from __future__ import annotations

import torch

from minicells.write_addressability import WriteAddressabilityConfig
from minicells.write_addressability_002b import (
    CoreValidation002BConfig,
    apply_global_ridge_write,
    apply_sparse_write,
    infer_sparse_write_address,
)
from minicells.write_addressability_002b_experiment import decide_run, oracle_latent_sanity
from minicells.write_addressability_models import SparseFunctionalModel


def tiny_base() -> WriteAddressabilityConfig:
    return WriteAddressabilityConfig(
        observation_dim=12,
        num_features=24,
        active_features=3,
        output_dim=5,
        latent_dim=10,
        latent_topk=4,
        edit_count=5,
        edit_examples=8,
        affected_examples=16,
        invariant_examples=16,
        retention_examples_per_edit=4,
        oracle_probe_examples=32,
        pretrain_steps=2,
        pretrain_examples=32,
        pretrain_batch_size=8,
    )


def tiny_config() -> CoreValidation002BConfig:
    return CoreValidation002BConfig(
        base=tiny_base(),
        address_widths=(1, 2, 4, 8),
        omp_als_steps=2,
        global_ridge_lambda=1e-4,
        global_scales=(0.25, 0.5, 1.0),
        maximum_sparse_base_normalized_mse=0.20,
        maximum_oracle_update_error=1e-10,
        maximum_oracle_write_leakage=1e-12,
        maximum_best_width_update_error=0.10,
        maximum_relative_update_error_vs_width1=0.60,
        maximum_absolute_write_leakage=0.005,
        maximum_matched_u_gap=0.05,
        maximum_leakage_ratio_vs_matched_global=0.50,
        maximum_repeated_update_error=0.20,
        maximum_repeated_write_leakage=0.01,
        maximum_assembly_fit_error=0.10,
    )


def identity_encoder_model() -> SparseFunctionalModel:
    config = tiny_base()
    model = SparseFunctionalModel(config)
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.weight[:, : config.latent_dim] = torch.eye(config.latent_dim)
        model.writer.weight.zero_()
    return model


def test_sparse_omp_recovers_two_coordinate_functional_assembly() -> None:
    model = identity_encoder_model()
    x = torch.zeros(8, model.config.observation_dim)
    x[:, 2] = torch.tensor([0.5, 0.8, 1.0, 1.2, -0.6, -0.9, -1.1, -1.4])
    x[:, 4] = torch.tensor([1.1, -0.7, 0.6, -1.0, 0.9, -1.2, 1.4, -0.5])
    scalar = x[:, 2] + 0.5 * x[:, 4]
    delta = torch.tensor([0.3, -0.2, 0.4, 0.1, -0.5])
    y = scalar[:, None] * delta[None, :]
    result = infer_sparse_write_address(model, x, y, width=2, als_steps=2)
    assert set(result["support"]) == {2, 4}
    assert result["support_size"] == 2
    assert result["rank1_residual_mse"] < 1e-10


def test_sparse_rank1_write_changes_exact_weighted_writer_subspace() -> None:
    model = identity_encoder_model()
    before = model.writer.weight.detach().clone()
    weights = torch.zeros(model.config.latent_dim)
    weights[1] = 0.8
    weights[3] = -0.6
    delta = torch.tensor([0.3, -0.2, 0.4, 0.1, -0.5])
    update_norm = apply_sparse_write(model, weights=weights, delta=delta)
    expected = before + delta[:, None] * weights[None, :]
    assert update_norm > 0
    assert torch.allclose(model.writer.weight, expected)


def test_global_ridge_write_updates_full_writer_without_encoder_change() -> None:
    model = identity_encoder_model()
    encoder_before = model.encoder.weight.detach().clone()
    x = torch.zeros(8, model.config.observation_dim)
    x[:, 1] = torch.linspace(0.5, 1.2, 8)
    x[:, 3] = torch.linspace(-1.0, 0.7, 8)
    y = torch.stack(
        (
            x[:, 1],
            x[:, 3],
            x[:, 1] + x[:, 3],
            x[:, 1] - x[:, 3],
            0.5 * x[:, 1],
        ),
        dim=1,
    )
    norm = apply_global_ridge_write(model, x, y, ridge_lambda=1e-4, scale=1.0)
    assert norm > 0
    assert torch.equal(model.encoder.weight, encoder_before)
    assert torch.count_nonzero(model.writer.weight).item() > 0


def test_oracle_latent_sanity_is_numerically_exact_and_local() -> None:
    result = oracle_latent_sanity(tiny_config(), seed=123)
    assert result["maximum_update_error"] <= 1e-10
    assert result["maximum_write_leakage"] <= 1e-12


def test_contextual_baselines_do_not_veto_002b_gate() -> None:
    config = tiny_config()
    summary = {
        "assembly_r1": {
            "variant_kind": "assembly",
            "address_width": 1,
            "global_scale": None,
            "median_update_error": 0.20,
            "median_write_leakage": 0.001,
            "mean_repeated_target_update_error": 0.18,
            "mean_repeated_target_write_leakage": 0.002,
            "median_assembly_fit_error": 0.15,
        },
        "assembly_r2": {
            "variant_kind": "assembly",
            "address_width": 2,
            "global_scale": None,
            "median_update_error": 0.08,
            "median_write_leakage": 0.001,
            "mean_repeated_target_update_error": 0.12,
            "mean_repeated_target_write_leakage": 0.002,
            "median_assembly_fit_error": 0.07,
        },
        "assembly_r4": {
            "variant_kind": "assembly",
            "address_width": 4,
            "global_scale": None,
            "median_update_error": 0.09,
            "median_write_leakage": 0.0015,
            "mean_repeated_target_update_error": 0.13,
            "mean_repeated_target_write_leakage": 0.002,
            "median_assembly_fit_error": 0.08,
        },
        "assembly_r8": {
            "variant_kind": "assembly",
            "address_width": 8,
            "global_scale": None,
            "median_update_error": 0.11,
            "median_write_leakage": 0.002,
            "mean_repeated_target_update_error": 0.15,
            "mean_repeated_target_write_leakage": 0.003,
            "median_assembly_fit_error": 0.09,
        },
        "global_ridge_0": {
            "variant_kind": "global_ridge",
            "address_width": None,
            "global_scale": 0.5,
            "median_update_error": 0.09,
            "median_write_leakage": 0.004,
        },
        "dense": {
            "variant_kind": "contextual",
            "median_update_error": 1.0,
            "median_write_leakage": 1.0,
        },
        "moe": {
            "variant_kind": "contextual",
            "median_update_error": 1.0,
            "median_write_leakage": 1.0,
        },
    }
    oracle = {"maximum_update_error": 0.0, "maximum_write_leakage": 0.0}
    gates = decide_run(
        config,
        {"sparse": 0.15, "dense": 999.0, "moe": 999.0},
        oracle,
        summary,
    )
    assert gates["pass"] is True
    assert gates["best_width"] == 2
