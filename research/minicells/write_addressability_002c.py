"""Oracle representation tomography for Core Validation 002C."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .write_addressability import SuperpositionWorld, WriteAddressabilityConfig, _EPS
from .write_addressability_models import (
    SparseFunctionalModel,
    _base_validation,
    _train_stream,
    build_models,
)


@dataclass(frozen=True)
class CoreValidation002CConfig:
    base: WriteAddressabilityConfig
    widths: tuple[int, ...]
    train_probe_examples: int
    test_probe_examples: int
    probe_batch_size: int
    omp_jitter: float
    dense_ridge_lambda: float
    maximum_sparse_base_normalized_mse: float
    maximum_sparse_affected_fit_error: float
    maximum_sparse_leakage: float
    maximum_relative_fit_error_vs_width1: float
    minimum_joint_feature_success_fraction: float
    minimum_feature_improvement_fraction: float

    @classmethod
    def from_protocol(cls, path: str | Path) -> "CoreValidation002CConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        world = payload["world"]
        model = payload["model"]
        pretrain = payload["pretraining"]
        tomography = payload["tomography"]
        gates = payload["gates"]
        base = WriteAddressabilityConfig(
            observation_dim=int(world["observation_dim"]),
            num_features=int(world["num_features"]),
            active_features=int(world["active_features"]),
            output_dim=int(world["output_dim"]),
            latent_dim=int(model["latent_dim"]),
            latent_topk=int(model["latent_topk"]),
            coefficient_min_abs=float(world["coefficient_min_abs"]),
            coefficient_max_abs=float(world["coefficient_max_abs"]),
            edit_scale=float(world["edit_scale"]),
            pretrain_steps=int(pretrain["steps"]),
            pretrain_examples=int(pretrain["examples"]),
            pretrain_batch_size=int(pretrain["batch_size"]),
            pretrain_learning_rate=float(pretrain["learning_rate"]),
            pretrain_weight_decay=float(pretrain["weight_decay"]),
            reconstruction_weight=float(pretrain["reconstruction_weight"]),
            gradient_clip_norm=float(pretrain["gradient_clip_norm"]),
            validation_examples=int(pretrain["validation_examples"]),
            moe_topk=int(model["moe_topk"]),
        )
        widths = tuple(int(value) for value in tomography["widths"])
        if not widths or widths[0] != 1 or any(value < 1 for value in widths):
            raise ValueError("002C widths must begin at 1 and be positive")
        if tuple(sorted(set(widths))) != widths:
            raise ValueError("002C widths must be unique and increasing")
        return cls(
            base=base,
            widths=widths,
            train_probe_examples=int(tomography["train_probe_examples"]),
            test_probe_examples=int(tomography["test_probe_examples"]),
            probe_batch_size=int(tomography["probe_batch_size"]),
            omp_jitter=float(tomography["omp_jitter"]),
            dense_ridge_lambda=float(tomography["dense_ridge_lambda"]),
            maximum_sparse_base_normalized_mse=float(gates["maximum_sparse_base_normalized_mse"]),
            maximum_sparse_affected_fit_error=float(gates["maximum_sparse_affected_fit_error"]),
            maximum_sparse_leakage=float(gates["maximum_sparse_leakage"]),
            maximum_relative_fit_error_vs_width1=float(gates["maximum_relative_fit_error_vs_width1"]),
            minimum_joint_feature_success_fraction=float(gates["minimum_joint_feature_success_fraction"]),
            minimum_feature_improvement_fraction=float(gates["minimum_feature_improvement_fraction"]),
        )


def pretrain_sparse_reference(
    config: WriteAddressabilityConfig,
    *,
    seed: int,
    device: torch.device,
) -> tuple[SuperpositionWorld, SparseFunctionalModel, dict[str, Any]]:
    """Reproduce the 002/002B sparse pretraining stream without training contextual baselines."""

    world = SuperpositionWorld(config, seed=seed + 101)
    models = build_models(config, seed=seed + 211)
    sparse = models["sparse"]
    assert isinstance(sparse, SparseFunctionalModel)
    generator = torch.Generator().manual_seed(seed + 293)
    training = world.sample_batch(config.pretrain_examples, generator=generator)
    history = _train_stream(
        sparse,
        training,
        config,
        seed=seed + 307,
        device=device,
        sparse_objective=True,
    )
    validation = _base_validation(
        sparse,
        world,
        config,
        seed=seed + 401,
        device=device,
    )
    sparse.eval()
    return world, sparse, {
        "loss_history": history,
        "base_normalized_mse": validation,
    }


@torch.no_grad()
def accumulate_oracle_moments(
    model: SparseFunctionalModel,
    world: SuperpositionWorld,
    *,
    examples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accumulate G=E[z z^T], C=E[z s^T], and E[s^2] on an oracle train stream."""

    if examples < 1 or batch_size < 1:
        raise ValueError("examples and batch_size must be positive")
    latent_dim = model.config.latent_dim
    num_features = model.config.num_features
    gram = torch.zeros(latent_dim, latent_dim, device=device)
    cross = torch.zeros(latent_dim, num_features, device=device)
    target_energy = torch.zeros(num_features, device=device)
    generator = torch.Generator().manual_seed(seed)
    consumed = 0
    model.eval()
    while consumed < examples:
        count = min(batch_size, examples - consumed)
        batch = world.sample_batch(count, generator=generator)
        z = model.encode(batch.x.to(device))
        s = batch.s.to(device)
        gram.addmm_(z.transpose(0, 1), z)
        cross.addmm_(z.transpose(0, 1), s)
        target_energy += s.square().sum(dim=0)
        consumed += count
    scale = 1.0 / float(examples)
    return (
        (gram * scale).detach().cpu().to(torch.float64),
        (cross * scale).detach().cpu().to(torch.float64),
        (target_energy * scale).detach().cpu().to(torch.float64),
    )


def _solve_support(
    gram: torch.Tensor,
    cross_column: torch.Tensor,
    support: list[int],
    *,
    jitter: float,
) -> torch.Tensor:
    indices = torch.tensor(support, dtype=torch.long)
    local_gram = gram.index_select(0, indices).index_select(1, indices)
    local_cross = cross_column.index_select(0, indices)
    eye = torch.eye(len(support), dtype=gram.dtype)
    try:
        return torch.linalg.solve(local_gram + jitter * eye, local_cross)
    except RuntimeError:
        return torch.linalg.lstsq(local_gram + jitter * eye, local_cross[:, None]).solution[:, 0]


def oracle_omp_decoders(
    gram: torch.Tensor,
    cross: torch.Tensor,
    *,
    widths: tuple[int, ...],
    jitter: float,
) -> tuple[dict[int, torch.Tensor], dict[int, list[list[int]]]]:
    """Fit all true feature columns with deterministic Gram-matrix OMP.

    Ground-truth targets are explicitly allowed here: 002C is evaluator-only tomography,
    not a deployable editing API.
    """

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be square")
    if cross.ndim != 2 or cross.shape[0] != gram.shape[0]:
        raise ValueError("cross must have shape [latent_dim, num_features]")
    if not widths or min(widths) < 1:
        raise ValueError("widths must be positive")
    max_width = max(widths)
    latent_dim, num_features = cross.shape
    predictor_energy = gram.diag()
    predictor_scale = predictor_energy.clamp_min(_EPS).sqrt()
    supports: list[list[int]] = [[] for _ in range(num_features)]
    coefficients = torch.zeros_like(cross)
    residual_correlation = cross.clone()
    decoder_snapshots: dict[int, torch.Tensor] = {}
    support_snapshots: dict[int, list[list[int]]] = {}

    for step in range(1, max_width + 1):
        scores = residual_correlation.abs() / predictor_scale[:, None]
        scores[predictor_energy <= _EPS, :] = float("-inf")
        for feature, support in enumerate(supports):
            if support:
                scores[torch.tensor(support, dtype=torch.long), feature] = float("-inf")
        selected = scores.argmax(dim=0)
        next_residual = torch.empty_like(residual_correlation)
        for feature in range(num_features):
            address = int(selected[feature].item())
            if not bool(torch.isfinite(scores[address, feature])):
                raise RuntimeError("oracle OMP exhausted non-degenerate latent coordinates")
            supports[feature].append(address)
            beta = _solve_support(
                gram,
                cross[:, feature],
                supports[feature],
                jitter=jitter,
            )
            coefficients[:, feature].zero_()
            indices = torch.tensor(supports[feature], dtype=torch.long)
            coefficients[indices, feature] = beta
            next_residual[:, feature] = cross[:, feature] - gram[:, indices] @ beta
        residual_correlation = next_residual
        if step in widths:
            decoder_snapshots[step] = coefficients.clone()
            support_snapshots[step] = [list(support) for support in supports]

    return decoder_snapshots, support_snapshots


def dense_linear_decoder(
    gram: torch.Tensor,
    cross: torch.Tensor,
    *,
    ridge_lambda: float,
) -> torch.Tensor:
    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    return torch.linalg.solve(gram + ridge_lambda * eye, cross)


@torch.no_grad()
def evaluate_decoders(
    model: SparseFunctionalModel,
    world: SuperpositionWorld,
    decoders: dict[str, torch.Tensor],
    *,
    examples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, list[float]]]:
    """Evaluate oracle decoders on an independent iid probe stream."""

    num_features = model.config.num_features
    device_decoders = {
        name: decoder.to(device=device, dtype=torch.float32)
        for name, decoder in decoders.items()
    }
    accumulators: dict[str, dict[str, torch.Tensor]] = {}
    for name in device_decoders:
        accumulators[name] = {
            "positive_error": torch.zeros(num_features, device=device),
            "positive_signal": torch.zeros(num_features, device=device),
            "negative_energy": torch.zeros(num_features, device=device),
            "positive_count": torch.zeros(num_features, device=device),
            "negative_count": torch.zeros(num_features, device=device),
            "all_error": torch.zeros(num_features, device=device),
            "ratio_sum": torch.zeros(num_features, device=device),
            "ratio_square_sum": torch.zeros(num_features, device=device),
        }

    generator = torch.Generator().manual_seed(seed)
    consumed = 0
    model.eval()
    while consumed < examples:
        count = min(batch_size, examples - consumed)
        batch = world.sample_batch(count, generator=generator)
        z = model.encode(batch.x.to(device))
        s = batch.s.to(device)
        active = s.ne(0)
        inactive = ~active
        positive_count = active.sum(dim=0)
        negative_count = inactive.sum(dim=0)
        signal = s.square().sum(dim=0)
        for name, decoder in device_decoders.items():
            h = z @ decoder
            difference = h - s
            stats = accumulators[name]
            stats["positive_error"] += (difference.square() * active).sum(dim=0)
            stats["positive_signal"] += signal
            stats["negative_energy"] += (h.square() * inactive).sum(dim=0)
            stats["positive_count"] += positive_count
            stats["negative_count"] += negative_count
            stats["all_error"] += difference.square().sum(dim=0)
            safe_s = torch.where(active, s, torch.ones_like(s))
            ratio = torch.where(active, h / safe_s, torch.zeros_like(h))
            stats["ratio_sum"] += ratio.sum(dim=0)
            stats["ratio_square_sum"] += ratio.square().sum(dim=0)
        consumed += count

    output: dict[str, dict[str, list[float]]] = {}
    for name, stats in accumulators.items():
        positive_signal = stats["positive_signal"].clamp_min(_EPS)
        positive_count = stats["positive_count"].clamp_min(1.0)
        negative_count = stats["negative_count"].clamp_min(1.0)
        affected_fit_error = stats["positive_error"] / positive_signal
        positive_signal_mean = positive_signal / positive_count
        negative_energy_mean = stats["negative_energy"] / negative_count
        leakage = negative_energy_mean / positive_signal_mean.clamp_min(_EPS)
        unconditional = stats["all_error"] / positive_signal
        ratio_mean = stats["ratio_sum"] / positive_count
        ratio_variance = (
            stats["ratio_square_sum"] / positive_count - ratio_mean.square()
        ).clamp_min(0.0)
        output[name] = {
            "affected_fit_error": affected_fit_error.detach().cpu().tolist(),
            "off_support_leakage": leakage.detach().cpu().tolist(),
            "unconditional_normalized_mse": unconditional.detach().cpu().tolist(),
            "context_ratio_variance": ratio_variance.detach().cpu().tolist(),
            "positive_examples": stats["positive_count"].detach().cpu().tolist(),
        }
    return output


def _median(values: list[float]) -> float:
    return float(torch.tensor(values, dtype=torch.float64).median().item())


def _quantile(values: list[float], q: float) -> float:
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q).item())


def summarize_decoder_metrics(
    metrics: dict[str, list[float]],
    *,
    maximum_fit_error: float,
    maximum_leakage: float,
) -> dict[str, float]:
    fit = [float(value) for value in metrics["affected_fit_error"]]
    leakage = [float(value) for value in metrics["off_support_leakage"]]
    unconditional = [float(value) for value in metrics["unconditional_normalized_mse"]]
    context = [float(value) for value in metrics["context_ratio_variance"]]
    success = [
        1.0 if u <= maximum_fit_error and l <= maximum_leakage else 0.0
        for u, l in zip(fit, leakage)
    ]
    return {
        "median_affected_fit_error": _median(fit),
        "p90_affected_fit_error": _quantile(fit, 0.90),
        "median_off_support_leakage": _median(leakage),
        "p90_off_support_leakage": _quantile(leakage, 0.90),
        "median_unconditional_normalized_mse": _median(unconditional),
        "median_context_ratio_variance": _median(context),
        "joint_feature_success_fraction": float(sum(success) / len(success)),
    }


def classify_seed(
    config: CoreValidation002CConfig,
    *,
    base_normalized_mse: float,
    metrics: dict[str, dict[str, list[float]]],
    summaries: dict[str, dict[str, float]],
) -> dict[str, Any]:
    base_valid = base_normalized_mse <= config.maximum_sparse_base_normalized_mse
    width1 = summaries["sparse_r1"]
    width1_fit = width1["median_affected_fit_error"]
    candidates: list[dict[str, Any]] = []
    width1_feature_fit = torch.tensor(
        metrics["sparse_r1"]["affected_fit_error"], dtype=torch.float64
    ).clamp_min(_EPS)
    for width in config.widths:
        if width == 1:
            continue
        name = f"sparse_r{width}"
        summary = summaries[name]
        feature_fit = torch.tensor(metrics[name]["affected_fit_error"], dtype=torch.float64)
        feature_ratio = feature_fit / width1_feature_fit
        median_ratio = summary["median_affected_fit_error"] / max(width1_fit, _EPS)
        improvement_fraction = float(
            (feature_ratio <= config.maximum_relative_fit_error_vs_width1)
            .to(torch.float64)
            .mean()
            .item()
        )
        passes = bool(
            summary["median_affected_fit_error"] <= config.maximum_sparse_affected_fit_error
            and summary["median_off_support_leakage"] <= config.maximum_sparse_leakage
            and median_ratio <= config.maximum_relative_fit_error_vs_width1
            and summary["joint_feature_success_fraction"] >= config.minimum_joint_feature_success_fraction
            and improvement_fraction >= config.minimum_feature_improvement_fraction
        )
        candidates.append(
            {
                "width": width,
                "median_relative_fit_error_vs_width1": median_ratio,
                "feature_improvement_fraction": improvement_fraction,
                "pass": passes,
                **summary,
            }
        )

    best = min(candidates, key=lambda row: row["median_affected_fit_error"])
    sparse_positive = base_valid and any(bool(row["pass"]) for row in candidates)
    dense = summaries["dense_linear"]
    dense_positive = bool(
        dense["median_affected_fit_error"] <= config.maximum_sparse_affected_fit_error
        and dense["median_off_support_leakage"] <= config.maximum_sparse_leakage
        and dense["joint_feature_success_fraction"] >= config.minimum_joint_feature_success_fraction
    )
    if not base_valid:
        regime = "INVALID_BASE_REPRESENTATION"
    elif sparse_positive:
        regime = "SPARSE_LINEAR"
    elif dense_positive:
        regime = "DENSE_LINEAR_ONLY"
    else:
        regime = "NO_LOW_ERROR_LINEAR_DECODER"
    return {
        "base_quality": base_valid,
        "sparse_assembly_present": sparse_positive,
        "representation_regime": regime,
        "width1_median_affected_fit_error": width1_fit,
        "best_sparse_width": int(best["width"]),
        "best_sparse_median_affected_fit_error": float(best["median_affected_fit_error"]),
        "best_sparse_median_off_support_leakage": float(best["median_off_support_leakage"]),
        "dense_linear_reference_passes_thresholds": dense_positive,
        "candidate_widths": candidates,
        "pass": bool(sparse_positive),
    }


def run_primary_seed(
    config: CoreValidation002CConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    world, model, pretraining = pretrain_sparse_reference(
        config.base,
        seed=seed,
        device=device,
    )
    gram, cross, target_energy = accumulate_oracle_moments(
        model,
        world,
        examples=config.train_probe_examples,
        batch_size=config.probe_batch_size,
        seed=seed + 1201,
        device=device,
    )
    sparse_decoders, supports = oracle_omp_decoders(
        gram,
        cross,
        widths=config.widths,
        jitter=config.omp_jitter,
    )
    dense = dense_linear_decoder(
        gram,
        cross,
        ridge_lambda=config.dense_ridge_lambda,
    )
    decoders = {
        **{f"sparse_r{width}": sparse_decoders[width] for width in config.widths},
        "dense_linear": dense,
    }
    metrics = evaluate_decoders(
        model,
        world,
        decoders,
        examples=config.test_probe_examples,
        batch_size=config.probe_batch_size,
        seed=seed + 1601,
        device=device,
    )
    summaries = {
        name: summarize_decoder_metrics(
            values,
            maximum_fit_error=config.maximum_sparse_affected_fit_error,
            maximum_leakage=config.maximum_sparse_leakage,
        )
        for name, values in metrics.items()
    }
    gates = classify_seed(
        config,
        base_normalized_mse=float(pretraining["base_normalized_mse"]),
        metrics=metrics,
        summaries=summaries,
    )
    support_examples = {
        f"r{width}": {
            str(feature): supports[width][feature]
            for feature in range(min(16, config.base.num_features))
        }
        for width in config.widths
    }
    return {
        "seed": seed,
        "config": {
            "base": asdict(config.base),
            "widths": list(config.widths),
            "train_probe_examples": config.train_probe_examples,
            "test_probe_examples": config.test_probe_examples,
            "probe_batch_size": config.probe_batch_size,
            "omp_jitter": config.omp_jitter,
            "dense_ridge_lambda": config.dense_ridge_lambda,
        },
        "pretraining": pretraining,
        "train_target_energy": target_energy.tolist(),
        "summaries": summaries,
        "gates": gates,
        "feature_metrics": metrics,
        "support_examples_first_16_features": support_examples,
    }


def summarize_experiment(
    runs: list[dict[str, Any]],
    *,
    positive_status: str,
    negative_status: str,
    invalid_status: str,
) -> dict[str, Any]:
    if not runs:
        return {
            "status": invalid_status,
            "pass": False,
            "scientific_decision": False,
            "reason": "no runs",
            "passed_seeds": 0,
            "total_seeds": 0,
        }
    if not all(bool(run["gates"]["base_quality"]) for run in runs):
        return {
            "status": invalid_status,
            "pass": False,
            "scientific_decision": False,
            "reason": "sparse base-quality gate failed",
            "passed_seeds": sum(bool(run["gates"]["pass"]) for run in runs),
            "total_seeds": len(runs),
        }
    passed = [bool(run["gates"]["pass"]) for run in runs]
    regimes = [str(run["gates"]["representation_regime"]) for run in runs]
    return {
        "status": positive_status if all(passed) else negative_status,
        "pass": all(passed),
        "scientific_decision": True,
        "passed_seeds": sum(passed),
        "total_seeds": len(passed),
        "require_all_seeds": True,
        "representation_regimes": regimes,
    }
