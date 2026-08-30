"""Sparse functional write assemblies for Core Validation 002B."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .write_addressability import Batch, WriteAddressabilityConfig, _EPS
from .write_addressability_models import SparseFunctionalModel


@dataclass(frozen=True)
class CoreValidation002BConfig:
    base: WriteAddressabilityConfig
    address_widths: tuple[int, ...]
    omp_als_steps: int
    global_ridge_lambda: float
    global_scales: tuple[float, ...]
    maximum_sparse_base_normalized_mse: float
    maximum_oracle_update_error: float
    maximum_oracle_write_leakage: float
    maximum_best_width_update_error: float
    maximum_relative_update_error_vs_width1: float
    maximum_absolute_write_leakage: float
    maximum_matched_u_gap: float
    maximum_leakage_ratio_vs_matched_global: float
    maximum_repeated_update_error: float
    maximum_repeated_write_leakage: float
    maximum_assembly_fit_error: float

    @classmethod
    def from_protocol(cls, path: str | Path) -> "CoreValidation002BConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        world = payload["world"]
        model = payload["model"]
        pretrain = payload["pretraining"]
        editing = payload["editing"]
        evaluation = payload["evaluation"]
        sparse_address = payload["sparse_address"]
        matched_global = payload["matched_global"]
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
            edit_count=int(editing["edit_count"]),
            edit_examples=int(editing["edit_examples"]),
            affected_examples=int(evaluation["affected_examples"]),
            invariant_examples=int(evaluation["invariant_examples"]),
            retention_examples_per_edit=int(evaluation["retention_examples_per_edit"]),
            repeat_every=int(editing["repeat_every"]),
            previous_target_distractor_every=int(editing["previous_target_distractor_every"]),
            address_min_shared_fraction=float(editing["address_min_shared_fraction"]),
            address_min_energy=float(editing["address_min_energy"]),
            global_edit_steps=int(editing["legacy_global_write"]["steps"]),
            global_edit_learning_rate=float(editing["legacy_global_write"]["learning_rate"]),
            dense_edit_steps=int(editing["dense"]["steps"]),
            dense_edit_learning_rate=float(editing["dense"]["learning_rate"]),
            moe_edit_steps=int(editing["moe"]["steps"]),
            moe_edit_learning_rate=float(editing["moe"]["learning_rate"]),
            moe_topk=int(model["moe_topk"]),
            oracle_probe_examples=int(evaluation["oracle_probe_examples"]),
        )
        widths = tuple(int(v) for v in sparse_address["widths"])
        if not widths or widths[0] != 1 or any(v < 1 for v in widths):
            raise ValueError("002B sparse-address widths must begin at 1 and be positive")
        return cls(
            base=base,
            address_widths=widths,
            omp_als_steps=int(sparse_address["als_refinement_steps"]),
            global_ridge_lambda=float(matched_global["ridge_lambda"]),
            global_scales=tuple(float(v) for v in matched_global["scales"]),
            maximum_sparse_base_normalized_mse=float(gates["maximum_sparse_base_normalized_mse"]),
            maximum_oracle_update_error=float(gates["maximum_oracle_update_error"]),
            maximum_oracle_write_leakage=float(gates["maximum_oracle_write_leakage"]),
            maximum_best_width_update_error=float(gates["maximum_best_width_update_error"]),
            maximum_relative_update_error_vs_width1=float(gates["maximum_relative_update_error_vs_width1"]),
            maximum_absolute_write_leakage=float(gates["maximum_absolute_write_leakage"]),
            maximum_matched_u_gap=float(gates["maximum_matched_u_gap"]),
            maximum_leakage_ratio_vs_matched_global=float(gates["maximum_leakage_ratio_vs_matched_global"]),
            maximum_repeated_update_error=float(gates["maximum_repeated_update_error"]),
            maximum_repeated_write_leakage=float(gates["maximum_repeated_write_leakage"]),
            maximum_assembly_fit_error=float(gates["maximum_assembly_fit_error"]),
        )


def _safe_lstsq(design: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(design, target[:, None]).solution[:, 0]


@torch.no_grad()
def infer_sparse_write_address(
    model: SparseFunctionalModel,
    edit_x: torch.Tensor,
    edit_y: torch.Tensor,
    *,
    width: int,
    als_steps: int = 3,
    minimum_energy: float = 1e-8,
) -> dict[str, Any]:
    """Infer a rank-1 sparse write assembly from edit residuals only.

    The editor receives only x/y edit examples. It first extracts the dominant
    residual direction, then uses deterministic OMP in latent space, followed by
    alternating least-squares refinement on the selected support.
    """

    if width < 1:
        raise ValueError("width must be positive")
    model.eval()
    z = model.encode(edit_x)
    residual = edit_y - model(edit_x)
    if z.shape[0] < 2:
        raise ValueError("sparse write inference requires at least two edit examples")

    u, singular_values, _ = torch.linalg.svd(residual, full_matrices=False)
    scalar_target = u[:, 0] * singular_values[0]
    energy = z.square().sum(dim=0)
    selected: list[int] = []
    scalar_residual = scalar_target.clone()
    coefficients = torch.empty(0, device=z.device, dtype=z.dtype)

    for _ in range(min(width, z.shape[1])):
        correlation = (z.transpose(0, 1) @ scalar_residual).abs()
        score = correlation / energy.clamp_min(_EPS).sqrt()
        score = score.masked_fill(energy < minimum_energy, float("-inf"))
        if selected:
            score[torch.tensor(selected, device=z.device, dtype=torch.long)] = float("-inf")
        address = int(score.argmax().item())
        if not bool(torch.isfinite(score[address])):
            break
        selected.append(address)
        design = z[:, selected]
        coefficients = _safe_lstsq(design, scalar_target)
        scalar_residual = scalar_target - design @ coefficients
        if float(scalar_residual.square().mean().item()) <= 1e-14:
            break

    if not selected:
        raise RuntimeError("no non-degenerate sparse write address was found")

    design = z[:, selected]
    for _ in range(max(0, als_steps)):
        h = design @ coefficients
        delta = (h[:, None] * residual).sum(dim=0) / h.square().sum().clamp_min(_EPS)
        scalar_target = (residual @ delta) / delta.square().sum().clamp_min(_EPS)
        coefficients = _safe_lstsq(design, scalar_target)

    norm = coefficients.norm().clamp_min(_EPS)
    coefficients = coefficients / norm
    h = design @ coefficients
    delta = (h[:, None] * residual).sum(dim=0) / h.square().sum().clamp_min(_EPS)
    rank1_residual = residual - h[:, None] * delta[None, :]
    weights = torch.zeros(z.shape[1], device=z.device, dtype=z.dtype)
    weights[torch.tensor(selected, device=z.device, dtype=torch.long)] = coefficients

    return {
        "support": selected,
        "weights": weights.detach().clone(),
        "delta": delta.detach().clone(),
        "support_size": len(selected),
        "rank1_residual_mse": float(rank1_residual.square().mean().item()),
        "assembly_energy": float(h.square().sum().item()),
    }


@torch.no_grad()
def apply_sparse_write(
    model: SparseFunctionalModel,
    *,
    weights: torch.Tensor,
    delta: torch.Tensor,
) -> float:
    if weights.ndim != 1 or weights.numel() != model.config.latent_dim:
        raise ValueError("weights must be a latent_dim vector")
    update = delta[:, None] * weights[None, :]
    model.writer.weight.add_(update)
    return float(update.norm().item())


@torch.no_grad()
def apply_global_ridge_write(
    model: SparseFunctionalModel,
    edit_x: torch.Tensor,
    edit_y: torch.Tensor,
    *,
    ridge_lambda: float,
    scale: float,
) -> float:
    """Apply a deterministic full-writer ridge update from the same edit examples."""

    model.eval()
    z = model.encode(edit_x)
    residual = edit_y - model(edit_x)
    eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
    dual = torch.linalg.solve(z @ z.transpose(0, 1) + ridge_lambda * eye, residual)
    update = scale * (dual.transpose(0, 1) @ z)
    model.writer.weight.add_(update)
    return float(update.norm().item())


@torch.no_grad()
def assembly_geometry(
    model: SparseFunctionalModel,
    affected: Batch,
    invariant: Batch,
    *,
    target_feature: int,
    weights: torch.Tensor,
) -> dict[str, float]:
    """Evaluator-only geometry diagnostics. Never use these values for editing."""

    h_affected = model.encode(affected.x) @ weights
    h_invariant = model.encode(invariant.x) @ weights
    target = affected.s[:, target_feature]
    beta = (h_affected * target).sum() / h_affected.square().sum().clamp_min(_EPS)
    fitted = beta * h_affected
    signal = target.square().mean().clamp_min(_EPS)
    fit_error = (fitted - target).square().mean() / signal
    off_support_ratio = h_invariant.square().mean() / h_affected.square().mean().clamp_min(_EPS)
    # Affected batches always contain the target and coefficients are bounded away from zero.
    ratio = fitted / target
    centered_h = h_affected - h_affected.mean()
    centered_target = target - target.mean()
    corr_den = torch.sqrt(centered_h.square().sum() * centered_target.square().sum()).clamp_min(_EPS)
    correlation = (centered_h * centered_target).sum().abs() / corr_den
    return {
        "assembly_fit_error": float(fit_error.item()),
        "assembly_off_support_energy_ratio": float(off_support_ratio.item()),
        "assembly_context_ratio_variance": float(ratio.var(unbiased=False).item()),
        "assembly_target_correlation": float(correlation.item()),
    }
