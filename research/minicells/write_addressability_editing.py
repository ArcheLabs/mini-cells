"""Editing helpers and metrics for Core Validation 002."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from .write_addressability import (
    Batch,
    EditTask,
    SuperpositionWorld,
    VariantName,
    WriteAddressabilityConfig,
    _EPS,
)
from .write_addressability_models import SparseFunctionalModel, normalized_mse


def _gradient_edit(
    model: nn.Module,
    edit: Batch,
    *,
    steps: int,
    learning_rate: float,
    parameters: list[nn.Parameter] | None = None,
) -> float:
    model.train()
    params = parameters if parameters is not None else list(model.parameters())
    optimizer = torch.optim.SGD(params, lr=learning_rate)
    final = float("nan")
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(edit.x)
        loss = F.mse_loss(prediction, edit.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        final = float(loss.detach().item())
    return final


def _predictions(model: nn.Module, batch: Batch) -> torch.Tensor:
    model.eval()
    return model(batch.x)


def _edit_metrics(
    model: nn.Module,
    edit: Batch,
    affected: Batch,
    invariant: Batch,
    *,
    target_feature: int,
    delta: torch.Tensor,
    pre_edit_prediction: torch.Tensor,
    pre_affected_prediction: torch.Tensor,
    pre_invariant_prediction: torch.Tensor,
) -> dict[str, float]:
    post_edit = _predictions(model, edit)
    post_affected = _predictions(model, affected)
    post_invariant = _predictions(model, invariant)
    desired_change = affected.s[:, target_feature, None] * delta[None, :]
    actual_change = post_affected - pre_affected_prediction
    signal = desired_change.square().mean().clamp_min(_EPS)
    update_error = (actual_change - desired_change).square().mean() / signal
    leakage = (post_invariant - pre_invariant_prediction).square().mean() / signal
    return {
        "edit_normalized_mse": normalized_mse(post_edit, edit.y),
        "affected_post_normalized_mse": normalized_mse(post_affected, affected.y),
        "invariant_post_normalized_mse": normalized_mse(post_invariant, invariant.y),
        "update_error": float(update_error.item()),
        "write_leakage": float(leakage.item()),
        "desired_change_energy": float(signal.item()),
        "edit_change_energy": float((post_edit - pre_edit_prediction).square().mean().item()),
    }


def _mechanistic_metrics(
    model: SparseFunctionalModel,
    invariant: Batch,
    affected: Batch,
    *,
    target_feature: int,
    address: int,
) -> dict[str, float]:
    z_invariant = model.encode(invariant.x)
    q = float(z_invariant[:, address].square().mean().item())
    target_signal = float(affected.s[:, target_feature].square().mean().item())
    proxy = q / max(target_signal, _EPS)
    return {
        "off_support_squared_activation": q,
        "target_coefficient_squared": target_signal,
        "leakage_proxy": proxy,
    }


def _retention_mse(
    model: nn.Module,
    retained_s: list[torch.Tensor],
    world: SuperpositionWorld,
    device: torch.device,
) -> float | None:
    if not retained_s:
        return None
    s = torch.cat(retained_s, dim=0)
    x = s @ world.A.transpose(0, 1)
    target = world.current_targets(s)
    return normalized_mse(model(x.to(device)), target.to(device))


def make_edit_schedule(
    config: WriteAddressabilityConfig,
    *,
    seed: int,
) -> list[EditTask]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(config.num_features, generator=generator).tolist()
    used: list[int] = []
    next_new = 0
    tasks: list[EditTask] = []
    previous_target: int | None = None
    for index in range(config.edit_count):
        should_repeat = (
            config.repeat_every > 0
            and index > 0
            and (index + 1) % config.repeat_every == 0
            and used
        )
        if should_repeat:
            target = used[(index // max(1, config.repeat_every)) % len(used)]
        else:
            if next_new >= len(order):
                target = order[index % len(order)]
            else:
                target = int(order[next_new])
                next_new += 1
                used.append(target)
        delta = torch.randn(config.output_dim, generator=generator)
        delta = F.normalize(delta, dim=0) * config.edit_scale
        forced = None
        if (
            config.previous_target_distractor_every > 0
            and previous_target is not None
            and previous_target != target
            and (index + 1) % config.previous_target_distractor_every == 0
        ):
            forced = previous_target
        tasks.append(
            EditTask(
                index=index, target_feature=target, delta=delta, forced_distractor=forced
            )
        )
        previous_target = target
    return tasks


def _cyclic_permutation(size: int, *, seed: int) -> torch.Tensor:
    if size < 2:
        raise ValueError("permutation control requires latent_dim >= 2")
    generator = torch.Generator().manual_seed(seed)
    offset = int(torch.randint(1, size, (1,), generator=generator).item())
    return (torch.arange(size) + offset) % size


def _variant_models(
    pretrained: dict[str, nn.Module],
    config: WriteAddressabilityConfig,
    *,
    device: torch.device,
) -> dict[VariantName, nn.Module]:
    sparse = pretrained["sparse"]
    variants: dict[VariantName, nn.Module] = {
        "inferred_address": copy.deepcopy(sparse),
        "oracle_address": copy.deepcopy(sparse),
        "permuted_address": copy.deepcopy(sparse),
        "global_write": copy.deepcopy(sparse),
        "dense": copy.deepcopy(pretrained["dense"]),
        "moe": copy.deepcopy(pretrained["moe"]),
    }
    for model in variants.values():
        model.to(device)
    for name in ("inferred_address", "oracle_address", "permuted_address", "global_write"):
        model = variants[name]
        assert isinstance(model, SparseFunctionalModel)
        model.encoder.requires_grad_(False)
        model.reconstructor.requires_grad_(False)
    return variants


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    x = torch.tensor(left, dtype=torch.float64)
    y = torch.tensor(right, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(x.square().sum() * y.square().sum())
    if float(denominator.item()) <= _EPS:
        return None
    return float((x * y).sum().div(denominator).item())


def _log_log_slope(proxy: list[float], leakage: list[float]) -> float | None:
    if len(proxy) < 3 or len(proxy) != len(leakage):
        return None
    x = torch.log10(torch.tensor(proxy, dtype=torch.float64).clamp_min(1e-12))
    y = torch.log10(torch.tensor(leakage, dtype=torch.float64).clamp_min(1e-12))
    x_centered = x - x.mean()
    denominator = x_centered.square().sum()
    if float(denominator.item()) <= _EPS:
        return None
    return float((x_centered * (y - y.mean())).sum().div(denominator).item())


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(torch.tensor(values, dtype=torch.float64).median().item())
