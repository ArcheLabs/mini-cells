"""Training and evaluation operations for Core Validation 003."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F

from .dependency_scoped_config import CoreValidation003Config, _EPS
from .dependency_scoped_model import RoutedCellModel
from .dependency_scoped_world import RoutedContinualWorld


def _to_device(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(t.to(device) for t in batch)  # type: ignore[return-value]


def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).square().mean() / target.square().mean().clamp_min(_EPS)


def pretrain_model(
    config: CoreValidation003Config,
    world: RoutedContinualWorld,
    *,
    granularity: int,
    seed: int,
    device: torch.device,
) -> tuple[RoutedCellModel, dict[str, float]]:
    model = RoutedCellModel(
        config, world, granularity=granularity, seed=seed + 101
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.pretrain_learning_rate,
        weight_decay=config.pretrain_weight_decay,
    )
    amplitudes = world.initial_amplitudes()
    generator = torch.Generator().manual_seed(seed + 211)
    final_loss = 0.0
    model.train()
    for _ in range(config.pretrain_steps):
        batch = world.sample(
            config.pretrain_batch_size,
            generator=generator,
            amplitudes=amplitudes,
        )
        x, context_ids, y = _to_device(batch, device)
        pred = model(x, context_ids)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    generator = torch.Generator().manual_seed(seed + 307)
    validation = world.sample(
        config.pretrain_validation_examples,
        generator=generator,
        amplitudes=amplitudes,
    )
    x, context_ids, y = _to_device(validation, device)
    model.eval()
    with torch.no_grad():
        validation_nrmse = float(normalized_mse(model(x, context_ids), y).item())
    return model, {
        "final_pretrain_mse": final_loss,
        "base_normalized_mse": validation_nrmse,
        "num_experts": float(model.num_experts),
        "expert_hidden": float(model.expert_hidden),
        "total_parameters": float(model.total_parameter_count()),
        "expert_block_parameters": float(model.expert_block_parameter_count()),
        "logical_active_expert_parameters": float(
            config.topk * model.expert_block_parameter_count()
        ),
    }


def _set_update_trainability(model: RoutedCellModel, *, update_shared: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(True)
    if not update_shared:
        for p in model.shared.parameters():
            p.requires_grad_(False)
        for p in model.shared_head.parameters():
            p.requires_grad_(False)


def train_candidate(
    model: RoutedCellModel,
    config: CoreValidation003Config,
    world: RoutedContinualWorld,
    *,
    amplitudes: torch.Tensor,
    target_context: int,
    seed: int,
    device: torch.device,
    update_shared: bool,
) -> RoutedCellModel:
    candidate = copy.deepcopy(model)
    _set_update_trainability(candidate, update_shared=update_shared)
    parameters = [p for p in candidate.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=config.update_learning_rate)
    generator = torch.Generator().manual_seed(seed)
    candidate.train()
    contexts = torch.tensor([target_context], dtype=torch.long)
    batch = world.sample(
        config.update_train_examples,
        generator=generator,
        amplitudes=amplitudes,
        context_ids=contexts,
    )
    x, context_ids, y = _to_device(batch, device)
    for _ in range(config.update_steps):
        pred = candidate(x, context_ids)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    candidate.eval()
    return candidate


def _masked_regression(
    before_pred: torch.Tensor,
    after_pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    if not bool(mask.any()):
        return 0.0
    denominator = target[mask].square().mean().clamp_min(_EPS)
    before = (before_pred[mask] - target[mask]).square().mean()
    after = (after_pred[mask] - target[mask]).square().mean()
    return float(((after - before) / denominator).item())


@torch.no_grad()
def evaluate_candidate(
    before: RoutedCellModel,
    candidate: RoutedCellModel,
    config: CoreValidation003Config,
    world: RoutedContinualWorld,
    *,
    old_amplitudes: torch.Tensor,
    new_amplitudes: torch.Tensor,
    target_context: int,
    historical_x: torch.Tensor,
    historical_contexts: torch.Tensor,
    new_validation_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    hx = historical_x.to(device)
    hc = historical_contexts.to(device)
    before_pred, before_routes = before(hx, hc, return_routes=True)
    after_pred, after_routes = candidate(hx, hc, return_routes=True)
    old_target = world.target(
        historical_x, historical_contexts, old_amplitudes
    ).to(device)

    target_tensor = torch.tensor([target_context], dtype=torch.long, device=device)
    dummy = torch.zeros(1, config.content_dim, device=device)
    _, touched_route = before(dummy, target_tensor, return_routes=True)
    touched = torch.unique(touched_route[0]).sort().values

    global_mask = hc != target_context
    overlap = (before_routes[:, :, None] == touched[None, None, :]).any(dim=(1, 2))
    local_mask = global_mask & overlap
    outside_mask = global_mask & ~overlap

    local_regression = _masked_regression(
        before_pred, after_pred, old_target, local_mask
    )
    global_regression = _masked_regression(
        before_pred, after_pred, old_target, global_mask
    )
    dependency_coverage = float(
        local_mask.float().sum().item() / max(global_mask.float().sum().item(), 1.0)
    )

    if bool(outside_mask.any()):
        changed = (
            (after_pred[outside_mask] - before_pred[outside_mask])
            .abs()
            .amax(dim=1)
            > 1e-6
        )
        structural_escape_rate = float(changed.float().mean().item())
        route_drift = (after_routes[outside_mask] != before_routes[outside_mask]).any(dim=1)
        routing_drift_rate = float(route_drift.float().mean().item())
    else:
        structural_escape_rate = 0.0
        routing_drift_rate = 0.0

    generator = torch.Generator().manual_seed(new_validation_seed)
    new_batch = world.sample(
        config.update_validation_examples,
        generator=generator,
        amplitudes=new_amplitudes,
        context_ids=torch.tensor([target_context], dtype=torch.long),
    )
    nx, nc, new_y = _to_device(new_batch, device)
    old_y = world.target(
        new_batch[0],
        new_batch[1],
        old_amplitudes,
    ).to(device)
    before_new = before(nx, nc)
    after_new = candidate(nx, nc)
    innovation_energy = (new_y - old_y).square().mean().clamp_min(_EPS)
    new_gain_fraction = float(
        (
            (
                (before_new - new_y).square().mean()
                - (after_new - new_y).square().mean()
            )
            / innovation_energy
        ).item()
    )

    local_pass = (
        new_gain_fraction >= config.minimum_new_gain_fraction
        and local_regression <= config.maximum_local_regression
    )
    oracle_pass = (
        new_gain_fraction >= config.minimum_new_gain_fraction
        and global_regression <= config.maximum_local_regression
    )

    return {
        "target_context": target_context,
        "touched_experts": [int(x) for x in touched.tolist()],
        "touched_expert_count": int(touched.numel()),
        "dependency_coverage": dependency_coverage,
        "local_regression": local_regression,
        "global_regression": global_regression,
        "new_gain_fraction": new_gain_fraction,
        "local_pass": bool(local_pass),
        "oracle_pass": bool(oracle_pass),
        "false_safe": bool(local_pass and not oracle_pass),
        "structural_escape_rate": structural_escape_rate,
        "routing_drift_rate": routing_drift_rate,
        "candidate_state_fraction": before.local_candidate_parameter_fraction(
            int(touched.numel())
        ),
    }


@torch.no_grad()
def evaluate_final_state(
    model: RoutedCellModel,
    config: CoreValidation003Config,
    world: RoutedContinualWorld,
    *,
    amplitudes: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    def score(contexts: torch.Tensor, offset: int) -> float:
        generator = torch.Generator().manual_seed(seed + offset)
        batch = world.sample(
            config.pretrain_validation_examples,
            generator=generator,
            amplitudes=amplitudes,
            context_ids=contexts,
        )
        x, c, y = _to_device(batch, device)
        return float(normalized_mse(model(x, c), y).item())

    return {
        "anchor_normalized_mse": score(world.anchor_context_ids, 1),
        "mutable_normalized_mse": score(world.mutable_context_ids, 2),
    }
