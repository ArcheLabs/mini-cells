"""Training and evaluation operations for Core Validation 004."""
from __future__ import annotations
import copy
from dataclasses import replace
from typing import Any
import torch
import torch.nn.functional as F
from .growth_plasticity_004_config import CoreValidation004Config, _EPS
from .growth_plasticity_004_world import GrowthPlasticityWorld
from .growth_plasticity_004_model import GrowingRoutedModel

def _to_device(batch, device: torch.device):
    return tuple(t.to(device) for t in batch)


def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).square().mean() / target.square().mean().clamp_min(_EPS)


def pretrain_model(
    config: CoreValidation004Config,
    world: GrowthPlasticityWorld,
    *,
    seed: int,
    device: torch.device,
) -> tuple[GrowingRoutedModel, dict[str, float]]:
    model = GrowingRoutedModel(config, world, seed=seed + 101).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.pretrain_learning_rate, weight_decay=config.pretrain_weight_decay
    )
    amps = world.initial_amplitudes()
    generator = torch.Generator().manual_seed(seed + 211)
    final_loss = 0.0
    model.train()
    for _ in range(config.pretrain_steps):
        batch = world.sample(config.pretrain_batch_size, generator=generator, amplitudes=amps)
        x, c, y = _to_device(batch, device)
        loss = F.mse_loss(model(x, c), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    vg = torch.Generator().manual_seed(seed + 307)
    x, c, y = _to_device(
        world.sample(config.pretrain_validation_examples, generator=vg, amplitudes=amps), device
    )
    model.eval()
    with torch.no_grad():
        base = float(normalized_mse(model(x, c), y).item())
    return model, {
        "final_pretrain_mse": final_loss,
        "base_normalized_mse": base,
        "num_base_experts": float(model.num_experts),
        "expert_hidden": float(model.expert_hidden),
        "base_total_parameters": float(model.total_parameter_count()),
        "growth_cell_parameters": float(model.growth_cell_parameter_count()),
    }


def _freeze_all(model: GrowingRoutedModel) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def train_direct_candidate(
    model: GrowingRoutedModel,
    config: CoreValidation004Config,
    world: GrowthPlasticityWorld,
    *,
    amplitudes: torch.Tensor,
    target_context: int,
    seed: int,
    device: torch.device,
) -> GrowingRoutedModel:
    candidate = copy.deepcopy(model)
    _freeze_all(candidate)
    for p in (candidate.expert_w1, candidate.expert_b1, candidate.expert_w2, candidate.expert_b2):
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(
        [candidate.expert_w1, candidate.expert_b1, candidate.expert_w2, candidate.expert_b2],
        lr=config.update_learning_rate,
    )
    g = torch.Generator().manual_seed(seed)
    batch = world.sample(
        config.update_train_examples,
        generator=g,
        amplitudes=amplitudes,
        context_ids=torch.tensor([target_context]),
    )
    x, c, y = _to_device(batch, device)
    candidate.train()
    for _ in range(config.update_steps):
        loss = F.mse_loss(candidate(x, c), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    candidate.eval()
    return candidate


def train_private_cell_candidate(
    model: GrowingRoutedModel,
    config: CoreValidation004Config,
    world: GrowthPlasticityWorld,
    *,
    amplitudes: torch.Tensor,
    target_context: int,
    seed: int,
    device: torch.device,
    spawn: bool,
) -> GrowingRoutedModel:
    candidate = copy.deepcopy(model)
    if spawn:
        cell_index = candidate.add_growth_cell(target_context=target_context, seed=seed + 17)
    else:
        existing = candidate.growth_cell_for_context(target_context)
        if existing is None:
            raise ValueError("private-cell update requested without an existing growth Cell")
        cell_index = existing
    _freeze_all(candidate)
    cell = candidate.growth_cells[cell_index]
    for p in cell.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(cell.parameters(), lr=config.growth_learning_rate)
    g = torch.Generator().manual_seed(seed)
    batch = world.sample(
        config.update_train_examples,
        generator=g,
        amplitudes=amplitudes,
        context_ids=torch.tensor([target_context]),
    )
    x, c, y = _to_device(batch, device)
    candidate.train()
    for _ in range(config.growth_steps):
        loss = F.mse_loss(candidate(x, c), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    candidate.eval()
    return candidate


def _masked_regression(before_pred, after_pred, target, mask) -> float:
    if not bool(mask.any()):
        return 0.0
    denom = target[mask].square().mean().clamp_min(_EPS)
    before = (before_pred[mask] - target[mask]).square().mean()
    after = (after_pred[mask] - target[mask]).square().mean()
    return float(((after - before) / denom).item())


@torch.no_grad()
def evaluate_candidate(
    before: GrowingRoutedModel,
    candidate: GrowingRoutedModel,
    config: CoreValidation004Config,
    world: GrowthPlasticityWorld,
    *,
    old_amplitudes: torch.Tensor,
    new_amplitudes: torch.Tensor,
    target_context: int,
    historical_x: torch.Tensor,
    historical_contexts: torch.Tensor,
    new_validation_seed: int,
    device: torch.device,
    candidate_kind: str,
) -> dict[str, Any]:
    hx = historical_x.to(device)
    hc = historical_contexts.to(device)
    before_pred, before_routes = before(hx, hc, return_routes=True)
    after_pred, _ = candidate(hx, hc, return_routes=True)
    old_target = world.target(historical_x, historical_contexts, old_amplitudes).to(device)
    global_mask = hc != target_context

    if candidate_kind == "direct":
        dummy = torch.zeros(1, config.content_dim, device=device)
        target_tensor = torch.tensor([target_context], dtype=torch.long, device=device)
        _, touched_route = before(dummy, target_tensor, return_routes=True)
        touched = torch.unique(touched_route[0]).sort().values
        overlap = (before_routes[:, :, None] == touched[None, None, :]).any(dim=(1, 2))
        local_mask = global_mask & overlap
        candidate_state_fraction = before.direct_candidate_state_fraction(int(touched.numel()))
        touched_experts = [int(x) for x in touched.tolist()]
    elif candidate_kind in {"spawn", "private"}:
        local_mask = torch.zeros_like(global_mask)
        candidate_state_fraction = candidate.growth_candidate_state_fraction()
        touched_experts = []
    else:
        raise ValueError(candidate_kind)

    outside_mask = global_mask & ~local_mask
    local_regression = _masked_regression(before_pred, after_pred, old_target, local_mask)
    global_regression = _masked_regression(before_pred, after_pred, old_target, global_mask)
    dependency_coverage = float(local_mask.float().sum().item() / max(global_mask.float().sum().item(), 1.0))
    if bool(outside_mask.any()):
        changed = (after_pred[outside_mask] - before_pred[outside_mask]).abs().amax(dim=1) > 1e-6
        structural_escape_rate = float(changed.float().mean().item())
    else:
        structural_escape_rate = 0.0

    g = torch.Generator().manual_seed(new_validation_seed)
    new_batch = world.sample(
        config.update_validation_examples,
        generator=g,
        amplitudes=new_amplitudes,
        context_ids=torch.tensor([target_context]),
    )
    nx, nc, new_y = _to_device(new_batch, device)
    old_y = world.target(new_batch[0], new_batch[1], old_amplitudes).to(device)
    before_new = before(nx, nc)
    after_new = candidate(nx, nc)
    innovation = (new_y - old_y).square().mean().clamp_min(_EPS)
    new_gain_fraction = float(
        (((before_new - new_y).square().mean() - (after_new - new_y).square().mean()) / innovation).item()
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
        "candidate_kind": candidate_kind,
        "target_context": int(target_context),
        "touched_experts": touched_experts,
        "dependency_coverage": dependency_coverage,
        "local_regression": local_regression,
        "global_regression": global_regression,
        "new_gain_fraction": new_gain_fraction,
        "local_pass": bool(local_pass),
        "oracle_pass": bool(oracle_pass),
        "false_safe": bool(local_pass and not oracle_pass),
        "structural_escape_rate": structural_escape_rate,
        "candidate_state_fraction": float(candidate_state_fraction),
    }


@torch.no_grad()
def evaluate_final_state(
    model: GrowingRoutedModel,
    config: CoreValidation004Config,
    world: GrowthPlasticityWorld,
    *,
    amplitudes: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    def score(contexts: torch.Tensor, offset: int) -> float:
        g = torch.Generator().manual_seed(seed + offset)
        x, c, y = _to_device(
            world.sample(config.pretrain_validation_examples, generator=g, amplitudes=amplitudes, context_ids=contexts),
            device,
        )
        return float(normalized_mse(model(x, c), y).item())
    return {
        "anchor_normalized_mse": score(world.anchor_context_ids, 1),
        "mutable_normalized_mse": score(world.mutable_context_ids, 2),
    }


def smoke_config(config: CoreValidation004Config) -> CoreValidation004Config:
    return replace(
        config,
        num_contexts=12,
        anchor_contexts=3,
        content_dim=8,
        model_dim=8,
        output_dim=3,
        basis_hidden=2,
        residual_hidden=2,
        router_dim=4,
        base_expert_hidden=16,
        granularity=4,
        growth_hidden=4,
        pretrain_steps=10,
        pretrain_batch_size=32,
        pretrain_validation_examples=64,
        transactions=8,
        update_train_examples=12,
        update_validation_examples=24,
        update_steps=3,
        growth_steps=6,
        historical_examples_per_context=4,
    )
