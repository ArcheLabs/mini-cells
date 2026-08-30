"""Models, pretraining, and non-oracle address inference for Core Validation 002."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .write_addressability import Batch, SuperpositionWorld, WriteAddressabilityConfig, _EPS

class SparseFunctionalModel(nn.Module):
    """Learned overcomplete sparse functional code with an aligned writer."""

    def __init__(self, config: WriteAddressabilityConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.observation_dim, config.latent_dim, bias=False)
        self.reconstructor = nn.Linear(config.latent_dim, config.observation_dim, bias=False)
        self.writer = nn.Linear(config.latent_dim, config.output_dim, bias=False)
        nn.init.normal_(self.encoder.weight, std=1.0 / math.sqrt(config.observation_dim))
        nn.init.normal_(self.reconstructor.weight, std=1.0 / math.sqrt(config.latent_dim))
        nn.init.normal_(self.writer.weight, std=1.0 / math.sqrt(config.latent_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.encoder(x)
        k = min(self.config.latent_topk, raw.shape[-1])
        indices = raw.abs().topk(k, dim=-1).indices
        mask = torch.zeros_like(raw).scatter_(1, indices, 1.0)
        return raw * mask

    def forward_with_latent(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.writer(z), self.reconstructor(z), z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.writer(self.encode(x))


class DenseMLP(nn.Module):
    def __init__(self, observation_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StandardMoE(nn.Module):
    """Top-k routed independent linear experts used as a contextual baseline."""

    def __init__(self, observation_dim: int, output_dim: int, num_experts: int, topk: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)
        self.router = nn.Linear(observation_dim, num_experts)
        self.expert_weight = nn.Parameter(
            torch.empty(num_experts, output_dim, observation_dim)
        )
        self.expert_bias = nn.Parameter(torch.zeros(num_experts, output_dim))
        nn.init.kaiming_uniform_(self.expert_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        selected = logits.topk(self.topk, dim=-1)
        gates = F.softmax(selected.values, dim=-1)
        weights = self.expert_weight[selected.indices]
        bias = self.expert_bias[selected.indices]
        outputs = torch.einsum("bd,bkod->bko", x, weights) + bias
        return (gates.unsqueeze(-1) * outputs).sum(dim=1)


@torch.no_grad()
def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _dense_hidden_for_budget(config: WriteAddressabilityConfig, budget: int) -> int:
    # d*h+h + h*m+m ~= budget
    denominator = config.observation_dim + config.output_dim + 1
    return max(1, int(round((budget - config.output_dim) / denominator)))


def _moe_experts_for_budget(config: WriteAddressabilityConfig, budget: int) -> int:
    # Per expert: expert matrix+bias plus one router row+bias.
    per_expert = (
        config.output_dim * config.observation_dim
        + config.output_dim
        + config.observation_dim
        + 1
    )
    return max(config.moe_topk, int(round(budget / per_expert)))


def build_models(config: WriteAddressabilityConfig, *, seed: int) -> dict[str, nn.Module]:
    torch.manual_seed(seed)
    sparse = SparseFunctionalModel(config)
    sparse_budget = parameter_count(sparse)
    dense = DenseMLP(
        config.observation_dim,
        config.output_dim,
        _dense_hidden_for_budget(config, sparse_budget),
    )
    moe = StandardMoE(
        config.observation_dim,
        config.output_dim,
        _moe_experts_for_budget(config, sparse_budget),
        config.moe_topk,
    )
    return {"sparse": sparse, "dense": dense, "moe": moe}


def normalized_mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = F.mse_loss(prediction, target)
    denominator = target.square().mean().clamp_min(_EPS)
    return float((numerator / denominator).item())


def _train_stream(
    model: nn.Module,
    training: Batch,
    config: WriteAddressabilityConfig,
    *,
    seed: int,
    device: torch.device,
    sparse_objective: bool,
) -> list[float]:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.pretrain_learning_rate,
        weight_decay=config.pretrain_weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    for step in range(config.pretrain_steps):
        indices = torch.randint(
            0, len(training.x), (config.pretrain_batch_size,), generator=generator
        )
        x = training.x[indices].to(device)
        y = training.y[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        if sparse_objective:
            assert isinstance(model, SparseFunctionalModel)
            prediction, reconstruction, _ = model.forward_with_latent(x)
            output_loss = F.mse_loss(prediction, y)
            reconstruction_loss = F.mse_loss(reconstruction, x)
            loss = output_loss + config.reconstruction_weight * reconstruction_loss
        else:
            loss = F.mse_loss(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, config.pretrain_steps // 20) == 0:
            losses.append(float(loss.detach().item()))
    return losses


@torch.no_grad()
def _base_validation(
    model: nn.Module,
    world: SuperpositionWorld,
    config: WriteAddressabilityConfig,
    *,
    seed: int,
    device: torch.device,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    batch = world.sample_batch(config.validation_examples, generator=generator).to(device)
    model.eval()
    return normalized_mse(model(batch.x), batch.y)


def pretrain_models(
    config: WriteAddressabilityConfig,
    *,
    seed: int,
    device: torch.device,
) -> tuple[SuperpositionWorld, dict[str, nn.Module], dict[str, Any]]:
    world = SuperpositionWorld(config, seed=seed + 101)
    models = build_models(config, seed=seed + 211)
    pretrain_generator = torch.Generator().manual_seed(seed + 293)
    training = world.sample_batch(config.pretrain_examples, generator=pretrain_generator)
    histories: dict[str, list[float]] = {}
    histories["sparse"] = _train_stream(
        models["sparse"], training, config, seed=seed + 307, device=device, sparse_objective=True
    )
    histories["dense"] = _train_stream(
        models["dense"], training, config, seed=seed + 307, device=device, sparse_objective=False
    )
    histories["moe"] = _train_stream(
        models["moe"], training, config, seed=seed + 307, device=device, sparse_objective=False
    )
    validation = {
        name: _base_validation(
            model,
            world,
            config,
            seed=seed + 401,
            device=device,
        )
        for name, model in models.items()
    }
    return world, models, {"loss_history": histories, "base_normalized_mse": validation}


@torch.no_grad()
def latent_feature_correlations(
    model: SparseFunctionalModel,
    world: SuperpositionWorld,
    config: WriteAddressabilityConfig,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    batch = world.sample_batch(config.oracle_probe_examples, generator=generator)
    z = model.encode(batch.x.to(device)).detach().cpu()
    s = batch.s
    z = z - z.mean(dim=0, keepdim=True)
    s = s - s.mean(dim=0, keepdim=True)
    numerator = z.transpose(0, 1) @ s
    denominator = torch.sqrt(
        z.square().sum(dim=0)[:, None] * s.square().sum(dim=0)[None, :]
    ).clamp_min(_EPS)
    return (numerator / denominator).abs()


@torch.no_grad()
def infer_write_address(
    model: SparseFunctionalModel,
    edit_x: torch.Tensor,
    edit_y: torch.Tensor,
    config: WriteAddressabilityConfig,
) -> dict[str, Any]:
    """Infer a shared latent coordinate from edit residuals only.

    No ground-truth sparse code or feature identity is accepted by this API.
    """

    model.eval()
    z = model.encode(edit_x)
    residual = edit_y - model(edit_x)
    energy = z.square().sum(dim=0)
    active_fraction = z.ne(0).float().mean(dim=0)
    denominator = energy.clamp_min(_EPS)
    delta = (z.transpose(0, 1) @ residual) / denominator[:, None]
    reconstruction = z[:, :, None] * delta[None, :, :]
    score = (residual[:, None, :] - reconstruction).square().mean(dim=(0, 2))
    valid = (energy >= config.address_min_energy) & (
        active_fraction >= config.address_min_shared_fraction
    )
    if not bool(valid.any()):
        # Preserve the no-oracle rule while making failure explicit and deterministic.
        valid = energy >= config.address_min_energy
    masked_score = score.masked_fill(~valid, float("inf"))
    address = int(masked_score.argmin().item())
    return {
        "address": address,
        "delta": delta[address].detach().clone(),
        "score": float(score[address].item()),
        "energy": float(energy[address].item()),
        "active_fraction": float(active_fraction[address].item()),
        "valid_candidates": int(valid.sum().item()),
    }


@torch.no_grad()
def least_squares_delta_for_address(
    model: SparseFunctionalModel,
    edit_x: torch.Tensor,
    edit_y: torch.Tensor,
    address: int,
) -> torch.Tensor:
    z = model.encode(edit_x)[:, address]
    residual = edit_y - model(edit_x)
    denominator = z.square().sum().clamp_min(_EPS)
    return (z[:, None] * residual).sum(dim=0) / denominator


@torch.no_grad()
def apply_addressed_write(
    model: SparseFunctionalModel,
    *,
    address: int,
    delta: torch.Tensor,
    destination: int | None = None,
) -> int:
    destination = address if destination is None else destination
    model.writer.weight[:, destination].add_(delta)
    return int(destination)

