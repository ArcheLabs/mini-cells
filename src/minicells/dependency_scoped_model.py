"""Sparse routed cell model for Core Validation 003."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .dependency_scoped_config import CoreValidation003Config
from .dependency_scoped_world import RoutedContinualWorld


class RoutedCellModel(nn.Module):
    """Structured sparse model whose expert replicas are independently mutable state blocks."""

    def __init__(
        self,
        config: CoreValidation003Config,
        world: RoutedContinualWorld,
        *,
        granularity: int,
        seed: int,
    ):
        super().__init__()
        if granularity not in config.granularities:
            raise ValueError(f"unexpected granularity: {granularity}")
        if config.base_expert_hidden % granularity != 0:
            raise ValueError("granularity must divide base_expert_hidden")

        self.config = config
        self.granularity = granularity
        self.num_experts = config.base_experts * granularity
        self.expert_hidden = config.base_expert_hidden // granularity

        torch.manual_seed(seed)
        self.shared = nn.Linear(config.content_dim, config.model_dim)
        self.shared_head = nn.Linear(config.model_dim, config.output_dim)
        nn.init.zeros_(self.shared_head.weight)
        nn.init.zeros_(self.shared_head.bias)

        self.expert_w1 = nn.Parameter(
            torch.randn(self.num_experts, self.expert_hidden, config.model_dim)
            / math.sqrt(config.model_dim)
        )
        self.expert_b1 = nn.Parameter(torch.zeros(self.num_experts, self.expert_hidden))
        self.expert_w2 = nn.Parameter(
            torch.randn(self.num_experts, config.output_dim, self.expert_hidden)
            / math.sqrt(self.expert_hidden)
        )
        self.expert_b2 = nn.Parameter(torch.zeros(self.num_experts, config.output_dim))

        router_generator = torch.Generator().manual_seed(seed + 17)
        router = torch.randn(
            config.num_function_families,
            granularity,
            config.router_dim,
            generator=router_generator,
        )
        router = F.normalize(router, dim=2)
        self.register_buffer("router_vectors", router)
        self.register_buffer("context_pairs", world.context_pairs.clone())
        self.register_buffer("context_coefficients", world.context_coefficients.clone())
        self.register_buffer("context_hashes", world.context_hashes.clone())

    def route(self, context_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context_ids = context_ids.to(dtype=torch.long, device=self.router_vectors.device)
        pair = self.context_pairs[context_ids]
        hashes = self.context_hashes[context_ids]
        expert_ids: list[torch.Tensor] = []
        for slot in range(2):
            family = pair[:, slot]
            vectors = self.router_vectors[family]
            score = torch.einsum("bgd,bd->bg", vectors, hashes)
            replica = score.argmax(dim=1)
            expert_ids.append(family * self.granularity + replica)
        return torch.stack(expert_ids, dim=1), self.context_coefficients[context_ids]

    def forward(
        self,
        x: torch.Tensor,
        context_ids: torch.Tensor,
        *,
        return_routes: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.shared(x))
        hidden = torch.tanh(
            torch.einsum("bd,ehd->beh", h, self.expert_w1) + self.expert_b1
        )
        expert_out = (
            torch.einsum("beh,eoh->beo", hidden, self.expert_w2) + self.expert_b2
        )
        routes, weights = self.route(context_ids)
        selected = expert_out.gather(
            1,
            routes[:, :, None].expand(-1, -1, self.config.output_dim),
        )
        out = self.shared_head(h) + (selected * weights[:, :, None]).sum(dim=1)
        if return_routes:
            return out, routes
        return out

    @torch.no_grad()
    def perturb_router(self, *, seed: int, noise_scale: float) -> None:
        if noise_scale <= 0:
            return
        g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(self.router_vectors.shape, generator=g).to(self.router_vectors.device)
        self.router_vectors.copy_(
            F.normalize(self.router_vectors + noise_scale * noise, dim=2)
        )

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def expert_block_parameter_count(self) -> int:
        return (
            self.expert_hidden * self.config.model_dim
            + self.expert_hidden
            + self.config.output_dim * self.expert_hidden
            + self.config.output_dim
        )

    def local_candidate_parameter_fraction(self, touched_experts: int) -> float:
        return (
            touched_experts * self.expert_block_parameter_count()
            / max(self.total_parameter_count(), 1)
        )
