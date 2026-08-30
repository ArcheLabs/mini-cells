"""Sparse base Cells plus monotonic context-scoped growth Cells."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from .growth_plasticity_004_config import CoreValidation004Config
from .growth_plasticity_004_world import GrowthPlasticityWorld

class GrowthCell(nn.Module):
    """Context-scoped additive Cell. Zero-output initialization preserves pre-growth behavior."""

    def __init__(self, model_dim: int, hidden: int, output_dim: int, *, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w1 = torch.randn(hidden, model_dim, generator=g) / math.sqrt(model_dim)
        self.w1 = nn.Parameter(w1)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.zeros(output_dim, hidden))
        self.b2 = nn.Parameter(torch.zeros(output_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(F.linear(h, self.w1, self.b1))
        return F.linear(hidden, self.w2, self.b2)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GrowingRoutedModel(nn.Module):
    """Sparse base model plus monotonic context-scoped growth routes."""

    def __init__(self, config: CoreValidation004Config, world: GrowthPlasticityWorld, *, seed: int):
        super().__init__()
        self.config = config
        self.granularity = config.granularity
        self.num_experts = config.base_experts * config.granularity
        self.expert_hidden = config.base_expert_hidden // config.granularity
        torch.manual_seed(seed)
        self.shared = nn.Linear(config.content_dim, config.model_dim)
        self.shared_head = nn.Linear(config.model_dim, config.output_dim)
        nn.init.zeros_(self.shared_head.weight)
        nn.init.zeros_(self.shared_head.bias)
        self.expert_w1 = nn.Parameter(
            torch.randn(self.num_experts, self.expert_hidden, config.model_dim) / math.sqrt(config.model_dim)
        )
        self.expert_b1 = nn.Parameter(torch.zeros(self.num_experts, self.expert_hidden))
        self.expert_w2 = nn.Parameter(
            torch.randn(self.num_experts, config.output_dim, self.expert_hidden) / math.sqrt(self.expert_hidden)
        )
        self.expert_b2 = nn.Parameter(torch.zeros(self.num_experts, config.output_dim))
        rg = torch.Generator().manual_seed(seed + 17)
        router = F.normalize(
            torch.randn(config.num_function_families, config.granularity, config.router_dim, generator=rg), dim=2
        )
        self.register_buffer("router_vectors", router)
        self.register_buffer("context_pairs", world.context_pairs.clone())
        self.register_buffer("context_coefficients", world.context_coefficients.clone())
        self.register_buffer("context_hashes", world.context_hashes.clone())
        self.growth_cells = nn.ModuleList()
        self.growth_routes: dict[int, int] = {}

    def base_route(self, context_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context_ids = context_ids.to(dtype=torch.long, device=self.router_vectors.device)
        pair = self.context_pairs[context_ids]
        hashes = self.context_hashes[context_ids]
        ids: list[torch.Tensor] = []
        for slot in range(2):
            family = pair[:, slot]
            vectors = self.router_vectors[family]
            score = torch.einsum("bgd,bd->bg", vectors, hashes)
            replica = score.argmax(dim=1)
            ids.append(family * self.granularity + replica)
        return torch.stack(ids, dim=1), self.context_coefficients[context_ids]

    def forward(self, x: torch.Tensor, context_ids: torch.Tensor, *, return_routes: bool = False):
        h = torch.tanh(self.shared(x))
        hidden = torch.tanh(torch.einsum("bd,ehd->beh", h, self.expert_w1) + self.expert_b1)
        expert_out = torch.einsum("beh,eoh->beo", hidden, self.expert_w2) + self.expert_b2
        routes, weights = self.base_route(context_ids)
        selected = expert_out.gather(1, routes[:, :, None].expand(-1, -1, self.config.output_dim))
        out = self.shared_head(h) + (selected * weights[:, :, None]).sum(dim=1)
        # Monotonic additive route: at most one private growth Cell per context.
        for context, cell_index in self.growth_routes.items():
            mask = context_ids == context
            if bool(mask.any()):
                out[mask] = out[mask] + self.growth_cells[cell_index](h[mask])
        if return_routes:
            return out, routes
        return out

    def add_growth_cell(self, *, target_context: int, seed: int) -> int:
        if target_context in self.growth_routes:
            raise ValueError("context already owns a growth Cell")
        cell = GrowthCell(
            self.config.model_dim,
            self.config.growth_hidden,
            self.config.output_dim,
            seed=seed,
        ).to(self.expert_w1.device)
        self.growth_cells.append(cell)
        index = len(self.growth_cells) - 1
        self.growth_routes[int(target_context)] = index
        return index

    def growth_cell_for_context(self, target_context: int) -> int | None:
        return self.growth_routes.get(int(target_context))

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def expert_block_parameter_count(self) -> int:
        return (
            self.expert_hidden * self.config.model_dim
            + self.expert_hidden
            + self.config.output_dim * self.expert_hidden
            + self.config.output_dim
        )

    def growth_cell_parameter_count(self) -> int:
        return (
            self.config.growth_hidden * self.config.model_dim
            + self.config.growth_hidden
            + self.config.output_dim * self.config.growth_hidden
            + self.config.output_dim
        )

    def direct_candidate_state_fraction(self, touched_experts: int) -> float:
        return touched_experts * self.expert_block_parameter_count() / max(self.total_parameter_count(), 1)

    def growth_candidate_state_fraction(self) -> float:
        return self.growth_cell_parameter_count() / max(self.total_parameter_count(), 1)

