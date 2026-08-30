"""Synthetic routed continual-learning world for Core Validation 003."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F

from .dependency_scoped_config import CoreValidation003Config


class RoutedContinualWorld:
    """Favorable synthetic world with shared base functions and context-local residual knowledge."""

    def __init__(self, config: CoreValidation003Config, *, seed: int):
        self.config = config
        g = torch.Generator().manual_seed(seed)

        def randn(*shape: int, scale: float) -> torch.Tensor:
            return torch.randn(*shape, generator=g) * scale

        f = config.num_function_families
        self.basis_w1 = randn(
            f, config.basis_hidden, config.content_dim,
            scale=1.0 / math.sqrt(config.content_dim),
        )
        self.basis_b1 = randn(f, config.basis_hidden, scale=0.1)
        self.basis_w2 = randn(
            f, config.output_dim, config.basis_hidden,
            scale=1.0 / math.sqrt(config.basis_hidden),
        )
        self.basis_b2 = randn(f, config.output_dim, scale=0.1)

        c = config.num_contexts
        self.residual_w1 = randn(
            c, config.residual_hidden, config.content_dim,
            scale=1.0 / math.sqrt(config.content_dim),
        )
        self.residual_b1 = randn(c, config.residual_hidden, scale=0.1)
        self.residual_w2 = randn(
            c, config.output_dim, config.residual_hidden,
            scale=1.0 / math.sqrt(config.residual_hidden),
        )
        self.residual_b2 = randn(c, config.output_dim, scale=0.1)

        combos = [(i, j) for i in range(f) for j in range(i + 1, f)]
        if not combos:
            raise ValueError("at least two function families are required")
        pairs = [combos[i % len(combos)] for i in range(c)]
        self.context_pairs = torch.tensor(pairs, dtype=torch.long)

        alpha = 0.30 + 0.40 * torch.rand(c, generator=g)
        self.context_coefficients = torch.stack([alpha, 1.0 - alpha], dim=1)

        hashes = torch.randn(c, config.router_dim, generator=g)
        self.context_hashes = F.normalize(hashes, dim=1)

        self.mutable_context_ids = torch.arange(config.anchor_contexts, c, dtype=torch.long)
        self.anchor_context_ids = torch.arange(config.anchor_contexts, dtype=torch.long)

    def initial_amplitudes(self) -> torch.Tensor:
        return torch.zeros(self.config.num_contexts, dtype=torch.float32)

    def target(
        self,
        x: torch.Tensor,
        context_ids: torch.Tensor,
        amplitudes: torch.Tensor,
    ) -> torch.Tensor:
        context_ids = context_ids.to(dtype=torch.long, device="cpu")
        x_cpu = x.to(device="cpu")
        amps = amplitudes.to(device="cpu")
        pair = self.context_pairs[context_ids]
        coeff = self.context_coefficients[context_ids]
        y = torch.zeros(x_cpu.shape[0], self.config.output_dim, dtype=x_cpu.dtype)

        for slot in range(2):
            family = pair[:, slot]
            hidden = torch.tanh(
                torch.einsum("bhd,bd->bh", self.basis_w1[family], x_cpu)
                + self.basis_b1[family]
            )
            value = (
                torch.einsum("boh,bh->bo", self.basis_w2[family], hidden)
                + self.basis_b2[family]
            )
            y += coeff[:, slot, None] * value

        hidden = torch.tanh(
            torch.einsum("bhd,bd->bh", self.residual_w1[context_ids], x_cpu)
            + self.residual_b1[context_ids]
        )
        residual = (
            torch.einsum("boh,bh->bo", self.residual_w2[context_ids], hidden)
            + self.residual_b2[context_ids]
        )
        y += amps[context_ids, None] * residual
        return y

    def sample(
        self,
        examples: int,
        *,
        generator: torch.Generator,
        amplitudes: torch.Tensor,
        context_ids: Iterable[int] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context_ids is None:
            pool = torch.arange(self.config.num_contexts, dtype=torch.long)
        else:
            values = context_ids if isinstance(context_ids, torch.Tensor) else list(context_ids)
            pool = torch.as_tensor(values).to(dtype=torch.long, device="cpu")
        indices = torch.randint(0, pool.numel(), (examples,), generator=generator)
        contexts = pool[indices]
        x = torch.randn(examples, self.config.content_dim, generator=generator)
        y = self.target(x, contexts, amplitudes)
        return x, contexts, y

    def fixed_historical_inputs(
        self, *, examples_per_context: int, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(seed)
        xs: list[torch.Tensor] = []
        contexts: list[torch.Tensor] = []
        for context in range(self.config.num_contexts):
            xs.append(torch.randn(examples_per_context, self.config.content_dim, generator=g))
            contexts.append(torch.full((examples_per_context,), context, dtype=torch.long))
        return torch.cat(xs, dim=0), torch.cat(contexts, dim=0)

    def transaction_stream(self, *, transactions: int, seed: int) -> list[int]:
        mutable = self.mutable_context_ids.tolist()
        if not mutable:
            raise ValueError("no mutable contexts")
        rng = torch.Generator().manual_seed(seed)
        out: list[int] = []
        while len(out) < transactions:
            perm = torch.randperm(len(mutable), generator=rng).tolist()
            out.extend(mutable[i] for i in perm)
        return out[:transactions]
