"""Experiment 005 causal LM arms; shared FFN/router components are reused."""
from __future__ import annotations

from typing import Any
import hashlib

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.common.models import FeedForward, GlobalRouter

from .cell import ResidualCell
from .routing import StateAwareRouter
from .topology import make_initial_graph


ARMS = ("M0", "P0", "S0", "R0", "G0")


class ArmFFN(nn.Module):
    def __init__(self, config: dict[str, Any], arm: str, seed: int, layer_index: int):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown Experiment 005 arm {arm!r}")
        self.arm = arm
        self.config = config
        self.layer_index = layer_index
        width = int(config["width"])
        self.cell_count = int(config["cell_count"])
        self.moe: GlobalRouter | None = None
        self.parallel_router: nn.Linear | None = None
        self.router: StateAwareRouter | None = None
        self.collect_trace = False
        self.last_trace: dict[str, Tensor] | None = None
        self.last_backup_cell_visits = 0

        if arm == "M0":
            self.moe = GlobalRouter(width, int(config["moe_hidden"]),
                                    int(config["moe_experts"]), top_k=2)
            self.router = None
        else:
            self._cells = nn.ModuleList(
                ResidualCell(width, int(config["cell_hidden"]),
                             float(config["residual_scale"]))
                for _ in range(self.cell_count)
            )
            if arm == "P0":
                self.parallel_router = nn.Linear(width, self.cell_count, bias=False)
            else:
                self.router = StateAwareRouter(width, self.cell_count,
                                               int(config["router_width"]))
                if arm in {"R0", "G0"}:
                    graph = make_initial_graph(
                        self.cell_count, int(config["graph_out_degree"]), seed + layer_index
                    )
                    self.register_buffer("adjacency", graph)

    @property
    def cells(self) -> nn.ModuleList:
        if self.arm == "M0":
            assert self.moe is not None
            return self.moe.cells
        return self._cells

    def _cell(self, flat_state: Tensor, ids: Tensor, norm: nn.LayerNorm) -> Tensor:
        normalized = norm(flat_state)
        output = torch.zeros_like(flat_state)
        for index, cell in enumerate(self.cells):
            rows = torch.where(ids == index)[0]
            if rows.numel():
                value = cell.compute(normalized.index_select(0, rows))
                # Autocast linear outputs can be Half/BFloat16 while residual state is Float.
                # Keep dispatch accumulation in the residual dtype; casts preserve gradients.
                output = output.index_copy(0, rows, value.to(output.dtype))
        return output

    def forward(self, x: Tensor, norm: nn.LayerNorm,
                forced_transition: tuple[int, int, int, int] | None = None
                ) -> tuple[Tensor, Tensor]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        zero = x.new_zeros(())
        if self.arm == "M0":
            assert self.moe is not None
            captured: dict[str, Tensor] = {}
            hook = self.moe.router.register_forward_hook(
                lambda _module, _inputs, output: captured.__setitem__("logits", output)
            )
            try:
                delta, _ = self.moe(norm(flat))
            finally:
                hook.remove()
            probabilities = captured["logits"].float().softmax(-1)
            auxiliary = probabilities.shape[-1] * probabilities.mean(0).square().sum()
            delta = delta.reshape(shape)
            if self.collect_trace:
                ids = captured["logits"].float().topk(2, dim=-1).indices.reshape(*shape[:-1], 2)
                self.last_trace = {"paths": ids, "candidates": ids, "backup_ids": ids}
            return delta, auxiliary

        if self.arm == "P0":
            assert self.parallel_router is not None
            probabilities = self.parallel_router(norm(flat)).float().softmax(-1)
            weights, ids = probabilities.topk(4, dim=-1)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
            values = []
            for slot in range(4):
                values.append(self._cell(flat, ids[:, slot], norm))
            stacked = torch.stack(values, dim=1)
            delta = (stacked * weights.to(stacked.dtype).unsqueeze(-1)).sum(1)
            if self.collect_trace:
                self.last_trace = {"paths": ids.reshape(*shape[:-1], 4),
                                   "candidates": ids.reshape(*shape[:-1], 4),
                                   "backup_ids": ids.reshape(*shape[:-1], 4)}
            return delta.reshape(shape), self.cell_count * probabilities.mean(0).square().sum()

        assert self.router is not None
        state = flat
        previous = torch.full((flat.shape[0],), self.cell_count, dtype=torch.long,
                              device=flat.device)
        path_rows: list[Tensor] = []
        candidate_rows: list[Tensor] = []
        backup_rows: list[Tensor] = []
        auxiliary = zero
        visits = int(self.config["cell_visits"])
        adjacency = self.adjacency if self.arm in {"R0", "G0"} else None
        backup_k = 2 if self.training and bool(self.config.get("top2_backup_gradient", True)) else 1
        self.last_backup_cell_visits = 0

        for step in range(visits):
            route = self.router(state, previous, adjacency)
            k = min(backup_k, route.logits.shape[-1])
            top_logits, top_slots = route.logits.topk(k, dim=-1)
            top_ids = route.candidate_ids.gather(1, top_slots)
            probabilities = route.logits.softmax(-1)
            importance = torch.zeros(self.cell_count, device=flat.device, dtype=probabilities.dtype)
            importance.scatter_add_(0, route.candidate_ids.reshape(-1),
                                    probabilities.reshape(-1) / flat.shape[0])
            auxiliary = auxiliary + self.cell_count * importance.square().sum()
            chosen = top_ids[:, 0]
            if forced_transition is not None:
                transitions = (forced_transition if isinstance(forced_transition, list)
                               else [forced_transition])
                for layer, forced_step, source, target in transitions:
                    if layer == self.layer_index and forced_step == step:
                        mask = torch.ones_like(previous, dtype=torch.bool) if source == -1 else previous == source
                        chosen = torch.where(mask, torch.full_like(chosen, target), chosen)
            hard_value = self._cell(state, chosen, norm)
            if k == 2:
                backup_value = self._cell(state, top_ids[:, 1], norm)
                backup_values = torch.stack((hard_value, backup_value), dim=1)
                weights = top_logits.softmax(-1).to(backup_values.dtype)
                surrogate = (backup_values * weights.unsqueeze(-1)).sum(1)
                hard_st = surrogate + (hard_value - surrogate).detach()
                # A forced edge intervention measures the actual forced forward;
                # the backup gradient still follows the unforced candidate policy.
                hard_st = torch.where((chosen != top_ids[:, 0])[:, None], hard_value, hard_st)
                self.last_backup_cell_visits += flat.shape[0]
            else:
                hard_st = hard_value
            state = state + self.cells[0].residual_scale.to(state.dtype) * hard_st
            path_rows.append(chosen)
            candidate_rows.append(route.candidate_ids)
            backup_rows.append(top_ids)
            previous = chosen

        if self.collect_trace:
            self.last_trace = {
                "paths": torch.stack(path_rows, dim=-1).reshape(*shape[:-1], visits),
                "candidates": torch.stack(candidate_rows, dim=1).reshape(
                    *shape[:-1], visits, -1
                ),
                "backup_ids": torch.stack(backup_rows, dim=1).reshape(
                    *shape[:-1], visits, -1
                ),
            }
        return (state - flat).reshape(shape), auxiliary / max(1, visits)


class CausalBlock(nn.Module):
    def __init__(self, config: dict[str, Any], arm: str, seed: int, layer_index: int):
        super().__init__()
        width = int(config["width"])
        self.heads = int(config["heads"])
        self.norm_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn = ArmFFN(config, arm, seed, layer_index)

    def forward(self, x: Tensor, past: tuple[Tensor, Tensor] | None = None,
                use_cache: bool = False,
                forced_transition: tuple[int, int, int, int] | None = None
                ) -> tuple[Tensor, tuple[Tensor, Tensor] | None, Tensor]:
        batch, length, width = x.shape
        qkv = self.qkv(self.norm_attention(x)).reshape(
            batch, length, 3, self.heads, width // self.heads
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if past is not None:
            if length != 1:
                raise ValueError("cached incremental execution accepts one new token at a time")
            key = torch.cat((past[0], key), dim=-2)
            value = torch.cat((past[1], value), dim=-2)
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=(past is None)
        )
        try:
            from .accounting import record_attention_flops
            record_attention_flops(query.shape, key.shape[-2])
        except ImportError:
            pass
        x = x + self.projection(attended.transpose(1, 2).reshape(batch, length, width))
        delta, auxiliary = self.ffn(x, self.norm_ffn, forced_transition)
        new_cache = (key, value) if use_cache else None
        return x + delta, new_cache, auxiliary


class Experiment005LanguageModel(nn.Module):
    def __init__(self, config: dict[str, Any], arm: str, seed: int = 0):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        self.config = dict(config)
        self.arm = arm
        width = int(config["width"])
        self.embedding = nn.Embedding(int(config["vocab_size"]), width)
        self.position = nn.Embedding(int(config["max_sequence_length"]), width)
        self.blocks = nn.ModuleList(
            CausalBlock(config, arm, seed, layer) for layer in range(int(config["layers"]))
        )
        self.norm = nn.LayerNorm(width)
        # Name-derived streams make shared tensors exactly equal across structures.
        # No process RNG is consumed by this canonical initialization.
        for name, parameter in self.named_parameters():
            digest = hashlib.sha256(f"exp005-init-v1:{seed}:{name}".encode()).digest()
            generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
            with torch.no_grad():
                if parameter.ndim >= 2:
                    parameter.normal_(std=0.02, generator=generator)
                elif name.endswith("weight"):
                    parameter.fill_(1.0)
                else:
                    parameter.zero_()

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens: Tensor, *, past_key_values: list[tuple[Tensor, Tensor]] | None = None,
                use_cache: bool = False,
                forced_transition: tuple[int, int, int, int] | None = None
                ) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor]] | None]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")
        if past_key_values is not None and tokens.shape[1] != 1:
            raise ValueError("KV-cache execution must append exactly one token")
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        positions = torch.arange(past_length, past_length + tokens.shape[1], device=tokens.device)
        if positions[-1] >= self.position.num_embeddings:
            raise ValueError("input exceeds configured maximum sequence length")
        x = self.embedding(tokens) + self.position(positions)[None, :, :]
        auxiliary = x.new_zeros(())
        caches: list[tuple[Tensor, Tensor]] = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            x, cache, balance = block(x, past, use_cache, forced_transition)
            auxiliary = auxiliary + balance
            if cache is not None:
                caches.append(cache)
        normalized = self.norm(x)
        try:
            from .accounting import record_linear_flops
            record_linear_flops("tied_output_projection", normalized.numel() // normalized.shape[-1],
                                normalized.shape[-1], self.embedding.num_embeddings)
        except ImportError:
            pass
        logits = F.linear(normalized, self.embedding.weight)
        traces = [block.ffn.last_trace for block in self.blocks]
        self.last_execution = traces
        self.last_backup_cell_visits = sum(block.ffn.last_backup_cell_visits
                                           for block in self.blocks)
        return logits, auxiliary / len(self.blocks), caches if use_cache else None

    def prefill(self, tokens: Tensor) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        logits, _, cache = self(tokens, use_cache=True)
        assert cache is not None
        return logits, cache

    def decode_step(self, token: Tensor, cache: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        logits, _, updated = self(token.reshape(token.shape[0], 1),
                                  past_key_values=cache, use_cache=True)
        assert updated is not None
        return logits[:, -1], updated

    def set_collect_trace(self, enabled: bool) -> None:
        for block in self.blocks:
            block.ffn.collect_trace = bool(enabled)

    def parameter_breakdown(self) -> dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        cell = sum(parameter.numel() for block in self.blocks for module in block.ffn.cells
                    for parameter in module.parameters())
        if self.arm == "M0":
            controller = sum(parameter.numel() for block in self.blocks
                             for parameter in block.ffn.moe.router.parameters())
        elif self.arm == "P0":
            controller = sum(parameter.numel() for block in self.blocks
                             for parameter in block.ffn.parallel_router.parameters())
        else:
            controller = sum(parameter.numel() for block in self.blocks
                             for parameter in block.ffn.router.parameters())
        return {"total": total, "ffn_cells": cell, "controller": controller,
                "shared": total - cell - controller}


def make_model(config: dict[str, Any], arm: str, seed: int = 0) -> Experiment005LanguageModel:
    return Experiment005LanguageModel(config, arm, seed)
