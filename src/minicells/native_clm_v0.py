"""Native CLM v0: a small token-predictive language model with sparse persistent Cells.

M0/M1 deliberately keep one Cellular Layer and a linear residual Cell operator family.
Continual-stream evaluation and autonomous growth are later milestones; this module
already provides dynamic Cell spawning and certificate-projected Cell gradients so
those later stages do not need a runtime rewrite.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NativeCLMConfig:
    vocab_size: int = 256
    max_seq_len: int = 256
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.0
    initial_cells: int = 8
    active_cells: int = 2
    cellular_layer_index: int = 3
    route_temperature: float = 0.7
    certificate_max_rank: int = 64
    cell_init_scale: float = 0.02
    tie_embeddings: bool = True

    def validate(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not (0 <= self.cellular_layer_index < self.n_layers):
            raise ValueError("cellular_layer_index must name an existing shared block")
        if self.initial_cells < 1:
            raise ValueError("initial_cells must be >= 1")
        if not (1 <= self.active_cells <= self.initial_cells):
            raise ValueError("active_cells must be in [1, initial_cells]")
        if self.route_temperature <= 0:
            raise ValueError("route_temperature must be positive")
        if not (0 <= self.certificate_max_rank <= self.d_model):
            raise ValueError("certificate_max_rank must be in [0, d_model]")


class ByteTokenizer:
    """Deterministic UTF-8 byte tokenizer used by Native CLM v0 M0/M1."""

    vocab_size = 256

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(tokens: list[int] | Tensor) -> str:
        if isinstance(tokens, Tensor):
            values = tokens.detach().cpu().view(-1).tolist()
        else:
            values = tokens
        data = bytes(int(v) % 256 for v in values)
        return data.decode("utf-8", errors="replace")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: NativeCLMConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, width = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def shape(tensor: Tensor) -> Tensor:
            return tensor.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = shape(q), shape(k), shape(v)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, width)
        return self.proj(y)


class FeedForward(nn.Module):
    def __init__(self, config: NativeCLMConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x), approximate="tanh")))


class TransformerBlock(nn.Module):
    def __init__(self, config: NativeCLMConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.ff(self.ln2(x))
        return x


class NativeCell(nn.Module):
    """Persistent linear residual operator with a bounded input certificate."""

    def __init__(
        self,
        d_model: int,
        certificate_max_rank: int,
        init_scale: float,
        *,
        parent_id: int = -1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.certificate_max_rank = int(certificate_max_rank)
        self.weight = nn.Parameter(torch.empty(d_model, d_model))
        self.route_key = nn.Parameter(torch.empty(d_model))
        self.register_buffer(
            "certificate_basis",
            torch.zeros(certificate_max_rank, d_model),
            persistent=True,
        )
        self.register_buffer("certificate_rank", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("usage_count", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer(
            "parent_id",
            torch.tensor(int(parent_id), dtype=torch.long),
            persistent=True,
        )
        nn.init.normal_(self.weight, mean=0.0, std=init_scale / math.sqrt(d_model))
        nn.init.normal_(self.route_key, mean=0.0, std=1.0 / math.sqrt(d_model))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight)

    @property
    def rank(self) -> int:
        return int(self.certificate_rank.item())

    @property
    def certificate_fill(self) -> float:
        if self.certificate_max_rank == 0:
            return 0.0
        return self.rank / self.certificate_max_rank

    @torch.no_grad()
    def add_certificate_vector(self, vector: Tensor, *, tolerance: float = 1e-6) -> bool:
        if self.rank >= self.certificate_max_rank:
            return False
        v = vector.detach().to(device=self.weight.device, dtype=self.weight.dtype).reshape(-1)
        if v.numel() != self.d_model:
            raise ValueError("certificate vector has wrong width")
        if self.rank:
            q = self.certificate_basis[: self.rank]
            v = v - q.transpose(0, 1).matmul(q.matmul(v))
        norm = torch.linalg.vector_norm(v)
        if not torch.isfinite(norm) or float(norm) <= tolerance:
            return False
        self.certificate_basis[self.rank].copy_(v / norm)
        self.certificate_rank.add_(1)
        return True

    @torch.no_grad()
    def project_weight_gradient_(self) -> float:
        """Project dW into the certificate nullspace.

        F.linear(x, W) changes registered context q by q @ dW.T.
        Enforcing dW q^T = 0 is achieved by dW <- dW (I - Q^T Q).
        Returns projected/raw gradient norm ratio.
        """
        grad = self.weight.grad
        if grad is None:
            return 1.0
        raw_norm = float(torch.linalg.vector_norm(grad).item())
        if self.rank:
            q = self.certificate_basis[: self.rank].to(dtype=grad.dtype)
            grad.sub_(grad.matmul(q.transpose(0, 1)).matmul(q))
        projected_norm = float(torch.linalg.vector_norm(grad).item())
        if raw_norm <= 1e-20:
            return 1.0
        return projected_norm / raw_norm


class CellularLayer(nn.Module):
    """Sparse top-k execution over a dynamic set of persistent Cells."""

    def __init__(self, config: NativeCLMConfig, *, cell_count: int | None = None) -> None:
        super().__init__()
        self.config = config
        count = config.initial_cells if cell_count is None else int(cell_count)
        if count < config.active_cells:
            raise ValueError("cell_count must be >= active_cells")
        self.norm = nn.LayerNorm(config.d_model)
        self.query_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.cells = nn.ModuleList(
            [
                NativeCell(
                    config.d_model,
                    config.certificate_max_rank,
                    config.cell_init_scale,
                )
                for _ in range(count)
            ]
        )

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    def route(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        route_input = self.norm(x)
        query = F.normalize(self.query_proj(route_input), dim=-1)
        keys = torch.stack([F.normalize(cell.route_key, dim=0) for cell in self.cells], dim=0)
        scores = query.matmul(keys.transpose(0, 1)) / self.config.route_temperature
        k = min(self.config.active_cells, self.cell_count)
        top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
        top_probs = F.softmax(top_scores, dim=-1)
        return route_input, top_idx, top_probs, top_scores

    def forward(self, x: Tensor, *, return_info: bool = False) -> tuple[Tensor, dict[str, Any] | None]:
        route_input, top_idx, top_probs, top_scores = self.route(x)
        batch, seq_len, width = x.shape
        flat_x = x.reshape(batch * seq_len, width)
        flat_idx = top_idx.reshape(batch * seq_len, -1)
        flat_probs = top_probs.reshape(batch * seq_len, -1)
        flat_out = torch.zeros_like(flat_x)

        with torch.no_grad():
            usage = torch.bincount(flat_idx.reshape(-1), minlength=self.cell_count)
            for cell_id, count in enumerate(usage.tolist()):
                if count:
                    self.cells[cell_id].usage_count.add_(count)

        for cell_id, cell in enumerate(self.cells):
            positions = torch.nonzero(flat_idx == cell_id, as_tuple=False)
            if positions.numel() == 0:
                continue
            token_rows = positions[:, 0]
            slots = positions[:, 1]
            selected = flat_x.index_select(0, token_rows)
            cell_out = cell(selected)
            gates = flat_probs[token_rows, slots].unsqueeze(-1)
            flat_out.index_add_(0, token_rows, cell_out * gates)

        out = x + flat_out.view(batch, seq_len, width)
        if not return_info:
            return out, None

        entropy = -(top_probs * torch.log(top_probs.clamp_min(1e-9))).sum(dim=-1).mean()
        confidence = top_probs[..., 0].mean()
        if top_scores.size(-1) > 1:
            margin = (top_scores[..., 0] - top_scores[..., 1]).mean()
        else:
            margin = top_scores.new_tensor(float("inf"))
        info: dict[str, Any] = {
            "top_idx": top_idx.detach(),
            "top_probs": top_probs.detach(),
            "cell_input": x.detach(),
            "route_input": route_input.detach(),
            "route_entropy": float(entropy.detach().cpu()),
            "top1_confidence": float(confidence.detach().cpu()),
            "route_margin": float(margin.detach().cpu()),
            "cell_count": self.cell_count,
            "active_cells": top_idx.size(-1),
            "active_fraction_vs_dense": top_idx.size(-1) / self.cell_count,
        }
        return out, info

    def spawn_cell(
        self,
        *,
        parent_id: int | None = None,
        route_key: Tensor | None = None,
        inherit_scale: float = 0.25,
    ) -> int:
        if parent_id is not None and not (0 <= parent_id < self.cell_count):
            raise ValueError("invalid parent_id")
        parent_value = -1 if parent_id is None else int(parent_id)
        cell = NativeCell(
            self.config.d_model,
            self.config.certificate_max_rank,
            self.config.cell_init_scale,
            parent_id=parent_value,
        ).to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
        with torch.no_grad():
            if parent_id is not None:
                cell.weight.copy_(self.cells[parent_id].weight * float(inherit_scale))
            if route_key is not None:
                key = route_key.detach().to(device=cell.route_key.device, dtype=cell.route_key.dtype)
                if key.numel() != self.config.d_model:
                    raise ValueError("route_key has wrong width")
                cell.route_key.copy_(key.reshape(-1))
        self.cells.append(cell)
        return self.cell_count - 1

    @torch.no_grad()
    def project_cell_gradients_(self) -> dict[int, float]:
        return {idx: cell.project_weight_gradient_() for idx, cell in enumerate(self.cells)}

    @torch.no_grad()
    def update_certificates(
        self,
        info: dict[str, Any],
        *,
        max_new_vectors_per_cell: int = 1,
    ) -> int:
        top_idx: Tensor = info["top_idx"]
        cell_input: Tensor = info["cell_input"]
        top1 = top_idx[..., 0]
        added = 0
        for cell_id, cell in enumerate(self.cells):
            if max_new_vectors_per_cell <= 0:
                continue
            mask = top1 == cell_id
            if not bool(mask.any()):
                continue
            selected = cell_input[mask]
            mean_context = selected.mean(dim=0)
            if cell.add_certificate_vector(mean_context):
                added += 1
        return added


class NativeCLM(nn.Module):
    def __init__(self, config: NativeCLMConfig, *, cell_count: int | None = None) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.cellular = CellularLayer(config, cell_count=cell_count)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if module is self.lm_head and self.config.tie_embeddings:
                return
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def cell_count(self) -> int:
        return self.cellular.cell_count

    def forward(
        self,
        tokens: Tensor,
        targets: Tensor | None = None,
        *,
        return_info: bool = False,
    ) -> dict[str, Any]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        batch, seq_len = tokens.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("sequence exceeds configured max_seq_len")
        positions = torch.arange(seq_len, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)

        cell_info = None
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index == self.config.cellular_layer_index:
                x, cell_info = self.cellular(x, return_info=return_info)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch * seq_len, -1),
                targets.reshape(batch * seq_len),
            )
        return {"logits": logits, "loss": loss, "cell_info": cell_info}

    @torch.no_grad()
    def generate(
        self,
        tokens: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        self.eval()
        output = tokens
        for _ in range(max_new_tokens):
            context = output[:, -self.config.max_seq_len :]
            logits = self(context)["logits"][:, -1, :]
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and 0 < top_k < logits.size(-1):
                    values, _ = torch.topk(logits, k=top_k, dim=-1)
                    cutoff = values[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            output = torch.cat([output, next_token], dim=1)
        return output

    @torch.no_grad()
    def project_cell_gradients_(self) -> dict[int, float]:
        return self.cellular.project_cell_gradients_()

    @torch.no_grad()
    def update_certificates(self, info: dict[str, Any]) -> int:
        return self.cellular.update_certificates(info)

    def spawn_cell(
        self,
        *,
        parent_id: int | None = None,
        route_key: Tensor | None = None,
        inherit_scale: float = 0.25,
    ) -> int:
        return self.cellular.spawn_cell(
            parent_id=parent_id,
            route_key=route_key,
            inherit_scale=inherit_scale,
        )

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        cell_weight_ids = {id(cell.weight) for cell in self.cellular.cells}
        route_ids = {id(cell.route_key) for cell in self.cellular.cells}
        route_ids.update(id(p) for p in self.cellular.query_proj.parameters())
        route_ids.update(id(p) for p in self.cellular.norm.parameters())

        groups: dict[str, list[nn.Parameter]] = {"shared": [], "router": [], "cells": []}
        seen: set[int] = set()
        for parameter in self.parameters():
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            if identity in cell_weight_ids:
                groups["cells"].append(parameter)
            elif identity in route_ids:
                groups["router"].append(parameter)
            else:
                groups["shared"].append(parameter)
        return groups

    def parameter_count(self) -> dict[str, int]:
        groups = self.parameter_groups()
        counts = {name: sum(p.numel() for p in params) for name, params in groups.items()}
        counts["total"] = sum(counts.values())
        return counts

    def certificate_summary(self) -> dict[str, float]:
        ranks = [cell.rank for cell in self.cellular.cells]
        fills = [cell.certificate_fill for cell in self.cellular.cells]
        return {
            "mean_rank": float(sum(ranks) / len(ranks)),
            "max_rank": float(max(ranks)),
            "mean_fill": float(sum(fills) / len(fills)),
            "max_fill": float(max(fills)),
        }

    def checkpoint_payload(self, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "format": "minicells.native-clm-v0.checkpoint.v1",
            "config": asdict(self.config),
            "cell_count": self.cell_count,
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }

    def save_checkpoint(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(extra=extra), target)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["NativeCLM", dict[str, Any]]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("format") != "minicells.native-clm-v0.checkpoint.v1":
            raise ValueError("unsupported Native CLM checkpoint format")
        config = NativeCLMConfig(**payload["config"])
        model = cls(config, cell_count=int(payload["cell_count"]))
        model.load_state_dict(payload["state_dict"])
        return model, payload.get("extra", {})
