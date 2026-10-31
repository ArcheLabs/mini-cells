"""Small learned memory models used only by the EPCLM-002-A0 experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    key_bits: int = 16
    value_bits: int = 16
    max_events: int = 96
    cell_dim: int = 32
    dropout: float = 0.0

    @property
    def state_budget_bytes(self) -> int:
        return self.max_events * self.cell_dim * 4 + self.max_events


class LearnedCellMemory(nn.Module):
    """Learned event encoder, key-conditioned readout, and latent Cell state.

    One event is written into one latent slot. Query scores are evaluated for
    every slot, so this model is intentionally a full-scan H2 candidate and
    cannot count as evidence for H3 selective retrieval.
    """

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        event_width = config.key_bits + config.value_bits + 2
        self.cell_encoder = nn.Sequential(
            nn.Linear(event_width, 96), nn.GELU(), nn.Linear(96, config.cell_dim), nn.Tanh()
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(config.key_bits, 64), nn.GELU(), nn.Linear(64, config.cell_dim)
        )
        self.key_decoder = nn.Linear(config.cell_dim, config.cell_dim, bias=False)
        self.value_decoder = nn.Linear(config.cell_dim, config.value_bits)
        self.recency_scale = nn.Parameter(torch.tensor(0.0))

    def encode_events(
        self,
        key_bits: torch.Tensor,
        value_bits: torch.Tensor,
        is_fact: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Encode [batch, events, bits] into persisted latent Cell states."""
        batch, count, _ = key_bits.shape
        if count > self.config.max_events:
            raise ValueError("episode exceeds max_events")
        kind = is_fact.to(key_bits.dtype).unsqueeze(-1)
        position = torch.linspace(0.0, 1.0, count, device=key_bits.device, dtype=key_bits.dtype)
        position = position.view(1, count, 1).expand(batch, -1, -1)
        encoded = self.cell_encoder(torch.cat((key_bits, value_bits, kind, position), dim=-1))
        return encoded * valid.to(encoded.dtype).unsqueeze(-1)

    def forward(
        self,
        state: torch.Tensor,
        valid: torch.Tensor,
        query_bits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_encoder(query_bits)
        cell_keys = self.key_decoder(state)
        scores = torch.einsum("bd,bnd->bn", query, cell_keys) / (self.config.cell_dim ** 0.5)
        position = torch.linspace(0.0, 1.0, state.shape[1], device=state.device, dtype=state.dtype)
        scores = scores + F.softplus(self.recency_scale) * position.unsqueeze(0)
        scores = scores.masked_fill(~valid.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        per_cell_values = self.value_decoder(state)
        value_logits = torch.einsum("bn,bnv->bv", weights, per_cell_values)
        return value_logits, scores


class FlatRecurrentMemory(nn.Module):
    """Single-vector recurrent comparator with a matched persistent byte budget."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        state_dim = config.state_budget_bytes // 4
        self.state_dim = state_dim
        event_width = config.key_bits + config.value_bits + 1
        self.event_encoder = nn.Sequential(nn.Linear(event_width, 128), nn.GELU(), nn.Linear(128, 64))
        self.update = nn.GRUCell(64, state_dim)
        self.query_encoder = nn.Sequential(nn.Linear(config.key_bits, 64), nn.GELU())
        self.readout = nn.Sequential(
            nn.Linear(state_dim + 64, 256), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(256, config.value_bits),
        )

    def write_events(
        self,
        key_bits: torch.Tensor,
        value_bits: torch.Tensor,
        is_fact: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, _ = key_bits.shape
        if count > self.config.max_events:
            raise ValueError("episode exceeds max_events")
        inputs = self.event_encoder(
            torch.cat((key_bits, value_bits, is_fact.to(key_bits.dtype).unsqueeze(-1)), dim=-1)
        )
        state = inputs.new_zeros(batch, self.state_dim)
        for index in range(count):
            updated = self.update(inputs[:, index], state)
            mask = valid[:, index].unsqueeze(-1)
            state = torch.where(mask, updated, state)
        return state

    def forward(
        self,
        state: torch.Tensor,
        valid: torch.Tensor,
        query_bits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_encoder(query_bits)
        logits = self.readout(torch.cat((state, query), dim=-1))
        return logits, logits.new_empty((logits.shape[0], 0))


def build_model(name: str, config: ModelConfig) -> nn.Module:
    if name == "learned-cells":
        return LearnedCellMemory(config)
    if name == "flat-recurrent":
        return FlatRecurrentMemory(config)
    raise ValueError(f"unsupported learned model: {name}")


def model_config_dict(config: ModelConfig) -> dict[str, int | float]:
    return asdict(config)
