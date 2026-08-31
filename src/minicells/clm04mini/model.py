"""Small token-level CLM primitives used by MiniCells experiments and preview runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MiniCLMConfig:
    vocab_size: int = 64
    max_seq_len: int = 16
    num_layers: int = 4
    d_model: int = 32
    n_heads: int = 4
    dense_ff_hidden: int = 64
    base_cells: int = 8
    cell_hidden: int = 8
    routing_salt: str = "clm-0.4-mini-m0"
    # Preview-only by default. Zero preserves all historical M0/M1 architectures.
    shared_cell_ff_hidden: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "MiniCLMConfig":
        return cls(**payload)


class StableAddressRouter:
    """Deterministic Top-2 address router independent of mutable hidden state."""

    def __init__(self, *, num_cells: int, salt: str) -> None:
        if num_cells < 2:
            raise ValueError("num_cells must be >= 2")
        self.num_cells = int(num_cells)
        self.salt = str(salt)

    def route(self, layer_id: int, address_id: str | int) -> tuple[int, int]:
        raw = f"{self.salt}|{layer_id}|{address_id}".encode("utf-8")
        digest = hashlib.sha256(raw).digest()
        first = int.from_bytes(digest[:8], "big") % self.num_cells
        second = int.from_bytes(digest[8:16], "big") % self.num_cells
        if second == first:
            second = (second + 1 + int.from_bytes(digest[16:18], "big")) % self.num_cells
            if second == first:
                second = (first + 1) % self.num_cells
        return (first, second)


class CellFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int, *, zero_output: bool = False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        if zero_output:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def _private_key(address_id: str | int) -> str:
    digest = hashlib.sha256(str(address_id).encode("utf-8")).hexdigest()[:20]
    return f"a_{digest}"


class SparseCellFFN(nn.Module):
    def __init__(self, cfg: MiniCLMConfig, *, layer_id: int, router: StableAddressRouter) -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        self.router = router
        self.base_cells = nn.ModuleList(
            [CellFFN(cfg.d_model, cfg.cell_hidden) for _ in range(cfg.base_cells)]
        )
        self.private_cells = nn.ModuleDict()
        self.private_owners: dict[str, str] = {}

    def base_route(self, address_id: str | int) -> tuple[int, int]:
        return self.router.route(self.layer_id, address_id)

    def has_private(self, address_id: str | int) -> bool:
        return _private_key(address_id) in self.private_cells

    def spawn_private(self, address_id: str | int, *, d_model: int, hidden: int) -> str:
        key = _private_key(address_id)
        if key in self.private_cells:
            raise ValueError(f"private cell already exists for {address_id}")
        reference = next(self.base_cells[0].parameters())
        cell = CellFFN(d_model, hidden, zero_output=True).to(
            device=reference.device,
            dtype=reference.dtype,
        )
        self.private_cells[key] = cell
        self.private_owners[key] = str(address_id)
        return key

    def private_module(self, address_id: str | int) -> CellFFN:
        return self.private_cells[_private_key(address_id)]

    def private_cell_id(self, address_id: str | int) -> str:
        return f"growth:L{self.layer_id}:{_private_key(address_id)}"

    def base_cell_id(self, cell_index: int) -> str:
        return f"base:L{self.layer_id}:C{cell_index:02d}"

    def forward(self, x: torch.Tensor, address_ids: list[str | int]) -> torch.Tensor:
        if x.size(0) != len(address_ids):
            raise ValueError("address_ids length must equal batch size")
        outputs: list[torch.Tensor] = []
        for batch_index, address_id in enumerate(address_ids):
            sample = x[batch_index : batch_index + 1]
            route = self.base_route(address_id)
            routed = 0.5 * (self.base_cells[route[0]](sample) + self.base_cells[route[1]](sample))
            key = _private_key(address_id)
            if key in self.private_cells:
                routed = routed + self.private_cells[key](sample)
            outputs.append(routed)
        return torch.cat(outputs, dim=0)


def sinusoidal_positions(length: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / width)
    )
    pe = torch.zeros(length, width, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


class CLMBlock(nn.Module):
    def __init__(
        self,
        cfg: MiniCLMConfig,
        *,
        layer_id: int,
        router: StableAddressRouter,
        sparse: bool,
    ) -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        self.sparse = bool(sparse)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.shared_ff: nn.Module | None = None
        if sparse:
            self.ff = SparseCellFFN(cfg, layer_id=layer_id, router=router)
            if int(cfg.shared_cell_ff_hidden) > 0:
                self.shared_ff = nn.Sequential(
                    nn.Linear(cfg.d_model, int(cfg.shared_cell_ff_hidden)),
                    nn.GELU(),
                    nn.Linear(int(cfg.shared_cell_ff_hidden), cfg.d_model),
                )
        else:
            self.ff = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.dense_ff_hidden),
                nn.GELU(),
                nn.Linear(cfg.dense_ff_hidden, cfg.d_model),
            )

    def forward(
        self,
        x: torch.Tensor,
        address_ids: list[str | int],
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self.norm1(x)
        attn_out, _ = self.attn(q, q, q, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        q = self.norm2(x)
        if self.sparse:
            if self.shared_ff is not None:
                x = x + self.shared_ff(q)
            x = x + self.ff(q, address_ids)
        else:
            x = x + self.ff(q)
        return x


class TinyCLMDecoder(nn.Module):
    """Four-block decoder preserving independently mutable Cell boundaries."""

    def __init__(self, cfg: MiniCLMConfig) -> None:
        super().__init__()
        if cfg.num_layers != 4:
            raise ValueError("MiniCells currently preserves a four-block topology")
        self.cfg = cfg
        self.router = StableAddressRouter(num_cells=cfg.base_cells, salt=cfg.routing_salt)
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                CLMBlock(cfg, layer_id=1, router=self.router, sparse=False),
                CLMBlock(cfg, layer_id=2, router=self.router, sparse=False),
                CLMBlock(cfg, layer_id=3, router=self.router, sparse=True),
                CLMBlock(cfg, layer_id=4, router=self.router, sparse=True),
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def sparse_layer(self, layer_id: int) -> SparseCellFFN:
        if layer_id not in (3, 4):
            raise ValueError("growth-capable layers are 3 and 4")
        block = self.blocks[layer_id - 1]
        assert isinstance(block.ff, SparseCellFFN)
        return block.ff

    def shared_cell_ffn_parameters(self) -> int:
        total = 0
        for layer_id in (3, 4):
            shared = self.blocks[layer_id - 1].shared_ff
            if shared is not None:
                total += sum(parameter.numel() for parameter in shared.parameters())
        return int(total)

    def base_routes(self, address_id: str | int) -> dict[str, list[int]]:
        return {
            str(layer_id): list(self.sparse_layer(layer_id).base_route(address_id))
            for layer_id in (3, 4)
        }

    def base_cell_ids(self, address_id: str | int) -> list[str]:
        ids: list[str] = []
        for layer_id in (3, 4):
            layer = self.sparse_layer(layer_id)
            ids.extend(layer.base_cell_id(index) for index in layer.base_route(address_id))
        return ids

    def has_private_bundle(self, address_id: str | int) -> bool:
        present = [self.sparse_layer(layer_id).has_private(address_id) for layer_id in (3, 4)]
        if present[0] != present[1]:
            raise RuntimeError("private bundle must be atomic across layers 3 and 4")
        return present[0]

    def spawn_growth_bundle(self, address_id: str | int) -> list[str]:
        if self.has_private_bundle(address_id):
            raise ValueError(f"growth bundle already exists for {address_id}")
        ids: list[str] = []
        for layer_id in (3, 4):
            layer = self.sparse_layer(layer_id)
            layer.spawn_private(
                address_id,
                d_model=self.cfg.d_model,
                hidden=self.cfg.cell_hidden,
            )
            ids.append(layer.private_cell_id(address_id))
        return ids

    def private_cell_ids(self, address_id: str | int) -> list[str]:
        if not self.has_private_bundle(address_id):
            return []
        return [self.sparse_layer(layer_id).private_cell_id(address_id) for layer_id in (3, 4)]

    def active_cell_ids(self, address_id: str | int) -> list[str]:
        return self.base_cell_ids(address_id) + self.private_cell_ids(address_id)

    def private_addresses(self) -> list[str]:
        owners = set(self.sparse_layer(3).private_owners.values())
        owners4 = set(self.sparse_layer(4).private_owners.values())
        if owners != owners4:
            raise RuntimeError("private bundle owner sets diverged")
        return sorted(owners)

    def modules_for_cell_ids(self, cell_ids: Iterable[str]) -> list[nn.Module]:
        modules: list[nn.Module] = []
        for cell_id in cell_ids:
            parts = cell_id.split(":")
            if parts[0] == "base":
                layer_id = int(parts[1][1:])
                cell_index = int(parts[2][1:])
                modules.append(self.sparse_layer(layer_id).base_cells[cell_index])
            elif parts[0] == "growth":
                layer_id = int(parts[1][1:])
                key = parts[2]
                modules.append(self.sparse_layer(layer_id).private_cells[key])
            else:
                raise ValueError(f"unknown cell id: {cell_id}")
        return modules

    def forward(self, token_ids: torch.Tensor, address_ids: list[str | int]) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape [batch, time]")
        if token_ids.size(1) > self.cfg.max_seq_len:
            raise ValueError("sequence exceeds configured maximum")
        x = self.token_embedding(token_ids)
        x = x + sinusoidal_positions(
            token_ids.size(1), self.cfg.d_model, x.device, x.dtype
        ).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(token_ids.size(1), token_ids.size(1), dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, address_ids, causal_mask)
        x = self.final_norm(x)
        return F.linear(x, self.token_embedding.weight)
