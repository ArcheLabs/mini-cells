"""State hashing, dependency indexing, and observability helpers for CLM-0.4-mini."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

import torch
from torch import nn

from .model import TinyCLMDecoder


@dataclass(frozen=True)
class TokenExample:
    example_id: str
    address_id: str
    tokens: tuple[int, ...]
    knowledge_key: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tokens"] = list(self.tokens)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "TokenExample":
        return cls(
            example_id=str(payload["example_id"]),
            address_id=str(payload["address_id"]),
            tokens=tuple(int(x) for x in payload["tokens"]),
            knowledge_key=payload.get("knowledge_key"),
        )


def _tensor_hash_update(hasher, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    hasher.update(str(value.dtype).encode("utf-8"))
    hasher.update(str(tuple(value.shape)).encode("utf-8"))
    hasher.update(value.numpy().tobytes())


def module_state_hash(module: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        _tensor_hash_update(hasher, tensor)
    return hasher.hexdigest()


def model_state_hash(model: TinyCLMDecoder) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        _tensor_hash_update(hasher, tensor)
    hasher.update(
        json.dumps(model.private_addresses(), separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    return hasher.hexdigest()


class DependencyIndex:
    """Exact inverted execution dependency index; false negatives are not permitted."""

    def __init__(self) -> None:
        self.cell_to_probes: dict[str, set[str]] = {}
        self.probe_to_cells: dict[str, set[str]] = {}

    def register(self, probe_id: str, cell_ids: Iterable[str]) -> None:
        cells = {str(cell_id) for cell_id in cell_ids}
        self.remove(probe_id)
        self.probe_to_cells[str(probe_id)] = cells
        for cell_id in cells:
            self.cell_to_probes.setdefault(cell_id, set()).add(str(probe_id))

    def remove(self, probe_id: str) -> None:
        probe_id = str(probe_id)
        prior = self.probe_to_cells.pop(probe_id, None)
        if not prior:
            return
        for cell_id in prior:
            bucket = self.cell_to_probes.get(cell_id)
            if bucket is None:
                continue
            bucket.discard(probe_id)
            if not bucket:
                self.cell_to_probes.pop(cell_id, None)

    def scope(self, touched_cells: Iterable[str]) -> set[str]:
        result: set[str] = set()
        for cell_id in touched_cells:
            result.update(self.cell_to_probes.get(str(cell_id), set()))
        return result

    def coverage(self, touched_cells: Iterable[str], total_probes: int) -> float:
        if total_probes <= 0:
            return 0.0
        return len(self.scope(touched_cells)) / float(total_probes)

    def probe_count(self, cell_id: str) -> int:
        return len(self.cell_to_probes.get(str(cell_id), set()))

    def to_dict(self) -> dict:
        return {
            "cell_to_probes": {
                key: sorted(values) for key, values in sorted(self.cell_to_probes.items())
            },
            "probe_to_cells": {
                key: sorted(values) for key, values in sorted(self.probe_to_cells.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DependencyIndex":
        obj = cls()
        for probe_id, cells in payload.get("probe_to_cells", {}).items():
            obj.register(str(probe_id), [str(cell) for cell in cells])
        return obj


class CellRegistry:
    """Mutable counters plus schema-compatible snapshots for all committed Cells."""

    def __init__(self, model: TinyCLMDecoder) -> None:
        self.stats: dict[str, dict] = {}
        for layer_id in (3, 4):
            layer = model.sparse_layer(layer_id)
            for index, module in enumerate(layer.base_cells):
                cell_id = layer.base_cell_id(index)
                self.stats[cell_id] = {
                    "cell_id": cell_id,
                    "layer_id": layer_id,
                    "cell_type": "base",
                    "owner_address_id": None,
                    "parent_base_route": None,
                    "birth_transaction": None,
                    "parameter_count": sum(p.numel() for p in module.parameters()),
                    "state_version": 0,
                    "activation_count": 0,
                    "accepted_updates": 0,
                    "rejected_updates": 0,
                    "growth_rescue_transaction": None,
                    "reuse_events": [],
                }

    def record_activation(self, cell_ids: Iterable[str], amount: int = 1) -> None:
        for cell_id in cell_ids:
            if cell_id in self.stats:
                self.stats[cell_id]["activation_count"] += int(amount)

    def record_rejected(self, cell_ids: Iterable[str]) -> None:
        for cell_id in cell_ids:
            if cell_id in self.stats:
                self.stats[cell_id]["rejected_updates"] += 1

    def record_accepted(self, cell_ids: Iterable[str], *, transaction_id: int, reuse: bool = False) -> None:
        for cell_id in cell_ids:
            if cell_id in self.stats:
                self.stats[cell_id]["accepted_updates"] += 1
                self.stats[cell_id]["state_version"] += 1
                if reuse:
                    self.stats[cell_id]["reuse_events"].append(int(transaction_id))

    def add_growth_bundle(
        self,
        model: TinyCLMDecoder,
        *,
        address_id: str,
        transaction_id: int,
    ) -> list[str]:
        ids = model.private_cell_ids(address_id)
        parent = model.base_cell_ids(address_id)
        for cell_id in ids:
            layer_id = int(cell_id.split(":")[1][1:])
            module = model.modules_for_cell_ids([cell_id])[0]
            self.stats[cell_id] = {
                "cell_id": cell_id,
                "layer_id": layer_id,
                "cell_type": "private-growth",
                "owner_address_id": str(address_id),
                "parent_base_route": list(parent),
                "birth_transaction": int(transaction_id),
                "parameter_count": sum(p.numel() for p in module.parameters()),
                "state_version": 1,
                "activation_count": 0,
                "accepted_updates": 1,
                "rejected_updates": 0,
                "growth_rescue_transaction": int(transaction_id),
                "reuse_events": [],
            }
        return ids

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self.stats))

    @classmethod
    def from_dict(cls, model: TinyCLMDecoder, payload: dict) -> "CellRegistry":
        obj = cls(model)
        obj.stats = json.loads(json.dumps(payload))
        return obj

    def snapshot(self, model: TinyCLMDecoder, dependency_index: DependencyIndex) -> list[dict]:
        entries: list[dict] = []
        for cell_id in sorted(self.stats):
            entry = dict(self.stats[cell_id])
            module = model.modules_for_cell_ids([cell_id])[0]
            entry["state_hash"] = module_state_hash(module)
            entry["dependency_probe_count"] = dependency_index.probe_count(cell_id)
            entries.append(entry)
        return entries
