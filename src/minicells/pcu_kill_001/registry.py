"""Versioned Cell lineage and registry-only merge operations."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor


SCHEMA = "minicells.pcu-cell-registry.v1"


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def module_tensor_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


@dataclass
class CellRecord:
    cell_id: str
    layer: int
    parent_expert: int
    slice_index: int
    slice_start: int
    slice_end: int
    foundation_model: str
    foundation_revision: str
    foundation_weight_hash: str
    state: str = "FOUNDATION"
    parents: list[str] = field(default_factory=list)
    generation: int = 0
    weight_hash: str = ""
    routing_mode: str = "INHERITED_PARENT"
    branch: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.state not in {"FOUNDATION", "FORKED"}:
            raise ValueError(f"unknown Cell state: {self.state}")
        if self.slice_start >= self.slice_end:
            raise ValueError("Cell slice must have positive width")
        if self.state == "FOUNDATION" and self.generation != 0:
            raise ValueError("foundation Cell generation must be zero")
        if self.state == "FORKED" and (not self.parents or self.generation < 1 or not self.branch):
            raise ValueError("forked Cell needs parent, generation, and branch")
        if self.state == "FORKED" and self.branch not in {"A", "B", "JOINT"}:
            raise ValueError("forked Cell has an unknown branch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "layer": self.layer,
            "parent_expert": self.parent_expert,
            "slice_index": self.slice_index,
            "slice_start": self.slice_start,
            "slice_end": self.slice_end,
            "foundation_model": self.foundation_model,
            "foundation_revision": self.foundation_revision,
            "foundation_weight_hash": self.foundation_weight_hash,
            "state": self.state,
            "parents": list(self.parents),
            "generation": self.generation,
            "weight_hash": self.weight_hash,
            "routing_mode": self.routing_mode,
            "branch": self.branch,
            "provenance": dict(self.provenance),
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CellRecord":
        return cls(
            cell_id=str(value["cell_id"]),
            layer=int(value["layer"]),
            parent_expert=int(value["parent_expert"]),
            slice_index=int(value["slice_index"]),
            slice_start=int(value["slice_start"]),
            slice_end=int(value["slice_end"]),
            foundation_model=str(value["foundation_model"]),
            foundation_revision=str(value["foundation_revision"]),
            foundation_weight_hash=str(value["foundation_weight_hash"]),
            state=str(value.get("state", "FOUNDATION")),
            parents=[str(item) for item in value.get("parents", [])],
            generation=int(value.get("generation", 0)),
            weight_hash=str(value.get("weight_hash", "")),
            routing_mode=str(value.get("routing_mode", "INHERITED_PARENT")),
            branch=str(value["branch"]) if value.get("branch") is not None else None,
            provenance=dict(value.get("provenance", {})),
            artifact_path=str(value["artifact_path"]) if value.get("artifact_path") is not None else None,
            artifact_sha256=str(value["artifact_sha256"]) if value.get("artifact_sha256") is not None else None,
        )


@dataclass
class CellRegistry:
    foundation_model: str
    foundation_revision: str
    foundation_hash: str
    protocol_sha256: str
    layer: int
    records: dict[str, CellRecord] = field(default_factory=dict)
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("unsupported Cell registry schema")
        if len(self.records) != len(set(self.records)):
            raise ValueError("duplicate Cell IDs")
        for key, record in self.records.items():
            if key != record.cell_id:
                raise ValueError("registry key does not match cell_id")
            if record.foundation_weight_hash != self.foundation_hash:
                raise ValueError("Cell foundation hash differs from registry foundation hash")

    @property
    def foundation_records(self) -> list[CellRecord]:
        return [record for record in self.records.values() if record.state == "FOUNDATION"]

    @property
    def fork_records(self) -> list[CellRecord]:
        return [record for record in self.records.values() if record.state == "FORKED"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "foundation_model": self.foundation_model,
            "foundation_revision": self.foundation_revision,
            "foundation_hash": self.foundation_hash,
            "protocol_sha256": self.protocol_sha256,
            "layer": self.layer,
            "cells": [self.records[key].to_dict() for key in sorted(self.records)],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CellRegistry":
        records = {
            str(item["cell_id"]): CellRecord.from_dict(item) for item in value.get("cells", [])
        }
        return cls(
            foundation_model=str(value["foundation_model"]),
            foundation_revision=str(value["foundation_revision"]),
            foundation_hash=str(value["foundation_hash"]),
            protocol_sha256=str(value["protocol_sha256"]),
            layer=int(value["layer"]),
            records=records,
            schema=str(value.get("schema", "")),
        )

    @classmethod
    def load(cls, path: str) -> "CellRegistry":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def copy(self) -> "CellRegistry":
        return CellRegistry.from_dict(self.to_dict())


def make_foundation_registry(
    *,
    layer: int,
    experts: int,
    cells_per_expert: int,
    cell_width: int,
    foundation_model: str,
    foundation_revision: str,
    foundation_hash: str,
    protocol_sha256: str,
    cell_weight_hashes: Mapping[tuple[int, int], str] | None = None,
) -> CellRegistry:
    records: dict[str, CellRecord] = {}
    for expert in range(experts):
        for index in range(cells_per_expert):
            start = index * cell_width
            end = start + cell_width
            cell_id = f"L{layer}:E{expert}:C{index}"
            records[cell_id] = CellRecord(
                cell_id=cell_id,
                layer=layer,
                parent_expert=expert,
                slice_index=index,
                slice_start=start,
                slice_end=end,
                foundation_model=foundation_model,
                foundation_revision=foundation_revision,
                foundation_weight_hash=foundation_hash,
                weight_hash=(cell_weight_hashes or {}).get((expert, index), ""),
            )
    return CellRegistry(
        foundation_model=foundation_model,
        foundation_revision=foundation_revision,
        foundation_hash=foundation_hash,
        protocol_sha256=protocol_sha256,
        layer=layer,
        records=records,
    )


def fork_registry(base: CellRegistry, selected: Iterable[str], branch: str) -> CellRegistry:
    if branch not in {"A", "B", "JOINT"}:
        raise ValueError("branch must be A, B, or JOINT")
    result = base.copy()
    for foundation_id in selected:
        parent = result.records.get(str(foundation_id))
        if parent is None or parent.state != "FOUNDATION":
            raise ValueError(f"selected Cell is not a foundation Cell: {foundation_id}")
        fork_id = f"{parent.cell_id}::fork::{branch}"
        result.records[fork_id] = CellRecord(
            cell_id=fork_id,
            layer=parent.layer,
            parent_expert=parent.parent_expert,
            slice_index=parent.slice_index,
            slice_start=parent.slice_start,
            slice_end=parent.slice_end,
            foundation_model=parent.foundation_model,
            foundation_revision=parent.foundation_revision,
            foundation_weight_hash=parent.foundation_weight_hash,
            state="FORKED",
            parents=[parent.cell_id],
            generation=parent.generation + 1,
            weight_hash=parent.weight_hash,
            branch=branch,
            provenance={
                "fork_initial_delta": "zero",
                "parent_cell_id": parent.cell_id,
                "foundation_model": base.foundation_model,
                "foundation_revision": base.foundation_revision,
                "foundation_hash": base.foundation_hash,
                "protocol_sha256": base.protocol_sha256,
            },
        )
    return result


def _check_merge_compatibility(*registries: CellRegistry) -> None:
    first = registries[0]
    for other in registries[1:]:
        if other.foundation_hash != first.foundation_hash:
            raise ValueError("merge requires the same foundation hash")
        if other.protocol_sha256 != first.protocol_sha256:
            raise ValueError("merge requires the same protocol hash")
        if other.layer != first.layer:
            raise ValueError("merge requires the same target layer")


def merge_registries(base: CellRegistry, branch_a: CellRegistry, branch_b: CellRegistry) -> CellRegistry:
    """Union branch artifacts; overlapping parent forks remain two records."""
    _check_merge_compatibility(base, branch_a, branch_b)
    result = base.copy()
    for source, expected_branch in ((branch_a, "A"), (branch_b, "B")):
        for record in source.fork_records:
            if record.branch != expected_branch:
                raise ValueError(f"unexpected branch record in {expected_branch} registry")
            if record.cell_id in result.records:
                existing = result.records[record.cell_id]
                if existing.to_dict() != record.to_dict():
                    raise ValueError(f"conflicting Cell artifact: {record.cell_id}")
            else:
                result.records[record.cell_id] = CellRecord.from_dict(record.to_dict())
    return result


def rollback_registry(merged: CellRegistry, branch: str) -> CellRegistry:
    """Remove only one branch's fork records, preserving parent and other branch."""
    if branch not in {"A", "B", "JOINT", "all"}:
        raise ValueError("branch must be A, B, or JOINT")
    result = merged.copy()
    result.records = {
        key: value
        for key, value in result.records.items()
        if value.state == "FOUNDATION" or (branch != "all" and value.branch != branch)
    }
    return result


def bind_fork_artifact(
    registry: CellRegistry,
    branch: str,
    artifact_path: str,
    artifact_sha256: str,
) -> CellRegistry:
    """Bind every selected fork to the exact serialized runtime artifact."""
    if branch not in {"A", "B", "JOINT"}:
        raise ValueError("branch must be A, B, or JOINT")
    if not artifact_path or not artifact_sha256:
        raise ValueError("fork artifact path and SHA-256 are required")
    result = registry.copy()
    records = [record for record in result.fork_records if record.branch == branch]
    if not records:
        raise ValueError(f"registry has no fork records for branch {branch}")
    for record in records:
        parent_id = record.parents[0] if record.parents else ""
        if record.provenance.get("parent_cell_id") != parent_id:
            raise ValueError(f"fork {record.cell_id} is not bound to its declared parent")
        if record.foundation_weight_hash != result.foundation_hash:
            raise ValueError(f"fork {record.cell_id} has a mismatched foundation hash")
        record.artifact_path = str(artifact_path)
        record.artifact_sha256 = str(artifact_sha256)
    return result


def validate_fork_artifacts(registry: CellRegistry, require_bound: bool = True) -> None:
    """Fail closed when a registry record cannot identify its runtime fork."""
    for record in registry.fork_records:
        if not record.parents or record.provenance.get("parent_cell_id") != record.parents[0]:
            raise ValueError(f"fork {record.cell_id} has invalid parent binding")
        if record.provenance.get("foundation_hash", registry.foundation_hash) != registry.foundation_hash:
            raise ValueError(f"fork {record.cell_id} has invalid foundation binding")
        if require_bound and (not record.artifact_path or not record.artifact_sha256):
            raise ValueError(f"fork {record.cell_id} has no bound runtime artifact")
        if record.artifact_path and record.artifact_sha256:
            path = Path(record.artifact_path)
            if not path.is_file():
                raise ValueError(f"fork {record.cell_id} artifact is missing: {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != record.artifact_sha256:
                raise ValueError(f"fork {record.cell_id} artifact SHA-256 mismatch")


def fork_delta(parent: Mapping[str, Tensor], fork: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Compute explicit fork-minus-parent tensors without modifying either input."""
    if set(parent) != set(fork):
        raise ValueError("parent and fork tensor keys differ")
    return {key: fork[key] - parent[key] for key in parent}
