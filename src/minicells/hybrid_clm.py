from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class HybridCLMError(RuntimeError):
    """Raised when a Hybrid CLM lifecycle invariant is violated."""


@dataclass(frozen=True)
class HybridGateTrace:
    probabilities: torch.Tensor
    active: torch.Tensor


@dataclass(frozen=True)
class HybridCellArtifact:
    cell_id: str
    parent_id: str | None
    version: int
    address_frozen: bool
    state: dict[str, torch.Tensor]

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.cell_id.encode())
        digest.update(b"\0")
        digest.update((self.parent_id or "").encode())
        digest.update(b"\0")
        digest.update(str(self.version).encode())
        digest.update(b"1" if self.address_frozen else b"0")
        for name in sorted(self.state):
            tensor = self.state[name].detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class HybridManifest:
    foundation_model_id: str
    foundation_revision: str
    cells: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "foundation_model_id": self.foundation_model_id,
            "foundation_revision": self.foundation_revision,
            "cells": [
                {"cell_id": cell_id, "digest": digest}
                for cell_id, digest in self.cells
            ],
        }

    def add(self, artifact: HybridCellArtifact) -> HybridManifest:
        mapping = dict(self.cells)
        digest = artifact.digest()
        existing = mapping.get(artifact.cell_id)
        if existing is not None and existing != digest:
            raise HybridCLMError(f"conflicting artifact for {artifact.cell_id}")
        mapping[artifact.cell_id] = digest
        return HybridManifest(
            foundation_model_id=self.foundation_model_id,
            foundation_revision=self.foundation_revision,
            cells=tuple(sorted(mapping.items())),
        )

    def remove(self, cell_id: str) -> HybridManifest:
        mapping = dict(self.cells)
        mapping.pop(cell_id, None)
        return HybridManifest(
            foundation_model_id=self.foundation_model_id,
            foundation_revision=self.foundation_revision,
            cells=tuple(sorted(mapping.items())),
        )

    def merge(self, other: HybridManifest) -> HybridManifest:
        if (
            self.foundation_model_id != other.foundation_model_id
            or self.foundation_revision != other.foundation_revision
        ):
            raise HybridCLMError("cannot merge manifests with different foundations")
        mapping = dict(self.cells)
        for cell_id, digest in other.cells:
            existing = mapping.get(cell_id)
            if existing is not None and existing != digest:
                raise HybridCLMError(f"manifest conflict for {cell_id}")
            mapping[cell_id] = digest
        return HybridManifest(
            foundation_model_id=self.foundation_model_id,
            foundation_revision=self.foundation_revision,
            cells=tuple(sorted(mapping.items())),
        )


def _clone_tensor_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


class HybridCellOverlay(nn.Module):
    """A non-competitive, commit-gated CLM evolution layer over a frozen model.

    Address predicates are evaluated once from an explicit prompt-anchor hidden
    state. The resulting applicability decision is then held fixed for the
    sequence and can affect only the anchor position and positions after it.
    This prevents answer/candidate tokens from changing routing and makes the
    runtime semantics match prompt-level address training.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        read_layer_index: int,
        write_layer_indices: Sequence[int],
        max_cells: int = 128,
        rank: int = 16,
        gate_threshold: float = 0.8,
        gate_temperature: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or rank <= 0 or max_cells <= 0:
            raise ValueError("hidden_size, rank and max_cells must be positive")
        if not write_layer_indices:
            raise ValueError("at least one write layer is required")
        writes = tuple(int(value) for value in write_layer_indices)
        if tuple(sorted(set(writes))) != writes:
            raise ValueError("write_layer_indices must be strictly increasing")
        if any(value <= int(read_layer_index) for value in writes):
            raise ValueError("all write layers must be after the read layer")
        if not 0.0 < gate_threshold < 1.0:
            raise ValueError("gate_threshold must be in (0, 1)")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")

        self.hidden_size = int(hidden_size)
        self.read_layer_index = int(read_layer_index)
        self.write_layer_indices = writes
        self.max_cells = int(max_cells)
        self.rank = int(rank)
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        gate_weight = torch.randn(max_cells, hidden_size, generator=generator)
        gate_weight = F.normalize(gate_weight, dim=-1)
        down = torch.randn(
            len(writes), max_cells, hidden_size, rank, generator=generator
        ) / (hidden_size**0.5)

        self.gate_weight = nn.Parameter(gate_weight)
        self.gate_bias = nn.Parameter(torch.full((max_cells,), -4.0))
        self.down = nn.Parameter(down)
        self.up = nn.Parameter(torch.zeros(len(writes), max_cells, rank, hidden_size))

        self.register_buffer("allocated_mask", torch.zeros(max_cells, dtype=torch.bool))
        self.register_buffer("committed_mask", torch.zeros(max_cells, dtype=torch.bool))
        self.register_buffer("address_frozen_mask", torch.zeros(max_cells, dtype=torch.bool))
        self.register_buffer("parent_slot", torch.full((max_cells,), -1, dtype=torch.long))

        self._cached_probabilities: torch.Tensor | None = None
        self._cached_active: torch.Tensor | None = None
        self._shadow_mask = torch.zeros(max_cells, dtype=torch.bool)
        self._prompt_positions: torch.Tensor | None = None
        self._installed_handles: list[Any] = []

    def next_free_slot(self) -> int:
        free = torch.nonzero(~self.allocated_mask, as_tuple=False).flatten()
        if not free.numel():
            raise HybridCLMError("no free Hybrid CLM cell slots remain")
        return int(free[0].item())

    @torch.no_grad()
    def allocate_cell(
        self,
        *,
        parent_slot: int | None = None,
        inherit_address: bool = False,
        inherit_transform: bool = False,
    ) -> int:
        slot = self.next_free_slot()
        if parent_slot is not None:
            parent = int(parent_slot)
            if not bool(self.allocated_mask[parent]):
                raise HybridCLMError("parent cell must already be allocated")
            self.parent_slot[slot] = parent
            if inherit_address:
                self.gate_weight[slot].copy_(self.gate_weight[parent])
                self.gate_bias[slot].copy_(self.gate_bias[parent])
            if inherit_transform:
                self.down[:, slot].copy_(self.down[:, parent])
                self.up[:, slot].copy_(self.up[:, parent])
        self.allocated_mask[slot] = True
        self.committed_mask[slot] = False
        self.address_frozen_mask[slot] = False
        return slot

    @torch.no_grad()
    def freeze_address_(self, slot: int) -> None:
        if not bool(self.allocated_mask[int(slot)]):
            raise HybridCLMError("cannot freeze an unallocated cell")
        self.address_frozen_mask[int(slot)] = True

    @torch.no_grad()
    def commit_cell_(self, slot: int) -> None:
        index = int(slot)
        if not bool(self.allocated_mask[index]):
            raise HybridCLMError("cannot commit an unallocated cell")
        if not bool(self.address_frozen_mask[index]):
            raise HybridCLMError("cell address must be frozen before commit")
        self.committed_mask[index] = True

    @torch.no_grad()
    def uncommit_cell_(self, slot: int) -> None:
        self.committed_mask[int(slot)] = False

    def clear_forward_cache(self) -> None:
        self._cached_probabilities = None
        self._cached_active = None

    def address_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.hidden_size:
            raise HybridCLMError("hidden size does not match Hybrid CLM overlay")
        query = F.normalize(hidden.float(), dim=-1)
        logits = torch.einsum("...d,kd->...k", query, self.gate_weight.float())
        logits = logits + self.gate_bias.float()
        return logits / self.gate_temperature

    def address_probabilities(self, hidden: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(self.address_logits(hidden))
        mask = self.allocated_mask.to(device=hidden.device)
        return probabilities * mask.to(dtype=probabilities.dtype)

    def address_probability_for_features(
        self, features: torch.Tensor, slot: int
    ) -> torch.Tensor:
        logits = self.address_logits(features)[..., int(slot)]
        return torch.sigmoid(logits)

    def _runtime_mask(self, device: torch.device) -> torch.Tensor:
        shadow = self._shadow_mask.to(device=device)
        committed = self.committed_mask.to(device=device)
        return committed | shadow

    def _validated_prompt_positions(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._prompt_positions is None:
            raise HybridCLMError("prompt_scope(anchor_positions) is required for overlay execution")
        positions = self._prompt_positions.to(device=hidden.device, dtype=torch.long)
        if positions.ndim != 1 or positions.numel() != hidden.shape[0]:
            raise HybridCLMError("prompt anchor positions do not match batch size")
        if bool((positions < 0).any()) or bool((positions >= hidden.shape[1]).any()):
            raise HybridCLMError("prompt anchor position is outside sequence bounds")
        return positions

    def _read(self, hidden: torch.Tensor) -> None:
        positions = self._validated_prompt_positions(hidden)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        anchor_hidden = hidden[rows, positions]
        anchor_probabilities = self.address_probabilities(anchor_hidden)
        anchor_active = anchor_probabilities >= self.gate_threshold
        runtime = self._runtime_mask(hidden.device).view(1, -1)
        anchor_active = anchor_active & runtime

        sequence_positions = torch.arange(hidden.shape[1], device=hidden.device)
        at_or_after_anchor = sequence_positions.view(1, -1, 1) >= positions.view(-1, 1, 1)
        self._cached_probabilities = anchor_probabilities[:, None, :].expand(
            -1, hidden.shape[1], -1
        )
        self._cached_active = anchor_active[:, None, :] & at_or_after_anchor

    def _transform(self, hidden: torch.Tensor, site: int) -> torch.Tensor:
        if self._cached_active is None or self._cached_probabilities is None:
            raise HybridCLMError("write site executed before Hybrid CLM read site")
        if hidden.shape[:2] != self._cached_active.shape[:2]:
            raise HybridCLMError("cached gate shape differs from downstream hidden state")
        gate = self._cached_active.to(dtype=torch.float32)
        work = hidden.float()
        low = F.silu(torch.einsum("btd,kdr->btkr", work, self.down[site].float()))
        delta = torch.einsum("btkr,krd->btkd", low, self.up[site].float())
        transformed = work + torch.einsum("btk,btkd->btd", gate, delta)
        return transformed.to(dtype=hidden.dtype)

    def prompt_gates(self, positions: torch.Tensor) -> HybridGateTrace:
        if self._cached_probabilities is None or self._cached_active is None:
            raise HybridCLMError("no Hybrid CLM gate trace is available")
        positions = positions.to(device=self._cached_probabilities.device, dtype=torch.long)
        if positions.ndim != 1 or positions.numel() != self._cached_probabilities.shape[0]:
            raise HybridCLMError("prompt positions do not match gate batch")
        rows = torch.arange(positions.numel(), device=positions.device)
        return HybridGateTrace(
            probabilities=self._cached_probabilities[rows, positions],
            active=self._cached_active[rows, positions],
        )

    @contextlib.contextmanager
    def prompt_scope(self, anchor_positions: torch.Tensor) -> Iterator[HybridCellOverlay]:
        if anchor_positions.ndim != 1:
            raise HybridCLMError("prompt anchor positions must be one-dimensional")
        previous = self._prompt_positions
        self._prompt_positions = anchor_positions.detach().clone()
        try:
            yield self
        finally:
            self._prompt_positions = previous
            self.clear_forward_cache()

    @contextlib.contextmanager
    def shadow(self, slots: Sequence[int]) -> Iterator[HybridCellOverlay]:
        previous = self._shadow_mask.clone()
        mask = previous.clone()
        for slot in slots:
            index = int(slot)
            if not bool(self.allocated_mask[index]):
                raise HybridCLMError("cannot shadow an unallocated cell")
            mask[index] = True
        self._shadow_mask = mask
        try:
            yield self
        finally:
            self._shadow_mask = previous
            self.clear_forward_cache()

    def cell_state(self, slot: int) -> dict[str, torch.Tensor]:
        index = int(slot)
        if not bool(self.allocated_mask[index]):
            raise HybridCLMError("cannot export an unallocated cell")
        return _clone_tensor_dict(
            {
                "gate_weight": self.gate_weight[index],
                "gate_bias": self.gate_bias[index : index + 1],
                "down": self.down[:, index],
                "up": self.up[:, index],
            }
        )

    def export_artifact(
        self,
        slot: int,
        *,
        cell_id: str,
        parent_id: str | None = None,
        version: int = 1,
    ) -> HybridCellArtifact:
        index = int(slot)
        return HybridCellArtifact(
            cell_id=str(cell_id),
            parent_id=parent_id,
            version=int(version),
            address_frozen=bool(self.address_frozen_mask[index]),
            state=self.cell_state(index),
        )

    @torch.no_grad()
    def apply_artifact_(self, artifact: HybridCellArtifact, *, commit: bool = True) -> int:
        slot = self.allocate_cell()
        device = self.gate_weight.device
        self.gate_weight[slot].copy_(
            artifact.state["gate_weight"].to(device=device, dtype=self.gate_weight.dtype)
        )
        self.gate_bias[slot].copy_(
            artifact.state["gate_bias"].to(device=device, dtype=self.gate_bias.dtype).reshape(())
        )
        self.down[:, slot].copy_(artifact.state["down"].to(device=device, dtype=self.down.dtype))
        self.up[:, slot].copy_(artifact.state["up"].to(device=device, dtype=self.up.dtype))
        self.address_frozen_mask[slot] = bool(artifact.address_frozen)
        if commit:
            self.commit_cell_(slot)
        return slot

    @contextlib.contextmanager
    def installed(self, model: nn.Module) -> Iterator[HybridCellOverlay]:
        if self._installed_handles:
            raise HybridCLMError("Hybrid CLM hooks are already installed")
        try:
            read_module = model.get_submodule(f"model.layers.{self.read_layer_index}")

            def read_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(hidden):
                    raise HybridCLMError("read layer did not return tensor hidden state")
                self._read(hidden)
                return output

            self._installed_handles.append(read_module.register_forward_hook(read_hook))

            for site, layer_index in enumerate(self.write_layer_indices):
                module = model.get_submodule(f"model.layers.{layer_index}")

                def write_hook(
                    _module: nn.Module,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    *,
                    site_index: int = site,
                ) -> Any:
                    hidden = output[0] if isinstance(output, tuple) else output
                    if not torch.is_tensor(hidden):
                        raise HybridCLMError("write layer did not return tensor hidden state")
                    transformed = self._transform(hidden, site_index)
                    if isinstance(output, tuple):
                        return (transformed, *output[1:])
                    return transformed

                self._installed_handles.append(module.register_forward_hook(write_hook))
            yield self
        finally:
            for handle in self._installed_handles:
                handle.remove()
            self._installed_handles.clear()
            self.clear_forward_cache()


def mask_address_gradients_(overlay: HybridCellOverlay, slot: int) -> None:
    index = int(slot)
    if overlay.gate_weight.grad is not None:
        keep = overlay.gate_weight.grad[index].clone()
        overlay.gate_weight.grad.zero_()
        overlay.gate_weight.grad[index].copy_(keep)
    if overlay.gate_bias.grad is not None:
        keep_bias = overlay.gate_bias.grad[index].clone()
        overlay.gate_bias.grad.zero_()
        overlay.gate_bias.grad[index].copy_(keep_bias)
    for parameter in (overlay.down, overlay.up):
        if parameter.grad is not None:
            parameter.grad.zero_()


def mask_transform_gradients_(overlay: HybridCellOverlay, slot: int) -> None:
    index = int(slot)
    for parameter in (overlay.gate_weight, overlay.gate_bias):
        if parameter.grad is not None:
            parameter.grad.zero_()
    for parameter in (overlay.down, overlay.up):
        if parameter.grad is not None:
            keep = parameter.grad[:, index].clone()
            parameter.grad.zero_()
            parameter.grad[:, index].copy_(keep)


def save_cell_artifact(path: Path, artifact: HybridCellArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cell_id": artifact.cell_id,
            "parent_id": artifact.parent_id,
            "version": artifact.version,
            "address_frozen": artifact.address_frozen,
            "digest": artifact.digest(),
            "state": artifact.state,
        },
        path,
    )


def load_cell_artifact(path: Path) -> HybridCellArtifact:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    artifact = HybridCellArtifact(
        cell_id=str(payload["cell_id"]),
        parent_id=payload["parent_id"],
        version=int(payload["version"]),
        address_frozen=bool(payload["address_frozen"]),
        state={name: tensor for name, tensor in payload["state"].items()},
    )
    if artifact.digest() != str(payload["digest"]):
        raise HybridCLMError(f"artifact digest mismatch: {path}")
    return artifact
