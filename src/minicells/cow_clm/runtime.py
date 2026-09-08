from __future__ import annotations

import contextlib
import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn.utils import parametrize


class COWCLMError(RuntimeError):
    """Raised when a copy-on-write Cell lifecycle invariant is violated."""


@dataclass(frozen=True, order=True)
class ExpertSite:
    layer: int
    expert: int

    def as_dict(self) -> dict[str, int]:
        return {"layer": int(self.layer), "expert": int(self.expert)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExpertSite:
        return cls(layer=int(payload["layer"]), expert=int(payload["expert"]))


class ExpertSliceDelta(nn.Module):
    """Additive COW parametrization over selected rows of a fused expert tensor.

    The parent tensor is never copied or mutated. A selected expert row starts with
    an exactly-zero delta, so the effective tensor is value-identical at Cell birth.
    """

    def __init__(self, base: torch.Tensor, expert_ids: Sequence[int]) -> None:
        super().__init__()
        ids = tuple(sorted({int(value) for value in expert_ids}))
        if not ids:
            raise COWCLMError("expert slice parametrization requires at least one expert")
        if min(ids) < 0 or max(ids) >= int(base.shape[0]):
            raise COWCLMError("expert index is outside fused expert tensor")
        self.register_buffer("expert_ids", torch.tensor(ids, dtype=torch.long))
        private_dtype = (
            torch.float32 if base.dtype in (torch.float16, torch.bfloat16) else base.dtype
        )
        self.delta = nn.Parameter(
            torch.zeros(
                (len(ids), *base.shape[1:]),
                dtype=private_dtype,
                device=base.device,
            )
        )

    def forward(self, parent: torch.Tensor) -> torch.Tensor:
        rows = self.expert_ids.to(device=parent.device)
        selected = parent.index_select(0, rows) + self.delta.to(dtype=parent.dtype)
        return parent.clone().index_copy(0, rows, selected)


@dataclass
class _PatchBinding:
    module_path: str
    parameter_name: str
    module: nn.Module
    parametrization: ExpertSliceDelta

    @property
    def state_key(self) -> str:
        return f"{self.module_path}::{self.parameter_name}"


@dataclass
class COWCell:
    cell_id: str
    parent_id: str
    parent_digest: str
    patch_modules: dict[str, nn.Module] = field(default_factory=dict)
    bindings: tuple[_PatchBinding, ...] = ()
    expert_sites: tuple[ExpertSite, ...] = ()

    def private_parameters(self) -> Iterator[nn.Parameter]:
        for binding in self.bindings:
            yield binding.parametrization.delta


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clone_module_shell(module: nn.Module) -> nn.Module:
    """Clone module metadata/registries while sharing immutable parent tensors/modules."""
    clone = copy.copy(module)
    clone._parameters = module._parameters.copy()
    clone._buffers = module._buffers.copy()
    clone._modules = module._modules.copy()
    clone._non_persistent_buffers_set = module._non_persistent_buffers_set.copy()
    return clone


def _set_submodule(model: nn.Module, path: str, module: nn.Module) -> None:
    if not path or "." not in path:
        parent_path, leaf = "", path
    else:
        parent_path, leaf = path.rsplit(".", 1)
    parent = model if not parent_path else model.get_submodule(parent_path)
    if leaf not in parent._modules:
        raise COWCLMError(f"target path is not a registered submodule: {path}")
    parent._modules[leaf] = module


def _granite_patch_targets(model: nn.Module, layer: int) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Discover the two Granite expert-storage layouts used by Transformers.

    Current Transformers stores gate/up + down tensors in one ``experts`` module.
    Older Granite 3.x releases use separate ``input_linear`` / ``output_linear``
    fused ParallelExperts modules. Both layouts keep expert id on tensor dimension 0.
    """
    block_path = f"model.layers.{int(layer)}.block_sparse_moe"
    try:
        block = model.get_submodule(block_path)
    except AttributeError as exc:
        raise COWCLMError(f"Granite MoE block not found: {block_path}") from exc

    experts = getattr(block, "experts", None)
    if experts is not None and hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        return ((f"{block_path}.experts", ("gate_up_proj", "down_proj")),)

    input_linear = getattr(block, "input_linear", None)
    output_linear = getattr(block, "output_linear", None)
    if (
        input_linear is not None
        and output_linear is not None
        and hasattr(input_linear, "weight")
        and hasattr(output_linear, "weight")
    ):
        return (
            (f"{block_path}.input_linear", ("weight",)),
            (f"{block_path}.output_linear", ("weight",)),
        )
    raise COWCLMError(
        f"unsupported Granite expert storage at {block_path}; "
        "expected experts.(gate_up_proj,down_proj) or input/output ParallelExperts"
    )


def _parameter_for(module: nn.Module, name: str) -> nn.Parameter:
    value = getattr(module, name, None)
    if not isinstance(value, nn.Parameter):
        raise COWCLMError(f"{type(module).__name__}.{name} is not a Parameter")
    if value.ndim < 2:
        raise COWCLMError(f"{type(module).__name__}.{name} is not a fused expert tensor")
    return value


class COWRuntime:
    """Persistent model-view runtime for root -> Cell expert-slice COW.

    v0.1 intentionally supports a single immutable Granite root and direct children.
    The public data model already records parent identity; deeper lineage is reserved
    for COW-CLM-002 so this experiment cannot silently invent merge semantics.
    """

    ROOT_CELL_ID = "granite-root"

    def __init__(
        self,
        model: nn.Module,
        *,
        foundation_model_id: str,
        foundation_revision: str,
    ) -> None:
        self.model = model
        self.foundation_model_id = str(foundation_model_id)
        self.foundation_revision = str(foundation_revision)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._active_cell_id: str | None = None
        self._cells: dict[str, COWCell] = {}
        self._foundation_guard: dict[str, tuple[int, int]] = {}
        self._refresh_foundation_guard()
        self.root_digest = _canonical_sha256(
            {
                "cell_id": self.ROOT_CELL_ID,
                "foundation_model_id": self.foundation_model_id,
                "foundation_revision": self.foundation_revision,
            }
        )

    @property
    def cells(self) -> tuple[str, ...]:
        return tuple(sorted(self._cells))

    def cell(self, cell_id: str) -> COWCell:
        try:
            return self._cells[str(cell_id)]
        except KeyError as exc:
            raise COWCLMError(f"unknown Cell: {cell_id}") from exc

    def fork_empty(self, cell_id: str, *, parent_id: str = ROOT_CELL_ID) -> COWCell:
        if parent_id != self.ROOT_CELL_ID:
            raise COWCLMError("COW-CLM v0.1 only permits direct forks from immutable root")
        return self._register_cell(
            COWCell(
                cell_id=str(cell_id),
                parent_id=self.ROOT_CELL_ID,
                parent_digest=self.root_digest,
            )
        )

    def fork_experts(
        self,
        cell_id: str,
        expert_sites: Sequence[ExpertSite | tuple[int, int]],
        *,
        parent_id: str = ROOT_CELL_ID,
    ) -> COWCell:
        if parent_id != self.ROOT_CELL_ID:
            raise COWCLMError("COW-CLM v0.1 only permits direct forks from immutable root")
        sites = tuple(
            sorted(
                {
                    value if isinstance(value, ExpertSite) else ExpertSite(*value)
                    for value in expert_sites
                }
            )
        )
        if not sites:
            return self.fork_empty(cell_id, parent_id=parent_id)

        by_layer: dict[int, list[int]] = defaultdict(list)
        for site in sites:
            by_layer[site.layer].append(site.expert)

        patch_modules: dict[str, nn.Module] = {}
        bindings: list[_PatchBinding] = []
        for layer, expert_ids in sorted(by_layer.items()):
            for module_path, parameter_names in _granite_patch_targets(self.model, layer):
                parent_module = self.model.get_submodule(module_path)
                clone = _clone_module_shell(parent_module)
                for parameter_name in parameter_names:
                    base = _parameter_for(parent_module, parameter_name)
                    if max(expert_ids) >= int(base.shape[0]):
                        highest = max(expert_ids)
                        raise COWCLMError(
                            f"expert index outside {module_path}.{parameter_name}: {highest}"
                        )
                    delta = ExpertSliceDelta(base, expert_ids)
                    parametrize.register_parametrization(clone, parameter_name, delta)
                    bindings.append(
                        _PatchBinding(
                            module_path=module_path,
                            parameter_name=parameter_name,
                            module=clone,
                            parametrization=delta,
                        )
                    )
                patch_modules[module_path] = clone

        cell = self._register_cell(
            COWCell(
                cell_id=str(cell_id),
                parent_id=self.ROOT_CELL_ID,
                parent_digest=self.root_digest,
                patch_modules=patch_modules,
                bindings=tuple(bindings),
                expert_sites=sites,
            )
        )
        self._refresh_foundation_guard()
        return cell

    def _register_cell(self, cell: COWCell) -> COWCell:
        if cell.cell_id == self.ROOT_CELL_ID:
            raise COWCLMError(f"{self.ROOT_CELL_ID!r} is reserved")
        if cell.cell_id in self._cells:
            raise COWCLMError(f"Cell already exists: {cell.cell_id}")
        self._cells[cell.cell_id] = cell
        return cell

    @contextlib.contextmanager
    def activate(self, cell_id: str) -> Iterator[COWCell | None]:
        """Select exactly one complete model view for the whole forward/training scope."""
        if self._active_cell_id is not None:
            raise COWCLMError("nested Cell activation is forbidden in COW-CLM v0.1")
        if cell_id == self.ROOT_CELL_ID:
            self._active_cell_id = self.ROOT_CELL_ID
            try:
                yield None
            finally:
                self._active_cell_id = None
            return

        cell = self.cell(cell_id)
        originals: dict[str, nn.Module] = {}
        self._active_cell_id = cell.cell_id
        try:
            for path, patch in cell.patch_modules.items():
                originals[path] = self.model.get_submodule(path)
                _set_submodule(self.model, path, patch)
            yield cell
        finally:
            for path, original in reversed(tuple(originals.items())):
                _set_submodule(self.model, path, original)
            self._active_cell_id = None

    def private_parameters(self, cell_id: str) -> tuple[nn.Parameter, ...]:
        return tuple(self.cell(cell_id).private_parameters())

    def private_parameter_count(self, cell_id: str) -> int:
        return sum(parameter.numel() for parameter in self.private_parameters(cell_id))

    def foundation_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def private_fraction(self, cell_id: str) -> float:
        denominator = self.foundation_parameter_count()
        return self.private_parameter_count(cell_id) / max(denominator, 1)

    def _refresh_foundation_guard(self) -> None:
        if self._active_cell_id is not None:
            raise COWCLMError("cannot seal foundation guard while a Cell is active")
        self._foundation_guard = {
            name: (id(parameter), int(parameter._version))
            for name, parameter in self.model.named_parameters()
        }

    def assert_foundation_unchanged(self) -> None:
        if self._active_cell_id is not None:
            raise COWCLMError("foundation guard must be checked with no active Cell")
        current = {
            name: (id(parameter), int(parameter._version))
            for name, parameter in self.model.named_parameters()
        }
        if current.keys() != self._foundation_guard.keys():
            raise COWCLMError("foundation parameter set changed")
        changed = [
            name
            for name, identity in current.items()
            if identity != self._foundation_guard[name]
        ]
        if changed:
            raise COWCLMError(f"foundation tensor mutated: {changed[0]}")

    def cell_state(self, cell_id: str) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for binding in self.cell(cell_id).bindings:
            state[binding.state_key] = binding.parametrization.delta.detach().cpu().clone()
        return state

    @torch.no_grad()
    def load_cell_state_(self, cell_id: str, state: dict[str, torch.Tensor]) -> None:
        cell = self.cell(cell_id)
        expected = {binding.state_key for binding in cell.bindings}
        if set(state) != expected:
            raise COWCLMError("Cell state keys differ from patch bindings")
        for binding in cell.bindings:
            source = state[binding.state_key]
            destination = binding.parametrization.delta
            if tuple(source.shape) != tuple(destination.shape):
                raise COWCLMError(f"Cell state shape mismatch: {binding.state_key}")
            destination.copy_(source.to(device=destination.device, dtype=destination.dtype))
