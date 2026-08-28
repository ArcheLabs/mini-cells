"""Optimized execution backends for CLM program-cell routing.

The CLM-0.3 research model intentionally keeps its scientific execution
semantics simple.  This module adds an optional runtime layer without changing
checkpoint formats, lineage structure, routing decisions, or birth semantics.

Backends provided by the installed runtime:

- ``masked_dense``: the original exact STE training path.
- ``batched_dense``: mathematically equivalent dense expert execution using
  batched matmuls.  It preserves gradients to every expert and to every router
  gate, so it is suitable for measuring a lower-overhead implementation of the
  existing training semantics.
- ``reference_sparse``: the original per-expert gather/execute/scatter path.
- ``sparse_dispatch``: optimized inference.  Under ``torch.no_grad()`` it
  groups tokens once and uses ``torch._grouped_mm`` for the two expert FFN
  projections.  If grouped GEMM is unavailable for the current device/shape,
  it falls back to ``reference_sparse``.  With gradients enabled it also falls
  back, because skipping unselected expert outputs would change the existing
  straight-through router gradient estimator.

The optimized runtime is installed explicitly with ``install_optimized_runtime``
so historical experiment semantics remain immutable.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any
from weakref import WeakKeyDictionary

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PackedExpertMLP:
    fingerprint: tuple[object, ...]
    expert_ids: tuple[str, ...]
    w1_t: torch.Tensor
    b1: torch.Tensor
    w2_t: torch.Tensor
    b2: torch.Tensor


_PACKED: WeakKeyDictionary[nn.Module, PackedExpertMLP] = WeakKeyDictionary()
_ORIGINAL_FORWARD: WeakKeyDictionary[nn.Module, Any] = WeakKeyDictionary()
_LAST_BACKEND: WeakKeyDictionary[nn.Module, str] = WeakKeyDictionary()
_LAST_FALLBACK: WeakKeyDictionary[nn.Module, str] = WeakKeyDictionary()


def _linear_gelu_linear(expert: nn.Module) -> tuple[nn.Linear, nn.Linear]:
    if not isinstance(expert, nn.Sequential) or len(expert) != 3:
        raise TypeError("optimized CLM runtime requires Linear-GELU-Linear experts")
    first, activation, second = expert[0], expert[1], expert[2]
    if not isinstance(first, nn.Linear) or not isinstance(activation, nn.GELU) or not isinstance(second, nn.Linear):
        raise TypeError("optimized CLM runtime requires Linear-GELU-Linear experts")
    if first.bias is None or second.bias is None:
        raise TypeError("optimized CLM runtime currently requires biased Linear experts")
    return first, second


def _fingerprint(bank: nn.Module, expert_ids: tuple[str, ...]) -> tuple[object, ...]:
    values: list[object] = [expert_ids]
    for expert_id in expert_ids:
        expert = bank.experts[expert_id]
        first, second = _linear_gelu_linear(expert)
        for tensor in (first.weight, first.bias, second.weight, second.bias):
            values.extend(
                (
                    int(tensor._version),
                    int(tensor.data_ptr()),
                    str(tensor.device),
                    str(tensor.dtype),
                    tuple(tensor.shape),
                )
            )
    return tuple(values)


def _pack(bank: nn.Module, expert_ids: tuple[str, ...]) -> PackedExpertMLP:
    fingerprint = _fingerprint(bank, expert_ids)
    cached = _PACKED.get(bank)
    if cached is not None and cached.fingerprint == fingerprint:
        return cached

    first_layers: list[nn.Linear] = []
    second_layers: list[nn.Linear] = []
    for expert_id in expert_ids:
        first, second = _linear_gelu_linear(bank.experts[expert_id])
        first_layers.append(first)
        second_layers.append(second)

    # torch._grouped_mm expects B in [groups, K, N] layout.  These detached
    # packed copies are inference-only and are invalidated whenever any source
    # parameter version, pointer, dtype, device, or expert-id set changes.
    packed = PackedExpertMLP(
        fingerprint=fingerprint,
        expert_ids=expert_ids,
        w1_t=torch.stack([layer.weight.detach().transpose(0, 1).contiguous() for layer in first_layers]),
        b1=torch.stack([layer.bias.detach() for layer in first_layers]),
        w2_t=torch.stack([layer.weight.detach().transpose(0, 1).contiguous() for layer in second_layers]),
        b2=torch.stack([layer.bias.detach() for layer in second_layers]),
    )
    _PACKED[bank] = packed
    return packed


def clear_runtime_cache(model: nn.Module) -> None:
    """Drop all packed inference weights associated with ``model``."""

    for stage in getattr(model, "stages", ()):
        bank = getattr(stage, "program_bank", None)
        if bank is not None:
            _PACKED.pop(bank, None)
            _LAST_BACKEND.pop(bank, None)
            _LAST_FALLBACK.pop(bank, None)


def _masked_dense(bank: nn.Module, perception: torch.Tensor, gates: torch.Tensor, expert_ids: tuple[str, ...]) -> torch.Tensor:
    output = torch.zeros_like(perception)
    for index, expert_id in enumerate(expert_ids):
        output = output + bank.experts[expert_id](perception) * gates[..., index, None]
    return output


def _batched_dense(bank: nn.Module, perception: torch.Tensor, gates: torch.Tensor, expert_ids: tuple[str, ...]) -> torch.Tensor:
    """Exact dense STE semantics with batched expert matmuls.

    Unlike the inference-only packed path, weights are stacked without detach so
    autograd still reaches every expert.  All expert outputs are computed and
    multiplied by the original straight-through gates, preserving the current
    router gradient estimator.
    """

    flat_inputs = perception.reshape(-1, perception.shape[-1])
    flat_gates = gates.reshape(-1, gates.shape[-1])
    first_layers: list[nn.Linear] = []
    second_layers: list[nn.Linear] = []
    for expert_id in expert_ids:
        first, second = _linear_gelu_linear(bank.experts[expert_id])
        first_layers.append(first)
        second_layers.append(second)

    w1_t = torch.stack([layer.weight.transpose(0, 1) for layer in first_layers])
    b1 = torch.stack([layer.bias for layer in first_layers])
    # [E, N, D] @ [E, D, F] -> [E, N, F], with the input broadcast over E.
    hidden = torch.matmul(flat_inputs.unsqueeze(0), w1_t) + b1[:, None, :]
    hidden = F.gelu(hidden)
    w2_t = torch.stack([layer.weight.transpose(0, 1) for layer in second_layers])
    b2 = torch.stack([layer.bias for layer in second_layers])
    expert_outputs = torch.matmul(hidden, w2_t) + b2[:, None, :]
    mixed = (expert_outputs * flat_gates.transpose(0, 1)[..., None]).sum(dim=0)
    return mixed.view_as(perception)


def _reference_sparse(bank: nn.Module, perception: torch.Tensor, gates: torch.Tensor, expert_ids: tuple[str, ...]) -> torch.Tensor:
    flat_inputs = perception.reshape(-1, perception.shape[-1])
    flat_gates = gates.reshape(-1, gates.shape[-1])
    flat_output = torch.zeros_like(flat_inputs)
    for index, expert_id in enumerate(expert_ids):
        active = flat_gates[:, index].detach().bool().nonzero(as_tuple=False).squeeze(-1)
        if active.numel() == 0:
            continue
        contribution = bank.experts[expert_id](flat_inputs.index_select(0, active))
        contribution = contribution * flat_gates.index_select(0, active)[:, index, None]
        flat_output.index_add_(0, active, contribution)
    return flat_output.view_as(perception)


def _grouped_sparse(bank: nn.Module, perception: torch.Tensor, gates: torch.Tensor, expert_ids: tuple[str, ...]) -> torch.Tensor:
    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("torch._grouped_mm is unavailable")
    if torch.is_grad_enabled():
        raise RuntimeError("grouped sparse path is inference-only under current STE semantics")

    flat_inputs = perception.reshape(-1, perception.shape[-1])
    flat_gates = gates.reshape(-1, gates.shape[-1])
    if flat_inputs.numel() == 0:
        return torch.zeros_like(perception)

    assignments = flat_gates.detach().argmax(dim=-1)
    order = torch.argsort(assignments, stable=True)
    sorted_assignments = assignments.index_select(0, order)
    sorted_inputs = flat_inputs.index_select(0, order).contiguous()
    counts = torch.bincount(sorted_assignments, minlength=len(expert_ids))
    offsets = counts.cumsum(dim=0, dtype=torch.int32)
    packed = _pack(bank, expert_ids)

    hidden = torch._grouped_mm(sorted_inputs, packed.w1_t, offs=offsets)
    hidden = hidden + packed.b1.index_select(0, sorted_assignments)
    hidden = F.gelu(hidden)
    sorted_output = torch._grouped_mm(hidden.contiguous(), packed.w2_t, offs=offsets)
    sorted_output = sorted_output + packed.b2.index_select(0, sorted_assignments)

    # Preserve exact hard-gate forward semantics, including merge-back paths.
    selected_gates = flat_gates.index_select(0, order).gather(
        1, sorted_assignments[:, None]
    ).to(dtype=sorted_output.dtype)
    sorted_output = sorted_output * selected_gates
    flat_output = torch.empty_like(sorted_output)
    flat_output.index_copy_(0, order, sorted_output)
    return flat_output.view_as(perception)


def _optimized_bank_forward(
    self: nn.Module,
    perception: torch.Tensor,
    *,
    backend: str = "masked_dense",
    merge_back_child: str | None = None,
):
    gates, root_indices, root_probabilities, choices, gates_by_id = self.route(
        perception, merge_back_child=merge_back_child
    )
    expert_ids = tuple(self.expert_ids)
    if backend == "masked_dense":
        output = _masked_dense(self, perception, gates, expert_ids)
        _LAST_BACKEND[self] = "masked_dense"
        _LAST_FALLBACK.pop(self, None)
    elif backend == "batched_dense":
        try:
            output = _batched_dense(self, perception, gates, expert_ids)
            _LAST_BACKEND[self] = "batched_dense"
            _LAST_FALLBACK.pop(self, None)
        except (TypeError, RuntimeError) as exc:
            output = _masked_dense(self, perception, gates, expert_ids)
            _LAST_BACKEND[self] = "masked_dense_fallback"
            _LAST_FALLBACK[self] = f"{type(exc).__name__}: {exc}"
    elif backend == "reference_sparse":
        output = _reference_sparse(self, perception, gates, expert_ids)
        _LAST_BACKEND[self] = "reference_sparse"
        _LAST_FALLBACK.pop(self, None)
    elif backend == "sparse_dispatch":
        if torch.is_grad_enabled():
            # Keep historical gradient semantics: the existing sparse path only
            # updates selected expert outputs and is not promoted to the formal
            # training backend by this runtime optimization.
            output = _reference_sparse(self, perception, gates, expert_ids)
            _LAST_BACKEND[self] = "reference_sparse_grad"
            _LAST_FALLBACK[self] = "grouped sparse is inference-only under current STE semantics"
        else:
            try:
                output = _grouped_sparse(self, perception, gates, expert_ids)
                _LAST_BACKEND[self] = "grouped_mm"
                _LAST_FALLBACK.pop(self, None)
            except (AttributeError, TypeError, RuntimeError, NotImplementedError) as exc:
                output = _reference_sparse(self, perception, gates, expert_ids)
                _LAST_BACKEND[self] = "reference_sparse_fallback"
                _LAST_FALLBACK[self] = f"{type(exc).__name__}: {exc}"
    else:
        raise ValueError(f"unknown execution backend: {backend}")
    return output, gates, root_indices, root_probabilities, choices, gates_by_id


def install_optimized_runtime(model: nn.Module) -> nn.Module:
    """Install optimized program-bank execution on a ProgressiveGrowthCLM.

    The operation is idempotent and changes no parameters or checkpoint state.
    """

    stages = getattr(model, "stages", None)
    if stages is None:
        raise TypeError("model does not expose CLM stages")
    installed = 0
    for stage in stages:
        bank = getattr(stage, "program_bank", None)
        if bank is None or not hasattr(bank, "route") or not hasattr(bank, "experts"):
            continue
        if bank not in _ORIGINAL_FORWARD:
            _ORIGINAL_FORWARD[bank] = bank.forward
            bank.forward = MethodType(_optimized_bank_forward, bank)
        installed += 1
    if installed == 0:
        raise TypeError("model contains no compatible CLM program banks")
    return model


def remove_optimized_runtime(model: nn.Module) -> nn.Module:
    for stage in getattr(model, "stages", ()):
        bank = getattr(stage, "program_bank", None)
        if bank is None:
            continue
        original = _ORIGINAL_FORWARD.pop(bank, None)
        if original is not None:
            bank.forward = original
        _PACKED.pop(bank, None)
        _LAST_BACKEND.pop(bank, None)
        _LAST_FALLBACK.pop(bank, None)
    return model


def runtime_status(model: nn.Module) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stage_index, stage in enumerate(getattr(model, "stages", ())):
        bank = getattr(stage, "program_bank", None)
        if bank is None:
            continue
        rows.append(
            {
                "stage": stage_index,
                "backend": _LAST_BACKEND.get(bank, "not_executed"),
                "fallback_reason": _LAST_FALLBACK.get(bank),
                "packed_inference_cache": bank in _PACKED,
                "expert_count": len(tuple(bank.expert_ids)),
            }
        )
    return rows
