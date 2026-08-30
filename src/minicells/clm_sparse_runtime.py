"""Optimized execution backends for CLM program-cell routing.

This module is an optional runtime layer over the frozen CLM-0.3 research
model. It changes no checkpoint format, routing decision, lineage relation,
birth rule, or probationary-growth semantics.

Backends installed on every CLM program bank:

- ``masked_dense``: original exact STE training semantics.
- ``batched_dense``: the same dense STE semantics expressed as two batched
  expert matmuls. All experts are still evaluated, so gradients to experts and
  router gates are preserved.
- ``reference_sparse``: original per-expert nonzero/index_select/index_add
  implementation.
- ``sparse_dispatch``: optimized inference policy. It one-time autotunes the
  available exact-forward implementations for each stage/token shape. On
  SM80+ this includes PyTorch grouped GEMM; on Tesla T4/SM75 it compares the
  original sparse path with a persistent packed-batched expert path and caches
  the faster choice. With gradients enabled it retains historical
  reference-sparse behavior rather than silently changing the STE estimator.

The installer also adds a telemetry-free fast model forward for ordinary calls
that do not request stats/debug state. Historical experiments remain immutable
because this runtime must be installed explicitly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable
from weakref import WeakKeyDictionary

import torch
from torch import nn
from torch.nn import functional as F

from .language_models import LanguageModelOutput


@dataclass(frozen=True)
class PackedExpertMLP:
    fingerprint: tuple[object, ...]
    expert_ids: tuple[str, ...]
    compute_dtype: torch.dtype
    w1_t: torch.Tensor
    b1: torch.Tensor
    w2_t: torch.Tensor
    b2: torch.Tensor


_PACKED: WeakKeyDictionary[nn.Module, PackedExpertMLP] = WeakKeyDictionary()
_ORIGINAL_BANK_FORWARD: WeakKeyDictionary[nn.Module, Any] = WeakKeyDictionary()
_ORIGINAL_MODEL_FORWARD: WeakKeyDictionary[nn.Module, Any] = WeakKeyDictionary()
_LAST_BACKEND: WeakKeyDictionary[nn.Module, str] = WeakKeyDictionary()
_LAST_FALLBACK: WeakKeyDictionary[nn.Module, str] = WeakKeyDictionary()
_AUTOTUNE: WeakKeyDictionary[nn.Module, dict[tuple[object, ...], str]] = WeakKeyDictionary()


def _linear_gelu_linear(expert: nn.Module) -> tuple[nn.Linear, nn.Linear]:
    if not isinstance(expert, nn.Sequential) or len(expert) != 3:
        raise TypeError("optimized CLM runtime requires Linear-GELU-Linear experts")
    first, activation, second = expert[0], expert[1], expert[2]
    if not isinstance(first, nn.Linear) or not isinstance(activation, nn.GELU) or not isinstance(second, nn.Linear):
        raise TypeError("optimized CLM runtime requires Linear-GELU-Linear experts")
    if first.bias is None or second.bias is None:
        raise TypeError("optimized CLM runtime currently requires biased Linear experts")
    return first, second


def _inference_compute_dtype(perception: torch.Tensor) -> torch.dtype:
    device_type = perception.device.type
    if device_type in {"cuda", "cpu"}:
        try:
            if torch.is_autocast_enabled(device_type):
                return torch.get_autocast_dtype(device_type)
        except TypeError:
            # Compatibility with older torch signatures; CUDA autocast was the
            # only device-global mode exposed there.
            if device_type == "cuda" and torch.is_autocast_enabled():
                return torch.float16
    return perception.dtype


def _fingerprint(
    bank: nn.Module,
    expert_ids: tuple[str, ...],
    compute_dtype: torch.dtype,
) -> tuple[object, ...]:
    values: list[object] = [expert_ids, str(compute_dtype)]
    for expert_id in expert_ids:
        first, second = _linear_gelu_linear(bank.experts[expert_id])
        for tensor in (first.weight, first.bias, second.weight, second.bias):
            values.append(
                (
                    int(tensor._version),
                    int(tensor.data_ptr()),
                    str(tensor.device),
                    str(tensor.dtype),
                    tuple(tensor.shape),
                )
            )
    return tuple(values)


def _pack(
    bank: nn.Module,
    expert_ids: tuple[str, ...],
    compute_dtype: torch.dtype,
) -> PackedExpertMLP:
    fingerprint = _fingerprint(bank, expert_ids, compute_dtype)
    cached = _PACKED.get(bank)
    if cached is not None and cached.fingerprint == fingerprint:
        return cached
    if cached is not None:
        _AUTOTUNE.pop(bank, None)

    first_layers: list[nn.Linear] = []
    second_layers: list[nn.Linear] = []
    for expert_id in expert_ids:
        first, second = _linear_gelu_linear(bank.experts[expert_id])
        first_layers.append(first)
        second_layers.append(second)

    packed = PackedExpertMLP(
        fingerprint=fingerprint,
        expert_ids=expert_ids,
        compute_dtype=compute_dtype,
        # grouped_mm consumes [E, K, N]. Detached copies are inference-only and
        # are materialized once in the active inference compute dtype so T4 does
        # not repeatedly cast persistent FP32 parameters on every recurrent step.
        w1_t=torch.stack(
            [layer.weight.detach().transpose(0, 1).contiguous() for layer in first_layers]
        ).to(dtype=compute_dtype),
        b1=torch.stack([layer.bias.detach() for layer in first_layers]).to(dtype=compute_dtype),
        w2_t=torch.stack(
            [layer.weight.detach().transpose(0, 1).contiguous() for layer in second_layers]
        ).to(dtype=compute_dtype),
        b2=torch.stack([layer.bias.detach() for layer in second_layers]).to(dtype=compute_dtype),
    )
    _PACKED[bank] = packed
    return packed


def clear_runtime_cache(model: nn.Module) -> None:
    for stage in getattr(model, "stages", ()):
        bank = getattr(stage, "program_bank", None)
        if bank is not None:
            _PACKED.pop(bank, None)
            _LAST_BACKEND.pop(bank, None)
            _LAST_FALLBACK.pop(bank, None)
            _AUTOTUNE.pop(bank, None)


def _masked_dense(bank: nn.Module, perception: torch.Tensor, gates: torch.Tensor, expert_ids: tuple[str, ...]) -> torch.Tensor:
    output = torch.zeros_like(perception)
    for index, expert_id in enumerate(expert_ids):
        output = output + bank.experts[expert_id](perception) * gates[..., index, None]
    return output


def _batched_dense(bank: nn.Module, perception: torch.Tensor, gates: torch.Tensor, expert_ids: tuple[str, ...]) -> torch.Tensor:
    """Exact dense STE semantics with batched expert matmuls and autograd."""

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
    hidden = torch.matmul(flat_inputs.unsqueeze(0), w1_t) + b1[:, None, :]
    hidden = F.gelu(hidden)
    w2_t = torch.stack([layer.weight.transpose(0, 1) for layer in second_layers])
    b2 = torch.stack([layer.bias for layer in second_layers])
    expert_outputs = torch.matmul(hidden, w2_t) + b2[:, None, :]
    mixed = (expert_outputs * flat_gates.transpose(0, 1)[..., None]).sum(dim=0)
    return mixed.view_as(perception)


def _packed_dense_inference(
    bank: nn.Module,
    perception: torch.Tensor,
    gates: torch.Tensor,
    expert_ids: tuple[str, ...],
) -> torch.Tensor:
    """Inference-only dense expert batching using persistent packed weights."""

    flat_inputs = perception.reshape(-1, perception.shape[-1])
    flat_gates = gates.reshape(-1, gates.shape[-1])
    compute_dtype = _inference_compute_dtype(perception)
    packed = _pack(bank, expert_ids, compute_dtype)
    runtime_inputs = flat_inputs.to(dtype=compute_dtype)
    hidden = torch.matmul(runtime_inputs.unsqueeze(0), packed.w1_t) + packed.b1[:, None, :]
    hidden = F.gelu(hidden)
    expert_outputs = torch.matmul(hidden, packed.w2_t) + packed.b2[:, None, :]
    mixed = (
        expert_outputs
        * flat_gates.transpose(0, 1)[..., None].to(dtype=expert_outputs.dtype)
    ).sum(dim=0)
    return mixed.to(dtype=perception.dtype).view_as(perception)


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


def _has_grouped_mm(perception: torch.Tensor) -> bool:
    has_api = hasattr(F, "grouped_mm") or hasattr(torch, "_grouped_mm")
    if not has_api:
        return False
    if perception.device.type == "cuda":
        # PyTorch >=2.9 grouped_mm CUDA kernels require Ampere (SM80) or newer.
        return torch.cuda.get_device_capability(perception.device) >= (8, 0)
    return True


def _grouped_mm(a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    if hasattr(F, "grouped_mm"):
        return F.grouped_mm(a, b, offs=offsets)
    if hasattr(torch, "_grouped_mm"):
        return torch._grouped_mm(a, b, offs=offsets)
    raise RuntimeError("grouped_mm API is unavailable")


def _grouped_sparse(bank: nn.Module, perception: torch.Tensor, gates: torch.Tensor, expert_ids: tuple[str, ...]) -> torch.Tensor:
    if torch.is_grad_enabled():
        raise RuntimeError("grouped sparse path is inference-only under current STE semantics")
    if not _has_grouped_mm(perception):
        raise RuntimeError("grouped_mm is unsupported on this device")

    flat_inputs = perception.reshape(-1, perception.shape[-1])
    flat_gates = gates.reshape(-1, gates.shape[-1])
    if flat_inputs.numel() == 0:
        return torch.zeros_like(perception)

    compute_dtype = _inference_compute_dtype(perception)
    assignments = flat_gates.detach().argmax(dim=-1)
    order = torch.argsort(assignments, stable=True)
    sorted_assignments = assignments.index_select(0, order)
    sorted_inputs = flat_inputs.index_select(0, order).to(dtype=compute_dtype).contiguous()
    counts = torch.bincount(sorted_assignments, minlength=len(expert_ids))
    offsets = counts.cumsum(dim=0, dtype=torch.int32)
    packed = _pack(bank, expert_ids, compute_dtype)

    hidden = _grouped_mm(sorted_inputs, packed.w1_t, offsets)
    hidden = hidden + packed.b1.index_select(0, sorted_assignments)
    hidden = F.gelu(hidden)
    sorted_output = _grouped_mm(hidden.contiguous(), packed.w2_t, offsets)
    sorted_output = sorted_output + packed.b2.index_select(0, sorted_assignments)

    selected_gates = flat_gates.index_select(0, order).gather(
        1, sorted_assignments[:, None]
    ).to(dtype=sorted_output.dtype)
    sorted_output = sorted_output * selected_gates
    flat_output = torch.empty_like(sorted_output)
    flat_output.index_copy_(0, order, sorted_output)
    return flat_output.to(dtype=perception.dtype).view_as(perception)


def _elapsed_cuda_or_cpu(fn: Callable[[], torch.Tensor], device: torch.device, repeats: int = 3) -> float:
    # Synchronization is intentional only during one-time backend selection.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        return max(float(start.elapsed_time(end)) / 1000.0, 1e-9)
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    return max(time.perf_counter() - started, 1e-9)


def _autotuned_inference(
    bank: nn.Module,
    perception: torch.Tensor,
    gates: torch.Tensor,
    expert_ids: tuple[str, ...],
) -> tuple[torch.Tensor, str, str | None]:
    compute_dtype = _inference_compute_dtype(perception)
    packed = _pack(bank, expert_ids, compute_dtype)
    flat_tokens = int(perception.numel() // perception.shape[-1])
    key = (
        str(perception.device),
        str(perception.dtype),
        str(compute_dtype),
        flat_tokens,
        len(expert_ids),
        packed.fingerprint,
    )
    choices = _AUTOTUNE.setdefault(bank, {})
    selected = choices.get(key)
    failures: list[str] = []

    candidates: dict[str, Callable[[], torch.Tensor]] = {
        "reference_sparse": lambda: _reference_sparse(bank, perception, gates, expert_ids),
        "packed_batched_dense": lambda: _packed_dense_inference(bank, perception, gates, expert_ids),
    }
    if _has_grouped_mm(perception):
        candidates["grouped_mm"] = lambda: _grouped_sparse(bank, perception, gates, expert_ids)

    if selected is None:
        timings: dict[str, float] = {}
        for name, fn in candidates.items():
            try:
                fn()  # warm the candidate
                timings[name] = _elapsed_cuda_or_cpu(fn, perception.device)
            except (AttributeError, TypeError, RuntimeError, NotImplementedError) as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        if not timings:
            raise RuntimeError("no CLM inference runtime candidate succeeded: " + "; ".join(failures))
        selected = min(timings, key=timings.get)
        choices[key] = selected

    output = candidates[selected]()
    return output, selected, "; ".join(failures) if failures else None


def _optimized_bank_forward(
    self: nn.Module,
    perception: torch.Tensor,
    *,
    backend: str = "masked_dense",
    merge_back_child: str | None = None,
):
    if self.training and torch.is_grad_enabled():
        # Any subsequent optimizer update would stale packed inference weights.
        _PACKED.pop(self, None)
        _AUTOTUNE.pop(self, None)

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
            output = _reference_sparse(self, perception, gates, expert_ids)
            _LAST_BACKEND[self] = "reference_sparse_grad"
            _LAST_FALLBACK[self] = "optimized sparse execution is inference-only under current STE semantics"
        else:
            output, selected, failure = _autotuned_inference(self, perception, gates, expert_ids)
            _LAST_BACKEND[self] = f"autotuned_{selected}"
            if failure is None:
                _LAST_FALLBACK.pop(self, None)
            else:
                _LAST_FALLBACK[self] = failure
    else:
        raise ValueError(f"unknown execution backend: {backend}")
    return output, gates, root_indices, root_probabilities, choices, gates_by_id


def _fast_model_forward(
    self: nn.Module,
    input_ids: torch.Tensor,
    *,
    execution_backend: str = "masked_dense",
    return_stats: bool = False,
    return_debug: bool = False,
    merge_back: tuple[int, str] | None = None,
):
    original = _ORIGINAL_MODEL_FORWARD[self]
    if return_stats or return_debug:
        return original(
            input_ids,
            execution_backend=execution_backend,
            return_stats=return_stats,
            return_debug=return_debug,
            merge_back=merge_back,
        )
    if input_ids.ndim != 2 or input_ids.shape[1] > self.max_context:
        raise ValueError(f"input_ids must be [batch, <= {self.max_context}]")

    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    state = self.token_embedding(input_ids) + self.position_embedding(positions)[None]
    intermediate: list[torch.Tensor] = []
    for stage_index, stage in enumerate(self.stages):
        child = merge_back[1] if merge_back is not None and merge_back[0] == stage_index else None
        batch, length, dim = state.shape
        for step in range(stage.iterations):
            conditioned = state + stage.step_embedding[step].view(1, 1, dim)
            attention_delta = stage.attention(stage.norm_attention(conditioned))
            perception = stage.norm_ffn(state + attention_delta)
            ffn_delta, *_ = stage.program_bank(
                perception,
                backend=execution_backend,
                merge_back_child=child,
            )
            proposal = attention_delta + ffn_delta
            state = stage.gru(
                proposal.reshape(batch * length, dim),
                state.reshape(batch * length, dim),
            ).view(batch, length, dim)
        if self.stage_supervision and stage_index < len(self.stages) - 1:
            intermediate.append(self.lm_head(self.final_norm(state)))

    logits = self.lm_head(self.final_norm(state))
    return LanguageModelOutput(
        logits,
        tuple([*intermediate, logits]) if self.stage_supervision else (),
    )


def install_optimized_runtime(model: nn.Module) -> nn.Module:
    """Install optimized program-bank and telemetry-free model execution."""

    stages = getattr(model, "stages", None)
    if stages is None:
        raise TypeError("model does not expose CLM stages")
    installed = 0
    for stage in stages:
        bank = getattr(stage, "program_bank", None)
        if bank is None or not hasattr(bank, "route") or not hasattr(bank, "experts"):
            continue
        if bank not in _ORIGINAL_BANK_FORWARD:
            _ORIGINAL_BANK_FORWARD[bank] = bank.forward
            bank.forward = MethodType(_optimized_bank_forward, bank)
        installed += 1
    if installed == 0:
        raise TypeError("model contains no compatible CLM program banks")
    if model not in _ORIGINAL_MODEL_FORWARD:
        _ORIGINAL_MODEL_FORWARD[model] = model.forward
        model.forward = MethodType(_fast_model_forward, model)
    return model


def remove_optimized_runtime(model: nn.Module) -> nn.Module:
    original_model = _ORIGINAL_MODEL_FORWARD.pop(model, None)
    if original_model is not None:
        model.forward = original_model
    for stage in getattr(model, "stages", ()):
        bank = getattr(stage, "program_bank", None)
        if bank is None:
            continue
        original = _ORIGINAL_BANK_FORWARD.pop(bank, None)
        if original is not None:
            bank.forward = original
        _PACKED.pop(bank, None)
        _LAST_BACKEND.pop(bank, None)
        _LAST_FALLBACK.pop(bank, None)
        _AUTOTUNE.pop(bank, None)
    return model


def runtime_status(model: nn.Module) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stage_index, stage in enumerate(getattr(model, "stages", ())):
        bank = getattr(stage, "program_bank", None)
        if bank is None:
            continue
        packed = _PACKED.get(bank)
        rows.append(
            {
                "stage": stage_index,
                "backend": _LAST_BACKEND.get(bank, "not_executed"),
                "fallback_reason": _LAST_FALLBACK.get(bank),
                "packed_inference_cache": packed is not None,
                "packed_compute_dtype": str(packed.compute_dtype) if packed is not None else None,
                "autotune_shapes": len(_AUTOTUNE.get(bank, {})),
                "expert_count": len(tuple(bank.expert_ids)),
                "fast_model_forward": model in _ORIGINAL_MODEL_FORWARD,
            }
        )
    return rows
