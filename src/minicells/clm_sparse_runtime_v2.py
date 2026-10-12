"""Second-generation CLM sparse runtime focused on Turing/Tesla T4.

This module layers on top of :mod:`clm_sparse_runtime` without changing the
frozen CLM-0.3 research model. The main addition is a true top-1 sparse
``padded_sparse`` inference backend that keeps the expert dimension batched for
T4 while avoiding dense expert-token work for inactive routes.

For N tokens and E experts, the path allocates a fixed per-expert capacity C,
executes two ``torch.bmm`` calls over [E, C, D], and sends only overflow routes
through the exact reference sparse path. No routed token is dropped. Several
capacity factors are considered by the one-time autotuner. Every candidate
must pass a numerical parity gate against ``reference_sparse`` before it may be
timed or selected.

Training semantics are unchanged: ``batched_dense`` from runtime v1 remains the
optimized exact-STE training candidate, while top-1 sparse execution remains
inference-only.
"""

from __future__ import annotations

import math
from types import MethodType
from typing import Callable
from weakref import WeakKeyDictionary

import torch
from torch import nn

from . import clm_sparse_runtime as v1


AUTOTUNE_RTOL = 3e-3
AUTOTUNE_ATOL = 3e-5
PADDED_CAPACITY_FACTORS = (1.0, 1.25, 1.5, 2.0)

# Reporting state is deliberately detached from checkpoint/model state.
_LAST_ROUTE_TENSORS: WeakKeyDictionary[nn.Module, dict[str, object]] = WeakKeyDictionary()
_LAST_AUTOTUNE: WeakKeyDictionary[nn.Module, dict[str, object]] = WeakKeyDictionary()


def _parity_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    rtol: float = AUTOTUNE_RTOL,
    atol: float = AUTOTUNE_ATOL,
) -> dict[str, object]:
    if reference.shape != candidate.shape:
        return {
            "ok": False,
            "reason": f"shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}",
            "max_abs_diff": float("inf"),
        }
    if not bool(torch.isfinite(candidate).all().item()):
        return {
            "ok": False,
            "reason": "candidate produced non-finite values",
            "max_abs_diff": float("inf"),
        }
    difference = (reference - candidate).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    tolerance = atol + rtol * reference.abs()
    ok = bool((difference <= tolerance).all().item())
    return {
        "ok": ok,
        "reason": None if ok else "candidate exceeded runtime parity tolerance",
        "max_abs_diff": max_abs,
        "rtol": rtol,
        "atol": atol,
    }


def _record_route_shape(
    bank: nn.Module,
    assignments: torch.Tensor,
    expert_count: int,
    *,
    selected_backend: str | None = None,
    capacity: int | None = None,
    overflow_mask: torch.Tensor | None = None,
) -> None:
    counts = torch.bincount(assignments, minlength=expert_count).detach()
    row: dict[str, object] = {
        "token_count": int(assignments.numel()),
        "expert_count": int(expert_count),
        "counts": counts,
        "selected_backend": selected_backend,
        "capacity": capacity,
        "overflow_count": overflow_mask.detach().sum() if overflow_mask is not None else None,
    }
    _LAST_ROUTE_TENSORS[bank] = row


def _padded_sparse(
    bank: nn.Module,
    perception: torch.Tensor,
    gates: torch.Tensor,
    expert_ids: tuple[str, ...],
    *,
    capacity_factor: float,
) -> torch.Tensor:
    """T4-friendly exact top-1 sparse FFN using fixed-shape batched GEMMs.

    Capacity is derived from tensor shape rather than observed expert load, so
    the recurrent hot path never needs a load-dependent GPU-to-CPU sync.
    Main-path packing uses fixed-length sort/index_add/gather tensors; overflow
    is represented as a fixed-shape gate mask and evaluated by the exact
    reference sparse path. No token is dropped.
    """

    if torch.is_grad_enabled():
        raise RuntimeError("padded sparse execution is inference-only under current STE semantics")
    if capacity_factor < 1.0:
        raise ValueError("capacity_factor must be >= 1")

    flat_inputs = perception.reshape(-1, perception.shape[-1])
    flat_gates = gates.reshape(-1, gates.shape[-1])
    token_count = int(flat_inputs.shape[0])
    expert_count = len(expert_ids)
    if token_count == 0:
        return torch.zeros_like(perception)

    compute_dtype = v1._inference_compute_dtype(perception)
    packed = v1._pack(bank, expert_ids, compute_dtype)
    assignments = flat_gates.detach().argmax(dim=-1)
    order = torch.argsort(assignments, stable=True)
    sorted_assignments = assignments.index_select(0, order)
    counts = torch.bincount(sorted_assignments, minlength=expert_count)
    starts = counts.cumsum(0) - counts
    positions = torch.arange(token_count, device=perception.device) - starts.index_select(
        0, sorted_assignments
    )

    capacity = max(1, int(math.ceil(token_count * capacity_factor / expert_count)))
    keep_sorted = positions < capacity
    overflow_sorted = ~keep_sorted
    safe_positions = positions.clamp_max(capacity - 1)

    # Every tensor below has a shape determined by N/E/capacity, not by the
    # observed number of kept or overflow routes. Overflow values are multiplied
    # by zero before index_add, so they cannot overwrite a valid capacity slot.
    sorted_inputs = flat_inputs.index_select(0, order).to(dtype=compute_dtype)
    kept_values = sorted_inputs * keep_sorted[:, None].to(dtype=compute_dtype)
    flat_slots = sorted_assignments * capacity + safe_positions
    padded_flat = torch.zeros(
        (expert_count * capacity, flat_inputs.shape[-1]),
        device=perception.device,
        dtype=compute_dtype,
    )
    padded_flat.index_add_(0, flat_slots, kept_values)
    padded = padded_flat.view(expert_count, capacity, flat_inputs.shape[-1])

    hidden = torch.bmm(padded, packed.w1_t) + packed.b1[:, None, :]
    hidden = torch.nn.functional.gelu(hidden)
    expert_output = torch.bmm(hidden, packed.w2_t) + packed.b2[:, None, :]

    selected_sorted = expert_output[sorted_assignments, safe_positions]
    selected_gates_sorted = flat_gates.index_select(0, order).gather(
        1, sorted_assignments[:, None]
    ).to(dtype=selected_sorted.dtype)
    selected_sorted = (
        selected_sorted
        * selected_gates_sorted
        * keep_sorted[:, None].to(dtype=selected_sorted.dtype)
    )

    flat_output = torch.zeros_like(flat_inputs)
    flat_output.index_copy_(0, order, selected_sorted.to(dtype=flat_output.dtype))

    # Fixed-size overflow mask in original token order. reference_sparse sees
    # zero gates for the already-computed routes and exact top-1 gates only for
    # overflow routes; this keeps correctness without a dynamic host branch.
    overflow_original = torch.zeros(token_count, device=perception.device, dtype=torch.bool)
    overflow_original.index_copy_(0, order, overflow_sorted)
    overflow_gates = flat_gates * overflow_original[:, None].to(dtype=flat_gates.dtype)
    overflow_output = v1._reference_sparse(
        bank,
        flat_inputs,
        overflow_gates,
        expert_ids,
    )
    flat_output = flat_output + overflow_output

    _record_route_shape(
        bank,
        assignments,
        expert_count,
        selected_backend=f"padded_sparse_{capacity_factor:.2f}",
        capacity=capacity,
        overflow_mask=overflow_sorted,
    )
    return flat_output.view_as(perception)


def _autotuned_inference_v2(
    bank: nn.Module,
    perception: torch.Tensor,
    gates: torch.Tensor,
    expert_ids: tuple[str, ...],
) -> tuple[torch.Tensor, str, str | None]:
    compute_dtype = v1._inference_compute_dtype(perception)
    packed = v1._pack(bank, expert_ids, compute_dtype)
    flat_tokens = int(perception.numel() // perception.shape[-1])
    key = (
        "runtime-v2-padded-sparse-fixed-shape",
        str(perception.device),
        str(perception.dtype),
        str(compute_dtype),
        flat_tokens,
        len(expert_ids),
        packed.fingerprint,
    )
    choices = v1._AUTOTUNE.setdefault(bank, {})
    selected = choices.get(key)

    flat_gates = gates.reshape(-1, gates.shape[-1])
    assignments = flat_gates.detach().argmax(dim=-1)
    _record_route_shape(bank, assignments, len(expert_ids), selected_backend=selected)

    candidates: dict[str, Callable[[], torch.Tensor]] = {
        "reference_sparse": lambda: v1._reference_sparse(bank, perception, gates, expert_ids),
        "packed_batched_dense": lambda: v1._packed_dense_inference(
            bank, perception, gates, expert_ids
        ),
    }
    for factor in PADDED_CAPACITY_FACTORS:
        name = f"padded_sparse_{factor:.2f}"
        candidates[name] = lambda factor=factor: _padded_sparse(
            bank,
            perception,
            gates,
            expert_ids,
            capacity_factor=factor,
        )
    if v1._has_grouped_mm(perception):
        candidates["grouped_mm"] = lambda: v1._grouped_sparse(
            bank, perception, gates, expert_ids
        )

    failures: list[str] = []
    if selected is None:
        reference = candidates["reference_sparse"]()
        timings: dict[str, float] = {
            "reference_sparse": v1._elapsed_cuda_or_cpu(
                candidates["reference_sparse"], perception.device
            )
        }
        parity: dict[str, dict[str, object]] = {
            "reference_sparse": {
                "ok": True,
                "reason": None,
                "max_abs_diff": 0.0,
                "rtol": AUTOTUNE_RTOL,
                "atol": AUTOTUNE_ATOL,
            }
        }

        for name, fn in candidates.items():
            if name == "reference_sparse":
                continue
            try:
                candidate_output = fn()
                metrics = _parity_metrics(reference, candidate_output)
                parity[name] = metrics
                if not bool(metrics["ok"]):
                    failures.append(
                        f"{name}: parity failed (max_abs_diff={metrics['max_abs_diff']})"
                    )
                    continue
                timings[name] = v1._elapsed_cuda_or_cpu(fn, perception.device)
            except (AttributeError, TypeError, RuntimeError, NotImplementedError) as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")

        if not timings:
            raise RuntimeError("no parity-safe CLM inference runtime candidate succeeded")
        selected = min(timings, key=timings.get)
        choices[key] = selected
        _LAST_AUTOTUNE[bank] = {
            "selected": selected,
            "timings_seconds": timings,
            "parity": parity,
            "failures": tuple(failures),
        }

    if selected not in candidates:
        choices.pop(key, None)
        return _autotuned_inference_v2(bank, perception, gates, expert_ids)

    output = candidates[selected]()
    if not selected.startswith("padded_sparse_"):
        _record_route_shape(
            bank,
            assignments,
            len(expert_ids),
            selected_backend=selected,
        )
    return output, selected, "; ".join(failures) if failures else None


def _optimized_bank_forward_v2(
    self: nn.Module,
    perception: torch.Tensor,
    *,
    backend: str = "masked_dense",
    merge_back_child: str | None = None,
):
    if self.training and torch.is_grad_enabled():
        v1._PACKED.pop(self, None)
        v1._AUTOTUNE.pop(self, None)
        _LAST_AUTOTUNE.pop(self, None)

    gates, root_indices, root_probabilities, choices, gates_by_id = self.route(
        perception, merge_back_child=merge_back_child
    )
    expert_ids = tuple(self.expert_ids)
    if backend == "masked_dense":
        output = v1._masked_dense(self, perception, gates, expert_ids)
        v1._LAST_BACKEND[self] = "masked_dense"
        v1._LAST_FALLBACK.pop(self, None)
    elif backend == "batched_dense":
        try:
            output = v1._batched_dense(self, perception, gates, expert_ids)
            v1._LAST_BACKEND[self] = "batched_dense"
            v1._LAST_FALLBACK.pop(self, None)
        except (TypeError, RuntimeError) as exc:
            output = v1._masked_dense(self, perception, gates, expert_ids)
            v1._LAST_BACKEND[self] = "masked_dense_fallback"
            v1._LAST_FALLBACK[self] = f"{type(exc).__name__}: {exc}"
    elif backend == "reference_sparse":
        output = v1._reference_sparse(self, perception, gates, expert_ids)
        v1._LAST_BACKEND[self] = "reference_sparse"
        v1._LAST_FALLBACK.pop(self, None)
    elif backend == "sparse_dispatch":
        if torch.is_grad_enabled():
            output = v1._reference_sparse(self, perception, gates, expert_ids)
            v1._LAST_BACKEND[self] = "reference_sparse_grad"
            v1._LAST_FALLBACK[self] = (
                "optimized sparse execution is inference-only under current STE semantics"
            )
        else:
            output, selected, failure = _autotuned_inference_v2(
                self, perception, gates, expert_ids
            )
            v1._LAST_BACKEND[self] = f"autotuned_{selected}"
            if failure is None:
                v1._LAST_FALLBACK.pop(self, None)
            else:
                v1._LAST_FALLBACK[self] = failure
    else:
        raise ValueError(f"unknown execution backend: {backend}")
    return output, gates, root_indices, root_probabilities, choices, gates_by_id


def install_optimized_runtime(model: nn.Module) -> nn.Module:
    """Install runtime v2 without mutating runtime-v1 process globals."""

    v1.install_optimized_runtime(model)
    for stage in getattr(model, "stages", ()):
        bank = getattr(stage, "program_bank", None)
        if bank is not None and bank in v1._ORIGINAL_BANK_FORWARD:
            bank.forward = MethodType(_optimized_bank_forward_v2, bank)
    return model


def clear_runtime_cache(model: nn.Module) -> None:
    v1.clear_runtime_cache(model)
    for stage in getattr(model, "stages", ()):
        bank = getattr(stage, "program_bank", None)
        if bank is not None:
            _LAST_ROUTE_TENSORS.pop(bank, None)
            _LAST_AUTOTUNE.pop(bank, None)


def runtime_status(model: nn.Module) -> list[dict[str, object]]:
    rows = v1.runtime_status(model)
    for row, stage in zip(rows, getattr(model, "stages", ())):
        bank = getattr(stage, "program_bank", None)
        if bank is None:
            continue
        route = _LAST_ROUTE_TENSORS.get(bank)
        if route is not None:
            counts_tensor = route["counts"]
            assert isinstance(counts_tensor, torch.Tensor)
            counts = [int(value) for value in counts_tensor.detach().cpu().tolist()]
            token_count = int(route["token_count"])
            expert_count = int(route["expert_count"])
            capacity = route.get("capacity")
            overflow_value = route.get("overflow_count")
            overflow_count = (
                int(overflow_value.detach().cpu().item())
                if isinstance(overflow_value, torch.Tensor)
                else 0
            )
            selected_backend = str(route.get("selected_backend") or row.get("backend"))
            dense_pairs = max(1, token_count * expert_count)
            if selected_backend.startswith("padded_sparse_") and isinstance(capacity, int):
                executed_pairs = expert_count * capacity + overflow_count
            elif selected_backend in {"reference_sparse", "grouped_mm"}:
                executed_pairs = token_count
            else:
                executed_pairs = dense_pairs
            row.update(
                {
                    "route_counts": counts,
                    "max_expert_load": max(counts) if counts else 0,
                    "min_expert_load": min(counts) if counts else 0,
                    "route_load_ratio": (
                        max(counts) / max(1.0, token_count / expert_count)
                        if counts
                        else 0.0
                    ),
                    "padded_capacity": capacity,
                    "overflow_tokens": overflow_count,
                    "expert_token_pairs_executed": int(executed_pairs),
                    "dense_expert_token_pairs": int(dense_pairs),
                    "expert_token_pair_fraction_vs_dense": executed_pairs / dense_pairs,
                }
            )
        autotune = _LAST_AUTOTUNE.get(bank)
        if autotune is not None:
            row["autotune_v2"] = autotune
    return rows


remove_optimized_runtime = v1.remove_optimized_runtime
