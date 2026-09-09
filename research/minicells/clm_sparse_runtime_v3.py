"""Third-generation CLM sparse runtime for Turing/Tesla T4.

Runtime v3 keeps the frozen CLM-0.3 research semantics unchanged while fixing
three shortcomings exposed by the T4 v2 benchmark:

1. Precision-aware parity.  FP16 candidates are compared with an error envelope
   appropriate for FP16 GEMM rather than an FP32-like absolute tolerance.
2. Calibrate once, execute hot.  Each stage/recurrent-step/token-shape is
   autotuned once.  Stable inference then jumps directly to the cached backend
   without rebuilding candidate dictionaries or collecting route telemetry.
3. Load-aware two-tier padding.  Calibration may split experts into high-load
   and low-load buckets, each with its own fixed capacity and one batched GEMM.
   This preserves batching while reducing padding under skewed router traffic.

Top-1 sparse execution remains inference-only under the current straight-
through router estimator. Training continues to use the exact-STE
``masked_dense`` or ``batched_dense`` backends from runtime v1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable
from weakref import WeakKeyDictionary

import torch
from torch import nn
from torch.nn import functional as F

from . import clm_sparse_runtime as v1
from .language_models import LanguageModelOutput


FP32_ATOL = 3e-5
FP32_RTOL = 3e-3
FP16_ATOL = 2e-3
FP16_RTOL = 8e-3
BF16_ATOL = 2e-2
BF16_RTOL = 3e-2
REL_L2_LIMIT = 2e-2
CAPACITY_ALIGNMENT = 32
FIXED_CAPACITY_FACTORS = (1.0, 1.25, 1.5, 2.0)
TIER_MARGINS = (1.0, 1.10, 1.25)


@dataclass(frozen=True)
class TieredCapacityPlan:
    groups: tuple[tuple[int, ...], ...]
    capacities: tuple[int, ...]
    source_counts: tuple[int, ...]
    margin: float
    alignment: int = CAPACITY_ALIGNMENT

    @property
    def padded_pairs(self) -> int:
        return sum(len(group) * capacity for group, capacity in zip(self.groups, self.capacities))

    @property
    def name(self) -> str:
        group_text = "-".join(str(len(group)) for group in self.groups)
        cap_text = "-".join(str(value) for value in self.capacities)
        return f"tiered_{group_text}_c{cap_text}_m{self.margin:.2f}"

    def to_dict(self) -> dict[str, object]:
        return {
            "groups": [list(group) for group in self.groups],
            "capacities": list(self.capacities),
            "source_counts": list(self.source_counts),
            "margin": self.margin,
            "alignment": self.alignment,
            "padded_pairs": self.padded_pairs,
        }


@dataclass(frozen=True)
class HotSelection:
    name: str
    kind: str
    capacity: int | None = None
    tiered_plan: TieredCapacityPlan | None = None


_HOT: WeakKeyDictionary[nn.Module, dict[tuple[object, ...], HotSelection]] = WeakKeyDictionary()
_CALIBRATION: WeakKeyDictionary[nn.Module, dict[tuple[object, ...], dict[str, object]]] = WeakKeyDictionary()


def _align(value: float, alignment: int = CAPACITY_ALIGNMENT) -> int:
    return max(alignment, int(math.ceil(value / alignment)) * alignment)


def _precision_tolerances(compute_dtype: torch.dtype) -> tuple[float, float]:
    if compute_dtype == torch.float16:
        return FP16_RTOL, FP16_ATOL
    if compute_dtype == torch.bfloat16:
        return BF16_RTOL, BF16_ATOL
    return FP32_RTOL, FP32_ATOL


def precision_aware_parity(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
) -> dict[str, object]:
    """Return a precision-aware local runtime parity decision.

    The router has already run before expert execution, so runtime candidates
    receive identical gates and routing assignments.  This check therefore
    focuses on numerical drift introduced by GEMM packing/execution order.
    """

    if reference.shape != candidate.shape:
        return {
            "ok": False,
            "reason": "shape mismatch",
            "max_abs_diff": float("inf"),
            "rms_diff": float("inf"),
            "relative_l2": float("inf"),
        }
    if not torch.isfinite(candidate).all():
        return {
            "ok": False,
            "reason": "candidate produced non-finite values",
            "max_abs_diff": float("inf"),
            "rms_diff": float("inf"),
            "relative_l2": float("inf"),
        }

    rtol, atol = _precision_tolerances(compute_dtype)
    difference = (reference.float() - candidate.float())
    abs_difference = difference.abs()
    max_abs = float(abs_difference.max().item()) if abs_difference.numel() else 0.0
    rms = float(difference.square().mean().sqrt().item()) if difference.numel() else 0.0
    reference_norm = float(reference.float().norm().item())
    relative_l2 = float(difference.norm().item()) / max(reference_norm, 1e-12)
    elementwise_ok = bool(
        torch.allclose(reference.float(), candidate.float(), rtol=rtol, atol=atol)
    )
    ok = elementwise_ok and relative_l2 <= REL_L2_LIMIT
    return {
        "ok": ok,
        "reason": None if ok else "candidate exceeded precision-aware parity envelope",
        "compute_dtype": str(compute_dtype),
        "rtol": rtol,
        "atol": atol,
        "max_abs_diff": max_abs,
        "rms_diff": rms,
        "relative_l2": relative_l2,
    }


def make_tiered_capacity_plans(
    counts: tuple[int, ...],
    *,
    token_count: int,
    alignment: int = CAPACITY_ALIGNMENT,
) -> tuple[TieredCapacityPlan, ...]:
    """Generate a small set of two-tier load-aware capacity plans."""

    expert_count = len(counts)
    if expert_count < 2:
        return ()
    ranked = tuple(sorted(range(expert_count), key=lambda index: counts[index], reverse=True))
    plans: list[TieredCapacityPlan] = []
    seen: set[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]] = set()
    for split in range(1, expert_count):
        groups = (ranked[:split], ranked[split:])
        for margin in TIER_MARGINS:
            capacities = tuple(
                min(
                    token_count,
                    _align(max(counts[index] for index in group) * margin, alignment),
                )
                for group in groups
            )
            signature = (groups, capacities)
            if signature in seen:
                continue
            seen.add(signature)
            plans.append(
                TieredCapacityPlan(
                    groups=groups,
                    capacities=capacities,
                    source_counts=counts,
                    margin=margin,
                    alignment=alignment,
                )
            )
    plans.sort(key=lambda plan: (plan.padded_pairs, len(plan.groups), plan.capacities))
    return tuple(plans[:6])


def _packed_dense_cached(
    perception: torch.Tensor,
    gates: torch.Tensor,
    packed: v1.PackedExpertMLP,
) -> torch.Tensor:
    flat_inputs = perception.reshape(-1, perception.shape[-1]).to(dtype=packed.compute_dtype)
    flat_gates = gates.reshape(-1, gates.shape[-1])
    hidden = torch.matmul(flat_inputs.unsqueeze(0), packed.w1_t) + packed.b1[:, None, :]
    hidden = F.gelu(hidden)
    expert_outputs = torch.matmul(hidden, packed.w2_t) + packed.b2[:, None, :]
    mixed = (
        expert_outputs
        * flat_gates.transpose(0, 1)[..., None].to(dtype=expert_outputs.dtype)
    ).sum(dim=0)
    return mixed.to(dtype=perception.dtype).view_as(perception)


def _route_geometry(gates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_gates = gates.reshape(-1, gates.shape[-1])
    assignments = flat_gates.detach().argmax(dim=-1)
    order = torch.argsort(assignments, stable=True)
    sorted_assignments = assignments.index_select(0, order)
    counts = torch.bincount(sorted_assignments, minlength=flat_gates.shape[-1])
    starts = counts.cumsum(0) - counts
    positions = torch.arange(order.shape[0], device=order.device) - starts.index_select(
        0, sorted_assignments
    )
    return assignments, order, sorted_assignments, positions


def _fixed_padded_sparse(
    bank: nn.Module,
    perception: torch.Tensor,
    gates: torch.Tensor,
    expert_ids: tuple[str, ...],
    packed: v1.PackedExpertMLP,
    *,
    capacity: int,
) -> torch.Tensor:
    """Exact fixed-capacity sparse execution with overflow fallback."""

    flat_inputs = perception.reshape(-1, perception.shape[-1])
    flat_gates = gates.reshape(-1, gates.shape[-1])
    token_count = flat_inputs.shape[0]
    expert_count = len(expert_ids)
    assignments, order, sorted_assignments, positions = _route_geometry(gates)
    keep = positions < capacity
    safe_positions = positions.clamp(max=max(capacity - 1, 0))
    slots = sorted_assignments * capacity + safe_positions

    padded_flat = torch.zeros(
        (expert_count * capacity, flat_inputs.shape[-1]),
        device=perception.device,
        dtype=packed.compute_dtype,
    )
    sorted_inputs = flat_inputs.index_select(0, order).to(dtype=packed.compute_dtype)
    padded_flat.index_add_(
        0,
        slots,
        sorted_inputs * keep[:, None].to(dtype=sorted_inputs.dtype),
    )
    padded = padded_flat.view(expert_count, capacity, flat_inputs.shape[-1])
    hidden = torch.bmm(padded, packed.w1_t) + packed.b1[:, None, :]
    hidden = F.gelu(hidden)
    expert_output = torch.bmm(hidden, packed.w2_t) + packed.b2[:, None, :]

    selected = expert_output.reshape(expert_count * capacity, -1).index_select(0, slots)
    selected_gates = flat_gates.index_select(0, order).gather(
        1, sorted_assignments[:, None]
    ).to(dtype=selected.dtype)
    selected = selected * selected_gates * keep[:, None].to(dtype=selected.dtype)

    flat_output = torch.zeros_like(flat_inputs)
    flat_output.index_copy_(0, order, selected.to(dtype=flat_output.dtype))

    overflow_sorted = ~keep
    overflow_original = torch.zeros(token_count, device=perception.device, dtype=torch.bool)
    overflow_original.index_copy_(0, order, overflow_sorted)
    overflow_gates = flat_gates * overflow_original[:, None].to(dtype=flat_gates.dtype)
    overflow_output = v1._reference_sparse(bank, flat_inputs, overflow_gates, expert_ids)
    return (flat_output + overflow_output).view_as(perception)


def _tiered_padded_sparse(
    bank: nn.Module,
    perception: torch.Tensor,
    gates: torch.Tensor,
    expert_ids: tuple[str, ...],
    packed: v1.PackedExpertMLP,
    plan: TieredCapacityPlan,
) -> torch.Tensor:
    """Two-tier sparse BMM execution derived from calibration route loads."""

    flat_inputs = perception.reshape(-1, perception.shape[-1])
    flat_gates = gates.reshape(-1, gates.shape[-1])
    token_count = flat_inputs.shape[0]
    expert_count = len(expert_ids)
    assignments, order, sorted_assignments, positions = _route_geometry(gates)
    sorted_inputs = flat_inputs.index_select(0, order).to(dtype=packed.compute_dtype)
    sorted_gate_values = flat_gates.index_select(0, order).gather(
        1, sorted_assignments[:, None]
    )

    flat_output = torch.zeros_like(flat_inputs)
    keep_any = torch.zeros(token_count, device=perception.device, dtype=torch.bool)

    for group, capacity in zip(plan.groups, plan.capacities):
        group_indices = torch.tensor(group, device=perception.device, dtype=torch.long)
        lookup = torch.full(
            (expert_count,),
            -1,
            device=perception.device,
            dtype=torch.long,
        )
        lookup[group_indices] = torch.arange(len(group), device=perception.device)
        local_assignments = lookup.index_select(0, sorted_assignments)
        in_group = local_assignments >= 0
        keep = in_group & (positions < capacity)
        keep_any |= keep
        safe_local = local_assignments.clamp_min(0)
        safe_positions = positions.clamp(max=max(capacity - 1, 0))
        slots = safe_local * capacity + safe_positions

        padded_flat = torch.zeros(
            (len(group) * capacity, flat_inputs.shape[-1]),
            device=perception.device,
            dtype=packed.compute_dtype,
        )
        padded_flat.index_add_(
            0,
            slots,
            sorted_inputs * keep[:, None].to(dtype=sorted_inputs.dtype),
        )
        padded = padded_flat.view(len(group), capacity, flat_inputs.shape[-1])
        w1 = packed.w1_t.index_select(0, group_indices)
        b1 = packed.b1.index_select(0, group_indices)
        w2 = packed.w2_t.index_select(0, group_indices)
        b2 = packed.b2.index_select(0, group_indices)
        hidden = torch.bmm(padded, w1) + b1[:, None, :]
        hidden = F.gelu(hidden)
        expert_output = torch.bmm(hidden, w2) + b2[:, None, :]
        selected = expert_output.reshape(len(group) * capacity, -1).index_select(0, slots)
        selected = (
            selected
            * sorted_gate_values.to(dtype=selected.dtype)
            * keep[:, None].to(dtype=selected.dtype)
        )
        flat_output.index_add_(0, order, selected.to(dtype=flat_output.dtype))

    overflow_sorted = ~keep_any
    overflow_original = torch.zeros(token_count, device=perception.device, dtype=torch.bool)
    overflow_original.index_copy_(0, order, overflow_sorted)
    overflow_gates = flat_gates * overflow_original[:, None].to(dtype=flat_gates.dtype)
    overflow_output = v1._reference_sparse(bank, flat_inputs, overflow_gates, expert_ids)
    return (flat_output + overflow_output).view_as(perception)


def _estimate_pairs(
    selection: HotSelection,
    counts: tuple[int, ...],
    *,
    token_count: int,
    expert_count: int,
) -> tuple[int, int]:
    if selection.kind in {"reference_sparse", "grouped_mm"}:
        return token_count, 0
    if selection.kind == "packed_batched_dense":
        return token_count * expert_count, 0
    if selection.kind == "fixed_padded":
        assert selection.capacity is not None
        overflow = sum(max(0, count - selection.capacity) for count in counts)
        return expert_count * selection.capacity + overflow, overflow
    if selection.kind == "tiered_padded":
        assert selection.tiered_plan is not None
        capacities_by_expert = [0] * expert_count
        for group, capacity in zip(
            selection.tiered_plan.groups, selection.tiered_plan.capacities
        ):
            for expert in group:
                capacities_by_expert[expert] = capacity
        overflow = sum(
            max(0, count - capacities_by_expert[index])
            for index, count in enumerate(counts)
        )
        return selection.tiered_plan.padded_pairs + overflow, overflow
    raise ValueError(f"unknown selection kind: {selection.kind}")


def _execute_selection(
    selection: HotSelection,
    bank: nn.Module,
    perception: torch.Tensor,
    gates: torch.Tensor,
    expert_ids: tuple[str, ...],
    packed: v1.PackedExpertMLP,
) -> torch.Tensor:
    if selection.kind == "reference_sparse":
        return v1._reference_sparse(bank, perception, gates, expert_ids)
    if selection.kind == "packed_batched_dense":
        return _packed_dense_cached(perception, gates, packed)
    if selection.kind == "fixed_padded":
        assert selection.capacity is not None
        return _fixed_padded_sparse(
            bank,
            perception,
            gates,
            expert_ids,
            packed,
            capacity=selection.capacity,
        )
    if selection.kind == "tiered_padded":
        assert selection.tiered_plan is not None
        return _tiered_padded_sparse(
            bank,
            perception,
            gates,
            expert_ids,
            packed,
            selection.tiered_plan,
        )
    if selection.kind == "grouped_mm":
        return v1._grouped_sparse(bank, perception, gates, expert_ids)
    raise ValueError(f"unknown selection kind: {selection.kind}")


def _runtime_key(
    bank: nn.Module,
    perception: torch.Tensor,
    expert_ids: tuple[str, ...],
    compute_dtype: torch.dtype,
) -> tuple[object, ...]:
    step = int(getattr(bank, "_clm_runtime_step_index", -1))
    flat_tokens = int(perception.numel() // perception.shape[-1])
    return (
        "runtime-v3",
        step,
        str(perception.device),
        str(perception.dtype),
        str(compute_dtype),
        flat_tokens,
        expert_ids,
    )


def _autotuned_inference_v3(
    bank: nn.Module,
    perception: torch.Tensor,
    gates: torch.Tensor,
    expert_ids: tuple[str, ...],
) -> tuple[torch.Tensor, str, str | None]:
    compute_dtype = v1._inference_compute_dtype(perception)
    key = _runtime_key(bank, perception, expert_ids, compute_dtype)
    hot_by_key = _HOT.setdefault(bank, {})
    selection = hot_by_key.get(key)
    packed = v1._PACKED.get(bank)

    if selection is not None and packed is not None:
        if packed.expert_ids == expert_ids and packed.compute_dtype == compute_dtype:
            output = _execute_selection(
                selection, bank, perception, gates, expert_ids, packed
            )
            return output, selection.name, None
        hot_by_key.pop(key, None)

    packed = v1._pack(bank, expert_ids, compute_dtype)
    flat_gates = gates.reshape(-1, gates.shape[-1])
    assignments = flat_gates.detach().argmax(dim=-1)
    counts_tensor = torch.bincount(assignments, minlength=len(expert_ids))
    # Calibration is intentionally allowed one device sync. Stable hot-path
    # inference never copies route counts to the host.
    counts = tuple(int(value) for value in counts_tensor.detach().cpu().tolist())
    token_count = int(assignments.numel())
    expert_count = len(expert_ids)

    candidates: list[tuple[HotSelection, Callable[[], torch.Tensor]]] = []
    reference_selection = HotSelection("reference_sparse", "reference_sparse")
    candidates.append(
        (
            reference_selection,
            lambda: v1._reference_sparse(bank, perception, gates, expert_ids),
        )
    )
    packed_selection = HotSelection("packed_batched_dense", "packed_batched_dense")
    candidates.append(
        (
            packed_selection,
            lambda: _packed_dense_cached(perception, gates, packed),
        )
    )
    for factor in FIXED_CAPACITY_FACTORS:
        capacity = max(1, int(math.ceil(token_count * factor / expert_count)))
        selection_row = HotSelection(
            f"fixed_padded_c{capacity}",
            "fixed_padded",
            capacity=capacity,
        )
        candidates.append(
            (
                selection_row,
                lambda capacity=capacity: _fixed_padded_sparse(
                    bank,
                    perception,
                    gates,
                    expert_ids,
                    packed,
                    capacity=capacity,
                ),
            )
        )
    for plan in make_tiered_capacity_plans(counts, token_count=token_count):
        selection_row = HotSelection(plan.name, "tiered_padded", tiered_plan=plan)
        candidates.append(
            (
                selection_row,
                lambda plan=plan: _tiered_padded_sparse(
                    bank,
                    perception,
                    gates,
                    expert_ids,
                    packed,
                    plan,
                ),
            )
        )
    if v1._has_grouped_mm(perception):
        grouped_selection = HotSelection("grouped_mm", "grouped_mm")
        candidates.append(
            (
                grouped_selection,
                lambda: v1._grouped_sparse(bank, perception, gates, expert_ids),
            )
        )

    reference = candidates[0][1]()
    timings: dict[str, float] = {}
    parity: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    safe: dict[str, HotSelection] = {}

    for candidate_selection, fn in candidates:
        try:
            candidate_output = reference if candidate_selection.kind == "reference_sparse" else fn()
            metrics = precision_aware_parity(
                reference,
                candidate_output,
                compute_dtype=compute_dtype,
            )
            parity[candidate_selection.name] = metrics
            if not bool(metrics["ok"]):
                failures.append(
                    f"{candidate_selection.name}: parity failed "
                    f"(max_abs_diff={metrics['max_abs_diff']})"
                )
                continue
            timings[candidate_selection.name] = v1._elapsed_cuda_or_cpu(
                fn,
                perception.device,
            )
            safe[candidate_selection.name] = candidate_selection
        except (AttributeError, TypeError, RuntimeError, NotImplementedError) as exc:
            failures.append(
                f"{candidate_selection.name}: {type(exc).__name__}: {exc}"
            )

    if not timings:
        raise RuntimeError("no precision-safe CLM runtime candidate succeeded")
    selected_name = min(timings, key=timings.get)
    selection = safe[selected_name]
    hot_by_key[key] = selection

    executed_pairs, overflow = _estimate_pairs(
        selection,
        counts,
        token_count=token_count,
        expert_count=expert_count,
    )
    calibration = _CALIBRATION.setdefault(bank, {})
    calibration[key] = {
        "step": int(getattr(bank, "_clm_runtime_step_index", -1)),
        "token_count": token_count,
        "expert_count": expert_count,
        "route_counts": list(counts),
        "route_load_ratio": max(counts) / max(1.0, token_count / expert_count),
        "selected": selection.name,
        "selected_kind": selection.kind,
        "selected_capacity": selection.capacity,
        "selected_tiered_plan": (
            selection.tiered_plan.to_dict() if selection.tiered_plan is not None else None
        ),
        "timings_seconds": timings,
        "parity": parity,
        "failures": failures,
        "dense_expert_token_pairs": token_count * expert_count,
        "expert_token_pairs_executed_estimate": executed_pairs,
        "expert_token_pair_fraction_vs_dense": executed_pairs / max(1, token_count * expert_count),
        "calibration_overflow_tokens": overflow,
    }

    output = _execute_selection(selection, bank, perception, gates, expert_ids, packed)
    return output, selection.name, "; ".join(failures) if failures else None


def _fast_model_forward_v3(
    self: nn.Module,
    input_ids: torch.Tensor,
    *,
    execution_backend: str = "masked_dense",
    return_stats: bool = False,
    return_debug: bool = False,
    merge_back: tuple[int, str] | None = None,
):
    original = v1._ORIGINAL_MODEL_FORWARD[self]
    if return_stats or return_debug:
        for stage in self.stages:
            setattr(stage.program_bank, "_clm_runtime_step_index", -1)
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
            setattr(stage.program_bank, "_clm_runtime_step_index", int(step))
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
    """Install runtime v3 without changing CLM research/checkpoint semantics."""

    v1._autotuned_inference = _autotuned_inference_v3
    v1.install_optimized_runtime(model)
    model.forward = MethodType(_fast_model_forward_v3, model)
    return model


def clear_runtime_cache(model: nn.Module) -> None:
    v1.clear_runtime_cache(model)
    for stage in getattr(model, "stages", ()):
        bank = getattr(stage, "program_bank", None)
        if bank is not None:
            _HOT.pop(bank, None)
            _CALIBRATION.pop(bank, None)


def remove_optimized_runtime(model: nn.Module) -> nn.Module:
    clear_runtime_cache(model)
    return v1.remove_optimized_runtime(model)


def runtime_status(model: nn.Module) -> list[dict[str, object]]:
    rows = v1.runtime_status(model)
    for row, stage in zip(rows, getattr(model, "stages", ())):
        bank = getattr(stage, "program_bank", None)
        if bank is None:
            continue
        profiles = list(_CALIBRATION.get(bank, {}).values())
        profiles.sort(key=lambda item: int(item["step"]))
        row["runtime_generation"] = 3
        row["hot_profile_count"] = len(_HOT.get(bank, {}))
        row["calibrated_profile_count"] = len(profiles)
        row["calibration_profiles"] = profiles
    return rows
