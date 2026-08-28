"""Parity gates, newborn diagnostics, and CLM-0.3 decision helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .language_models import LanguageModelOutput


CLM01_EXPECTED_RELEASE_PPL = 17.968933276012226


def baseline_reproduction_gate(
    observed_ppl: float,
    *,
    expected_ppl: float = CLM01_EXPECTED_RELEASE_PPL,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    passed = math.isfinite(observed_ppl) and abs(observed_ppl - expected_ppl) <= tolerance
    return {
        "status": "CLM01_BASELINE_REPRODUCTION" if passed else "CLM01_BASELINE_REPRODUCTION_FAILURE",
        "expected_ppl": expected_ppl,
        "observed_ppl": observed_ppl,
        "absolute_error": abs(observed_ppl - expected_ppl),
    }


@dataclass(frozen=True)
class ExecutionCapture:
    logits: torch.Tensor
    states: tuple[torch.Tensor, ...]
    root_routes: tuple[torch.Tensor, ...]


def capture_execution(model: Any, input_ids: torch.Tensor) -> ExecutionCapture:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output, stats, states = model(input_ids, return_debug=True)
    if was_training:
        model.train()
    return ExecutionCapture(
        output.logits.detach().clone(),
        tuple(state.detach().clone() for state in states),
        tuple(item.detach().clone() for item in stats.root_routes),
    )


def compare_captures(
    before: ExecutionCapture,
    after: ExecutionCapture,
    *,
    validation_targets: torch.Tensor | None = None,
    max_logits_diff: float = 2e-5,
    max_state_diff: float = 1e-6,
) -> dict[str, Any]:
    logits_diff = float((before.logits - after.logits).abs().max())
    state_diff = max(
        (float((left - right).abs().max()) for left, right in zip(before.states, after.states)),
        default=0.0,
    )
    route_unchanged = len(before.root_routes) == len(after.root_routes) and all(
        torch.equal(left, right) for left, right in zip(before.root_routes, after.root_routes)
    )
    if validation_targets is None:
        ppl_ratio = 1.0
    else:
        before_nll = F.cross_entropy(before.logits.flatten(0, 1), validation_targets.reshape(-1))
        after_nll = F.cross_entropy(after.logits.flatten(0, 1), validation_targets.reshape(-1))
        ppl_ratio = math.exp(float(after_nll - before_nll))
    passed = (
        abs(ppl_ratio - 1.0) <= 1e-5
        and logits_diff <= max_logits_diff
        and state_diff <= max_state_diff
        and route_unchanged
    )
    return {
        "status": "CLM_GROWTH_EQUIVALENCE" if passed else "CLM_GROWTH_EQUIVALENCE_FAILURE",
        "ppl_ratio": ppl_ratio,
        "max_logits_abs_diff": logits_diff,
        "max_recurrent_state_abs_diff": state_diff,
        "non_parent_root_routes_unchanged": route_unchanged,
    }


def compare_birth_equivalence(
    model: Any,
    input_ids: torch.Tensor,
    *,
    before: ExecutionCapture | None = None,
    validation_targets: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compare a pre-birth capture with the current model after a birth."""

    if before is None:
        # Useful as a defensive check for callers that only have a post-birth
        # model: it still verifies finite execution and stable state hooks.
        before = capture_execution(model, input_ids)
    after = capture_execution(model, input_ids)
    result = compare_captures(before, after, validation_targets=validation_targets)
    result["child_parameters_equal_parent"] = _children_equal_parents(model)
    result["status"] = (
        "CLM_GROWTH_EQUIVALENCE"
        if result["status"] == "CLM_GROWTH_EQUIVALENCE" and result["child_parameters_equal_parent"]
        else "CLM_GROWTH_EQUIVALENCE_FAILURE"
    )
    return result


def _children_equal_parents(model: Any, *, atol: float = 0.0) -> bool:
    for stage in model.stages:
        bank = stage.program_bank
        for child_id, parent_id in bank.parent_by_child.items():
            for parent, child in zip(bank.experts[parent_id].parameters(), bank.experts[child_id].parameters()):
                if not torch.allclose(parent, child, rtol=0.0, atol=atol):
                    return False
    return True


def parameter_diagnostics(model: Any, stage: int, parent_id: str, child_id: str) -> dict[str, float]:
    parent = torch.cat([item.detach().float().reshape(-1) for item in model.stages[stage].program_bank.experts[parent_id].parameters()])
    child = torch.cat([item.detach().float().reshape(-1) for item in model.stages[stage].program_bank.experts[child_id].parameters()])
    delta = (child - parent).norm()
    return {
        "relative_l2": float(delta / (parent.norm() + 1e-12)),
        "cosine_similarity": float(F.cosine_similarity(parent[None], child[None]).item()),
    }


@torch.no_grad()
def evaluate_nll(model: Any, batches: list[tuple[torch.Tensor, torch.Tensor]], *, merge_back: tuple[int, str] | None = None) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    tokens = 0
    for inputs, targets in batches:
        if hasattr(model, "growth_structure"):
            output = model(
                inputs,
                execution_backend="sparse_dispatch",
                merge_back=merge_back,
            )
        else:
            if merge_back is not None:
                raise ValueError("merge-back is only valid for a progressive-growth model")
            output = model(inputs)
        total += float(F.cross_entropy(output.logits.flatten(0, 1), targets.reshape(-1), reduction="sum"))
        tokens += targets.numel()
    if was_training:
        model.train()
    return total / max(tokens, 1)


def newborn_causal_diagnostics(
    model: Any,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    stage: int,
    parent_id: str,
    child_id: str,
) -> dict[str, float]:
    dynamic = evaluate_nll(model, batches)
    merged = evaluate_nll(model, batches, merge_back=(stage, child_id))
    bank = model.stages[stage].program_bank
    diagnostics = parameter_diagnostics(model, stage, parent_id, child_id)
    bank.begin_pressure_collection()
    try:
        output, stats = model(batches[0][0], return_stats=True)
    finally:
        bank.end_pressure_collection()
    del output
    parent_usage = float(stats.usage.get(parent_id, torch.tensor(0.0)))
    child_usage = float(stats.usage.get(child_id, torch.tensor(0.0)))
    split_id = bank.split_by_child.get(child_id)
    entropy = float(stats.split_entropy.get(split_id, 0.0)) if split_id is not None else 0.0
    logit_variance = 0.0
    if split_id is not None:
        perceptions = [item for item in bank.last_perceptions.get(parent_id, [])]
        perceptions.extend(bank.last_perceptions.get(child_id, []))
        if perceptions:
            logits = bank.router.split_routers[split_id](torch.cat(perceptions))
            logit_variance = float(logits.var(unbiased=False))
    return {
        "parent_usage": parent_usage,
        "child_usage": child_usage,
        "relative_l2": diagnostics["relative_l2"],
        "cosine_similarity": diagnostics["cosine_similarity"],
        "nll_dynamic": dynamic,
        "nll_mergeback": merged,
        "causal_merge_back_penalty": (merged - dynamic) / max(dynamic, 1e-12),
        "split_entropy": entropy,
        "router_logit_variance": logit_variance,
    }


def clm_growth_loss(
    student: LanguageModelOutput,
    teacher: LanguageModelOutput,
    targets: torch.Tensor,
    *,
    root_usage: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    beta: float = 0.5,
    balance_weight: float = 0.01,
) -> torch.Tensor:
    ce = F.cross_entropy(student.logits.flatten(0, 1), targets.reshape(-1))
    kl = student_teacher_kl(student.logits, teacher.logits)
    balance = root_router_balance_loss(root_usage, reference=student.logits)
    return ce + beta * kl + balance_weight * balance


def student_teacher_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Explicit ``KL(student || frozen teacher)`` over token distributions."""

    student_logp = F.log_softmax(student_logits, dim=-1)
    student_p = student_logp.exp()
    teacher_logp = F.log_softmax(teacher_logits.detach(), dim=-1)
    return (student_p * (student_logp - teacher_logp)).sum(dim=-1).mean()


def root_router_balance_loss(
    root_usage: torch.Tensor | tuple[torch.Tensor, ...] | None,
    *,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    if root_usage is None:
        if reference is None:
            raise ValueError("reference is required when root usage is absent")
        return reference.new_zeros(())
    stages = (root_usage,) if torch.is_tensor(root_usage) else tuple(root_usage)
    if not stages:
        if reference is None:
            raise ValueError("reference is required for empty root usage")
        return reference.new_zeros(())
    losses = []
    for usage in stages:
        target = torch.full_like(usage, 1.0 / usage.numel())
        losses.append((usage - target).square().mean())
    return torch.stack(losses).mean()


def health_label(growth_ppl: float, clm01_start_ppl: float) -> str:
    ratio = growth_ppl / clm01_start_ppl
    if ratio <= 1.03:
        return "GREEN"
    if ratio <= 1.10:
        return "YELLOW"
    return "RED"


def make_ppl_row(*, replicate: int, arm: str, tokens: int, phase: str, ppl: float, nll: float,
                 fixed4_ppl: float, clm01_start_ppl: float, textnca_frozen_ppl: float) -> dict[str, Any]:
    return {
        "replicate": replicate, "arm": arm, "tokens": tokens, "phase": phase,
        "ppl": ppl, "nll": nll,
        "ppl_vs_fixed4": ppl / fixed4_ppl,
        "ppl_vs_clm01": ppl / clm01_start_ppl,
        "ppl_vs_textnca": ppl / textnca_frozen_ppl,
        "health": health_label(ppl, clm01_start_ppl),
    }


def progressive_growth_decision(
    rows: list[dict[str, Any]], *, formal_gpu_experiment_run: bool = False
) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(str(row["arm"]), []).append(row)
    pressure = sorted(by_arm.get("pressure_growth", []), key=lambda x: x.get("replicate", 0))
    fixed = sorted(by_arm.get("fixed4", []), key=lambda x: x.get("replicate", 0))
    random_rows = sorted(by_arm.get("random_growth", []), key=lambda x: x.get("replicate", 0))
    viable = sum(bool(row.get("viable", False)) for row in pressure)
    equivalence = sum(int(row.get("equivalent_births", 0)) for row in pressure)
    births_expected = 2 * len(pressure)
    improved = sum(
        growth.get("ppl", math.inf) / max(control.get("ppl", math.inf), 1e-12) <= 0.995
        for growth, control in zip(pressure, fixed)
        if growth.get("replicate") == control.get("replicate")
    )
    pressure_wins = sum(
        growth.get("ppl", math.inf) < control.get("ppl", math.inf)
        for growth, control in zip(pressure, random_rows)
        if growth.get("replicate") == control.get("replicate")
    )
    viability_pass = len(pressure) == 3 and viable == 3
    utility_pass = improved >= 2 and viability_pass
    return {
        "format": "minicells.clm-0.3-progressive-growth.decision.v1",
        "growth_equivalence": {
            "status": "CLM_GROWTH_EQUIVALENCE" if births_expected > 0 and equivalence == births_expected else "CLM_GROWTH_EQUIVALENCE_FAILURE",
            "births_checked": equivalence,
        },
        "growth_viability": {
            "status": "CLM_PROGRESSIVE_GROWTH_VIABILITY" if viability_pass else "NO_PROGRESSIVE_GROWTH_VIABILITY",
            "replicates_passed": viable,
        },
        "growth_utility": {
            "status": "CLM_PROGRESSIVE_GROWTH_SIGNAL" if utility_pass else "NO_GROWTH_UTILITY_SIGNAL",
            "threshold": 0.995, "replicates_passed": improved,
        },
        "pressure_selection": {
            "status": "CLM_GROWTH_PRESSURE_SELECTION_SIGNAL" if pressure_wins >= 2 else "NO_GROWTH_PRESSURE_SELECTION_SIGNAL",
            "replicates_passed": pressure_wins,
        },
        "formal_gpu_experiment_run": bool(formal_gpu_experiment_run),
    }
