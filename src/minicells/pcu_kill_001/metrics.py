"""Metrics and machine-readable kill-test decision tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor


def exact_accuracy(predictions: Iterable[str], answers: Iterable[str]) -> float:
    predictions, answers = list(predictions), list(answers)
    if len(predictions) != len(answers) or not answers:
        return 0.0
    return sum(str(left).strip() == str(right).strip() for left, right in zip(predictions, answers)) / len(answers)


def branch_gain(base: float, branch: float) -> float:
    return float(branch) - float(base)


def merge_retention(base: float, branch: float, merged: float) -> float | None:
    denominator = float(branch) - float(base)
    return None if denominator <= 0 else (float(merged) - float(base)) / denominator


def retention_or_undefined(base: float, branch: float, merged: float) -> float | None:
    """Return retention only for a positive branch gain."""
    denominator = float(branch) - float(base)
    if denominator <= 0:
        return None
    return (float(merged) - float(base)) / denominator


def composition_synergy_same_task(base_ab: float, a_ab: float, b_ab: float, ab_ab: float) -> float:
    """Synergy on T_AB only; direct-task scores never enter this formula."""
    return float(ab_ab) - max(float(base_ab), float(a_ab), float(b_ab))


def composition_synergy(base: float, branch_a: float, branch_b: float, merged: float) -> float:
    return float(merged) - max(float(base), float(branch_a), float(branch_b))


def interaction_metric(logits_0: Tensor, logits_a: Tensor, logits_b: Tensor, logits_ab: Tensor) -> float:
    return float(((logits_ab.float() - logits_0.float()) - ((logits_a.float() - logits_0.float()) + (logits_b.float() - logits_0.float()))).norm())


@dataclass(frozen=True)
class Decision:
    status: str
    scientific_decision: bool
    valid_run: bool
    gates: dict[str, bool]
    metrics: dict[str, float]
    baseline: dict[str, float | None]
    reason: str
    schema: str = "minicells.pcu-kill-001.decision.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "scientific_decision": self.scientific_decision,
            "valid_run": self.valid_run,
            "gates": dict(self.gates),
            "metrics": dict(self.metrics),
            "baseline": dict(self.baseline),
            "reason": self.reason,
        }


DEFAULT_THRESHOLDS = {
    "min_direct_accuracy": 0.80,
    "min_retention": 0.90,
    "min_composition_accuracy": 0.50,
    "min_composition_synergy": 0.30,
    "max_anchor_regression": 0.01,
}


def decide(
    metrics: Mapping[str, float],
    validity: Mapping[str, bool],
    baseline: Mapping[str, float] | None = None,
    thresholds: Mapping[str, float] | None = None,
    joint_oracle_pass: bool | None = None,
) -> Decision:
    values = {key: float(value) for key, value in metrics.items()}
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    base = {key: (None if value is None else float(value)) for key, value in (baseline or {}).items()}
    if not all(bool(value) for value in validity.values()):
        return Decision("INVALID_FORMAL_RUN", False, False, dict(validity), values, base, "protocol or run validity gate failed")
    g0 = bool(values.get("g0_exact_embedding", 0.0))
    if not g0:
        return Decision("EXACT_CELL_EMBEDDING_FAILED", True, True, {"g0_exact_embedding": False}, values, base, "cellularized foundation is not numerically equivalent")
    if values.get("context_oracle_accuracy", 0.0) < 0.90:
        return Decision("TESTBED_COMPOSITION_CAPACITY_INADEQUATE", True, True, {"context_oracle": False}, values, base, "in-context oracle is below the registered capacity floor")
    direct_a = values.get("acc_a", 0.0) >= limits["min_direct_accuracy"] and values.get("acc_a", 0.0) > values.get("base_a", -1.0)
    direct_b = values.get("acc_b", 0.0) >= limits["min_direct_accuracy"] and values.get("acc_b", 0.0) > values.get("base_b", -1.0)
    if not direct_a or not direct_b:
        return Decision("LOCAL_CELL_MUTATION_UNSUPPORTED", True, True, {"branch_a_learned": direct_a, "branch_b_learned": direct_b}, values, base, "one or both direct branch capabilities failed")
    retention_a = values.get("retention_a", 0.0) >= limits["min_retention"]
    retention_b = values.get("retention_b", 0.0) >= limits["min_retention"]
    if not retention_a or not retention_b:
        return Decision("PARAMETER_LOCALITY_ONLY_MERGEABILITY_FAILED", True, True, {"merge_retention_a": retention_a, "merge_retention_b": retention_b}, values, base, "registry-only union lost a branch gain")
    regression = values.get("anchor_regression", float("inf")) <= limits["max_anchor_regression"]
    if not regression:
        return Decision("MERGEABLE_BUT_FOUNDATION_REGRESSION_UNSAFE", True, True, {"background_regression": False}, values, base, "merged branch exceeded anchor regression budget")
    composition = values.get("composition_acc", 0.0) >= limits["min_composition_accuracy"] and values.get("composition_synergy", 0.0) >= limits["min_composition_synergy"]
    if not composition:
        if joint_oracle_pass is True:
            status = "INDEPENDENT_CELL_COMPOSITION_UNSUPPORTED"
            reason = "joint oracle passed but independent union failed unseen composition"
        else:
            status = "COMPOSITION_TESTBED_INCONCLUSIVE"
            reason = "independent composition failed and joint oracle did not establish capacity"
        return Decision(status, True, True, {"composition": False, "joint_oracle": bool(joint_oracle_pass)}, values, base, reason)
    lora = base.get("lora_composition_acc")
    if lora is None:
        return Decision("PCU_MECHANISM_SUPPORTED_ADVANTAGE_UNPROVEN", True, True, {"composition": True, "lora_distinctiveness": False}, values, base, "matched LoRA baseline metrics were not supplied")
    if lora >= values.get("composition_acc", 0.0):
        return Decision("PCU_MECHANISM_SUPPORTED_ADVANTAGE_UNPROVEN", True, True, {"composition": True, "lora_distinctiveness": False}, values, base, "matched LoRA is as good or better")
    return Decision("PCU_COMPOSABILITY_CONSTRUCTIVE_EVIDENCE", True, True, {"composition": True, "lora_distinctiveness": True}, values, base, "all registered PCU gates passed")


def evaluation_matrix(rows: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    fields = ("a_direct", "b_direct", "composition", "anchor_regression")
    return [{"model": name, **{field: float(values.get(field, 0.0)) for field in fields}} for name, values in rows.items()]
