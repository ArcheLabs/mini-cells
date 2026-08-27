from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .clm_upcycling_validation import UpcyclingEvaluation


QUALITY_RATIO_MAX = 1.03
NORMALIZED_ADVANTAGE_MIN = 0.002
ALIGNED_DISAGREEMENT_MIN = 0.10
USAGE_ENTROPY_MIN = 0.80


@dataclass(frozen=True)
class Conditionality002Evidence:
    replicate: int
    quality_ratio_to_dense_continued: float
    aligned_route_disagreement: float
    static_advantage: float
    shuffled_advantage: float
    usage_entropy: float
    passed: bool


def aligned_route_disagreement(masks: list[torch.Tensor]) -> float:
    """Pairwise sample disagreement at aligned recurrent-step and token positions.

    Each mask is [batch, position, expert]. Unlike the older sample-profile metric,
    this preserves recurrent-time and position identity instead of averaging them away.
    """
    if not masks:
        raise ValueError("routing masks must not be empty")
    shape = masks[0].shape
    if len(shape) != 3:
        raise ValueError("routing masks must have shape [batch, position, expert]")
    if any(mask.shape != shape for mask in masks):
        raise ValueError("all routing masks must have identical shapes")
    batch = shape[0]
    if batch < 2:
        return 0.0
    values: list[torch.Tensor] = []
    for mask in masks:
        routes = mask.detach().float().argmax(-1)  # [batch, position]
        one_hot = torch.nn.functional.one_hot(routes, num_classes=shape[-1]).float()
        counts = one_hot.sum(0)  # [position, expert]
        pair_count = float(batch * (batch - 1))
        agreement = (counts * (counts - 1)).sum(-1) / pair_count
        values.append(1.0 - agreement)
    return float(torch.stack(values).mean())


def normalized_advantage(dynamic_nll: float, control_nll: float, dense_nll: float) -> float:
    return (control_nll - dynamic_nll) / dense_nll


def evaluate_conditionality_evidence(
    *,
    replicate: int,
    dense_ppl: float,
    dense_nll: float,
    dynamic: UpcyclingEvaluation | dict[str, object],
    static: UpcyclingEvaluation | dict[str, object],
    shuffled: UpcyclingEvaluation | dict[str, object],
    aligned_disagreement: float,
) -> Conditionality002Evidence:
    def get(row, key: str) -> float:
        return float(getattr(row, key) if hasattr(row, key) else row[key])

    quality = get(dynamic, "ppl") / dense_ppl
    static_adv = normalized_advantage(get(dynamic, "nll"), get(static, "nll"), dense_nll)
    shuffled_adv = normalized_advantage(get(dynamic, "nll"), get(shuffled, "nll"), dense_nll)
    entropy = get(dynamic, "usage_entropy")
    passed = (
        quality <= QUALITY_RATIO_MAX
        and aligned_disagreement >= ALIGNED_DISAGREEMENT_MIN
        and static_adv >= NORMALIZED_ADVANTAGE_MIN
        and shuffled_adv >= NORMALIZED_ADVANTAGE_MIN
        and entropy >= USAGE_ENTROPY_MIN
    )
    return Conditionality002Evidence(
        replicate=replicate,
        quality_ratio_to_dense_continued=quality,
        aligned_route_disagreement=aligned_disagreement,
        static_advantage=static_adv,
        shuffled_advantage=shuffled_adv,
        usage_entropy=entropy,
        passed=passed,
    )


def make_conditionality_002_decision(evidence: list[Conditionality002Evidence]) -> dict[str, object]:
    successful = sum(int(row.passed) for row in evidence)
    diagnosis = (
        "CLM_LOCAL_CONDITIONALITY_SIGNAL"
        if successful >= 2
        else "CLM_LOCAL_CONDITIONALITY_NOT_ESTABLISHED"
    )
    return {
        "format": "minicells.clm-conditionality-002.v1",
        "experiment": "CLM Conditionality Validation 002 — Aligned Local Routing",
        "status": "PASS" if successful >= 2 else "FAIL",
        "diagnosis": diagnosis,
        "successful_replicates": successful,
        "thresholds": {
            "quality_ratio_to_dense_continued_max": QUALITY_RATIO_MAX,
            "aligned_route_disagreement_min": ALIGNED_DISAGREEMENT_MIN,
            "normalized_advantage_min": NORMALIZED_ADVANTAGE_MIN,
            "usage_entropy_min": USAGE_ENTROPY_MIN,
        },
        "evidence": [asdict(row) for row in evidence],
    }
