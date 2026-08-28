"""Probationary-mitosis decision utilities for CLM-0.3d."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable, Sequence

from .growth_counterfactual import PairedUtility

FORMAL_CONDITIONS = ("stationary_story", "story_arithmetic_shift")
FORMAL_HORIZONS = (50_000, 100_000, 200_000, 300_000, 500_000)
SHORTLIST_HORIZON = 100_000
SHORTLIST_K = 4
LATE_HORIZONS = (200_000, 300_000, 500_000)
PRACTICAL_PPL_RATIO_THRESHOLD = 0.995
STORY_RETENTION_RATIO_THRESHOLD = 1.01
SHIFT_ARITHMETIC_FRACTION = 0.50


@dataclass(frozen=True)
class ProbationPoint:
    tokens: int
    utility: PairedUtility
    control_ppl: float
    candidate_ppl: float

    @property
    def ppl_ratio(self) -> float:
        return float(self.candidate_ppl / self.control_ppl)

    def to_dict(self) -> dict[str, object]:
        return {
            "tokens": int(self.tokens),
            **self.utility.to_dict(),
            "control_ppl": float(self.control_ppl),
            "candidate_ppl": float(self.candidate_ppl),
            "ppl_ratio": self.ppl_ratio,
        }


@dataclass(frozen=True)
class ProbationDecision:
    expert_id: str
    sustained_positive: bool
    cumulative_positive: bool
    practical_effect: bool
    accepted_on_probe_holdout: bool
    mean_late_relative_improvement: float
    final_probe_ci95_low: float
    final_probe_ppl_ratio: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def condition_domains(
    condition: str,
    *,
    steps: int,
    seed: int,
    arithmetic_fraction: float = SHIFT_ARITHMETIC_FRACTION,
) -> tuple[str, ...]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if condition == "stationary_story":
        return tuple("story" for _ in range(steps))
    if condition != "story_arithmetic_shift":
        raise ValueError(f"unknown CLM-0.3d condition: {condition}")
    if not 0.0 < arithmetic_fraction < 1.0:
        raise ValueError("arithmetic_fraction must be between zero and one")
    arithmetic = int(round(steps * arithmetic_fraction))
    rows = ["story"] * (steps - arithmetic) + ["arithmetic"] * arithmetic
    random.Random(int(seed)).shuffle(rows)
    return tuple(rows)


def shortlist_candidates(
    rows: Sequence[dict[str, object]],
    *,
    k: int = SHORTLIST_K,
) -> list[dict[str, object]]:
    if k <= 0:
        raise ValueError("shortlist size must be positive")
    if not rows:
        raise ValueError("candidate rows are empty")
    ordered = sorted(
        rows,
        key=lambda row: (-float(row["relative_improvement"]), str(row["expert_id"])),
    )
    return ordered[: min(k, len(ordered))]


def summarize_probation(
    expert_id: str,
    points: Sequence[ProbationPoint],
    *,
    practical_ppl_ratio_threshold: float = PRACTICAL_PPL_RATIO_THRESHOLD,
) -> ProbationDecision:
    by_horizon = {int(point.tokens): point for point in points}
    missing = [token for token in LATE_HORIZONS if token not in by_horizon]
    if missing:
        raise ValueError(f"probation trajectory is missing late horizons: {missing}")
    p200, p300, p500 = (by_horizon[token] for token in LATE_HORIZONS)
    sustained = bool(p300.utility.ci95_low > 0.0 and p500.utility.ci95_low > 0.0)
    late_values = [
        p200.utility.relative_improvement,
        p300.utility.relative_improvement,
        p500.utility.relative_improvement,
    ]
    cumulative = bool(mean(late_values) > 0.0)
    practical = bool(p500.ppl_ratio <= practical_ppl_ratio_threshold)
    return ProbationDecision(
        expert_id=str(expert_id),
        sustained_positive=sustained,
        cumulative_positive=cumulative,
        practical_effect=practical,
        accepted_on_probe_holdout=bool(sustained and cumulative and practical),
        mean_late_relative_improvement=float(mean(late_values)),
        final_probe_ci95_low=float(p500.utility.ci95_low),
        final_probe_ppl_ratio=float(p500.ppl_ratio),
    )


def select_promotion_candidate(
    decisions: Iterable[ProbationDecision],
) -> ProbationDecision | None:
    accepted = [item for item in decisions if item.accepted_on_probe_holdout]
    if not accepted:
        return None
    return max(
        accepted,
        key=lambda item: (
            item.mean_late_relative_improvement,
            item.final_probe_ci95_low,
            -item.final_probe_ppl_ratio,
            item.expert_id,
        ),
    )


def independent_confirmation(
    *,
    utility: PairedUtility,
    control_ppl: float,
    candidate_ppl: float,
    story_control_nll: float | None = None,
    story_candidate_nll: float | None = None,
    practical_ppl_ratio_threshold: float = PRACTICAL_PPL_RATIO_THRESHOLD,
    story_retention_ratio_threshold: float = STORY_RETENTION_RATIO_THRESHOLD,
) -> dict[str, object]:
    ppl_ratio = float(candidate_ppl / control_ppl)
    if story_control_nll is None or story_candidate_nll is None:
        story_ratio = None
        retained = True
    else:
        story_ratio = float(story_candidate_nll / max(story_control_nll, 1e-12))
        retained = bool(story_ratio <= story_retention_ratio_threshold)
    confirmed = bool(
        utility.ci95_low > 0.0
        and ppl_ratio <= practical_ppl_ratio_threshold
        and retained
    )
    return {
        **utility.to_dict(),
        "control_ppl": float(control_ppl),
        "candidate_ppl": float(candidate_ppl),
        "ppl_ratio": ppl_ratio,
        "story_nll_ratio_vs_control": story_ratio,
        "story_retention_pass": retained,
        "confirmed": confirmed,
    }


def absorption_diagnostic(
    *,
    baseline_story_nll: float,
    baseline_arithmetic_nll: float,
    control_story_nll: float,
    control_arithmetic_nll: float,
    learnability_min: float = 0.02,
    story_damage_max: float = 0.01,
) -> dict[str, object]:
    arithmetic_gain = (
        float(baseline_arithmetic_nll) - float(control_arithmetic_nll)
    ) / max(abs(float(baseline_arithmetic_nll)), 1e-12)
    story_damage = max(
        (float(control_story_nll) - float(baseline_story_nll))
        / max(abs(float(baseline_story_nll)), 1e-12),
        0.0,
    )
    return {
        "arithmetic_gain": float(arithmetic_gain),
        "story_damage": float(story_damage),
        "absorption_value": float(arithmetic_gain - story_damage),
        "learnability_min": float(learnability_min),
        "story_damage_max": float(story_damage_max),
        "absorbable_without_mitosis": bool(
            arithmetic_gain >= learnability_min and story_damage <= story_damage_max
        ),
    }


def maturation_rescue(
    early_point: ProbationPoint | None,
    final_confirmation: dict[str, object] | None,
) -> bool:
    if early_point is None or final_confirmation is None:
        return False
    return bool(
        early_point.tokens == SHORTLIST_HORIZON
        and early_point.utility.ci95_low <= 0.0
        and final_confirmation.get("confirmed") is True
    )
