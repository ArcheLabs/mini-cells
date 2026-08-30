from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RECRUITMENT_GRID = (
    0.0,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    2e-2,
    5e-2,
    1e-1,
    2.5e-1,
    5e-1,
    1.0,
)
SMALL_PROBE_MAX = 2e-2
LOW_PROBE_MAX = 1e-1
FULL_BENEFIT_MIN = 5e-2
BARRIER_ABS_HARM_MIN = 5e-3
BARRIER_RELATIVE_HARM = 5e-2


@dataclass(frozen=True)
class ResponseSummary:
    full_value: float
    best_value: float
    best_recruitment: float
    min_small_value: float
    first_positive_recruitment: float | None
    full_beneficial: bool
    activation_barrier: bool
    nonmonotonic: bool


def summarize_response(recruitment: np.ndarray, value: np.ndarray) -> ResponseSummary:
    recruitment = np.asarray(recruitment, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    if recruitment.ndim != 1 or value.ndim != 1 or len(recruitment) != len(value):
        raise ValueError("recruitment and value must be aligned 1-D arrays")
    if len(recruitment) < 2 or not np.isfinite(recruitment).all() or not np.isfinite(value).all():
        raise ValueError("response curve must contain at least two finite points")
    order = np.argsort(recruitment)
    recruitment = recruitment[order]
    value = value[order]
    if abs(float(recruitment[0])) > 1e-12 or abs(float(recruitment[-1]) - 1.0) > 1e-12:
        raise ValueError("response curve must include recruitment=0 and recruitment=1")

    full_value = float(value[-1])
    best_pos = int(np.argmax(value))
    small = (recruitment > 0.0) & (recruitment <= SMALL_PROBE_MAX + 1e-12)
    min_small = float(value[small].min()) if bool(small.any()) else 0.0
    positive = np.flatnonzero((recruitment > 0.0) & (value > 0.0))
    first_positive = float(recruitment[int(positive[0])]) if len(positive) else None
    full_beneficial = full_value > FULL_BENEFIT_MIN
    harm_threshold = max(BARRIER_ABS_HARM_MIN, BARRIER_RELATIVE_HARM * max(full_value, 0.0))
    barrier = bool(full_beneficial and min_small < -harm_threshold)
    diffs = np.diff(value)
    nonmonotonic = bool((diffs > 1e-6).any() and (diffs < -1e-6).any())
    return ResponseSummary(
        full_value=full_value,
        best_value=float(value[best_pos]),
        best_recruitment=float(recruitment[best_pos]),
        min_small_value=min_small,
        first_positive_recruitment=first_positive,
        full_beneficial=full_beneficial,
        activation_barrier=barrier,
        nonmonotonic=nonmonotonic,
    )


def normalized_regret(best: float, chosen: float) -> float:
    return float((best - chosen) / max(abs(best), 1e-6))
