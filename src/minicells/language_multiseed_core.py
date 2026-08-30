from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import math
from typing import Iterable, Mapping


CORE_VARIANT_CODES = ("A", "B", "F", "H")
N_REPLICATES = 5

# Pre-registered confirmation thresholds. Aggregate thresholds are primary;
# the per-seed joint criterion prevents one unusually favorable seed from
# carrying the whole conclusion.
CORE_PPL_RATIO_MAX = 1.00
CORE_COST_RATIO_MAX = 0.90
PER_SEED_PPL_RATIO_MAX = 1.01
PER_SEED_COST_RATIO_MAX = 0.95
MIN_JOINT_PASS_REPLICATES = 4


@dataclass(frozen=True)
class SeedBundle:
    replicate: int
    model_seed_1d: int
    model_seed_2d: int
    schedule_seed: int
    depth_seed: int
    validation_seed: int
    depth_eval_seed: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def seed_bundle(replicate: int) -> SeedBundle:
    """Return one matched Experiment 014 seed bundle.

    A/B/F/H share every stochastic stream within a topology and replicate.
    1D and 2D share schedules/evaluation samples while retaining topology-
    specific model seeds compatible with the Experiment 013 initialization.
    """

    if replicate < 0 or replicate >= N_REPLICATES:
        raise ValueError(f"replicate must be in [0, {N_REPLICATES - 1}]")
    offset = 1_000 * replicate
    return SeedBundle(
        replicate=replicate,
        model_seed_1d=61_011 + offset,
        model_seed_2d=61_015 + offset,
        schedule_seed=11_011 + offset,
        depth_seed=21_011 + offset,
        validation_seed=41_011 + offset,
        depth_eval_seed=51_013 + offset,
    )


def model_seed(bundle: SeedBundle, topology: str) -> int:
    if topology == "1d":
        return bundle.model_seed_1d
    if topology == "2d":
        return bundle.model_seed_2d
    raise ValueError(f"unknown topology: {topology}")


def _positive(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def core_recipe_ratio(values: Mapping[str, float]) -> float:
    """Return H/A for one matched topology/replicate."""

    return _positive(values["H"], "H") / _positive(values["A"], "A")


def factor_ratio(values: Mapping[str, float], factor: str) -> float:
    """Return the balanced 2x2 main-effect ratio for A/B/F/H.

    random_depth compares B/A and H/F. stability_loss compares F/A and H/B.
    The geometric mean keeps multiplicative metrics (PPL, wall time, residual
    ratios) symmetric and matches Experiment 013's log-contrast convention.
    """

    required = set(CORE_VARIANT_CODES)
    if set(values) != required:
        raise ValueError(f"values must contain exactly {sorted(required)}")
    logs = {key: math.log(_positive(values[key], key)) for key in required}
    if factor == "random_depth":
        contrast = ((logs["B"] - logs["A"]) + (logs["H"] - logs["F"])) / 2.0
    elif factor == "stability_loss":
        contrast = ((logs["F"] - logs["A"]) + (logs["H"] - logs["B"])) / 2.0
    else:
        raise ValueError("factor must be random_depth or stability_loss")
    return math.exp(contrast)


def geometric_mean(values: Iterable[float]) -> float:
    values = tuple(_positive(value, "ratio") for value in values)
    if not values:
        raise ValueError("at least one value is required")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    if q <= 0.0:
        return sorted_values[0]
    if q >= 1.0:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def exact_bootstrap_geometric_ci(
    ratios: Iterable[float],
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Exact percentile bootstrap CI for a small paired ratio sample.

    With the pre-registered five replicates this enumerates 5**5 = 3125
    bootstrap resamples, so the result is deterministic and needs no RNG or
    scipy dependency.
    """

    values = tuple(_positive(value, "ratio") for value in ratios)
    if len(values) < 2:
        raise ValueError("at least two ratios are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    log_values = tuple(math.log(value) for value in values)
    n = len(log_values)
    samples: list[float] = []
    for indices in product(range(n), repeat=n):
        samples.append(math.exp(sum(log_values[index] for index in indices) / n))
    samples.sort()
    tail = (1.0 - confidence) / 2.0
    return _quantile(samples, tail), _quantile(samples, 1.0 - tail)


def ratio_summary(
    ratios: Iterable[float],
    *,
    aggregate_threshold: float | None = None,
) -> dict[str, float | int | bool]:
    values = tuple(_positive(value, "ratio") for value in ratios)
    if len(values) != N_REPLICATES:
        raise ValueError(f"expected {N_REPLICATES} replicate ratios")
    mean = geometric_mean(values)
    lower, upper = exact_bootstrap_geometric_ci(values)
    result: dict[str, float | int | bool] = {
        "replicates": len(values),
        "geometric_mean_ratio": mean,
        "bootstrap_95_lower": lower,
        "bootstrap_95_upper": upper,
        "min_ratio": min(values),
        "max_ratio": max(values),
    }
    if aggregate_threshold is not None:
        result["aggregate_threshold"] = float(aggregate_threshold)
        result["aggregate_pass"] = mean <= aggregate_threshold
    return result


def core_recipe_confirmation(
    ppl_ratios: Iterable[float],
    cost_ratios: Iterable[float],
) -> dict[str, object]:
    ppl = tuple(_positive(value, "ppl ratio") for value in ppl_ratios)
    cost = tuple(_positive(value, "cost ratio") for value in cost_ratios)
    if len(ppl) != N_REPLICATES or len(cost) != N_REPLICATES:
        raise ValueError(f"expected {N_REPLICATES} paired replicate ratios")
    joint_passes = sum(
        p <= PER_SEED_PPL_RATIO_MAX and c <= PER_SEED_COST_RATIO_MAX
        for p, c in zip(ppl, cost, strict=True)
    )
    ppl_summary = ratio_summary(ppl, aggregate_threshold=CORE_PPL_RATIO_MAX)
    cost_summary = ratio_summary(cost, aggregate_threshold=CORE_COST_RATIO_MAX)
    confirmed = bool(
        ppl_summary["aggregate_pass"]
        and cost_summary["aggregate_pass"]
        and joint_passes >= MIN_JOINT_PASS_REPLICATES
    )
    return {
        "confirmed": confirmed,
        "ppl": ppl_summary,
        "cost": cost_summary,
        "joint_pass_replicates": joint_passes,
        "joint_pass_required": MIN_JOINT_PASS_REPLICATES,
        "per_seed_ppl_ratio_max": PER_SEED_PPL_RATIO_MAX,
        "per_seed_cost_ratio_max": PER_SEED_COST_RATIO_MAX,
    }
