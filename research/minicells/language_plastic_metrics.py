from __future__ import annotations

import numpy as np

from .language_topology_metrics import permutation_mi_null


def paired_bootstrap_delta(
    normal: np.ndarray,
    intervention: np.ndarray,
    *,
    seed: int,
    samples: int = 2000,
) -> dict[str, float]:
    """Paired bootstrap for intervention NLL - normal NLL.

    Positive values mean the normal plastic dynamics outperforms the intervention.
    """
    normal = np.asarray(normal, dtype=np.float64)
    intervention = np.asarray(intervention, dtype=np.float64)
    if normal.ndim != 1 or intervention.ndim != 1 or len(normal) != len(intervention):
        raise ValueError("normal and intervention must be paired 1D arrays")
    if len(normal) < 2:
        raise ValueError("at least two paired examples are required")
    if samples < 100:
        raise ValueError("samples must be at least 100")
    delta = intervention - normal
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    means = delta[indices].mean(axis=1)
    return {
        "mean_delta_nll": float(delta.mean()),
        "bootstrap_95_lower": float(np.quantile(means, 0.025)),
        "bootstrap_95_upper": float(np.quantile(means, 0.975)),
        "positive_fraction": float((delta > 0).mean()),
    }


def task_signal_null(
    task_ids: np.ndarray,
    features: np.ndarray,
    *,
    seed: int,
    permutations: int = 1000,
) -> dict[str, float]:
    return permutation_mi_null(
        np.asarray(task_ids, dtype=np.int64),
        np.asarray(features, dtype=np.float64),
        seed=seed,
        permutations=permutations,
    )


def effective_fraction(weights: np.ndarray, *, axis: int = -1) -> float:
    """Participation ratio divided by support size; 1.0 means uniform use."""
    values = np.asarray(weights, dtype=np.float64)
    numerator = values.sum(axis=axis) ** 2
    denominator = np.square(values).sum(axis=axis)
    participation = numerator / np.clip(denominator, 1e-12, None)
    return float(np.mean(participation / values.shape[axis]))
