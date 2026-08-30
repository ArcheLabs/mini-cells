from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def weighted_mutual_information(task_ids: np.ndarray, features: np.ndarray) -> float:
    """Mutual information between discrete task labels and non-negative event weights."""
    task_ids = np.asarray(task_ids, dtype=np.int64)
    features = np.asarray(features, dtype=np.float64)
    if task_ids.ndim != 1 or features.ndim != 2 or len(task_ids) != len(features):
        raise ValueError("task_ids must be [n] and features [n, f]")
    if np.any(features < 0) or not np.all(np.isfinite(features)):
        raise ValueError("features must be finite and non-negative")
    if features.sum() <= 0:
        return 0.0
    labels = np.unique(task_ids)
    counts = np.stack([features[task_ids == label].sum(axis=0) for label in labels], axis=0)
    total = counts.sum()
    joint = counts / total
    p_task = joint.sum(axis=1, keepdims=True)
    p_feature = joint.sum(axis=0, keepdims=True)
    expected = p_task @ p_feature
    mask = joint > 0
    return float((joint[mask] * np.log(joint[mask] / expected[mask])).sum())


def permutation_mi_null(
    task_ids: np.ndarray,
    features: np.ndarray,
    *,
    seed: int,
    permutations: int = 1000,
) -> dict[str, float]:
    if permutations < 10:
        raise ValueError("permutations must be at least ten")
    rng = np.random.default_rng(seed)
    observed = weighted_mutual_information(task_ids, features)
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        null[index] = weighted_mutual_information(rng.permutation(task_ids), features)
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "null_p99": float(np.quantile(null, 0.99)),
        "empirical_p": float((1.0 + np.count_nonzero(null >= observed)) / (permutations + 1.0)),
    }


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    aa = np.asarray(tuple(a), dtype=np.float64)
    bb = np.asarray(tuple(b), dtype=np.float64)
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denominator <= 0:
        return 0.0
    return float(np.dot(aa, bb) / denominator)


def composition_reuse_scores(
    task_activity: Mapping[str, Iterable[float]],
    composition_map: Mapping[str, tuple[str, str]],
    base_tasks: Iterable[str],
) -> list[dict[str, float | str]]:
    base = tuple(base_tasks)
    rows: list[dict[str, float | str]] = []
    for composite, true_pair in composition_map.items():
        composite_activity = np.asarray(tuple(task_activity[composite]), dtype=np.float64)
        true_activity = np.asarray(tuple(task_activity[true_pair[0]]), dtype=np.float64) + np.asarray(
            tuple(task_activity[true_pair[1]]), dtype=np.float64
        )
        true_score = cosine_similarity(composite_activity, true_activity)
        wrong_scores: list[float] = []
        for left_index, left in enumerate(base):
            for right in base[left_index + 1 :]:
                pair = (left, right)
                if set(pair) == set(true_pair):
                    continue
                candidate = np.asarray(tuple(task_activity[left]), dtype=np.float64) + np.asarray(
                    tuple(task_activity[right]), dtype=np.float64
                )
                wrong_scores.append(cosine_similarity(composite_activity, candidate))
        wrong_max = max(wrong_scores) if wrong_scores else 0.0
        wrong_mean = float(np.mean(wrong_scores)) if wrong_scores else 0.0
        rows.append(
            {
                "composite": composite,
                "left": true_pair[0],
                "right": true_pair[1],
                "true_reuse": true_score,
                "wrong_pair_mean": wrong_mean,
                "wrong_pair_max": wrong_max,
                "reuse_margin_vs_best_wrong": true_score - wrong_max,
            }
        )
    return rows
