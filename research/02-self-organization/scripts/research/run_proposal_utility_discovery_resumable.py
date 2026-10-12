from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "scripts"))

import run_proposal_utility_discovery as base  # noqa: E402


REQUIRED_WORKER_FILES = (
    "worker.json",
    "phase1-checkpoints.csv",
    "donor-summary.csv",
    "utility-observations.csv",
)


def _replicate_complete(replicate: int) -> bool:
    for suffix in REQUIRED_WORKER_FILES:
        path = base.OUT / f"r{replicate}-{suffix}"
        if not path.is_file() or path.stat().st_size == 0:
            return False
    return True


def _finite_audit(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    numeric = ("oracle_gradient", "oracle_fd", *base.BOUNDARY_FEATURES)
    for column in numeric:
        values = observations[column].to_numpy(float)
        finite = np.isfinite(values)
        rows.append({
            "column": column,
            "rows": len(values),
            "finite_rows": int(finite.sum()),
            "nonfinite_rows": int((~finite).sum()),
            "nonfinite_fraction": float((~finite).mean()),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(base.OUT / "finite-audit.csv", index=False)
    return frame


def _sanitize_feature_matrices(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold-local finite repair with no held-out leakage.

    Non-finite feature values are replaced by the finite training median for
    that feature. A feature with no finite training values is replaced by zero
    in both train and test and marked unusable. Test statistics never influence
    the replacement value.
    """
    train = np.asarray(train, dtype=np.float64).copy()
    test = np.asarray(test, dtype=np.float64).copy()
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("feature matrices must be 2-D with matching columns")
    usable = np.ones(train.shape[1], dtype=bool)
    for column in range(train.shape[1]):
        train_finite = np.isfinite(train[:, column])
        if train_finite.any():
            fill = float(np.median(train[train_finite, column]))
        else:
            fill = 0.0
            usable[column] = False
        train[~train_finite, column] = fill
        test_finite = np.isfinite(test[:, column])
        test[~test_finite, column] = fill
    return train, test, usable


def _standardize_finite(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train, test, _ = _sanitize_feature_matrices(train, test)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where((~np.isfinite(scale)) | (scale < 1e-8), 1.0, scale)
    train_z = np.nan_to_num((train - mean) / scale, nan=0.0, posinf=20.0, neginf=-20.0)
    test_z = np.nan_to_num((test - mean) / scale, nan=0.0, posinf=20.0, neginf=-20.0)
    train_z = np.clip(train_z, -20.0, 20.0)
    test_z = np.clip(test_z, -20.0, 20.0)
    return train_z, test_z, mean, scale


def _selection_metrics_finite(test: pd.DataFrame, predictions: np.ndarray) -> tuple[float, float]:
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != (len(test),):
        raise ValueError("prediction shape does not match held-out rows")
    if not np.isfinite(predictions).all():
        bad = int((~np.isfinite(predictions)).sum())
        raise RuntimeError(f"estimator produced {bad}/{len(predictions)} non-finite predictions after feature repair")

    scored = test[["replicate", "example", "input_family", "candidate_family", "oracle_gradient"]].copy()
    scored["prediction"] = predictions
    correct = 0
    regrets: list[float] = []
    groups = 0
    for key, group in scored.groupby(["replicate", "example", "input_family"], sort=False):
        oracle_values = group["oracle_gradient"].to_numpy(float)
        prediction_values = group["prediction"].to_numpy(float)
        if not np.isfinite(oracle_values).all():
            raise RuntimeError(f"non-finite oracle utility in selection group {key}")
        oracle_pos = int(np.argmax(oracle_values))
        prediction_pos = int(np.argmax(prediction_values))
        oracle_row = group.iloc[oracle_pos]
        prediction_row = group.iloc[prediction_pos]
        correct += int(oracle_row["candidate_family"] == prediction_row["candidate_family"])
        best = float(oracle_row["oracle_gradient"])
        chosen = float(prediction_row["oracle_gradient"])
        regrets.append((best - chosen) / max(abs(best), 1e-6))
        groups += 1
    return correct / max(1, groups), float(np.median(regrets)) if regrets else float("nan")


def _validate_oracles(observations: pd.DataFrame) -> None:
    for column in ("oracle_gradient", "oracle_fd"):
        values = observations[column].to_numpy(float)
        if not np.isfinite(values).all():
            bad = int((~np.isfinite(values)).sum())
            raise RuntimeError(
                f"{column} contains {bad}/{len(values)} non-finite values; "
                "this is an oracle-generation failure and must not be imputed"
            )


def _postprocess(gpu_count: int) -> dict[str, object]:
    collected = base.collect()
    observations = collected["observations"]
    donors = collected["donor_summary"]
    audit = _finite_audit(observations)
    print("finite audit:")
    bad = audit.loc[audit["nonfinite_rows"] > 0]
    print(bad.to_string(index=False) if not bad.empty else "all finite")
    _validate_oracles(observations)

    base._standardize = _standardize_finite
    base._selection_metrics = _selection_metrics_finite

    oracle = base.oracle_consistency(observations)
    estimators = base.evaluate_estimators(observations)
    matrix = base.utility_matrix(observations)
    correlations = base.feature_correlations(observations)

    base.plot_oracle(observations)
    base.plot_matrix(matrix)
    base.plot_estimator_metric(estimators, "spearman", "heldout-spearman.png", "Spearman", threshold=0.50)
    base.plot_estimator_metric(estimators, "auc_positive_utility", "heldout-auc.png", "AUC(U*>0)", threshold=0.75)
    base.plot_estimator_metric(
        estimators,
        "top1_selection_accuracy",
        "heldout-top1.png",
        "top-1 tissue selection accuracy",
        threshold=0.60,
    )
    base.plot_estimator_metric(
        estimators,
        "median_normalized_regret",
        "heldout-regret.png",
        "median normalized regret",
        threshold=0.35,
    )
    base.plot_feature_correlations(correlations)
    base.write_task_spec(gpu_count)
    decision = base.make_decision(observations, donors, oracle, estimators, gpu_count)
    decision.setdefault("numerical_validation", {})
    decision["numerical_validation"].update({
        "finite_audit_file": "finite-audit.csv",
        "feature_nonfinite_rows_total": int(
            audit.loc[audit["column"].isin(base.BOUNDARY_FEATURES), "nonfinite_rows"].sum()
        ),
        "repair": "fold-local finite-training-median imputation; all-nonfinite training columns -> zero",
        "oracle_imputation": False,
    })
    (base.OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return decision


def main() -> int:
    cache, _ = base.prepare_corpus()
    complete = [replicate for replicate in range(base.N_REPLICATES) if _replicate_complete(replicate)]
    if len(complete) == base.N_REPLICATES:
        gpu_count = min(2, max(1, torch.cuda.device_count()))
        print(f"reusing completed Experiment 019 workers: {complete}; no GPU donor retraining")
    else:
        missing = [replicate for replicate in range(base.N_REPLICATES) if replicate not in complete]
        print(f"missing/incomplete Experiment 019 workers {missing}; running worker stage")
        gpu_count = base.run_workers(cache)
    _postprocess(gpu_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
