from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_proposal_utility_discovery_resumable.py"
spec = importlib.util.spec_from_file_location("e019_resume", SCRIPT)
assert spec is not None and spec.loader is not None
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


def test_feature_repair_uses_training_statistics_only() -> None:
    train = np.array([[1.0, np.nan, np.nan], [3.0, 5.0, np.nan], [5.0, np.inf, np.nan]])
    test = np.array([[np.nan, 1000.0, np.nan], [7.0, -np.inf, np.nan]])
    repaired_train, repaired_test, usable = resume._sanitize_feature_matrices(train, test)

    assert np.isfinite(repaired_train).all()
    assert np.isfinite(repaired_test).all()
    assert repaired_test[0, 0] == 3.0
    assert repaired_train[0, 1] == 5.0
    assert repaired_train[2, 1] == 5.0
    assert repaired_test[1, 1] == 5.0
    assert repaired_test[:, 2].tolist() == [0.0, 0.0]
    assert usable.tolist() == [True, True, False]


def test_standardization_never_propagates_nonfinite_features() -> None:
    train = np.array([[1.0, np.nan], [1.0, 2.0], [1.0, np.inf]])
    test = np.array([[1.0, -np.inf], [np.nan, 4.0]])
    train_z, test_z, _, scale = resume._standardize_finite(train, test)
    assert np.isfinite(train_z).all()
    assert np.isfinite(test_z).all()
    assert np.isfinite(scale).all()
    assert (np.abs(train_z) <= 20.0).all()
    assert (np.abs(test_z) <= 20.0).all()


def test_selection_metrics_reject_nonfinite_predictions_explicitly() -> None:
    test = pd.DataFrame(
        {
            "replicate": [0, 0],
            "example": [0, 0],
            "input_family": ["A", "A"],
            "candidate_family": ["A", "B"],
            "oracle_gradient": [1.0, 0.0],
        }
    )
    with pytest.raises(RuntimeError, match="non-finite predictions"):
        resume._selection_metrics_finite(test, np.array([np.nan, np.nan]))


def test_selection_metrics_uses_positional_argmax_not_nan_index() -> None:
    test = pd.DataFrame(
        {
            "replicate": [0, 0, 0],
            "example": [0, 0, 0],
            "input_family": ["A", "A", "A"],
            "candidate_family": ["A", "B", "RANDOM"],
            "oracle_gradient": [1.0, 0.5, -0.5],
        },
        index=[11, 29, 42],
    )
    top1, regret = resume._selection_metrics_finite(test, np.array([0.1, 0.8, 0.0]))
    assert top1 == 0.0
    assert regret == pytest.approx(0.5)


def test_oracle_nonfinite_is_never_imputed() -> None:
    observations = pd.DataFrame({"oracle_gradient": [1.0, np.nan], "oracle_fd": [1.0, 1.0]})
    with pytest.raises(RuntimeError, match="oracle_gradient contains"):
        resume._validate_oracles(observations)
