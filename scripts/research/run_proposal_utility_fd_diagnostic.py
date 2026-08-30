from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "scripts"))

import run_proposal_utility_discovery as base  # noqa: E402
import run_proposal_utility_discovery_resumable as resumable  # noqa: E402


SOURCE_OUT = ROOT / "results" / "proposal-utility-discovery-v1"
OUT = ROOT / "results" / "proposal-utility-fd-diagnostic-v1"


def _pass_count(results: pd.DataFrame, estimator: str) -> int:
    selected = results.loc[results["estimator"] == estimator]
    passes = (
        (selected["spearman"] >= 0.50)
        & (selected["auc_positive_utility"] >= 0.75)
        & (selected["top1_selection_accuracy"] >= 0.60)
        & (selected["median_normalized_regret"] <= 0.35)
    )
    return int(passes.sum())


def main() -> int:
    # Read the already-completed failed-gradient run without modifying it.
    base.OUT = SOURCE_OUT
    collected = base.collect()
    observations = collected["observations"].copy()
    fd = observations["oracle_fd"].to_numpy(float)
    if not np.isfinite(fd).all():
        bad = int((~np.isfinite(fd)).sum())
        raise RuntimeError(f"oracle_fd contains {bad}/{len(fd)} non-finite values; FD diagnostic is invalid")

    original_gradient = observations["oracle_gradient"].to_numpy(float)
    gradient_finite = np.isfinite(original_gradient)

    # This is explicitly exploratory. We replace the estimator target only in a
    # copy of the observations. The original 019 artifacts and decision remain
    # untouched and cannot be reported as a preregistered gradient-oracle result.
    observations["oracle_gradient_original"] = original_gradient
    observations["oracle_gradient"] = observations["oracle_fd"]

    OUT.mkdir(parents=True, exist_ok=True)
    observations.to_csv(OUT / "fd-utility-observations.csv", index=False)
    base.OUT = OUT
    base._standardize = resumable._standardize_finite
    base._selection_metrics = resumable._selection_metrics_finite

    estimators = base.evaluate_estimators(observations)
    matrix = base.utility_matrix(observations)
    correlations = base.feature_correlations(observations)
    base.plot_matrix(matrix)
    base.plot_estimator_metric(estimators, "spearman", "heldout-spearman.png", "Spearman", threshold=0.50)
    base.plot_estimator_metric(estimators, "auc_positive_utility", "heldout-auc.png", "AUC(FD utility > 0)", threshold=0.75)
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

    counts = {name: _pass_count(estimators, name) for name in ("local-ridge", "local-mlp", "boundary-ridge", "boundary-mlp")}
    best_local = max(counts["local-ridge"], counts["local-mlp"])
    best_boundary = max(counts["boundary-ridge"], counts["boundary-mlp"])
    if best_local >= 4:
        status = "EXPLORATORY_FD_LOCAL_SIGNAL"
    elif best_boundary >= 4:
        status = "EXPLORATORY_FD_BOUNDARY_SIGNAL"
    elif max(best_local, best_boundary) >= 2:
        status = "EXPLORATORY_FD_PARTIAL_SIGNAL"
    else:
        status = "EXPLORATORY_FD_NO_GENERAL_SIGNAL"

    decision = {
        "format": "minicells.proposal-utility-fd-diagnostic.v1",
        "experiment": "019 numerical diagnostic",
        "status": status,
        "official_019_result": "INVALID_PENDING_STABLE_GRADIENT_ORACLE",
        "exploratory_only": True,
        "source_run": str(SOURCE_OUT.relative_to(ROOT)),
        "utility_definition": "[L(e=0)-L(e=0.02)]/0.02",
        "interpretation_limit": "This diagnostic may guide whether a stable-gradient rerun is worth the GPU cost; it must not be reported as the preregistered Experiment 019 result.",
        "numerics": {
            "rows": len(observations),
            "original_gradient_finite_rows": int(gradient_finite.sum()),
            "original_gradient_nonfinite_rows": int((~gradient_finite).sum()),
            "fd_finite_rows": int(np.isfinite(fd).sum()),
        },
        "heldout_pass_counts": counts,
    }
    (OUT / "fd-diagnostic-decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
