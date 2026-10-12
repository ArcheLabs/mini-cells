from __future__ import annotations

import argparse

from publish_experiment_results import ExperimentSpec, prepare_artifacts, push_results, repo_root, DEFAULT_SECRET_NAME


WORKERS = tuple(f"r{replicate}-worker.json" for replicate in range(3))

SPEC = ExperimentSpec(
    source_dir="results/proposal-utility-discovery-stable-v1",
    artifact_dir="artifacts/experiments/019-proposal-utility-discovery-stable",
    files=(
        "decision.json",
        "task-spec.json",
        "checkpoint-manifest.json",
        "finite-audit.csv",
        "corpus-manifest.json",
        "tokenizer.json",
        "phase1-checkpoints.csv",
        "phase1-events.csv",
        "donor-summary.csv",
        "donor-events.csv",
        "utility-observations.csv",
        "oracle-consistency.csv",
        "utility-matrix.csv",
        "feature-correlations.csv",
        "estimator-results.csv",
        "oracle-gradient-vs-fd.png",
        "candidate-utility-matrix.png",
        "heldout-spearman.png",
        "heldout-auc.png",
        "heldout-top1.png",
        "heldout-regret.png",
        "feature-oracle-correlations.png",
        *WORKERS,
    ),
    branch="kaggle/experiment-019-stable-results",
    expected_format="minicells.proposal-utility-discovery.v1",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    args = parser.parse_args()
    root = repo_root()
    destination = prepare_artifacts(root, "019-stable", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    print("Kaggle-local model checkpoints are intentionally excluded from curated GitHub artifacts.")
    if args.push:
        push_results(root, destination, "019-stable", args.branch or SPEC.branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
