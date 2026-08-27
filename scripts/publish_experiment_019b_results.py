from __future__ import annotations

import argparse

from publish_experiment_results import DEFAULT_SECRET_NAME, ExperimentSpec, prepare_artifacts, push_results, repo_root


WORKERS = tuple(f"r{replicate}-worker.json" for replicate in range(3))

SPEC = ExperimentSpec(
    source_dir="results/recruitment-response-curves-v1",
    artifact_dir="artifacts/experiments/019b-recruitment-response-curves",
    files=(
        "decision.json",
        "task-spec.json",
        "source-019-decision.json",
        "invariants.json",
        "response-observations.csv.gz",
        "response-curves.csv",
        "response-example-summary.csv",
        "response-curve-summary.csv",
        "probe-metrics.csv",
        "barrier-summary.csv",
        "matching-response-curves.png",
        "probe-family-pass-count.png",
        "probe-predicts-full.png",
        *WORKERS,
    ),
    branch="kaggle/experiment-019b-results",
    expected_format="minicells.recruitment-response-curves.v1",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    args = parser.parse_args()
    root = repo_root()
    destination = prepare_artifacts(root, "019b", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    print("019-stable model checkpoints remain Kaggle-local and are intentionally not duplicated in 019b artifacts.")
    if args.push:
        push_results(root, destination, "019b", args.branch or SPEC.branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
