from __future__ import annotations

import argparse

from publish_experiment_results import (
    DEFAULT_SECRET_NAME,
    ExperimentSpec,
    prepare_artifacts,
    push_results,
    repo_root,
)


SPEC = ExperimentSpec(
    source_dir="results/clm-validation-001-program-conditionality",
    artifact_dir="artifacts/experiments/clm-validation-001-program-conditionality",
    files=(
        "decision.json",
        "task-spec.json",
        "corpus-manifest.json",
        "arms.csv",
        "phases.csv",
        "quality-vs-compute.png",
        "routing-controls-nll.png",
        "routing-variation.png",
        "program-usage.png",
        "program-coactivation.png",
        "throughput-vs-compute.png",
        "r0-worker.json",
        "r1-worker.json",
        "r2-worker.json",
    ),
    branch="kaggle/clm-validation-001-results",
    expected_format="minicells.clm-validation-001.v1",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish curated CLM Validation 001 results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    args = parser.parse_args()
    root = repo_root()
    destination = prepare_artifacts(root, "clm-validation-001", SPEC,
                                    args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    if args.push:
        push_results(root, destination, "clm-validation-001",
                     args.branch or SPEC.branch, args.secret_name)
    else:
        print("Not pushed. Review the results and re-run with --push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
