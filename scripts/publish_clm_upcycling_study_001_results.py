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
    source_dir="results/clm-upcycling-study-001-inherit-then-differentiate",
    artifact_dir="artifacts/experiments/clm-upcycling-study-001-inherit-then-differentiate",
    files=(
        "decision.json",
        "task-spec.json",
        "runtime.json",
        "corpus-manifest.json",
        "progression.csv",
        "controls.csv",
        "continuation-quality.png",
        "final-quality.png",
        "expert-divergence.png",
        "usage-entropy.png",
        "routing-controls.png",
        "routing-variation.png",
        "capacity-vs-active.png",
        "r0-worker.json", "r1-worker.json", "r2-worker.json",
        "r0-geometry-init.json", "r1-geometry-init.json", "r2-geometry-init.json",
    ),
    branch="kaggle/clm-upcycling-study-001-results",
    expected_format="minicells.clm-upcycling-study-001.v1",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish CLM Upcycling Study 001 results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    args = parser.parse_args()
    root = repo_root()
    destination = prepare_artifacts(
        root, "clm-upcycling-study-001", SPEC, args.kaggle_script_version_id
    )
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    if args.push:
        push_results(
            root,
            destination,
            "clm-upcycling-study-001",
            args.branch or SPEC.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Review the results and re-run with --push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
