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
    source_dir="results/clm-validation-001b-stable-program-conditionality",
    artifact_dir="artifacts/experiments/clm-validation-001b-stable-program-conditionality",
    files=(
        "decision.json",
        "task-spec.json",
        "runtime.json",
        "corpus-manifest.json",
        "progression.csv",
        "arms.csv",
        "router-diagnostics.csv",
        "VALIDATION_001_DIAGNOSIS.md",
        "quality-vs-k.png",
        "routing-controls-nll.png",
        "routing-variation.png",
        "program-usage.png",
        "program-coactivation.png",
        "compute-vs-quality.png",
        "throughput.png",
        "r0-stage0.json",
        "r1-stage0.json",
        "r2-stage0.json",
        "r0-worker.json",
        "r1-worker.json",
        "r2-worker.json",
    ),
    branch="kaggle/clm-validation-001b-results",
    expected_format="minicells.clm-validation-001b.v1",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish curated CLM Validation 001b results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    args = parser.parse_args()
    root = repo_root()
    destination = prepare_artifacts(
        root, "clm-validation-001b", SPEC, args.kaggle_script_version_id
    )
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    if args.push:
        push_results(
            root, destination, "clm-validation-001b",
            args.branch or SPEC.branch, args.secret_name,
        )
    else:
        print("Not pushed. Review the results and re-run with --push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
