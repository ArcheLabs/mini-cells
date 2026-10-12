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
    source_dir="results/language-adaptive-halting-v1",
    artifact_dir="artifacts/experiments/010-adaptive-halting",
    files=(
        "decision.json",
        "task-spec.json",
        "halting-sweep.csv",
        "ppl-vs-iterations.png",
        "walltime-vs-iterations.png",
    ),
    branch="kaggle/experiment-010-results",
    expected_format="minicells.language-adaptive-halting.v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Experiment 010 adaptive-halting results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    destination = prepare_artifacts(root, "010", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    if args.push:
        push_results(
            root,
            destination,
            "010",
            args.branch or SPEC.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push after reviewing the results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
