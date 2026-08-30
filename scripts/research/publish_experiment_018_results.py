from __future__ import annotations

import argparse

from publish_experiment_results import (
    DEFAULT_SECRET_NAME,
    ExperimentSpec,
    prepare_artifacts,
    push_results,
    repo_root,
)


WORKERS = tuple(f"r{replicate}-worker.json" for replicate in range(3))

SPEC = ExperimentSpec(
    source_dir="results/conditional-tissue-recruitment-v1",
    artifact_dir="artifacts/experiments/018-conditional-tissue-recruitment",
    files=(
        "decision.json",
        "task-spec.json",
        "corpus-manifest.json",
        "tokenizer.json",
        "policy-summary.csv",
        "paired-policy-comparisons.csv",
        "phase1-checkpoints.csv",
        "structural-events.csv",
        "local-learning.csv",
        "transplantation.csv",
        "localization.csv",
        "recruitment-interventions.csv",
        "skill-learning-recruitment-comparison.png",
        "language-retention-recruitment-comparison.png",
        "recruitment-selectivity.png",
        "recruitment-causal-interventions.png",
        "conditional-transplantation-recovery.png",
        "conditional-tissue-atlas.png",
        *WORKERS,
    ),
    branch="kaggle/experiment-018-results",
    expected_format="minicells.conditional-tissue-recruitment.v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Experiment 018 conditional tissue recruitment results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    destination = prepare_artifacts(root, "018", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    if args.push:
        push_results(root, destination, "018", args.branch or SPEC.branch, args.secret_name)
    else:
        print("Not pushed. Re-run with --push after reviewing the results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
