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
    source_dir="results/localized-cellular-learning-v1",
    artifact_dir="artifacts/experiments/017-localized-cellular-learning",
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
        "tissue-ablation.csv",
        "skill-learning-policy-comparison.png",
        "language-retention-policy-comparison.png",
        "localized-memory-updates.png",
        "transplantation-recovery.png",
        "localized-tissue-atlas.png",
        *WORKERS,
    ),
    branch="kaggle/experiment-017-results",
    expected_format="minicells.localized-cellular-learning.v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Experiment 017 localized cellular learning results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    destination = prepare_artifacts(root, "017", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    if args.push:
        push_results(root, destination, "017", args.branch or SPEC.branch, args.secret_name)
    else:
        print("Not pushed. Re-run with --push after reviewing the results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
