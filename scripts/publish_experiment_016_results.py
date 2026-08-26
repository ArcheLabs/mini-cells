from __future__ import annotations

import argparse

from publish_experiment_results import (
    DEFAULT_SECRET_NAME,
    ExperimentSpec,
    prepare_artifacts,
    push_results,
    repo_root,
)


WORKERS = tuple(
    f"r{replicate}-{variant}-worker.json"
    for replicate in range(3)
    for variant in ("T", "F", "G")
)

SPEC = ExperimentSpec(
    source_dir="results/growing-cellular-lm-v1",
    artifact_dir="artifacts/experiments/016-growing-cellular-lm",
    files=(
        "decision.json",
        "task-spec.json",
        "corpus-manifest.json",
        "tokenizer.json",
        "model-summary.csv",
        "paired-comparisons.csv",
        "checkpoints.csv",
        "structural-events.csv",
        "structural-probes.csv",
        "interventions.csv",
        "cells.csv",
        "edges.csv",
        "local-learning.csv",
        "transplantation.csv",
        "skill-localization.csv",
        "language-learning-curves.png",
        "quality-vs-gpu-cost.png",
        "cells-edges-over-training.png",
        "final-organism-atlas.png",
        "lineage-tree.png",
        "cell-activity-heatmap.png",
        "cell-gradient-conflict.png",
        "local-learning-transplantation.png",
        *WORKERS,
    ),
    branch="kaggle/experiment-016-results",
    expected_format="minicells.growing-cellular-lm.v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Experiment 016 growing cellular LM results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    destination = prepare_artifacts(root, "016", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    if args.push:
        push_results(root, destination, "016", args.branch or SPEC.branch, args.secret_name)
    else:
        print("Not pushed. Re-run with --push after reviewing the results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
