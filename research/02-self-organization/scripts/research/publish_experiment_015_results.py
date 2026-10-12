from __future__ import annotations

import argparse

from publish_experiment_results import (
    DEFAULT_SECRET_NAME,
    ExperimentSpec,
    prepare_artifacts,
    push_results,
    repo_root,
)


WORKERS = tuple(f"r{replicate}-{variant}-worker.json" for replicate in range(3) for variant in "ABC")

SPEC = ExperimentSpec(
    source_dir="results/language-sparse-topology-v1",
    artifact_dir="artifacts/experiments/015-emergent-sparse-topology",
    files=(
        "decision.json",
        "task-spec.json",
        "corpus-manifest.json",
        "model-summary.csv",
        "paired-ratios.csv",
        "checkpoints.csv",
        "task-metrics.csv",
        "task-region.csv",
        "task-edge.csv",
        "ablation.csv",
        "composition-reuse.csv",
        "quality-vs-sparsity.png",
        "task-region-heatmap.png",
        "ablation-heatmap.png",
        "composition-reuse.png",
        *WORKERS,
    ),
    branch="kaggle/experiment-015-results",
    expected_format="minicells.language-sparse-topology.v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Experiment 015 sparse-topology results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    destination = prepare_artifacts(root, "015", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    if args.push:
        push_results(
            root,
            destination,
            "015",
            args.branch or SPEC.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push after reviewing the results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
