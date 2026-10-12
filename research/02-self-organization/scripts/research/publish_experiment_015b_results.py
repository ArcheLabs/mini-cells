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
    for variant in ("B", "D")
)

SPEC = ExperimentSpec(
    source_dir="results/language-plastic-reaction-diffusion-v1",
    artifact_dir="artifacts/experiments/015b-plastic-reaction-diffusion",
    files=(
        "decision.json",
        "task-spec.json",
        "corpus-manifest.json",
        "model-summary.csv",
        "paired-ratios.csv",
        "checkpoints.csv",
        "task-metrics.csv",
        "depth-eval.csv",
        "activity.csv",
        "connectome.csv",
        "final-activity.csv",
        "final-connectome.csv",
        "dynamics.csv",
        "interventions.csv",
        "ablation.csv",
        "final-node-topology.png",
        "task-node-topology-atlas.png",
        "final-connectome-heatmap.png",
        "activity-dynamics.png",
        "connectome-dynamics.png",
        "reaction-diffusion-dynamics.png",
        *WORKERS,
    ),
    branch="kaggle/experiment-015b-results",
    expected_format="minicells.language-plastic-reaction-diffusion.v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish Experiment 015b plastic reaction-diffusion results."
    )
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    destination = prepare_artifacts(
        root,
        "015b",
        SPEC,
        args.kaggle_script_version_id,
    )
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    if args.push:
        push_results(
            root,
            destination,
            "015b",
            args.branch or SPEC.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push after reviewing the results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
