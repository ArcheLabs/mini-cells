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
    source_dir="results/language-stabilizing-cost-v1",
    artifact_dir="artifacts/experiments/011-stabilizing-cost",
    files=(
        "decision.json",
        "task-spec.json",
        "corpus-manifest.json",
        "checkpoints.csv",
        "model-summary.csv",
        "cost-to-quality.csv",
        "halting-sweep.csv",
        "transformer-s-checkpoints.csv",
        "minicells-v2-fixed-checkpoints.csv",
        "minicells-v2-stable-checkpoints.csv",
        "minicells-2d-k4-fixed-checkpoints.csv",
        "minicells-2d-k4-stable-checkpoints.csv",
        "minicells-v2-fixed-halting.csv",
        "minicells-v2-stable-halting.csv",
        "minicells-2d-k4-fixed-halting.csv",
        "minicells-2d-k4-stable-halting.csv",
        "transformer-s-worker.json",
        "minicells-v2-fixed-worker.json",
        "minicells-v2-stable-worker.json",
        "minicells-2d-k4-fixed-worker.json",
        "minicells-2d-k4-stable-worker.json",
        "minicells-v2-stable-2m.pt",
        "minicells-2d-k4-stable-2m.pt",
        "ppl-vs-training-tokens.png",
        "ppl-vs-training-seconds.png",
        "training-cost-per-million.png",
        "peak-vram.png",
        "quality-cost-frontier.png",
        "cost-to-quality.png",
        "adaptive-iterations.png",
    ),
    branch="kaggle/experiment-011-results",
    expected_format="minicells.language-stabilizing-cost.v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Experiment 011 stabilizing/cost results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    destination = prepare_artifacts(root, "011", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    if args.push:
        push_results(
            root,
            destination,
            "011",
            args.branch or SPEC.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push after reviewing the results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
