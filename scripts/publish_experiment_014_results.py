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
    f"{topology}-r{replicate}-{code}-worker.json"
    for topology in ("1d", "2d")
    for replicate in range(5)
    for code in ("A", "B", "F", "H")
)

SPEC = ExperimentSpec(
    source_dir="results/language-multiseed-core-v1",
    artifact_dir="artifacts/experiments/014-multiseed-core-recipe",
    files=(
        "decision.json",
        "task-spec.json",
        "corpus-manifest.json",
        "checkpoints.csv",
        "depth-eval.csv",
        "model-summary.csv",
        "paired-ratios.csv",
        "effect-summary.csv",
        "core-recipe-seed-ratios.png",
        "core-recipe-ppl-cost.png",
        "factor-main-effects.png",
        *WORKERS,
    ),
    branch="kaggle/experiment-014-results",
    expected_format="minicells.language-multiseed-core.v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Experiment 014 multi-seed core recipe results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    destination = prepare_artifacts(root, "014", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    if args.push:
        push_results(
            root,
            destination,
            "014",
            args.branch or SPEC.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push after reviewing the results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
