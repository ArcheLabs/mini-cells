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
    f"{topology}-{code}-worker.json"
    for topology in ("1d", "2d")
    for code in "ABCDEFGH"
)

SPEC = ExperimentSpec(
    source_dir="results/language-depth-ablation-v1",
    artifact_dir="artifacts/experiments/013-random-depth-ablation",
    files=(
        "decision.json",
        "task-spec.json",
        "corpus-manifest.json",
        "checkpoints.csv",
        "depth-eval.csv",
        "model-summary.csv",
        "factorial-effects.csv",
        "pure-contrasts.csv",
        "replication-check.csv",
        "factorial-ppl.png",
        "factorial-training-cost.png",
        "factor-main-effects-ppl.png",
        "random-depth-isolation.png",
        "1d-depth-robustness.png",
        "2d-depth-robustness.png",
        "step-embedding-growth.png",
        *WORKERS,
    ),
    branch="kaggle/experiment-013-results",
    expected_format="minicells.language-depth-ablation.v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Experiment 013 random-depth ablation results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    destination = prepare_artifacts(root, "013", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    if args.push:
        push_results(
            root,
            destination,
            "013",
            args.branch or SPEC.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push after reviewing the results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
