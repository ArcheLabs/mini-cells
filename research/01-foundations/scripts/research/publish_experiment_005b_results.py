from __future__ import annotations

import argparse
from pathlib import Path

from publish_experiment_results import (
    DEFAULT_SECRET_NAME,
    ExperimentSpec,
    prepare_artifacts,
    push_results,
    repo_root,
)


SPEC = ExperimentSpec(
    source_dir="results/consumer-language-ablation-v1",
    artifact_dir="artifacts/experiments/005b-consumer-language-ablation",
    files=(
        "decision.json",
        "task-spec.json",
        "corpus-manifest.json",
        "tokenizer.json",
        "model-configs.json",
        "factorial-results.csv",
        "factorial-effects.csv",
        "checkpoints.csv",
        "replication.csv",
        "generation-samples.json",
        "generation-progression.md",
        "best-500k.pt",
        "factorial-ppl.png",
        "factorial-learning-curves.png",
        "main-effects.png",
        "interaction-effects.png",
        "triple-interaction.png",
        "replication.png",
    ),
    branch="kaggle/experiment-005b-results",
    expected_format="minicells.consumer-language-ablation.v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Experiment 005B curated Kaggle results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    destination = prepare_artifacts(root, "005b", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    if args.push:
        push_results(
            root,
            destination,
            "005b",
            args.branch or SPEC.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push after reviewing the results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
