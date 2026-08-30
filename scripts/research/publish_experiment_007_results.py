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
    source_dir="results/consumer-language-30m-v1",
    artifact_dir="artifacts/experiments/007-minicells-30m",
    files=(
        "decision.json",
        "task-spec.json",
        "corpus-manifest.json",
        "tokenizer.json",
        "model-configs.json",
        "checkpoints.csv",
        "model-summary.csv",
        "relative-gap.csv",
        "generation-samples.json",
        "generation-progression.md",
        "MODEL_CARD.md",
        "minicells-30m-v0-fp16.pt",
        "ppl-scaling.png",
        "nll-scaling.png",
        "relative-gap.png",
        "throughput.png",
    ),
    branch="kaggle/experiment-007-results",
    expected_format="minicells.language-30m.v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Experiment 007 MiniCells-30M results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    destination = prepare_artifacts(root, "007", SPEC, args.kaggle_script_version_id)
    model_path = destination / "minicells-30m-v0-fp16.pt"
    if model_path.stat().st_size >= 95 * 1024 * 1024:
        raise RuntimeError(
            "MiniCells-30M inference artifact is >=95 MiB; do not push it to normal Git. "
            "Move the model to external model storage before publishing."
        )
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    if args.push:
        push_results(
            root,
            destination,
            "007",
            args.branch or SPEC.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push after reviewing the results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
