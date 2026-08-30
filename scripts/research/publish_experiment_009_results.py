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
    source_dir="results/language-2d-latent-tissue-v1",
    artifact_dir="artifacts/experiments/009-2d-latent-tissue",
    files=(
        "decision.json",
        "task-spec.json",
        "corpus-manifest.json",
        "checkpoints.csv",
        "model-summary.csv",
        "minicells-v2-checkpoints.csv",
        "minicells-2d-k4-checkpoints.csv",
        "minicells-v2-generations.json",
        "minicells-2d-k4-generations.json",
        "minicells-v2-worker.json",
        "minicells-2d-k4-worker.json",
        "minicells-2d-k4-tissue-diagnostics.json",
        "minicells-v2-2m.pt",
        "minicells-2d-k4-2m.pt",
        "ppl-comparison.png",
        "throughput.png",
        "tissue-cosine.png",
    ),
    branch="kaggle/experiment-009-results",
    expected_format="minicells.language-2d-latent-tissue.v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Experiment 009 2D latent-tissue results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    destination = prepare_artifacts(root, "009", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    if args.push:
        push_results(
            root,
            destination,
            "009",
            args.branch or SPEC.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push after reviewing the results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
