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
    source_dir="results/language-settling-dynamics-v1",
    artifact_dir="artifacts/experiments/012-settling-dynamics",
    files=(
        "decision.json",
        "task-spec.json",
        "corpus-manifest.json",
        "relaxation-sweep.csv",
        "model-summary.csv",
        "minicells-v2-settling-checkpoints.csv",
        "minicells-2d-k4-settling-checkpoints.csv",
        "minicells-v2-settling-worker.json",
        "minicells-2d-k4-settling-worker.json",
        "minicells-v2-settling-2m.pt",
        "minicells-2d-k4-settling-2m.pt",
        "ppl-vs-relaxation-depth.png",
        "residual-vs-relaxation-depth.png",
        "late-ppl-drift.png",
        "residual-contraction.png",
        "quality-vs-relaxation-compute.png",
    ),
    branch="kaggle/experiment-012-results",
    expected_format="minicells.language-settling-dynamics.v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Experiment 012 settling results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    destination = prepare_artifacts(root, "012", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    if args.push:
        push_results(
            root,
            destination,
            "012",
            args.branch or SPEC.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push after reviewing the results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
