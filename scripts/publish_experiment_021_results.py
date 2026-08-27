from __future__ import annotations

import argparse
import json
from pathlib import Path

from publish_experiment_results import DEFAULT_SECRET_NAME, ExperimentSpec, prepare_artifacts, push_results, repo_root


WORKERS = tuple(f"r{replicate}-worker.json" for replicate in range(3))

SPEC = ExperimentSpec(
    source_dir="results/conflict-driven-differentiation-v1",
    artifact_dir="artifacts/experiments/021-conflict-driven-differentiation",
    files=(
        "decision.json",
        "task-spec.json",
        "checkpoint-manifest.json",
        "corpus-manifest.json",
        "arithmetic-manifest.json",
        "tokenizer.json",
        "arm-summary.csv",
        "evaluation.csv",
        "learning-curve.csv",
        "routing.csv",
        "conflict-windows.csv",
        "pretrain.csv",
        "identity-margin.png",
        "conflict-interference.png",
        "capacity-utility-matrix.png",
        "differentiation-utility-matrix.png",
        "routing-projections.png",
        *WORKERS,
    ),
    branch="kaggle/experiment-021-results",
    expected_format="minicells.conflict-driven-differentiation.v1",
)


def validate_checkpoint_manifest(root: Path) -> Path:
    source = root / SPEC.source_dir
    path = source / "checkpoint-manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Experiment 021 checkpoint-manifest.json is missing; run the experiment first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("file_count", -1)) != int(payload.get("expected_file_count", -2)):
        raise RuntimeError(
            f"Experiment 021 checkpoint set incomplete: {payload.get('file_count')}/{payload.get('expected_file_count')}"
        )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    args = parser.parse_args()
    root = repo_root()
    validate_checkpoint_manifest(root)
    destination = prepare_artifacts(root, "021", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    print("12 Experiment 021 model checkpoints remain Kaggle-local and are intentionally excluded from GitHub artifacts.")
    if args.push:
        push_results(root, destination, "021", args.branch or SPEC.branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
