from __future__ import annotations

import argparse
import json
from pathlib import Path

from publish_experiment_results import DEFAULT_SECRET_NAME, ExperimentSpec, prepare_artifacts, push_results, repo_root


WORKERS = tuple(f"r{replicate}-worker.json" for replicate in range(3))

SPEC = ExperimentSpec(
    source_dir="results/probationary-trait-genesis-v1",
    artifact_dir="artifacts/experiments/023b-probationary-trait-genesis",
    files=(
        "decision.json",
        "task-spec.json",
        "invariants.json",
        "checkpoint-manifest.json",
        "source-022-decision.json",
        "source-023-decision.json",
        "corpus-manifest.json",
        "arithmetic-manifest.json",
        "tokenizer.json",
        "proposal.csv",
        "condition-summary.csv",
        "probation-windows.csv",
        "learning-curve.csv",
        "routing.csv",
        "evaluation.csv",
        "pretrain.csv",
        "probation-utility.png",
        "geometry-advantage.png",
        *WORKERS,
    ),
    branch="kaggle/experiment-023b-results",
    expected_format="minicells.probationary-trait-genesis.v1",
)


def validate_checkpoint_manifest(root: Path) -> Path:
    source = root / SPEC.source_dir
    path = source / "checkpoint-manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Experiment 023b checkpoint-manifest.json is missing; run the experiment first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("file_count", -1)) != int(payload.get("expected_file_count", -2)):
        raise RuntimeError(
            f"Experiment 023b checkpoint set incomplete: {payload.get('file_count')}/{payload.get('expected_file_count')}"
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
    destination = prepare_artifacts(root, "023b", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    print("15 Experiment 023b checkpoints remain Kaggle-local and are intentionally excluded from GitHub artifacts.")
    if args.push:
        push_results(root, destination, "023b", args.branch or SPEC.branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
