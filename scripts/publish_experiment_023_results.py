from __future__ import annotations

import argparse
import json
from pathlib import Path

from publish_experiment_results import DEFAULT_SECRET_NAME, ExperimentSpec, prepare_artifacts, push_results, repo_root


WORKERS = tuple(f"r{replicate}-worker.json" for replicate in range(3))

SPEC = ExperimentSpec(
    source_dir="results/online-nonparametric-trait-genesis-v1",
    artifact_dir="artifacts/experiments/023-online-nonparametric-trait-genesis",
    files=(
        "decision.json",
        "task-spec.json",
        "invariants.json",
        "checkpoint-manifest.json",
        "source-022-decision.json",
        "corpus-manifest.json",
        "arithmetic-manifest.json",
        "transform-manifest.json",
        "tokenizer.json",
        "stage-summary.csv",
        "structure.csv",
        "routing.csv",
        "genesis.csv",
        "evaluation.csv",
        "pretrain.csv",
        "replicate-decision.csv",
        "trait-count-trajectory.png",
        "structural-objectives.png",
        "stage-identity.png",
        *WORKERS,
    ),
    branch="kaggle/experiment-023-results",
    expected_format="minicells.online-nonparametric-trait-genesis.v1",
)


def validate_checkpoint_manifest(root: Path) -> Path:
    source = root / SPEC.source_dir
    path = source / "checkpoint-manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Experiment 023 checkpoint-manifest.json is missing; run the experiment first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("file_count", -1)) != int(payload.get("expected_file_count", -2)):
        raise RuntimeError(
            f"Experiment 023 checkpoint set incomplete: {payload.get('file_count')}/{payload.get('expected_file_count')}"
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
    destination = prepare_artifacts(root, "023", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    print("15 Experiment 023 model checkpoints remain Kaggle-local and are intentionally excluded from GitHub artifacts.")
    if args.push:
        push_results(root, destination, "023", args.branch or SPEC.branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
