from __future__ import annotations

import argparse
import json
from pathlib import Path

from publish_experiment_results import (
    DEFAULT_SECRET_NAME,
    ExperimentSpec,
    prepare_artifacts,
    push_results,
    repo_root,
)


SCREEN_WORKERS = tuple(f"r{replicate}-screen-worker.json" for replicate in range(3))
CHALLENGE_WORKERS = tuple(f"r{replicate}-challenge-worker.json" for replicate in range(3))

SPEC = ExperimentSpec(
    source_dir="results/independent-third-trait-v1",
    artifact_dir="artifacts/experiments/024b-independent-third-trait",
    files=(
        "decision.json",
        "selection.json",
        "task-spec.json",
        "invariants.json",
        "checkpoint-manifest.json",
        "source-022-decision.json",
        "source-023-decision.json",
        "source-023b-decision.json",
        "source-024-decision.json",
        "source-024-diagnosis.md",
        "corpus-manifest.json",
        "arithmetic-manifest.json",
        "candidate-manifests.json",
        "tokenizer.json",
        "screening.csv",
        "stage-summary.csv",
        "proposal.csv",
        "windows.csv",
        "learning.csv",
        "routing.csv",
        "evaluation.csv",
        "replicate-decision.csv",
        "candidate-independence-screen.png",
        "selected-capability-probation.png",
        *SCREEN_WORKERS,
        *CHALLENGE_WORKERS,
    ),
    branch="kaggle/experiment-024b-results",
    expected_format="minicells.independent-third-trait.v1",
)


def validate_checkpoint_manifest(root: Path) -> Path:
    source = root / SPEC.source_dir
    path = source / "checkpoint-manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Experiment 024b checkpoint-manifest.json is missing; run the experiment first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not bool(payload.get("observed_path_complete", False)):
        raise RuntimeError(
            f"Experiment 024b observed execution path has missing checkpoints: {payload.get('missing_required')}"
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
    destination = prepare_artifacts(root, "024b", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    print("Experiment 024b model/stage checkpoints remain Kaggle-local and are intentionally excluded from GitHub artifacts.")
    if args.push:
        push_results(root, destination, "024b", args.branch or SPEC.branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
