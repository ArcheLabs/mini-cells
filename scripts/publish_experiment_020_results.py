from __future__ import annotations

import argparse
import json
from pathlib import Path

from publish_experiment_results import DEFAULT_SECRET_NAME, ExperimentSpec, prepare_artifacts, push_results, repo_root


WORKERS = tuple(f"r{replicate}-worker.json" for replicate in range(3))

SPEC = ExperimentSpec(
    source_dir="results/capability-tissue-specificity-v1",
    artifact_dir="artifacts/experiments/020-capability-tissue-specificity",
    files=(
        "decision.json",
        "task-spec.json",
        "invariants.json",
        "checkpoint-manifest.json",
        "corpus-manifest.json",
        "tokenizer.json",
        "baseline-019b-specificity.csv",
        "donor-summary.csv",
        "specificity-observations.csv.gz",
        "utility-matrix.csv",
        "replicate-specificity.csv",
        "family-specificity.csv",
        "geometry-comparison.csv",
        "one-cell-utility-matrix.png",
        "three-cell-utility-matrix.png",
        "specificity-by-family.png",
        "identity-recovery.png",
        *WORKERS,
    ),
    branch="kaggle/experiment-020-results",
    expected_format="minicells.capability-tissue-specificity.v1",
)


def write_checkpoint_manifest(root: Path) -> Path:
    source = root / SPEC.source_dir
    checkpoint_dir = source / "checkpoints"
    files = []
    if checkpoint_dir.is_dir():
        for path in sorted(checkpoint_dir.glob("*.pt")):
            files.append({"name": path.name, "bytes": path.stat().st_size})
    manifest = {
        "format": "minicells.capability-tissue-specificity-checkpoint-manifest.v1",
        "experiment": "020",
        "checkpoint_dir": str(checkpoint_dir.relative_to(root)),
        "file_count": len(files),
        "expected_file_count": 36,
        "files": files,
        "published_model_checkpoints": False,
        "purpose": "Kaggle-local donor recovery; checkpoints are not part of curated GitHub artifacts",
    }
    path = source / "checkpoint-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if len(files) != 36:
        raise RuntimeError(f"Experiment 020 checkpoint set incomplete: {len(files)}/36")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    args = parser.parse_args()
    root = repo_root()
    write_checkpoint_manifest(root)
    destination = prepare_artifacts(root, "020", SPEC, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    print("36 Experiment 020 donor checkpoints remain Kaggle-local and are intentionally excluded from GitHub artifacts.")
    if args.push:
        push_results(root, destination, "020", args.branch or SPEC.branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
