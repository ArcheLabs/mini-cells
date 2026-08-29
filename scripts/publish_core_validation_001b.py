#!/usr/bin/env python3
"""Curate and optionally publish Core Validation 001b Kaggle results."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from publish_experiment_results import DEFAULT_SECRET_NAME, push_results, repo_root

SOURCE_DIR = Path("results/core-validation-001b-residual-memorization")
ARTIFACT_DIR = Path("artifacts/experiments/core-validation-001b-residual-memorization")
DEFAULT_BRANCH = "kaggle/core-validation-001b-residual-memorization-results"
EXPECTED_FORMAT = "minicells.core-validation.residual-memorization.v1"
REQUIRED_FILES = (
    "raw.json",
    "decision.json",
    "runs.csv",
    "frequency-sweep.csv",
    "oracle-frequency-sweep.csv",
    "frequency-exclusion-trajectories.png",
    "membership-gap-trajectories.png",
    "oracle-exclusion-trajectory.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Core Validation 001b")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    root = repo_root()
    source = root / SOURCE_DIR
    destination = root / ARTIFACT_DIR
    if not source.is_dir():
        raise FileNotFoundError(source)
    missing = [source / name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing Core Validation 001b outputs: {missing}")

    raw = json.loads((source / "raw.json").read_text(encoding="utf-8"))
    decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    if raw.get("format") != EXPECTED_FORMAT or decision.get("format") != EXPECTED_FORMAT:
        raise RuntimeError("unexpected Core Validation 001b result format")
    if raw.get("mode") != "formal" or decision.get("mode") != "formal":
        raise RuntimeError("refusing to publish a non-formal Core Validation 001b result")
    if raw.get("provenance", {}).get("tracked_tree_dirty") is not False:
        raise RuntimeError("refusing publication from dirty tracked analysis provenance")
    if raw.get("parent_training_provenance", {}).get("tracked_tree_dirty") is not False:
        raise RuntimeError("refusing publication from dirty parent training provenance")
    if decision.get("status") not in {
        "NO_MATERIAL_RESIDUAL_MEMORIZATION_DETECTED",
        "RESIDUAL_MEMORIZATION_OR_INCONCLUSIVE",
    }:
        raise RuntimeError(f"unexpected decision status: {decision.get('status')!r}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    copied: list[Path] = []
    for name in REQUIRED_FILES:
        target = destination / name
        shutil.copy2(source / name, target)
        copied.append(target)
    protocol_target = destination / "protocol.json"
    shutil.copy2(root / "research" / "core-validation-001b-protocol.json", protocol_target)
    copied.append(protocol_target)
    parent_protocol_target = destination / "parent-core-validation-001-protocol.json"
    shutil.copy2(root / "research" / "core-validation-001-protocol.json", parent_protocol_target)
    copied.append(parent_protocol_target)

    metadata = {
        "format": "minicells.experiment-publication.v1",
        "experiment_id": "core-validation-001b",
        "experiment_format": EXPECTED_FORMAT,
        "status": decision.get("status"),
        "source_results_dir": SOURCE_DIR.as_posix(),
        "source_commit": raw.get("provenance", {}).get("code_commit"),
        "source_tree": raw.get("provenance", {}).get("code_tree"),
        "parent_training_commit": raw.get("parent_training_provenance", {}).get(
            "code_commit"
        ),
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "kaggle_script_version_id": args.kaggle_script_version_id,
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(copied)
        ],
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "RESULTS.md").write_text(
        "# Core Validation 001b Results\n\n"
        f"- Status: `{decision.get('status')}`\n"
        f"- Primary passes: `{decision.get('primary_passes')}/{decision.get('primary_runs')}`\n"
        f"- Control false positives: `{decision.get('control_false_positives')}`\n"
        f"- Oracle assay valid: `{decision.get('oracle_valid')}`\n"
        "- Scope: fresh rerun of Core Validation 001 training followed by cumulative Fourier-pair removal/retention diagnostics.\n"
        "- No training mechanism changed; no replay, routing, growth, mitosis, tissue, or apoptosis was added.\n"
        "- Positive interpretation is limited to absence of a material old-vs-heldout membership advantage under this preregistered assay.\n\n"
        "See `protocol.json`, `decision.json`, `runs.csv`, and `frequency-sweep.csv` for the frozen gate and measurements.\n",
        encoding="utf-8",
    )

    print(f"Prepared {destination.relative_to(root)}")
    if args.push:
        push_results(root, destination, "core-validation-001b", args.branch, args.secret_name)
    else:
        print("Not pushed. Re-run with --push to publish the result branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
