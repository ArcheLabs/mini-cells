#!/usr/bin/env python3
"""Curate and optionally publish Core Validation 002B Kaggle results."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from publish_experiment_results import DEFAULT_SECRET_NAME, push_results, repo_root

SOURCE_DIR = Path("results/core-validation-002b-sparse-write-assembly")
ARTIFACT_DIR = Path("artifacts/experiments/core-validation-002b-sparse-write-assembly")
DEFAULT_BRANCH = "kaggle/core-validation-002b-sparse-write-assembly-results"
EXPECTED_FORMAT = "minicells.core-validation.sparse-write-assembly.v1"
REQUIRED_FILES = (
    "raw.json",
    "decision.json",
    "edit-records.csv",
    "seed-summary.csv",
    "gate-summary.csv",
    "update-error-vs-address-width.png",
    "matched-update-leakage-frontier.png",
    "representation-geometry-vs-width.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Core Validation 002B")
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
        raise FileNotFoundError(f"missing Core Validation 002B outputs: {missing}")

    raw = json.loads((source / "raw.json").read_text(encoding="utf-8"))
    decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    if raw.get("format") != EXPECTED_FORMAT or decision.get("format") != EXPECTED_FORMAT:
        raise RuntimeError("unexpected Core Validation 002B result format")
    if raw.get("mode") != "formal" or decision.get("mode") != "formal":
        raise RuntimeError("refusing to publish a smoke run as a formal result")
    if raw.get("provenance", {}).get("tracked_tree_dirty") is not False:
        raise RuntimeError("refusing publication from dirty tracked training provenance")
    if decision.get("scientific_decision") is not True:
        raise RuntimeError("refusing to publish a non-scientific decision")
    if decision.get("status") not in {
        "SPARSE_WRITE_ASSEMBLY_SUPPORTED",
        "SPARSE_WRITE_ASSEMBLY_NOT_SUPPORTED",
    }:
        raise RuntimeError(f"unexpected decision status: {decision.get('status')!r}")
    parent = raw.get("parent_experiment", {})
    if parent.get("frozen_outcome") != "WRITE_ADDRESSABILITY_NOT_SUPPORTED":
        raise RuntimeError("002B provenance must preserve the frozen Core Validation 002 outcome")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    copied: list[Path] = []
    for name in REQUIRED_FILES:
        target = destination / name
        shutil.copy2(source / name, target)
        copied.append(target)
    protocol_target = destination / "protocol.json"
    shutil.copy2(root / "research" / "core-validation-002b-protocol.json", protocol_target)
    copied.append(protocol_target)

    metadata = {
        "format": "minicells.experiment-publication.v1",
        "experiment_id": "core-validation-002b",
        "experiment_format": EXPECTED_FORMAT,
        "status": decision.get("status"),
        "parent_experiment": parent,
        "protocol_sha256": raw.get("protocol_sha256"),
        "source_results_dir": SOURCE_DIR.as_posix(),
        "source_commit": raw.get("provenance", {}).get("code_commit"),
        "source_tree": raw.get("provenance", {}).get("code_tree"),
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "kaggle_script_version_id": args.kaggle_script_version_id,
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(copied)
        ],
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "RESULTS.md").write_text(
        "# Core Validation 002B Results\n\n"
        f"- Status: `{decision.get('status')}`\n"
        f"- Primary seed passes: `{decision.get('passed_seeds')}/{decision.get('total_seeds')}`\n"
        "- Parent Core Validation 002 remains `WRITE_ADDRESSABILITY_NOT_SUPPORTED`.\n"
        "- Candidate: non-oracle rank-1 sparse functional write assemblies with r in {1,2,4,8}.\n"
        "- Decisive baseline: full-writer ridge update curve compared at matched median Update Error.\n"
        "- Scope: favorable synthetic additive Gaussian-superposition world; frozen encoder; no replay or growth.\n\n"
        "See `protocol.json`, `decision.json`, `gate-summary.csv`, `seed-summary.csv`, and `edit-records.csv`.\n",
        encoding="utf-8",
    )

    print(f"Prepared {destination.relative_to(root)}")
    if args.push:
        push_results(root, destination, "core-validation-002b", args.branch, args.secret_name)
    else:
        print("Not pushed. Re-run with --push to publish the result branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
