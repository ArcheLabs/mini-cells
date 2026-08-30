#!/usr/bin/env python3
"""Curate and optionally publish Core Validation 002C results."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from publish_experiment_results import DEFAULT_SECRET_NAME, push_results, repo_root

SOURCE_DIR = Path("results/core-validation-002c-oracle-tomography")
ARTIFACT_DIR = Path("artifacts/experiments/core-validation-002c-oracle-tomography")
DEFAULT_BRANCH = "kaggle/core-validation-002c-oracle-tomography-results"
EXPECTED_FORMAT = "minicells.core-validation.oracle-tomography.v1"
REQUIRED_FILES = (
    "raw.json",
    "decision.json",
    "seed-summary.csv",
    "feature-metrics.csv",
    "gate-summary.csv",
    "oracle-fit-vs-width.png",
    "oracle-leakage-vs-width.png",
    "featurewise-improvement-vs-width.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Core Validation 002C")
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
        raise FileNotFoundError(f"missing Core Validation 002C outputs: {missing}")

    raw = json.loads((source / "raw.json").read_text(encoding="utf-8"))
    decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    if raw.get("format") != EXPECTED_FORMAT or decision.get("format") != EXPECTED_FORMAT:
        raise RuntimeError("unexpected Core Validation 002C result format")
    if raw.get("mode") != "formal" or decision.get("mode") != "formal":
        raise RuntimeError("refusing to publish a smoke run as a formal result")
    if raw.get("provenance", {}).get("tracked_tree_dirty") is not False:
        raise RuntimeError("refusing publication from dirty tracked provenance")
    if decision.get("status") not in {
        "ORACLE_SPARSE_ASSEMBLY_PRESENT",
        "ORACLE_SPARSE_ASSEMBLY_NOT_SUPPORTED",
        "ORACLE_TOMOGRAPHY_INVALID",
    }:
        raise RuntimeError(f"unexpected 002C status: {decision.get('status')!r}")
    if decision.get("status") != "ORACLE_TOMOGRAPHY_INVALID" and decision.get("scientific_decision") is not True:
        raise RuntimeError("formal non-invalid 002C output must be a scientific decision")
    parents = raw.get("parent_experiments", [])
    frozen = {item.get("experiment_id"): item.get("frozen_outcome") for item in parents}
    if frozen.get("core-validation-002") != "WRITE_ADDRESSABILITY_NOT_SUPPORTED":
        raise RuntimeError("002C must preserve the frozen 002 outcome")
    if frozen.get("core-validation-002b") != "SPARSE_WRITE_ASSEMBLY_NOT_SUPPORTED":
        raise RuntimeError("002C must preserve the frozen 002B outcome")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    copied: list[Path] = []
    for name in REQUIRED_FILES:
        target = destination / name
        shutil.copy2(source / name, target)
        copied.append(target)
    protocol_target = destination / "protocol.json"
    shutil.copy2(root / "research" / "core-validation-002c-protocol.json", protocol_target)
    copied.append(protocol_target)

    metadata = {
        "format": "minicells.experiment-publication.v1",
        "experiment_id": "core-validation-002c",
        "experiment_format": EXPECTED_FORMAT,
        "status": decision.get("status"),
        "parent_experiments": parents,
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
    regimes = decision.get("representation_regimes")
    (destination / "RESULTS.md").write_text(
        "# Core Validation 002C Results\n\n"
        f"- Status: `{decision.get('status')}`\n"
        f"- Primary seed passes: `{decision.get('passed_seeds')}/{decision.get('total_seeds')}`\n"
        f"- Representation regimes: `{regimes}`\n"
        "- Parent 002 remains `WRITE_ADDRESSABILITY_NOT_SUPPORTED`.\n"
        "- Parent 002B remains `SPARSE_WRITE_ASSEMBLY_NOT_SUPPORTED`.\n"
        "- Diagnostic: evaluator-only oracle sparse tomography at r in {1,2,4,8,16}.\n"
        "- Dense linear oracle is interpretive only and cannot rescue the sparse gate.\n\n"
        "See `protocol.json`, `decision.json`, `gate-summary.csv`, `seed-summary.csv`, and `feature-metrics.csv`.\n",
        encoding="utf-8",
    )

    print(f"Prepared {destination.relative_to(root)}")
    if args.push:
        push_results(root, destination, "core-validation-002c", args.branch, args.secret_name)
    else:
        print("Not pushed. Re-run with --push to publish the result branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
