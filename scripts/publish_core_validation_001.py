#!/usr/bin/env python3
"""Curate and optionally publish Core Validation 001 Kaggle results."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from publish_experiment_results import DEFAULT_SECRET_NAME, push_results, repo_root

SOURCE_DIR = Path("results/core-validation-001-knowledge-subsumption")
ARTIFACT_DIR = Path("artifacts/experiments/core-validation-001-knowledge-subsumption")
DEFAULT_BRANCH = "kaggle/core-validation-001-knowledge-subsumption-results"
EXPECTED_FORMAT = "minicells.core-validation.knowledge-subsumption.v1"
REQUIRED_FILES = (
    "raw.json",
    "decision.json",
    "runs.csv",
    "fourier-circuit-concentration.png",
    "fourier-circuit-interventions.png",
    "causal-path-reuse.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Core Validation 001")
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
        raise FileNotFoundError(f"missing Core Validation 001 outputs: {missing}")

    raw = json.loads((source / "raw.json").read_text(encoding="utf-8"))
    decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    if raw.get("format") != EXPECTED_FORMAT or decision.get("format") != EXPECTED_FORMAT:
        raise RuntimeError("unexpected Core Validation 001 result format")
    if raw.get("mode") != "formal" or decision.get("mode") != "formal":
        raise RuntimeError("refusing to publish a smoke run as a formal result")
    if raw.get("provenance", {}).get("tracked_tree_dirty") is not False:
        raise RuntimeError("refusing publication from dirty tracked training provenance")
    if decision.get("status") not in {
        "KNOWLEDGE_SUBSUMPTION_SUPPORTED",
        "KNOWLEDGE_SUBSUMPTION_NOT_SUPPORTED",
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
    shutil.copy2(root / "research" / "core-validation-001-protocol.json", protocol_target)
    copied.append(protocol_target)

    metadata = {
        "format": "minicells.experiment-publication.v1",
        "experiment_id": "core-validation-001",
        "experiment_format": EXPECTED_FORMAT,
        "status": decision.get("status"),
        "source_results_dir": SOURCE_DIR.as_posix(),
        "source_commit": raw.get("provenance", {}).get("code_commit"),
        "source_tree": raw.get("provenance", {}).get("code_tree"),
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
        "# Core Validation 001 Results\n\n"
        f"- Status: `{decision.get('status')}`\n"
        f"- Primary passes: `{decision.get('primary_passes')}/{decision.get('primary_runs')}`\n"
        f"- Control false positives: `{decision.get('control_false_positives')}`\n"
        "- Scope: fixed factored MLP, sequential modular addition, no replay and no growth.\n"
        "- Mechanistic test: Fourier restricted/excluded circuit formation and cleanup.\n"
        "- Interpretation: this does not establish that MiniCells uniquely causes the effect.\n\n"
        "See `protocol.json`, `decision.json`, and `runs.csv` for the frozen gate and measurements.\n",
        encoding="utf-8",
    )

    print(f"Prepared {destination.relative_to(root)}")
    if args.push:
        push_results(root, destination, "core-validation-001", args.branch, args.secret_name)
    else:
        print("Not pushed. Re-run with --push to publish the result branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
