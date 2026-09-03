#!/usr/bin/env python3
"""Curate and optionally publish Shadow Cell Validation 001 v2 results.

The runner owns scientific execution.  This module owns the publication
boundary: it verifies protocol identity, aggregates the registered formal
seeds, excludes binary checkpoints from Git, writes provenance, and delegates
authenticated GitHub publication to the repository's shared publisher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from publish_experiment_results import (
    DEFAULT_SECRET_NAME,
    push_results,
    repo_root,
)


VALIDATION_ID = "shadow-cell-validation-001-v2-developmental-maturation"
FORMAL_SEEDS = (95311, 95312, 95313)
DEFAULT_SOURCE = Path("results") / VALIDATION_ID
DEFAULT_ARTIFACT = Path("artifacts/experiments") / VALIDATION_ID
DEFAULT_BRANCH = "kaggle/shadow-cell-validation-001-v2-results"
AGGREGATOR = Path("scripts/research/aggregate_shadow_cell_validation_001_v2.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def _run_aggregate(root: Path, source: Path) -> dict:
    protocol_path = root / "research/validations" / VALIDATION_ID / "protocol.json"
    protocol_sha = sha256_file(protocol_path)
    subprocess.run(
        [
            sys.executable,
            str(root / AGGREGATOR),
            "--results-root",
            str(source),
            "--protocol-sha256",
            protocol_sha,
        ],
        cwd=root,
        check=True,
    )
    return json.loads((source / "aggregate.json").read_text(encoding="utf-8"))


def _load_protocol(root: Path) -> tuple[dict, str]:
    path = root / "research/validations" / VALIDATION_ID / "protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("validation_id") != VALIDATION_ID:
        raise RuntimeError("unexpected Shadow v2 validation id")
    return payload, sha256_file(path)


def _validate_formal_results(source: Path, protocol_sha: str) -> list[dict]:
    results: list[dict] = []
    for seed in FORMAL_SEEDS:
        path = source / f"seed-{seed}" / "result.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("validation_id") != VALIDATION_ID:
            raise RuntimeError(f"unexpected validation id in {path}")
        if int(result.get("seed", -1)) != seed:
            raise RuntimeError(f"seed identity mismatch in {path}")
        if result.get("phase") != "formal":
            raise RuntimeError(f"non-formal result cannot be published: {path}")
        if result.get("protocol_sha256") != protocol_sha:
            raise RuntimeError(f"protocol identity mismatch in {path}")
        results.append(result)
    if not results:
        raise RuntimeError(
            "no completed formal result found; run at least one registered seed first"
        )
    return results


def _copy_curated_files(source: Path, destination: Path) -> list[dict[str, object]]:
    """Copy text/figure evidence while deliberately excluding checkpoint binaries."""
    allowed_suffixes = {".csv", ".json", ".md", ".png", ".txt", ".log"}
    copied: list[dict[str, object]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(
            {
                "path": relative.as_posix(),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    return copied


def _checkpoint_manifest(source: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(source.rglob("*.pt")):
        records.append(
            {
                "path": path.relative_to(source).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "published_to_git": False,
            }
        )
    return records


def prepare_publication(
    root: Path,
    source: Path,
    destination: Path,
    *,
    kaggle_script_version_id: str | None = None,
) -> Path:
    source = source if source.is_absolute() else root / source
    destination = destination if destination.is_absolute() else root / destination
    if not source.is_dir():
        raise FileNotFoundError(source)

    protocol, protocol_sha = _load_protocol(root)
    results = _validate_formal_results(source, protocol_sha)
    aggregate = _run_aggregate(root, source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{VALIDATION_ID}-", dir=destination.parent))
    try:
        copied = _copy_curated_files(source, staging)
        checkpoint_records = _checkpoint_manifest(source)
        metadata = {
            "format": "minicells.shadow-cell-validation-001-v2.publication.v1",
            "validation_id": VALIDATION_ID,
            "status": aggregate.get("status", "INCOMPLETE"),
            "scientific_decision": bool(aggregate.get("scientific_decision", False)),
            "protocol_sha256": protocol_sha,
            "formal_seeds": list(FORMAL_SEEDS),
            "completed_formal_seeds": [int(result["seed"]) for result in results],
            "source_results_dir": str(source.relative_to(root)),
            "source_commit": _git(root, "rev-parse", "HEAD"),
            "source_branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "published_at_utc": datetime.now(timezone.utc).isoformat(),
            "kaggle": {
                "script_version_id": kaggle_script_version_id,
                "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "protocol": {
                "model": protocol.get("model"),
                "maturity_grid": protocol.get("maturity_grid"),
            },
            "files": copied,
            "checkpoint_files": checkpoint_records,
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "RESULTS.md").write_text(
            "\n".join(
                [
                    "# Shadow Cell Validation 001 v2 Results",
                    "",
                    f"- Status: `{metadata['status']}`",
                    f"- Scientific decision: `{metadata['scientific_decision']}`",
                    f"- Completed formal seeds: `{metadata['completed_formal_seeds']}`",
                    f"- Required formal seeds: `{list(FORMAL_SEEDS)}`",
                    f"- Protocol SHA-256: `{protocol_sha}`",
                    "",
                    "This directory is generated from machine-readable result artifacts.",
                    "Binary checkpoint files remain in the Kaggle results directory and are",
                    "listed with SHA-256 hashes in `metadata.json`; they are not committed to Git.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def publish_results(
    root: Path,
    source: Path,
    *,
    branch: str = DEFAULT_BRANCH,
    secret_name: str = DEFAULT_SECRET_NAME,
    kaggle_script_version_id: str | None = None,
) -> Path:
    destination = root / DEFAULT_ARTIFACT
    prepared = prepare_publication(
        root,
        source,
        destination,
        kaggle_script_version_id=kaggle_script_version_id,
    )
    push_results(root, prepared, VALIDATION_ID, branch, secret_name)
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Shadow v2 evidence and optionally push it to GitHub."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--kaggle-script-version-id")
    args = parser.parse_args()
    root = repo_root()
    destination = prepare_publication(
        root,
        args.source,
        root / DEFAULT_ARTIFACT,
        kaggle_script_version_id=args.kaggle_script_version_id,
    )
    print(f"Prepared curated Shadow v2 artifacts: {destination.relative_to(root)}")
    if args.push:
        push_results(
            root,
            destination,
            VALIDATION_ID,
            args.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push to publish to GitHub.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
