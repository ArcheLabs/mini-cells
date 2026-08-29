#!/usr/bin/env python3
"""Curate and optionally publish Experiment-026 Kaggle results."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from publish_experiment_results import DEFAULT_SECRET_NAME, push_results, repo_root


SOURCE_DIR = Path("results/experiment-026-cell-granularity")
ARTIFACT_DIR = Path("artifacts/experiments/026-cell-granularity")
DEFAULT_BRANCH = "kaggle/experiment-026-cell-granularity-results"
EXPECTED_FORMAT = "minicells.cell-granularity-30m.v1"

TOP_LEVEL = (
    "protocol.json",
    "decision.json",
    "run-provenance.json",
    "worker-summary.json",
    "granularity-trajectory.csv",
    "granularity-final.csv",
    "cell-diagnostics.csv",
    "tissue-diagnostics.csv",
    "performance-by-granularity.png",
    "differentiation-by-granularity.png",
    "granularity-frontier.png",
)
ARM_FILES = (
    "metrics.csv",
    "cell-diagnostics.csv",
    "tissue-diagnostics.csv",
    "worker-summary.json",
    "age-zero-parity.json",
)


def parser() -> argparse.Namespace:
    result = argparse.ArgumentParser(description="Publish Experiment 026")
    result.add_argument("--push", action="store_true")
    result.add_argument("--branch", default=DEFAULT_BRANCH)
    result.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    result.add_argument("--kaggle-script-version-id")
    return result.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parser()
    root = repo_root()
    source = root / SOURCE_DIR
    destination = root / ARTIFACT_DIR
    if not source.is_dir():
        raise FileNotFoundError(source)

    missing = [source / name for name in TOP_LEVEL if not (source / name).is_file()]
    for granularity in (1, 2, 4, 8):
        missing.extend(
            source / f"g{granularity}" / name
            for name in ARM_FILES
            if not (source / f"g{granularity}" / name).is_file()
        )
    if missing:
        raise FileNotFoundError(f"missing Experiment-026 outputs: {[str(path) for path in missing]}")

    decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    if decision.get("format") != EXPECTED_FORMAT:
        raise RuntimeError(f"unexpected Experiment-026 decision format: {decision.get('format')!r}")
    workers = json.loads((source / "worker-summary.json").read_text(encoding="utf-8"))
    if workers.get("complete") is not True:
        raise RuntimeError("refusing publication: all Experiment-026 arms must be complete")
    provenance = json.loads((source / "run-provenance.json").read_text(encoding="utf-8"))
    if provenance.get("tracked_tree_dirty") is not False:
        raise RuntimeError("refusing publication from dirty tracked training provenance")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    copied: list[Path] = []
    for name in TOP_LEVEL:
        target = destination / name
        shutil.copy2(source / name, target)
        copied.append(target)
    for granularity in (1, 2, 4, 8):
        for name in ARM_FILES:
            target = destination / f"g{granularity}-{name}"
            shutil.copy2(source / f"g{granularity}" / name, target)
            copied.append(target)

    metadata = {
        "format": "minicells.experiment-publication.v1",
        "experiment_id": "026",
        "experiment_format": EXPECTED_FORMAT,
        "status": decision.get("status"),
        "selected_granularity": decision.get("selected_granularity"),
        "source_results_dir": SOURCE_DIR.as_posix(),
        "source_commit": provenance.get("code_commit"),
        "source_tree_sha": provenance.get("code_tree_sha"),
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
        "# Experiment 026 Results\n\n"
        f"- Status: `{decision.get('status')}`\n"
        f"- Selected qualifying granularity: `{decision.get('selected_granularity')}`\n"
        f"- Parameter parity: `{decision.get('parameter_parity')}`\n"
        f"- Max age-zero logit difference: `{decision.get('max_age_zero_logit_abs_diff')}`\n"
        "- Scope: fixed-capacity G={1,2,4,8} cell-granularity ablation under a balanced four-domain continuation.\n"
        "- Persistent growth is disabled; this experiment does not test autonomous mitosis.\n\n"
        "See `protocol.json` for the frozen preregistration and `decision.json` for the machine-readable result.\n",
        encoding="utf-8",
    )

    print(f"Prepared curated Experiment-026 artifacts: {destination.relative_to(root)}")
    if args.push:
        push_results(root, destination, "026", args.branch, args.secret_name)
    else:
        print("Not pushed. Re-run with --push to publish the result branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
