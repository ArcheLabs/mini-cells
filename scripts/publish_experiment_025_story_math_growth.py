#!/usr/bin/env python3
"""Curate and optionally publish Experiment-025 Kaggle results."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from publish_experiment_results import DEFAULT_SECRET_NAME, push_results, repo_root, run_git


SOURCE_DIR = Path("results/experiment-025-story-math-growth")
ARTIFACT_DIR = Path("artifacts/experiments/025-story-math-growth")
DEFAULT_BRANCH = "kaggle/experiment-025-story-math-growth-results"
EXPECTED_FORMAT = "minicells.story-math-shift-30m.v1"

TOP_LEVEL = (
    "decision.json",
    "run-provenance.json",
    "worker-summary.json",
    "panel-a-starting-comparability.csv",
    "panel-a-starting-comparability.png",
    "llm-vs-clm-trajectory.csv",
    "story-math-performance.png",
    "growth-timeline.png",
)
ARM_FILES = (
    ("llm", "metrics.csv", "llm-metrics.csv"),
    ("llm", "events.json", "llm-events.json"),
    ("llm", "worker-summary.json", "llm-worker-summary.json"),
    ("clm", "metrics.csv", "clm-metrics.csv"),
    ("clm", "events.json", "clm-events.json"),
    ("clm", "worker-summary.json", "clm-worker-summary.json"),
)


def parser() -> argparse.Namespace:
    result = argparse.ArgumentParser(description="Publish Experiment 025")
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
    missing.extend(source / arm / name for arm, name, _ in ARM_FILES if not (source / arm / name).is_file())
    if missing:
        raise FileNotFoundError(f"missing Experiment-025 outputs: {[str(path) for path in missing]}")

    decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    if decision.get("format") != EXPECTED_FORMAT:
        raise RuntimeError(f"unexpected Experiment-025 decision format: {decision.get('format')!r}")
    workers = json.loads((source / "worker-summary.json").read_text(encoding="utf-8"))
    if workers.get("complete") is not True:
        raise RuntimeError("refusing publication: both Experiment-025 arms must be complete")
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
    animation = source / "growth-animation.gif"
    if animation.is_file():
        target = destination / animation.name
        shutil.copy2(animation, target)
        copied.append(target)
    for arm, name, target_name in ARM_FILES:
        target = destination / target_name
        shutil.copy2(source / arm / name, target)
        copied.append(target)

    metadata = {
        "format": "minicells.experiment-publication.v1",
        "experiment_id": "025",
        "experiment_format": EXPECTED_FORMAT,
        "status": decision.get("status"),
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
        "# Experiment 025 Results\n\n"
        f"- Status: `{decision.get('status')}`\n"
        f"- Pareto crossover: `{decision.get('pareto_crossover') is not None}`\n"
        f"- Persistent promotions: `{decision.get('growth', {}).get('promotions')}`\n"
        "- Main comparison: fixed 30M Transformer LLM vs growing 30M-source CLM.\n"
        "- Math scope: held-out synthetic integer arithmetic, not general mathematical reasoning.\n\n"
        "See `decision.json` for the preregistered crossover definition and machine-readable result.\n",
        encoding="utf-8",
    )

    print(f"Prepared curated Experiment-025 artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    if args.push:
        push_results(root, destination, "025", args.branch, args.secret_name)
    else:
        print("Not pushed. Re-run with --push to publish the result branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
