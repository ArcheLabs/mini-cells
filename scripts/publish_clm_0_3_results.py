from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

from publish_experiment_results import (
    DEFAULT_SECRET_NAME,
    push_results,
    repo_root,
    run_git,
)


SOURCE_DIR = Path("results/clm-0.3-progressive-growth")
ARTIFACT_DIR = Path("artifacts/experiments/clm-0.3-progressive-growth")
DEFAULT_BRANCH = "kaggle/clm-0.3-progressive-growth-results"
EXPECTED_DECISION_FORMAT = "minicells.clm-0.3-progressive-growth.decision.v1"
FORMAL_ARMS = ("fixed4", "pressure_growth", "random_growth")
FORMAL_REPLICATES = (0, 1, 2)

TOP_LEVEL_FILES = (
    "decision.json",
    "formal-ppl-history.csv",
    "replicate-summary.json",
)
WORKER_COMMON_FILES = (
    "events.jsonl",
    "ppl-history.csv",
)
WORKER_GROWTH_FILES = (
    "growth-history.json",
    "newborn-diagnostics.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _required_worker_files(source: Path, replicate: int, arm: str) -> list[Path]:
    worker = source / f"r{replicate}-{arm}"
    required = [worker / name for name in WORKER_COMMON_FILES]
    if arm != "fixed4":
        required.extend(worker / name for name in WORKER_GROWTH_FILES)
    return required


def _read_events(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _validate_formal_results(source: Path) -> dict[str, object]:
    missing = [source / name for name in TOP_LEVEL_FILES if not (source / name).is_file()]
    for replicate in FORMAL_REPLICATES:
        for arm in FORMAL_ARMS:
            missing.extend(
                path for path in _required_worker_files(source, replicate, arm) if not path.is_file()
            )
    if missing:
        relative = [str(path.relative_to(source)) for path in missing]
        raise FileNotFoundError(f"Missing CLM-0.3 formal artifacts: {relative}")

    decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    if decision.get("format") != EXPECTED_DECISION_FORMAT:
        raise RuntimeError(
            f"Unexpected CLM-0.3 decision format {decision.get('format')!r}; "
            f"expected {EXPECTED_DECISION_FORMAT!r}"
        )
    if decision.get("formal_gpu_experiment_run") is not True:
        raise RuntimeError("Refusing publication: CLM-0.3 decision is not a formal full GPU matrix")

    for replicate in FORMAL_REPLICATES:
        for arm in FORMAL_ARMS:
            events_path = source / f"r{replicate}-{arm}" / "events.jsonl"
            complete = [
                event for event in _read_events(events_path)
                if event.get("type") == "worker_complete"
                and event.get("mode") != "preflight_only"
            ]
            if not complete:
                raise RuntimeError(f"r{replicate}-{arm} has no completed formal worker event")
            final = complete[-1]
            if int(final.get("consumed_tokens", 0)) < 1_500_000:
                raise RuntimeError(f"r{replicate}-{arm} did not reach 1.5M tokens")
    return decision


def prepare_clm_0_3_artifacts(
    root: Path,
    *,
    kaggle_script_version_id: str | None = None,
) -> Path:
    source = root / SOURCE_DIR
    destination = root / ARTIFACT_DIR
    if not source.is_dir():
        raise FileNotFoundError(f"CLM-0.3 results directory does not exist: {source}")
    decision = _validate_formal_results(source)

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for name in TOP_LEVEL_FILES:
        _copy(source / name, destination / name)

    for replicate in FORMAL_REPLICATES:
        for arm in FORMAL_ARMS:
            worker_name = f"r{replicate}-{arm}"
            worker_source = source / worker_name
            worker_destination = destination / worker_name
            for name in WORKER_COMMON_FILES:
                _copy(worker_source / name, worker_destination / name)
            if arm != "fixed4":
                for name in WORKER_GROWTH_FILES:
                    _copy(worker_source / name, worker_destination / name)
                for pressure_scan in sorted(worker_source.glob("pressure-scan-*.csv")):
                    _copy(pressure_scan, worker_destination / pressure_scan.name)

    # Copy any formal plots produced by the run, but never transient checkpoints.
    for plot in sorted(source.glob("*.png")):
        _copy(plot, destination / plot.name)

    source_commit = run_git(root, "rev-parse", "HEAD").stdout.strip()
    source_branch = run_git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    files = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            files.append({
                "path": path.relative_to(destination).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })

    metadata = {
        "format": "minicells.experiment-publication.v1",
        "experiment_id": "clm-0.3-progressive-growth",
        "experiment_format": decision.get("format"),
        "source_results_dir": SOURCE_DIR.as_posix(),
        "source_commit": source_commit,
        "source_branch": source_branch,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "kaggle": {
            "script_version_id": kaggle_script_version_id,
            "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "decision": decision,
        "files": files,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    results_md = f"""# CLM-0.3 Progressive Growth Results

This directory contains the curated formal evidence from the paired 3×3 Kaggle GPU matrix.
Transient caches and training/resume checkpoints are intentionally excluded.

## Formal decision

- Growth equivalence: `{decision.get('growth_equivalence', {}).get('status', 'unknown')}`
- Growth viability: `{decision.get('growth_viability', {}).get('status', 'unknown')}`
- Growth utility: `{decision.get('growth_utility', {}).get('status', 'unknown')}`
- Pressure selection: `{decision.get('pressure_selection', {}).get('status', 'unknown')}`
- Formal GPU experiment: `{decision.get('formal_gpu_experiment_run')}`

## Matrix

- Replicates: 3
- Arms: fixed4, pressure_growth, random_growth
- Continuation budget: 1,500,000 tokens per worker
- Formal control: paired fixed4 continuation within each replicate

## Provenance

- Source commit: `{source_commit}`
- Source branch: `{source_branch}`
- Source results directory: `{SOURCE_DIR.as_posix()}`
- Kaggle script version ID: `{kaggle_script_version_id or 'not recorded'}`

Machine-readable SHA-256 hashes and runtime provenance are in `metadata.json`.
The authoritative formal decision is `decision.json`.
"""
    (destination / "RESULTS.md").write_text(results_md, encoding="utf-8")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish CLM-0.3 progressive-growth Kaggle results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()

    # A previous manual `git add -f results/...` must not leak ignored runtime
    # outputs into the publication commit. Reset only this experiment's result
    # path in the index; working files remain untouched.
    run_git(root, "reset", "--", SOURCE_DIR.as_posix(), check=False)

    destination = prepare_clm_0_3_artifacts(
        root,
        kaggle_script_version_id=args.kaggle_script_version_id,
    )
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(destination)} ({path.stat().st_size} bytes)")

    if args.push:
        push_results(
            root,
            destination,
            "clm-0.3-progressive-growth",
            args.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push after reviewing the curated artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
