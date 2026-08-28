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


SOURCE_DIR = Path("results/clm-0.3b-marginal-growth-utility")
ARTIFACT_DIR = Path("artifacts/experiments/clm-0.3b-marginal-growth-utility")
DEFAULT_BRANCH = "kaggle/clm-0.3b-marginal-growth-utility-results"
EXPECTED_DECISION_FORMAT = "minicells.clm-0.3b-marginal-growth-utility.decision.v1"
FORMAL_ARMS = ("fixed4", "marginal_growth", "random_growth")
FORMAL_REPLICATES = (0, 1, 2)

TOP_LEVEL_FILES = (
    "decision.json",
    "formal-ppl-history.csv",
    "replicate-summary.json",
)
WORKER_COMMON_FILES = (
    "events.jsonl",
    "ppl-history.csv",
    "saturation.json",
)
WORKER_GROWTH_FILES = (
    "growth-history.json",
    "newborn-diagnostics.json",
    "marginal-scan-1.csv",
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


def _read_events(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _validate_formal_results(source: Path) -> tuple[dict[str, object], str, str]:
    missing = [source / name for name in TOP_LEVEL_FILES if not (source / name).is_file()]
    worker_commits: set[str] = set()
    worker_trees: set[str] = set()
    for replicate in FORMAL_REPLICATES:
        for arm in FORMAL_ARMS:
            worker = source / f"r{replicate}-{arm}"
            required = [worker / name for name in WORKER_COMMON_FILES]
            if arm != "fixed4":
                required.extend(worker / name for name in WORKER_GROWTH_FILES)
            missing.extend(path for path in required if not path.is_file())
    if missing:
        relative = [str(path.relative_to(source)) for path in missing]
        raise FileNotFoundError(f"Missing CLM-0.3b formal artifacts: {relative}")

    decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    if decision.get("format") != EXPECTED_DECISION_FORMAT:
        raise RuntimeError(
            f"Unexpected CLM-0.3b decision format {decision.get('format')!r}; "
            f"expected {EXPECTED_DECISION_FORMAT!r}"
        )
    if decision.get("formal_gpu_experiment_run") is not True:
        raise RuntimeError("Refusing publication: CLM-0.3b was not run with the formal matrix defaults")

    for replicate in FORMAL_REPLICATES:
        for arm in FORMAL_ARMS:
            events = _read_events(source / f"r{replicate}-{arm}" / "events.jsonl")
            complete = [
                event for event in events
                if event.get("type") == "worker_complete" and event.get("mode") != "preflight_only"
            ]
            if not complete:
                raise RuntimeError(f"r{replicate}-{arm} has no completed formal worker event")
            final = complete[-1]
            code_commit = final.get("code_commit")
            code_tree = final.get("code_tree_sha")
            if not code_commit or not code_tree:
                raise RuntimeError(f"r{replicate}-{arm} is missing immutable code provenance")
            worker_commits.add(str(code_commit))
            worker_trees.add(str(code_tree))
    if len(worker_commits) != 1 or len(worker_trees) != 1:
        raise RuntimeError(
            f"CLM-0.3b workers used mixed code provenance: commits={sorted(worker_commits)}, trees={sorted(worker_trees)}"
        )
    return decision, next(iter(worker_commits)), next(iter(worker_trees))


def prepare_clm_0_3b_artifacts(
    root: Path,
    *,
    kaggle_script_version_id: str | None = None,
) -> Path:
    source = root / SOURCE_DIR
    destination = root / ARTIFACT_DIR
    if not source.is_dir():
        raise FileNotFoundError(f"CLM-0.3b results directory does not exist: {source}")
    decision, worker_commit, worker_tree = _validate_formal_results(source)

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
    for plot in sorted(source.glob("*.png")):
        _copy(plot, destination / plot.name)

    publishing_commit = run_git(root, "rev-parse", "HEAD").stdout.strip()
    publishing_branch = run_git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
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
        "experiment_id": "clm-0.3b-marginal-growth-utility",
        "experiment_format": decision.get("format"),
        "source_results_dir": SOURCE_DIR.as_posix(),
        "training_code_commit": worker_commit,
        "training_code_tree_sha": worker_tree,
        "publishing_commit": publishing_commit,
        "publishing_branch": publishing_branch,
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

    results_md = f"""# CLM-0.3b Marginal Growth Utility Results

This directory contains the curated formal evidence from the paired 3×3 Kaggle GPU matrix.
Training/resume checkpoints and corpus caches are intentionally excluded.

## Formal decision

- Saturation regime: `{decision.get('saturation_regime', {}).get('status', 'unknown')}`
- Growth equivalence: `{decision.get('growth_equivalence', {}).get('status', 'unknown')}`
- Marginal growth viability: `{decision.get('growth_viability', {}).get('status', 'unknown')}`
- Marginal growth utility: `{decision.get('marginal_growth_utility', {}).get('status', 'unknown')}`
- Marginal selection: `{decision.get('marginal_selection', {}).get('status', 'unknown')}`
- Newborn causal utility: `{decision.get('causal_utility', {}).get('status', 'unknown')}`
- Formal GPU experiment: `{decision.get('formal_gpu_experiment_run')}`

## Matrix

- Replicates: 3
- Arms: fixed4, marginal_growth, random_growth
- Earliest saturation check: 1.5M tokens
- Latest allowed pre-birth boundary: 3.0M tokens
- Matched newborn age: 1.0M tokens
- Formal validation: 32 batches / 32K target tokens
- Auxiliary root balance weight: 0.0

## Immutable training provenance

- Training code commit: `{worker_commit}`
- Training code tree: `{worker_tree}`
- Publishing commit: `{publishing_commit}`
- Publishing branch: `{publishing_branch}`
- Kaggle script version ID: `{kaggle_script_version_id or 'not recorded'}`

Machine-readable SHA-256 hashes and provenance are in `metadata.json`.
The authoritative formal decision is `decision.json`.
"""
    (destination / "RESULTS.md").write_text(results_md, encoding="utf-8")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish CLM-0.3b Kaggle results.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    run_git(root, "reset", "--", SOURCE_DIR.as_posix(), check=False)
    destination = prepare_clm_0_3b_artifacts(
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
            "clm-0.3b-marginal-growth-utility",
            args.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push after reviewing the curated artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
