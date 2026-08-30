#!/usr/bin/env python3
"""Curate and publish the CLM-0.3 public release benchmark evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

from publish_experiment_results import DEFAULT_SECRET_NAME, push_results, repo_root, run_git


SOURCE_DIR = Path("results/clm-0.3-release-benchmark")
ARTIFACT_DIR = Path("artifacts/experiments/clm-0.3-release-benchmark")
DEFAULT_BRANCH = "kaggle/clm-0.3-release-benchmark-results"
EXPECTED_DECISION_FORMAT = "minicells.clm-0.3-release-benchmark.v1"
EXPECTED_BRIDGE_FORMAT = "minicells.clm-0.3-release-bridge-summary.v1"
ARMS = ("textnca_continuation", "clm_fixed4")
TOP_LEVEL = (
    "decision.json",
    "bridge-summary.json",
    "historical-evidence.json",
    "capability-evidence.json",
    "PUBLIC-RELEASE-SUMMARY.md",
)
WORKER_FILES = (
    "run-provenance.json",
    "bridge-checkpoints.json",
    "worker-result.json",
)
FIGURES = (
    "figure-1-language-quality.png",
    "figure-1-language-quality.svg",
    "figure-2-machinery-bridge.png",
    "figure-2-machinery-bridge.svg",
    "figure-3-developmental-selectivity.png",
    "figure-3-developmental-selectivity.svg",
    "figure-4-reference-cost.png",
    "figure-4-reference-cost.svg",
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


def _validate(source: Path) -> tuple[dict[str, object], dict[str, object], str, str]:
    missing = [source / name for name in TOP_LEVEL if not (source / name).is_file()]
    missing.extend(source / "figures" / name for name in FIGURES if not (source / "figures" / name).is_file())
    for arm in ARMS:
        directory = source / "bridge" / arm
        missing.extend(directory / name for name in WORKER_FILES if not (directory / name).is_file())
    if missing:
        raise FileNotFoundError(
            "Missing CLM-0.3 release artifacts: "
            + ", ".join(str(path.relative_to(source)) for path in missing)
        )

    decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    bridge = json.loads((source / "bridge-summary.json").read_text(encoding="utf-8"))
    capability = json.loads((source / "capability-evidence.json").read_text(encoding="utf-8"))
    historical = json.loads((source / "historical-evidence.json").read_text(encoding="utf-8"))
    if decision.get("format") != EXPECTED_DECISION_FORMAT:
        raise RuntimeError(f"unexpected release decision format: {decision.get('format')!r}")
    if bridge.get("format") != EXPECTED_BRIDGE_FORMAT:
        raise RuntimeError(f"unexpected release bridge format: {bridge.get('format')!r}")
    if decision.get("formal_gpu_experiment_run") is not True or bridge.get("formal_gpu_experiment_run") is not True:
        raise RuntimeError("refusing publication: release bridge is not a formal GPU run")
    if historical.get("status") != "CLM_RELEASE_TEXTNCA_LANGUAGE_FOUNDATION_CONFIRMED":
        raise RuntimeError("historical TextNCA language foundation is not confirmed")
    if capability.get("status") != "CLM_RELEASE_DEVELOPMENTAL_CAPABILITY_CONFIRMED":
        raise RuntimeError("developmental capability evidence is not confirmed")
    if int(capability.get("births_checked", -1)) != 72 or int(capability.get("births_equivalent", -1)) != 72:
        raise RuntimeError("developmental capability parity matrix is incomplete")

    commits: set[str] = set()
    trees: set[str] = set()
    for arm in ARMS:
        directory = source / "bridge" / arm
        identity = json.loads((directory / "run-provenance.json").read_text(encoding="utf-8"))
        result = json.loads((directory / "worker-result.json").read_text(encoding="utf-8"))
        checkpoints = json.loads((directory / "bridge-checkpoints.json").read_text(encoding="utf-8"))
        if result.get("formal_gpu_experiment_run") is not True:
            raise RuntimeError(f"{arm} is not a formal GPU bridge worker")
        if result.get("code_commit") != identity.get("code_commit") or result.get("code_tree_sha") != identity.get("code_tree_sha"):
            raise RuntimeError(f"{arm} worker provenance does not match its immutable identity")
        if identity.get("tracked_tree_dirty") is not False:
            raise RuntimeError(f"{arm} ran from a dirty tracked tree")
        if [int(row["consumed_tokens"]) for row in checkpoints] != [0, 100000, 250000, 500000, 1000000]:
            raise RuntimeError(f"{arm} bridge checkpoints are incomplete")
        commits.add(str(result["code_commit"]))
        trees.add(str(result["code_tree_sha"]))
    if len(commits) != 1 or len(trees) != 1:
        raise RuntimeError(f"mixed release bridge provenance: commits={sorted(commits)}, trees={sorted(trees)}")
    training_commit = next(iter(commits))
    training_tree = next(iter(trees))
    if bridge.get("training_commit") != training_commit or bridge.get("training_tree_sha") != training_tree:
        raise RuntimeError("bridge summary provenance does not match worker provenance")
    if decision.get("bridge", {}).get("training_commit") != training_commit:
        raise RuntimeError("release decision training commit does not match bridge")
    if bridge.get("age_zero_equivalence", {}).get("status") != "CLM_RELEASE_BRIDGE_EQUIVALENCE":
        raise RuntimeError("release bridge age-zero equivalence failed")

    listed_figures = set(decision.get("figures", []))
    if listed_figures != set(FIGURES):
        raise RuntimeError("release decision figure manifest is incomplete")
    return decision, bridge, training_commit, training_tree


def prepare_artifacts(root: Path, *, kaggle_script_version_id: str | None = None) -> Path:
    source = root / SOURCE_DIR
    destination = root / ARTIFACT_DIR
    if not source.is_dir():
        raise FileNotFoundError(f"release benchmark results directory does not exist: {source}")
    decision, bridge, training_commit, training_tree = _validate(source)

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in TOP_LEVEL:
        _copy(source / name, destination / name)
    for name in FIGURES:
        _copy(source / "figures" / name, destination / "figures" / name)
    for arm in ARMS:
        for name in WORKER_FILES:
            _copy(source / "bridge" / arm / name, destination / "bridge" / arm / name)
        age_zero = source / "bridge" / arm / "age-zero-equivalence.json"
        if age_zero.exists():
            _copy(age_zero, destination / "bridge" / arm / age_zero.name)

    publishing_commit = run_git(root, "rev-parse", "HEAD").stdout.strip()
    publishing_branch = run_git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    files = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(destination).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    metadata = {
        "format": "minicells.experiment-publication.v1",
        "experiment_id": "clm-0.3-release-benchmark",
        "experiment_format": decision.get("format"),
        "source_results_dir": SOURCE_DIR.as_posix(),
        "training_code_commit": training_commit,
        "training_code_tree_sha": training_tree,
        "publishing_commit": publishing_commit,
        "publishing_branch": publishing_branch,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "kaggle": {
            "script_version_id": kaggle_script_version_id,
            "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "decision": decision,
        "bridge": {
            "source_checkpoint_sha256": bridge.get("source_checkpoint_sha256"),
            "budget_tokens_per_arm": bridge.get("budget_tokens_per_arm"),
        },
        "files": files,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    results = f"""# CLM-0.3 Public Release Benchmark Results

The primary human-readable artifact is [PUBLIC-RELEASE-SUMMARY.md](PUBLIC-RELEASE-SUMMARY.md).

## Release recommendation

`{decision.get('overall', {}).get('status', 'unknown')}`

## Core statuses

- Language quality: `{decision.get('language_quality', {}).get('status', 'unknown')}`
- Reference runtime: `{decision.get('reference_runtime', {}).get('status', 'unknown')}`
- Developmental capability: `{decision.get('developmental_capability', {}).get('status', 'unknown')}`
- Age-zero bridge: `{bridge.get('age_zero_equivalence', {}).get('status', 'unknown')}`

## Immutable bridge provenance

- Training commit: `{training_commit}`
- Training tree: `{training_tree}`
- Source checkpoint SHA-256: `{bridge.get('source_checkpoint_sha256')}`
- Publishing commit: `{publishing_commit}`
- Publishing branch: `{publishing_branch}`
- Kaggle script version ID: `{kaggle_script_version_id or 'not recorded'}`

Training/resume checkpoints (`*.pt`), corpus caches, and worker logs are deliberately excluded from publication.
"""
    (destination / "RESULTS.md").write_text(results, encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish CLM-0.3 release benchmark")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    args = parser.parse_args()
    root = repo_root()
    run_git(root, "reset", "--", SOURCE_DIR.as_posix(), check=False)
    destination = prepare_artifacts(root, kaggle_script_version_id=args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(destination)} ({path.stat().st_size} bytes)")
    if args.push:
        push_results(
            root,
            destination,
            "clm-0.3-release-benchmark",
            args.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push after reviewing the curated artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
