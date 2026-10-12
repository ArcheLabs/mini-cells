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


SOURCE_DIR = Path("results/clm-0.3c-counterfactual-mitosis")
ARTIFACT_DIR = Path("artifacts/experiments/clm-0.3c-counterfactual-mitosis")
DEFAULT_BRANCH = "kaggle/clm-0.3c-counterfactual-mitosis-results"
EXPECTED_DECISION_FORMAT = "minicells.clm-0.3c-counterfactual-mitosis.decision.v1"

TOP_LEVEL = ("decision.json", "replicate-summary.json")
WORKER_FILES = (
    "run-provenance.json",
    "events.jsonl",
    "trunk-history.json",
    "split-regret.csv",
    "probe-control.json",
    "probe-results.json",
    "growth-equivalence.json",
    "policy-decision.json",
    "confirm-control.json",
    "confirm-candidate.json",
    "replicate-result.json",
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


def _validate(source: Path) -> tuple[dict[str, object], str, str]:
    missing = [source / name for name in TOP_LEVEL if not (source / name).is_file()]
    commits: set[str] = set()
    trees: set[str] = set()
    for replicate in range(3):
        worker = source / f"r{replicate}-counterfactual"
        missing.extend(worker / name for name in WORKER_FILES if not (worker / name).is_file())
        probes = worker / "probes"
        if not probes.is_dir() or len(list(probes.glob("*.json"))) != 12:
            missing.append(probes)
    if missing:
        relative = [str(path.relative_to(source)) for path in missing]
        raise FileNotFoundError(f"Missing CLM-0.3c formal artifacts: {relative}")

    decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    if decision.get("format") != EXPECTED_DECISION_FORMAT:
        raise RuntimeError(f"unexpected CLM-0.3c decision format: {decision.get('format')!r}")
    if decision.get("formal_gpu_experiment_run") is not True:
        raise RuntimeError("refusing publication: CLM-0.3c was not aggregated as a formal GPU run")

    for replicate in range(3):
        worker = source / f"r{replicate}-counterfactual"
        result = json.loads((worker / "replicate-result.json").read_text(encoding="utf-8"))
        identity = json.loads((worker / "run-provenance.json").read_text(encoding="utf-8"))
        if int(result.get("births_checked", -1)) != 13:
            raise RuntimeError(f"r{replicate} counterfactual birth evidence is incomplete")
        commit = result.get("code_commit")
        tree = result.get("code_tree_sha")
        if not commit or not tree:
            raise RuntimeError(f"r{replicate} missing immutable code provenance")
        if identity.get("code_commit") != commit or identity.get("code_tree_sha") != tree:
            raise RuntimeError(f"r{replicate} run identity does not match final result provenance")

        probe_hash = identity.get("probe_validation_schedule_sha256")
        confirm_hash = identity.get("confirm_validation_schedule_sha256")
        if not probe_hash or not confirm_hash or probe_hash == confirm_hash:
            raise RuntimeError(f"r{replicate} validation holdout provenance is invalid")
        for path in sorted((worker / "probes").glob("*.json")):
            probe = json.loads(path.read_text(encoding="utf-8"))
            if probe.get("code_commit") != commit:
                raise RuntimeError(f"r{replicate} stale probe commit in {path.name}")
            if probe.get("probe_validation_schedule_sha256") != probe_hash:
                raise RuntimeError(f"r{replicate} probe holdout mismatch in {path.name}")
        probe_control = json.loads((worker / "probe-control.json").read_text(encoding="utf-8"))
        confirm_control = json.loads((worker / "confirm-control.json").read_text(encoding="utf-8"))
        confirm_candidate = json.loads((worker / "confirm-candidate.json").read_text(encoding="utf-8"))
        if probe_control.get("code_commit") != commit or probe_control.get("validation_schedule_sha256") != probe_hash:
            raise RuntimeError(f"r{replicate} probe control provenance mismatch")
        if confirm_control.get("code_commit") != commit or confirm_control.get("validation_schedule_sha256") != confirm_hash:
            raise RuntimeError(f"r{replicate} confirmation control provenance mismatch")
        if confirm_candidate.get("code_commit") != commit or confirm_candidate.get("confirm_validation_schedule_sha256") != confirm_hash:
            raise RuntimeError(f"r{replicate} confirmation candidate provenance mismatch")

        commits.add(str(commit))
        trees.add(str(tree))
    if len(commits) != 1 or len(trees) != 1:
        raise RuntimeError(f"mixed CLM-0.3c provenance: commits={sorted(commits)}, trees={sorted(trees)}")
    return decision, next(iter(commits)), next(iter(trees))


def prepare_artifacts(root: Path, *, kaggle_script_version_id: str | None = None) -> Path:
    source = root / SOURCE_DIR
    destination = root / ARTIFACT_DIR
    if not source.is_dir():
        raise FileNotFoundError(f"CLM-0.3c results directory does not exist: {source}")
    decision, training_commit, training_tree = _validate(source)

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for name in TOP_LEVEL:
        _copy(source / name, destination / name)
    for replicate in range(3):
        worker_name = f"r{replicate}-counterfactual"
        worker_source = source / worker_name
        worker_destination = destination / worker_name
        for name in WORKER_FILES:
            _copy(worker_source / name, worker_destination / name)
        for path in sorted((worker_source / "probes").glob("*.json")):
            _copy(path, worker_destination / "probes" / path.name)

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
        "experiment_id": "clm-0.3c-counterfactual-mitosis",
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
        "files": files,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results_md = f"""# CLM-0.3c Counterfactual Mitosis Results

This directory contains the curated evidence from the formal 3-replicate counterfactual experiment. Training checkpoints and corpus caches are excluded.

## Formal decision

- Growth equivalence: `{decision.get('growth_equivalence', {}).get('status', 'unknown')}`
- Split-regret prediction: `{decision.get('split_regret_prediction', {}).get('status', 'unknown')}`
- Counterfactual decision: `{decision.get('counterfactual_decision', {}).get('status', 'unknown')}`
- Capacity value: `{decision.get('capacity_value', {}).get('status', 'unknown')}`
- Practical growth: `{decision.get('practical_growth', {}).get('status', 'unknown')}`
- Formal GPU experiment: `{decision.get('formal_gpu_experiment_run')}`

## Frozen formal parameters

- Decision checkpoint: 1.5M continuation tokens
- Shadow probe horizon: 100K tokens
- Candidates: all 12 CLM-0.1 root lineages
- Confirmation horizon: 500K tokens
- Probe holdout: 32 batches / 32K target tokens
- Confirmation holdout: separate 32 batches / 32K target tokens
- Bootstrap: 2,000 paired resamples
- Practical PPL threshold: 0.995

## Immutable training provenance

- Training commit: `{training_commit}`
- Training tree: `{training_tree}`
- Publishing commit: `{publishing_commit}`
- Publishing branch: `{publishing_branch}`
- Kaggle script version ID: `{kaggle_script_version_id or 'not recorded'}`

Machine-readable hashes are in `metadata.json`. The authoritative formal decision is `decision.json`.
"""
    (destination / "RESULTS.md").write_text(results_md, encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish CLM-0.3c Kaggle results")
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
            "clm-0.3c-counterfactual-mitosis",
            args.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push after reviewing the curated artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
