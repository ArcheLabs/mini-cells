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


SOURCE_DIR = Path("results/clm-0.3d-probationary-mitosis")
ARTIFACT_DIR = Path("artifacts/experiments/clm-0.3d-probationary-mitosis")
DEFAULT_BRANCH = "kaggle/clm-0.3d-probationary-mitosis-results"
EXPECTED_DECISION_FORMAT = "minicells.clm-0.3d-probationary-mitosis.decision.v1"
CONDITIONS = ("stationary_story", "story_arithmetic_shift")

TOP_LEVEL = ("decision.json", "replicate-summary.json")
WORKER_FILES = (
    "run-provenance.json",
    "events.jsonl",
    "trunk-history.json",
    "geometry-calibration.json",
    "replicate-result.json",
)
CONDITION_FILES = (
    "baseline-evaluation.json",
    "control-trajectory.json",
    "initial-shadow-results.json",
    "shortlist.json",
    "probation-trajectories.json",
    "probation-decisions.json",
    "promotion-decision.json",
    "growth-equivalence.json",
    "final-control.json",
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
        worker = source / f"r{replicate}-probationary"
        missing.extend(worker / name for name in WORKER_FILES if not (worker / name).is_file())
        for condition in CONDITIONS:
            cdir = worker / condition
            missing.extend(cdir / name for name in CONDITION_FILES if not (cdir / name).is_file())
            if condition == "story_arithmetic_shift" and not (cdir / "absorption-diagnostic.json").is_file():
                missing.append(cdir / "absorption-diagnostic.json")
            shadows = cdir / "shadows"
            if not shadows.is_dir() or len(list(shadows.glob("*-initial.json"))) != 12:
                missing.append(shadows)
    if missing:
        relative = [str(path.relative_to(source)) for path in missing]
        raise FileNotFoundError(f"Missing CLM-0.3d formal artifacts: {relative}")

    decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    if decision.get("format") != EXPECTED_DECISION_FORMAT:
        raise RuntimeError(f"unexpected CLM-0.3d decision format: {decision.get('format')!r}")
    if decision.get("formal_gpu_experiment_run") is not True:
        raise RuntimeError("refusing publication: CLM-0.3d was not aggregated as a formal GPU run")

    for replicate in range(3):
        worker = source / f"r{replicate}-probationary"
        result = json.loads((worker / "replicate-result.json").read_text(encoding="utf-8"))
        identity = json.loads((worker / "run-provenance.json").read_text(encoding="utf-8"))
        if int(result.get("births_checked", -1)) != 24:
            raise RuntimeError(f"r{replicate} probationary birth evidence is incomplete")
        if int(result.get("births_equivalent", -1)) != 24:
            raise RuntimeError(f"r{replicate} contains a non-equivalent shadow birth")
        commit = result.get("code_commit")
        tree = result.get("code_tree_sha")
        if not commit or not tree:
            raise RuntimeError(f"r{replicate} missing immutable code provenance")
        if identity.get("code_commit") != commit or identity.get("code_tree_sha") != tree:
            raise RuntimeError(f"r{replicate} run identity does not match final provenance")
        if identity.get("tracked_tree_dirty") is not False:
            raise RuntimeError(f"r{replicate} formal worker ran from a dirty tracked tree")

        for condition in CONDITIONS:
            cdir = worker / condition
            hashes = identity.get("conditions", {}).get(condition, {})
            if not hashes.get("future_schedule_sha256"):
                raise RuntimeError(f"r{replicate} {condition} future schedule hash missing")
            holdout_a = hashes.get("holdout_a_sha256")
            holdout_b = hashes.get("holdout_b_sha256")
            if not holdout_a or not holdout_b or holdout_a == holdout_b:
                raise RuntimeError(f"r{replicate} {condition} holdout provenance is invalid")
            parity = json.loads((cdir / "growth-equivalence.json").read_text(encoding="utf-8"))
            initial = json.loads((cdir / "initial-shadow-results.json").read_text(encoding="utf-8"))
            shortlist = json.loads((cdir / "shortlist.json").read_text(encoding="utf-8"))
            promotion = json.loads((cdir / "promotion-decision.json").read_text(encoding="utf-8"))
            if len(parity) != 12 or sum(bool(row.get("equivalent")) for row in parity) != 12:
                raise RuntimeError(f"r{replicate} {condition} parity matrix is incomplete")
            if len(initial) != 12:
                raise RuntimeError(f"r{replicate} {condition} does not contain all 12 shadows")
            if len(shortlist.get("experts", [])) != 4:
                raise RuntimeError(f"r{replicate} {condition} shortlist is not K=4")
            if promotion.get("code_commit") != commit:
                raise RuntimeError(f"r{replicate} {condition} promotion evidence is stale")
            for path in sorted((cdir / "shadows").glob("*-initial.json")):
                shadow = json.loads(path.read_text(encoding="utf-8"))
                if shadow.get("code_commit") != commit:
                    raise RuntimeError(f"r{replicate} stale shadow evidence in {path.name}")

        commits.add(str(commit))
        trees.add(str(tree))

    if len(commits) != 1 or len(trees) != 1:
        raise RuntimeError(f"mixed CLM-0.3d provenance: commits={sorted(commits)}, trees={sorted(trees)}")
    return decision, next(iter(commits)), next(iter(trees))


def prepare_artifacts(root: Path, *, kaggle_script_version_id: str | None = None) -> Path:
    source = root / SOURCE_DIR
    destination = root / ARTIFACT_DIR
    if not source.is_dir():
        raise FileNotFoundError(f"CLM-0.3d results directory does not exist: {source}")
    decision, training_commit, training_tree = _validate(source)

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for name in TOP_LEVEL:
        _copy(source / name, destination / name)
    for replicate in range(3):
        worker_name = f"r{replicate}-probationary"
        worker_source = source / worker_name
        worker_destination = destination / worker_name
        for name in WORKER_FILES:
            _copy(worker_source / name, worker_destination / name)
        for condition in CONDITIONS:
            csource = worker_source / condition
            cdestination = worker_destination / condition
            for name in CONDITION_FILES:
                _copy(csource / name, cdestination / name)
            absorption = csource / "absorption-diagnostic.json"
            if absorption.exists():
                _copy(absorption, cdestination / absorption.name)
            final_candidate = csource / "final-candidate.json"
            if final_candidate.exists():
                _copy(final_candidate, cdestination / final_candidate.name)
            for path in sorted((csource / "shadows").glob("*.json")):
                _copy(path, cdestination / "shadows" / path.name)

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
        "experiment_id": "clm-0.3d-probationary-mitosis",
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
    results_md = f"""# CLM-0.3d Probationary Mitosis Results

This directory contains curated evidence from the formal three-replicate, two-environment probationary-mitosis experiment. Training checkpoints and corpus caches are excluded.

## Formal decision

- Overall: `{decision.get('overall', {}).get('status', 'unknown')}`
- Growth equivalence: `{decision.get('growth_equivalence', {}).get('status', 'unknown')}`
- Stationary specificity: `{decision.get('stationary_specificity', {}).get('status', 'unknown')}`
- Shift sensitivity: `{decision.get('shift_sensitivity', {}).get('status', 'unknown')}`
- Maturation: `{decision.get('maturation', {}).get('status', 'unknown')}`
- Formal GPU experiment: `{decision.get('formal_gpu_experiment_run')}`

## Frozen formal parameters

- Decision checkpoint: 1.5M TinyStories continuation tokens
- Conditions: stationary TinyStories; 50/50 Story + Arithmetic capability shift
- Shadow ages: 50K / 100K / 200K / 300K / 500K
- Initial shadows: all 12 CLM-0.1 root lineages per condition
- Shortlist: top 4 by realized 100K point utility
- Promotion gate: positive 300K and 500K LCB95, positive mean 200K/300K/500K utility, final PPL ratio <= 0.995
- Independent holdout B: required for persistent promotion
- Shift Story-retention ratio: <= 1.01
- Bootstrap: 2,000 paired resamples

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
    parser = argparse.ArgumentParser(description="Publish CLM-0.3d Kaggle results")
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
            "clm-0.3d-probationary-mitosis",
            args.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push after reviewing the curated artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
