#!/usr/bin/env python3
"""Append-only same-branch publication for Shadow v2 formal evidence."""
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

from publish_experiment_results import DEFAULT_SECRET_NAME, EXPECTED_ORIGIN, load_github_token, repo_root

VALIDATION_ID = "shadow-cell-validation-001-v2-developmental-maturation"
FORMAL_SEEDS = (95311, 95312, 95313)
DEFAULT_SOURCE = Path("results") / VALIDATION_ID
DEFAULT_ARTIFACT = Path("artifacts/experiments") / VALIDATION_ID
DEFAULT_BRANCH = "codex/shadow-cell-validation-001-v2-amendment"
REGISTERED_BRANCH = DEFAULT_BRANCH
AGGREGATOR = Path("scripts/research/aggregate_shadow_cell_validation_001_v2.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)


def current_branch(root: Path) -> str:
    value = git(root, "branch", "--show-current").stdout.strip()
    if not value:
        raise RuntimeError("canonical Shadow publication refuses detached HEAD")
    return value


def validate_branch(root: Path, branch: str) -> None:
    actual = current_branch(root)
    if actual != REGISTERED_BRANCH or branch != REGISTERED_BRANCH:
        raise RuntimeError(f"Shadow formal publication requires branch {REGISTERED_BRANCH!r}; current={actual!r}, target={branch!r}")


def validate_origin(root: Path) -> None:
    origin = git(root, "remote", "get-url", "origin").stdout.strip()
    normalized = origin[:-4] if origin.endswith(".git") else origin
    normalized = normalized.rstrip("/")
    if normalized not in {EXPECTED_ORIGIN, "git@github.com:ArcheLabs/mini-cells", "ssh://git@github.com/ArcheLabs/mini-cells"}:
        raise RuntimeError(f"unexpected origin: {origin!r}")


def _askpass(token: str) -> tuple[Path, dict[str, str]]:
    handle = tempfile.NamedTemporaryFile(mode="w", prefix="minicells-askpass-", suffix=".sh", delete=False, encoding="utf-8")
    handle.write('#!/bin/sh\ncase "$1" in\n  *Username*) printf "%s\\n" "x-access-token" ;;\n  *) printf "%s\\n" "$GITHUB_TOKEN" ;;\nesac\n')
    handle.close()
    path = Path(handle.name)
    path.chmod(0o700)
    env = os.environ.copy()
    env["GITHUB_TOKEN"] = token
    env["GIT_ASKPASS"] = str(path)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return path, env


def github_write_preflight(root: Path, branch: str, secret_name: str) -> None:
    validate_origin(root)
    validate_branch(root, branch)
    token = load_github_token(secret_name)
    askpass, env = _askpass(token)
    try:
        result = git(root, "push", "--dry-run", EXPECTED_ORIGIN + ".git", f"HEAD:refs/heads/{branch}", check=False, env=env)
    finally:
        askpass.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"GitHub write preflight failed: {result.stderr.strip() or result.stdout.strip()}")
    print(f"GitHub write preflight passed for {branch}")


def _run_aggregate(root: Path, source: Path) -> dict:
    subprocess.run([sys.executable, str(root / AGGREGATOR), "--results-root", str(source)], cwd=root, check=True)
    return json.loads((source / "aggregate.json").read_text(encoding="utf-8"))


def _copy_one_append_only(source: Path, destination: Path) -> bool:
    if destination.exists():
        if not source.is_file() or sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"published formal evidence is immutable and differs: {destination}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _seed_files(source_seed: Path, destination_seed: Path) -> list[str]:
    changed: list[str] = []
    allowed = {".csv", ".json", ".md", ".png", ".txt", ".log"}
    for path in sorted(source_seed.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        target = destination_seed / path.relative_to(source_seed)
        if _copy_one_append_only(path, target):
            changed.append(target.as_posix())
    for path in sorted(source_seed.rglob("*.pt")):
        record = destination_seed / "provenance.json"
        payload = json.loads(record.read_text(encoding="utf-8")) if record.is_file() else {}
        payload.setdefault("checkpoint_files", []).append({"path": path.relative_to(source_seed).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path), "published_to_git": False})
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if "provenance.json" not in changed:
            changed.append(record.as_posix())
    return changed


def prepare_publication(root: Path, source: Path, destination: Path, *, kaggle_script_version_id: str | None = None) -> Path:
    source = source if source.is_absolute() else root / source
    destination = destination if destination.is_absolute() else root / destination
    if not source.is_dir():
        raise FileNotFoundError(source)
    protocol_path = root / "research/validations" / VALIDATION_ID / "protocol.json"
    lock_path = protocol_path.with_name("protocol-lock.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("validation_id") != VALIDATION_ID:
        raise RuntimeError("unexpected validation id")
    results = []
    for seed in FORMAL_SEEDS:
        result_path = source / f"seed-{seed}" / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("phase") != "formal" or int(result.get("seed", -1)) != seed:
                raise RuntimeError(f"invalid formal result identity: {result_path}")
            results.append(result)
    aggregate = _run_aggregate(root, source)
    destination.mkdir(parents=True, exist_ok=True)
    _copy_one_append_only(protocol_path, destination / "protocol.json")
    _copy_one_append_only(lock_path, destination / "protocol-lock.json")
    manifest = destination / "implementation-manifest.json"
    manifest_payload = {"format": "minicells.shadow-cell-validation-001-v2.implementation-manifest.v1", "files": json.loads(lock_path.read_text(encoding="utf-8")).get("implementation_files", {}), "sha256": json.loads(lock_path.read_text(encoding="utf-8")).get("implementation_manifest_sha256")}
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != manifest_payload:
        raise RuntimeError("published implementation manifest is immutable and differs")
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    changed = []
    for result in results:
        changed.extend(_seed_files(source / f"seed-{int(result['seed'])}", destination / "formal/seeds" / f"seed-{int(result['seed'])}"))
    for failure in sorted(source.glob("seed-*/failure.json")):
        seed_dir = destination / "formal/seeds" / failure.parent.name
        _copy_one_append_only(failure, seed_dir / "failure.json")
    decision = destination / "formal/decision.json"
    decision.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completed = sorted(int(result["seed"]) for result in results)
    metadata = {"format": "minicells.shadow-cell-validation-001-v2.publication.v2", "validation_id": VALIDATION_ID, "status": aggregate.get("status"), "scientific_decision": False if len(completed) < 3 else bool(aggregate.get("scientific_decision", False)), "formal_seeds": list(FORMAL_SEEDS), "completed_formal_seeds": completed, "source_branch": current_branch(root), "published_at_utc": datetime.now(timezone.utc).isoformat(), "runtime": {"python": platform.python_version(), "platform": platform.platform()}, "kaggle": {"script_version_id": kaggle_script_version_id}}
    (destination / "formal/manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "formal/RESULTS.md").write_text(f"# Shadow Cell Validation 001 v2 Results\n\nStatus: `{aggregate.get('status')}`\n\nCompleted formal seeds: `{completed}` / `{list(FORMAL_SEEDS)}`\n", encoding="utf-8")
    return destination


def push_results(root: Path, destination: Path, *, branch: str = DEFAULT_BRANCH, secret_name: str = DEFAULT_SECRET_NAME) -> None:
    validate_origin(root)
    validate_branch(root, branch)
    # Observe the remote before creating the publication commit; never force-push.
    git(root, "fetch", "origin", branch)
    remote = git(root, "rev-parse", f"refs/remotes/origin/{branch}", check=False).stdout.strip()
    head = git(root, "rev-parse", "HEAD").stdout.strip()
    if remote and remote != head:
        ancestor = git(root, "merge-base", "--is-ancestor", remote, "HEAD", check=False)
        if ancestor.returncode != 0:
            raise RuntimeError("remote publication branch advanced unexpectedly; refusing to overwrite evidence")
    relative = destination.relative_to(root).as_posix()
    git(root, "add", "--", relative)
    if git(root, "diff", "--cached", "--quiet", check=False).returncode != 0:
        completed = len(list((destination / "formal/seeds").glob("seed-*/result.json")))
        message = f"research: {'record' if completed == 3 else 'checkpoint'} Shadow Cell Validation 001 v2 {'formal decision' if completed == 3 else f'({completed}/3)'}"
        git(root, "config", "user.name", "MiniCells Kaggle")
        git(root, "config", "user.email", "kaggle@minicells.local")
        git(root, "commit", "-m", message)
    token = load_github_token(secret_name)
    askpass, env = _askpass(token)
    try:
        result = git(root, "push", EXPECTED_ORIGIN + ".git", f"HEAD:refs/heads/{branch}", env=env, check=False)
    finally:
        askpass.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    print(f"Pushed Shadow v2 evidence to {branch}")


def publish_results(root: Path, source: Path, *, branch: str = DEFAULT_BRANCH, secret_name: str = DEFAULT_SECRET_NAME, kaggle_script_version_id: str | None = None) -> Path:
    validate_branch(root, branch)
    destination = root / DEFAULT_ARTIFACT
    prepared = prepare_publication(root, source, destination, kaggle_script_version_id=kaggle_script_version_id)
    push_results(root, prepared, branch=branch, secret_name=secret_name)
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--kaggle-script-version-id")
    args = parser.parse_args()
    root = repo_root()
    if args.preflight_only:
        github_write_preflight(root, args.branch, args.secret_name)
        return 0
    validate_branch(root, args.branch)
    destination = prepare_publication(root, args.source, root / DEFAULT_ARTIFACT, kaggle_script_version_id=args.kaggle_script_version_id)
    print(f"Prepared curated Shadow v2 artifacts: {destination.relative_to(root)}")
    if args.push:
        push_results(root, destination, branch=args.branch, secret_name=args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
