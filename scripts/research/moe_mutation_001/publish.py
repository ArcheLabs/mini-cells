#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESEARCH_SCRIPTS = ROOT / "scripts" / "research"
if str(RESEARCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SCRIPTS))

from publish_core_validation_007 import _authenticated_git_env, _check_branch  # noqa: E402
from publish_experiment_results import DEFAULT_SECRET_NAME, EXPECTED_ORIGIN, run_git  # noqa: E402

from aggregate import aggregate  # noqa: E402

RESULTS = ROOT / "results" / "moe-mutation-001"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "moe-mutation-001"
VALIDATION = ROOT / "research" / "validations" / "moe-mutation-001"
DEFAULT_BRANCH = "codex/moe-mutation-001-kaggle"


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def preflight_push(branch: str, secret_name: str) -> None:
    _check_branch(branch)
    with _authenticated_git_env(secret_name) as env:
        result = run_git(
            ROOT,
            "push",
            "--dry-run",
            EXPECTED_ORIGIN + ".git",
            f"HEAD:refs/heads/{branch}",
            env=env,
            check=False,
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"GitHub write preflight failed: {detail}")
    print(f"GitHub write preflight passed for {branch}")


def _copy_seed(seed: int) -> dict:
    source = RESULTS / f"seed-{seed}"
    result_path = source / "result.json"
    training_path = source / "training.jsonl"
    mutation_dir = source / "mutation"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    if not training_path.is_file():
        raise FileNotFoundError(training_path)
    if not (mutation_dir / "mutation.json").is_file():
        raise FileNotFoundError(mutation_dir / "mutation.json")
    if not (mutation_dir / "mutation.safetensors").is_file():
        raise FileNotFoundError(mutation_dir / "mutation.safetensors")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if int(result.get("seed", -1)) != seed:
        raise RuntimeError("seed result identity mismatch")
    if result.get("experiment") != "MOE_MUTATION_001":
        raise RuntimeError("refusing to publish non-MoE-Mutation-001 result")

    destination = ARTIFACTS / f"seed-{seed}"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copy2(result_path, destination / "result.json")
    shutil.copy2(training_path, destination / "training.jsonl")
    shutil.copytree(mutation_dir, destination / "mutation")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(VALIDATION / "protocol.json", ARTIFACTS / "protocol.json")
    shutil.copy2(VALIDATION / "PROTOCOL.md", ARTIFACTS / "PROTOCOL.md")
    aggregate()
    return result


def _commit(seed: int, result: dict) -> str | None:
    run_git(ROOT, "config", "user.name", "MiniCells Kaggle")
    run_git(ROOT, "config", "user.email", "kaggle@minicells.local")
    run_git(ROOT, "add", "--", ARTIFACTS.relative_to(ROOT).as_posix())
    changed = run_git(ROOT, "diff", "--cached", "--quiet", check=False).returncode != 0
    if not changed:
        return None
    status = result.get("status", "UNKNOWN")
    run_git(ROOT, "commit", "-m", f"research: record MoE Mutation 001 seed {seed} ({status})")
    return _git_output(["rev-parse", "HEAD"])


def _push(branch: str, secret_name: str) -> None:
    _check_branch(branch)
    with _authenticated_git_env(secret_name) as env:
        result = run_git(
            ROOT,
            "push",
            EXPECTED_ORIGIN + ".git",
            f"HEAD:refs/heads/{branch}",
            env=env,
            check=False,
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"authenticated push failed: {detail}")
    print(f"pushed MoE Mutation 001 artifacts to {branch}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish one MoE Mutation 001 seed")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if args.preflight_only:
        preflight_push(args.branch, args.secret_name)
        return 0
    if args.seed is None:
        parser.error("--seed is required unless --preflight-only is used")

    result = _copy_seed(args.seed)
    commit = _commit(args.seed, result)
    print(f"commit={commit or 'no changes'}")
    _push(args.branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
