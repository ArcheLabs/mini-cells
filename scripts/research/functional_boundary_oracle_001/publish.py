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

from aggregate import aggregate
from publish_core_validation_007 import _authenticated_git_env, _check_branch
from publish_experiment_results import DEFAULT_SECRET_NAME, EXPECTED_ORIGIN, run_git

RESULTS = ROOT / "results" / "functional-boundary-oracle-001"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "functional-boundary-oracle-001"
VALIDATION = ROOT / "research" / "validations" / "functional-boundary-oracle-001"
DEFAULT_BRANCH = "codex/functional-boundary-oracle-001"


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
    required = (
        source / "result.json",
        source / "training.jsonl",
        source / "mutation" / "mutation.json",
        source / "mutation" / "mutation.safetensors",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing formal artifacts: {missing}")
    result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    if int(result.get("seed", -1)) != seed:
        raise RuntimeError("seed result identity mismatch")
    if result.get("experiment") != "FUNCTIONAL_BOUNDARY_ORACLE_001":
        raise RuntimeError("refusing to publish another experiment")

    destination = ARTIFACTS / f"seed-{seed}"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copy2(source / "result.json", destination / "result.json")
    shutil.copy2(source / "training.jsonl", destination / "training.jsonl")
    shutil.copytree(source / "mutation", destination / "mutation")
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
    message = f"research: record Functional Boundary Oracle seed {seed} ({status})"
    run_git(ROOT, "commit", "-m", message)
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
    print(f"pushed Functional Boundary Oracle artifacts to {branch}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish Functional Boundary Oracle seed")
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
