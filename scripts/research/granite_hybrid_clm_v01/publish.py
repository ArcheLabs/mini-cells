from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESEARCH_SCRIPTS = ROOT / "scripts" / "research"
LOCAL_ROOT = Path(__file__).resolve().parent
for path in (RESEARCH_SCRIPTS, LOCAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from publish_core_validation_007 import _authenticated_git_env, _check_branch
from publish_experiment_results import (
    DEFAULT_SECRET_NAME,
    EXPECTED_ORIGIN,
    run_git,
)
from validate_result import _load_json, validate_payload

RESULTS = ROOT / "results" / "granite-hybrid-clm-v0.1"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "granite-hybrid-clm-v0.1"
PROTOCOL = ROOT / "research" / "validations" / "granite-hybrid-clm-v0.1" / "protocol.json"
DEFAULT_BRANCH = "codex/granite-hybrid-clm-v0.1"


def _git_output(*args: str) -> str:
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
    print(f"[granite-hybrid-clm-v0.1] GitHub write preflight passed for {branch}")


def _validate() -> None:
    result = _load_json(RESULTS / "result.json")
    protocol = _load_json(PROTOCOL)
    errors = validate_payload(result, protocol, result_dir=RESULTS)
    if errors:
        raise RuntimeError("refusing to publish rejected milestone: " + "; ".join(errors))


def _copy_artifacts() -> None:
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    shutil.copytree(RESULTS, ARTIFACTS)
    shutil.copy2(PROTOCOL, ARTIFACTS / "protocol.json")


def _commit() -> str | None:
    run_git(ROOT, "config", "user.name", "MiniCells Kaggle")
    run_git(ROOT, "config", "user.email", "kaggle@minicells.local")
    run_git(ROOT, "add", "--", ARTIFACTS.relative_to(ROOT).as_posix())
    changed = run_git(ROOT, "diff", "--cached", "--quiet", check=False).returncode != 0
    if not changed:
        return None
    run_git(ROOT, "commit", "-m", "artifacts: record Granite Hybrid CLM v0.1 milestone")
    return _git_output("rev-parse", "HEAD")


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
    print(f"[granite-hybrid-clm-v0.1] pushed accepted artifacts to {branch}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish accepted Granite Hybrid CLM v0.1 artifacts")
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight_push(args.branch, args.secret_name)
        return 0
    _validate()
    _copy_artifacts()
    commit = _commit()
    print(commit or "no artifact changes")
    _push(args.branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
