#!/usr/bin/env python3
"""Publish Core 008 postmortem diagnostics without changing Core 008."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from publish_core_validation_007 import _authenticated_git_env, _check_branch
from publish_experiment_results import DEFAULT_SECRET_NAME, EXPECTED_ORIGIN, run_git

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "core-008-postmortem-functional-capacity"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "core-008-postmortem-functional-capacity"
PROTOCOL = ROOT / "research" / "validations" / "core-008-postmortem-functional-capacity" / "protocol.json"
EXCLUDE = {"frozen-hidden.pt"}


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _ignore(_directory: str, names: list[str]) -> list[str]:
    return [name for name in names if name in EXCLUDE or name.endswith(".tmp")]


def preflight_push(branch: str, secret_name: str) -> None:
    _check_branch(branch)
    with _authenticated_git_env(secret_name) as env:
        result = run_git(ROOT, "push", "--dry-run", EXPECTED_ORIGIN + ".git", f"HEAD:refs/heads/{branch}", env=env, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"GitHub write preflight failed: {detail}")
    print(f"GitHub write preflight passed for {branch}")


def _copy_results() -> None:
    decision_path = RESULTS / "decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError(decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("scientific_decision") is not False:
        raise RuntimeError("Core 008 postmortem must remain non-confirmatory")
    if decision.get("source_core008_status_changed") is not False:
        raise RuntimeError("postmortem must not change Core 008 status")
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    shutil.copytree(RESULTS, ARTIFACTS, ignore=_ignore)
    shutil.copy2(PROTOCOL, ARTIFACTS / "protocol.json")


def _commit() -> str | None:
    run_git(ROOT, "config", "user.name", "MiniCells Kaggle")
    run_git(ROOT, "config", "user.email", "kaggle@minicells.local")
    run_git(ROOT, "add", "--", ARTIFACTS.relative_to(ROOT).as_posix())
    changed = run_git(ROOT, "diff", "--cached", "--quiet", check=False).returncode != 0
    if not changed:
        return None
    run_git(ROOT, "commit", "-m", "research: record Core 008 postmortem functional capacity")
    return _git_output(["rev-parse", "HEAD"])


def _push(branch: str, secret_name: str) -> None:
    _check_branch(branch)
    with _authenticated_git_env(secret_name) as env:
        result = run_git(ROOT, "push", EXPECTED_ORIGIN + ".git", f"HEAD:refs/heads/{branch}", env=env, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"authenticated push failed: {detail}")
    print(f"pushed Core 008 postmortem results to {branch}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--branch", default=None)
    p.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--push-results", action="store_true")
    args = p.parse_args()
    branch = args.branch or _git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    if args.preflight_only:
        preflight_push(branch, args.secret_name)
        return 0
    _copy_results()
    commit = _commit()
    print(f"commit={commit or 'no changes'}")
    if args.push_results:
        _push(branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
