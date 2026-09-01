#!/usr/bin/env python3
"""Publish Core Validation 007 artifacts with the repository's proven Kaggle auth path."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from publish_experiment_results import (
    DEFAULT_SECRET_NAME,
    EXPECTED_ORIGIN,
    load_github_token,
    run_git,
    validate_origin,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "core-validation-007-functional-boundary-discovery"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "core-validation-007-functional-boundary-discovery"
VALIDATION = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery"
PROTOCOL = VALIDATION / "protocol.json"
CONFIRMATION_AMENDMENT = VALIDATION / "confirmation-protocol-v1.1.json"
WINNER_LOCK = VALIDATION / "winner-lock.json"
EXCLUDE_NAMES = {"frozen-hidden.pt", "frozen-hidden-smoke.pt"}


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _ignore(_directory: str, names: list[str]) -> list[str]:
    return [name for name in names if name in EXCLUDE_NAMES or name.endswith(".tmp")]


def _copy_phase(phase: str, *, allow_partial: bool) -> tuple[Path, dict]:
    src = RESULTS / phase
    decision_path = src / "decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError(decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if phase == "discovery":
        if decision.get("scientific_decision") is not False:
            raise RuntimeError("discovery artifacts must remain non-scientific")
    else:
        scientific = decision.get("scientific_decision") is True
        partial = decision.get("status") == "CONFIRMATION_INCOMPLETE"
        if not scientific and not (allow_partial and partial):
            raise RuntimeError(
                "confirmation publication requires a final scientific decision or "
                "--allow-partial with CONFIRMATION_INCOMPLETE"
            )

    dest = ARTIFACTS / phase
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=_ignore)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROTOCOL, ARTIFACTS / "protocol.json")
    if WINNER_LOCK.is_file():
        shutil.copy2(WINNER_LOCK, ARTIFACTS / "winner-lock.json")
    if CONFIRMATION_AMENDMENT.is_file():
        shutil.copy2(CONFIRMATION_AMENDMENT, ARTIFACTS / "confirmation-protocol-v1.1.json")
    return dest, decision


def _commit(phase: str, decision: dict) -> str | None:
    run_git(ROOT, "config", "user.name", "MiniCells Kaggle")
    run_git(ROOT, "config", "user.email", "kaggle@minicells.local")
    run_git(ROOT, "add", "--", ARTIFACTS.relative_to(ROOT).as_posix())
    changed = run_git(ROOT, "diff", "--cached", "--quiet", check=False).returncode != 0
    if not changed:
        return None
    if phase == "discovery":
        message = "research: lock Core Validation 007 discovery winner"
    elif decision.get("scientific_decision") is True:
        message = "research: record Core Validation 007 final confirmation results"
    else:
        completed = len(decision.get("completed_seeds", []))
        message = f"research: checkpoint Core Validation 007 confirmation ({completed}/3)"
    run_git(ROOT, "commit", "-m", message)
    return _git_output(["rev-parse", "HEAD"])


@contextmanager
def _authenticated_git_env(secret_name: str) -> Iterator[dict[str, str]]:
    """Use the same GIT_ASKPASS flow as the historical unified publisher."""
    token = load_github_token(secret_name)
    askpass_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="minicells-git-askpass-",
            suffix=".sh",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(
                "#!/bin/sh\n"
                "case \"$1\" in\n"
                "  *Username*) printf \"%s\\n\" \"x-access-token\" ;;\n"
                "  *) printf \"%s\\n\" \"$GITHUB_TOKEN\" ;;\n"
                "esac\n"
            )
            askpass_path = Path(handle.name)
        askpass_path.chmod(0o700)
        env = os.environ.copy()
        env["GITHUB_TOKEN"] = token
        env["GIT_ASKPASS"] = str(askpass_path)
        env["GIT_TERMINAL_PROMPT"] = "0"
        yield env
    finally:
        if askpass_path is not None:
            askpass_path.unlink(missing_ok=True)
        token = ""


def _check_branch(branch: str) -> None:
    current = _git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    if current != branch:
        raise RuntimeError(f"expected branch {branch!r}, current branch is {current!r}")
    validate_origin(ROOT)
    fetch = run_git(
        ROOT,
        "fetch",
        "origin",
        f"refs/heads/{branch}:refs/remotes/origin/{branch}",
        check=False,
    )
    if fetch.returncode == 0:
        ancestor = run_git(
            ROOT,
            "merge-base",
            "--is-ancestor",
            f"origin/{branch}",
            "HEAD",
            check=False,
        )
        if ancestor.returncode != 0:
            raise RuntimeError(
                f"origin/{branch} advanced independently; use a fresh clone before publishing"
            )


def preflight_push(branch: str, secret_name: str) -> None:
    """Fail before GPU work unless the historical GITHUB_TOKEN path can write."""
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
        raise RuntimeError(f"GitHub write preflight failed before GPU work: {detail}")
    print(f"GitHub write preflight passed for {branch} using Kaggle Secret {secret_name}")


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
        raise RuntimeError(
            f"results remain committed locally but authenticated push failed: {detail}"
        )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    print(f"pushed Core 007 results to {branch}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("discovery", "confirmation"))
    p.add_argument("--commit-results", action="store_true")
    p.add_argument("--push-results", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--branch", default=None)
    p.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    branch = args.branch or _git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    if not branch or branch == "HEAD":
        raise RuntimeError("detached HEAD: pass --branch explicitly")
    if args.preflight_only:
        preflight_push(branch, args.secret_name)
        return 0
    if args.phase is None:
        raise RuntimeError("--phase is required unless --preflight-only is used")
    if args.push_results:
        args.commit_results = True
    dest, decision = _copy_phase(args.phase, allow_partial=args.allow_partial)
    print(dest)
    if args.commit_results:
        commit = _commit(args.phase, decision)
        print(f"commit={commit or 'no changes'}")
    if args.push_results:
        _push(branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
