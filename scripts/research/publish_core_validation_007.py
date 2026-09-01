#!/usr/bin/env python3
"""Publish Core Validation 007 artifacts and optionally commit/push them."""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "core-validation-007-functional-boundary-discovery"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "core-validation-007-functional-boundary-discovery"
VALIDATION = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery"
PROTOCOL = VALIDATION / "protocol.json"
CONFIRMATION_PROTOCOL = VALIDATION / "confirmation-protocol-v1.1.json"
WINNER_LOCK = VALIDATION / "winner-lock.json"
EXCLUDE_NAMES = {"frozen-hidden.pt", "frozen-hidden-smoke.pt"}


def _run(command: list[str], *, check: bool = True, capture: bool = False, env=None):
    return subprocess.run(
        command,
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=capture,
        env=env,
    )


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _ignore(_directory: str, names: list[str]) -> list[str]:
    return [name for name in names if name in EXCLUDE_NAMES or name.endswith(".tmp")]


def _copy_phase(phase: str) -> tuple[Path, dict]:
    src = RESULTS / phase
    decision_path = src / "decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError(decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if phase == "discovery":
        if decision.get("scientific_decision") is not False:
            raise RuntimeError("discovery artifacts must remain non-scientific")
    else:
        scientific = decision.get("scientific_decision")
        status = decision.get("status")
        if scientific is False and status != "CONFIRMATION_INCOMPLETE":
            raise RuntimeError("non-scientific confirmation publication must be CONFIRMATION_INCOMPLETE")
        if scientific not in {False, True}:
            raise RuntimeError("confirmation decision must explicitly declare scientific_decision")

    dest = ARTIFACTS / phase
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=_ignore)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROTOCOL, ARTIFACTS / "protocol.json")
    if CONFIRMATION_PROTOCOL.is_file():
        shutil.copy2(CONFIRMATION_PROTOCOL, ARTIFACTS / "confirmation-protocol-v1.1.json")
    if WINNER_LOCK.is_file():
        shutil.copy2(WINNER_LOCK, ARTIFACTS / "winner-lock.json")
    return dest, decision


def _commit(phase: str, decision: dict) -> str | None:
    _run(["git", "config", "user.name", "MiniCells Kaggle Runner"], check=False)
    _run(["git", "config", "user.email", "minicells-kaggle@users.noreply.github.com"], check=False)
    paths = [str(ARTIFACTS.relative_to(ROOT))]
    _run(["git", "add", *paths])
    changed = _run(["git", "diff", "--cached", "--quiet"], check=False).returncode != 0
    if not changed:
        return None
    if phase == "discovery":
        message = "research: lock Core Validation 007 discovery winner"
    elif decision.get("scientific_decision") is True:
        message = "research: record Core Validation 007 final confirmation results"
    else:
        completed = len(decision.get("completed_seeds", []))
        message = f"research: checkpoint Core Validation 007 confirmation ({completed}/3)"
    _run(["git", "commit", "-m", message])
    return _git_output(["rev-parse", "HEAD"])


def _auth_env(token: str) -> dict[str, str]:
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = os.environ.copy()
    env.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.extraHeader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: Basic {encoded}",
        }
    )
    return env


def _push(branch: str) -> None:
    first = _run(["git", "push", "origin", f"HEAD:{branch}"], check=False, capture=True)
    if first.returncode == 0:
        print(f"pushed origin/{branch}")
        return
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        second = _run(
            [
                "git",
                "push",
                "https://github.com/ArcheLabs/mini-cells.git",
                f"HEAD:{branch}",
            ],
            check=False,
            capture=True,
            env=_auth_env(token),
        )
        if second.returncode == 0:
            print(f"pushed ArcheLabs/mini-cells:{branch} using token environment")
            return
        detail = (second.stderr or second.stdout or "").strip().splitlines()[-1:]
        suffix = detail[0] if detail else "unknown git error"
        raise RuntimeError(
            f"results were committed locally but token push failed: {suffix}. "
            f"Run: git push origin HEAD:{branch}"
        )
    detail = (first.stderr or first.stdout or "").strip().splitlines()[-1:]
    suffix = detail[0] if detail else "no git credentials"
    raise RuntimeError(
        f"results were committed locally but push credentials are unavailable ({suffix}). "
        f"Set GH_TOKEN/GITHUB_TOKEN or run: git push origin HEAD:{branch}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    p.add_argument("--commit-results", action="store_true")
    p.add_argument("--push-results", action="store_true")
    p.add_argument("--branch", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.push_results:
        args.commit_results = True
    dest, decision = _copy_phase(args.phase)
    print(dest)
    commit = None
    if args.commit_results:
        commit = _commit(args.phase, decision)
        print(f"commit={commit or 'no changes'}")
    if args.push_results:
        branch = args.branch or _git_output(["rev-parse", "--abbrev-ref", "HEAD"])
        if not branch or branch == "HEAD":
            raise RuntimeError("detached HEAD: pass --branch explicitly for result push")
        _push(branch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
