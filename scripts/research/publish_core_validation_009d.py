#!/usr/bin/env python3
"""Publish Core Validation 009D checkpoints and representation lock."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from publish_core_validation_007 import _authenticated_git_env, _check_branch
from publish_experiment_results import DEFAULT_SECRET_NAME, EXPECTED_ORIGIN, run_git

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "core-validation-009d-compositional-operator-geometry"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "core-validation-009d-compositional-operator-geometry"
VALIDATION = ROOT / "research" / "validations" / "core-009d-compositional-operator-geometry"
PROTOCOL = VALIDATION / "protocol.json"
LOCK = VALIDATION / "representation-lock.json"
EXCLUDE = {"frozen-hidden.pt", "operator-pack.pt"}


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


def _copy_phase(phase: str) -> dict:
    src = RESULTS / phase; decision_path = src / "decision.json"
    if not decision_path.is_file(): raise FileNotFoundError(decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if phase == "discovery" and decision.get("scientific_decision") is not False: raise RuntimeError("009D discovery must remain non-scientific")
    if phase == "confirmation":
        scientific = decision.get("scientific_decision") is True; partial = decision.get("status") == "CONFIRMATION_INCOMPLETE"
        if not scientific and not partial: raise RuntimeError("009D confirmation publication requires final decision or partial checkpoint")
    dest = ARTIFACTS / phase
    if dest.exists(): shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(src, dest, ignore=_ignore)
    ARTIFACTS.mkdir(parents=True, exist_ok=True); shutil.copy2(PROTOCOL, ARTIFACTS / "protocol.json")
    manifest = RESULTS / "data-manifest.json"
    if manifest.is_file(): shutil.copy2(manifest, ARTIFACTS / "data-manifest.json")
    if phase == "discovery":
        generated = src / "representation-lock.json"
        if generated.is_file():
            lock = json.loads(generated.read_text(encoding="utf-8"))
            if lock.get("confirmation_allowed") is not True: raise RuntimeError("refusing to publish disabled 009D representation lock")
            shutil.copy2(generated, LOCK); shutil.copy2(generated, ARTIFACTS / "representation-lock.json")
    elif LOCK.is_file(): shutil.copy2(LOCK, ARTIFACTS / "representation-lock.json")
    return decision


def _commit(phase: str, decision: dict) -> str | None:
    run_git(ROOT, "config", "user.name", "MiniCells Kaggle"); run_git(ROOT, "config", "user.email", "kaggle@minicells.local")
    paths = [ARTIFACTS.relative_to(ROOT).as_posix()]
    if LOCK.is_file(): paths.append(LOCK.relative_to(ROOT).as_posix())
    run_git(ROOT, "add", "--", *paths)
    if run_git(ROOT, "diff", "--cached", "--quiet", check=False).returncode == 0: return None
    if phase == "discovery": message = "research: lock Core Validation 009D representation" if decision.get("confirmation_allowed") else f"research: checkpoint Core Validation 009D discovery ({len(decision.get('completed_seeds', []))}/2)"
    elif decision.get("scientific_decision") is True: message = "research: record Core Validation 009D confirmation"
    else: message = f"research: checkpoint Core Validation 009D confirmation ({len(decision.get('completed_seeds', []))}/3)"
    run_git(ROOT, "commit", "-m", message); return _git_output(["rev-parse", "HEAD"])


def _push(branch: str, secret_name: str) -> None:
    _check_branch(branch)
    with _authenticated_git_env(secret_name) as env:
        result = run_git(ROOT, "push", EXPECTED_ORIGIN + ".git", f"HEAD:refs/heads/{branch}", env=env, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip(); raise RuntimeError(f"authenticated push failed: {detail}")
    print(f"pushed Core 009D artifacts to {branch}")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--phase", choices=("discovery", "confirmation"), required=True); p.add_argument("--branch", default=None); p.add_argument("--secret-name", default=DEFAULT_SECRET_NAME); p.add_argument("--preflight-only", action="store_true"); p.add_argument("--push-results", action="store_true"); args = p.parse_args()
    branch = args.branch or _git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    if args.preflight_only: preflight_push(branch, args.secret_name); return 0
    decision = _copy_phase(args.phase); commit = _commit(args.phase, decision); print(f"commit={commit or 'no changes'}")
    if args.push_results: _push(branch, args.secret_name)
    return 0


if __name__ == "__main__": raise SystemExit(main())
