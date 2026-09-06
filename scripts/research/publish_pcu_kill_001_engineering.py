#!/usr/bin/env python3
"""Publish PCU-KILL-001 engineering evidence to the registered GitHub branch.

This publisher is intentionally engineering-only. Scientific negative outcomes
are publishable evidence; formal/frozen artifacts and binary mutation/cache
artifacts are never staged.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

BRANCH = "codex/pcu-composability-kill-001"
ENGINEERING_SEED = 26090501
FORMAL_SEEDS = (26090511, 26090512, 26090513)
OUTPUT = Path(f"artifacts/research/pcu-kill-001/engineering/{ENGINEERING_SEED}")
SEED_REGISTRY = Path("research/formal_seed_registry.json")
SAFE_SUFFIXES = {".json", ".csv", ".md", ".txt"}
FORBIDDEN_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".npy", ".npz"}


def run(command: list[str], *, env: dict[str, str] | None = None, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        check=True,
        env=env,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def assert_formal_seeds_untouched() -> None:
    payload = json.loads(SEED_REGISTRY.read_text(encoding="utf-8"))
    rows = {int(row["seed"]): str(row["state"]) for row in payload.get("seeds", [])}
    expected = {seed: "RESERVED_UNTOUCHED" for seed in FORMAL_SEEDS}
    if rows != expected:
        raise RuntimeError(f"formal seed registry changed: expected {expected}, got {rows}")


def validate_engineering_evidence() -> dict:
    decision_path = OUTPUT / "ENGINEERING_DECISION.json"
    run_manifest_path = OUTPUT / "RUN_MANIFEST.json"
    if not decision_path.is_file():
        raise RuntimeError("ENGINEERING_DECISION.json is required before publication")
    if not run_manifest_path.is_file():
        raise RuntimeError("RUN_MANIFEST.json is required before publication")

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("phase") != "engineering":
        raise RuntimeError("refusing to publish a non-engineering run")
    if int(manifest.get("seed", -1)) != ENGINEERING_SEED:
        raise RuntimeError("refusing to publish an unexpected engineering seed")
    if manifest.get("backend") != "granite":
        raise RuntimeError("refusing to publish a non-Granite PCU E0 run")
    if decision.get("scientific_evidence") is not False:
        raise RuntimeError("engineering evidence must not be labelled scientific/formal evidence")
    if decision.get("formal_execution_not_started") is not True:
        raise RuntimeError("formal execution boundary was not preserved")
    if decision.get("status") in {None, "REAL_GRANITE_E0_NOT_RUN", "FORMAL_EXECUTION_FAILED"}:
        raise RuntimeError(f"engineering run did not produce an interpretable E0 result: {decision.get('status')}")
    return decision


def stage_engineering_evidence() -> list[str]:
    assert_formal_seeds_untouched()
    decision = validate_engineering_evidence()

    paths = sorted(
        path for path in OUTPUT.rglob("*")
        if path.is_file() and path.suffix.lower() in SAFE_SUFFIXES
    )
    if not paths:
        raise RuntimeError("no lightweight PCU engineering evidence found")

    forbidden_present = sorted(
        str(path) for path in OUTPUT.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    )
    if forbidden_present:
        print(
            "Binary runtime artifacts remain local/Kaggle-only and will not be staged:\n  "
            + "\n  ".join(forbidden_present)
        )

    run(["git", "reset"])
    run(["git", "add", "-f", "--", *[str(path) for path in paths]])
    staged = run(["git", "diff", "--cached", "--name-only"], capture=True).splitlines()
    prefix = str(OUTPUT) + "/"
    unexpected = [path for path in staged if not path.startswith(prefix)]
    if unexpected:
        raise RuntimeError("unexpected staged files: " + ", ".join(unexpected))
    forbidden = [path for path in staged if Path(path).suffix.lower() in FORBIDDEN_SUFFIXES]
    if forbidden:
        raise RuntimeError("binary PCU artifacts must not enter Git: " + ", ".join(forbidden))
    if str(SEED_REGISTRY) in staged:
        raise RuntimeError("formal seed registry must never be staged by engineering publisher")

    changed = run(["git", "diff", "HEAD", "--name-only"], capture=True).splitlines()
    unexpected_changed = [path for path in changed if not path.startswith(prefix)]
    if unexpected_changed:
        raise RuntimeError(
            "working tree contains tracked changes outside the engineering evidence prefix: "
            + ", ".join(unexpected_changed)
        )

    print(json.dumps({
        "engineering_status": decision.get("status"),
        "staged_files": staged,
        "formal_seeds": "RESERVED_UNTOUCHED",
    }, indent=2))
    return staged


def remote_already_has_canonical_e0(branch: str) -> bool:
    remote_ref = f"origin/{branch}:{OUTPUT}/ENGINEERING_DECISION.json"
    probe = subprocess.run(
        ["git", "show", remote_ref],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return probe.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()

    current = run(["git", "branch", "--show-current"], capture=True)
    if current != args.branch:
        raise RuntimeError(f"expected branch {args.branch}, got {current}")

    run(["git", "fetch", "origin"])
    if remote_already_has_canonical_e0(args.branch):
        raise RuntimeError(
            "canonical PCU-KILL-001 engineering E0 evidence is already tracked on the remote branch; "
            "refusing to overwrite it"
        )

    staged = stage_engineering_evidence()
    if not staged:
        raise RuntimeError("nothing staged for PCU engineering publication")

    run(["git", "config", "user.name", "MiniCells Research"])
    run(["git", "config", "user.email", "research@minicells.local"])
    run(["git", "commit", "-m", "artifacts: publish PCU-KILL-001 engineering E0 evidence"])
    run(["git", "rebase", f"origin/{args.branch}"])

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    askpass = Path("/tmp/minicells-pcu-kill-001-askpass.sh")
    askpass.write_text(
        '#!/bin/sh\n'
        'case "$1" in\n'
        '  *Username*) echo "x-access-token" ;;\n'
        '  *) echo "$GITHUB_TOKEN" ;;\n'
        'esac\n',
        encoding="utf-8",
    )
    askpass.chmod(0o700)
    env = os.environ.copy()
    env["GITHUB_TOKEN"] = token
    env["GIT_ASKPASS"] = str(askpass)
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        run(
            [
                "git",
                "push",
                "https://github.com/ArcheLabs/mini-cells.git",
                f"HEAD:{args.branch}",
            ],
            env=env,
        )
    finally:
        askpass.unlink(missing_ok=True)

    print(json.dumps({
        "published": True,
        "branch": args.branch,
        "commit": run(["git", "rev-parse", "HEAD"], capture=True),
        "engineering_seed": ENGINEERING_SEED,
        "formal_execution_not_started": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
