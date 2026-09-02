#!/usr/bin/env python3
"""Publish Native CLM v0 M0/M1 lightweight artifacts from Kaggle.

Model checkpoints are intentionally not committed. The published M1 summary records
the final checkpoint SHA-256 and byte size so a future weight store can verify the
exact runtime artifact without turning the Git repository into a model registry.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


BRANCH = "codex/native-clm-v0-m0-m1"
M0_DIR = Path("artifacts/experiments/native-clm-v0-m0-execution-smoke")
M1_DIR = Path("artifacts/experiments/native-clm-v0-m1-next-token")
M0_ALLOWED = {"decision.json", "RESULTS.md"}
M1_ALLOWED = {
    "summary.json",
    "metrics.csv",
    "run-config.json",
    "sample.txt",
    "RESULTS.md",
    "data-manifest.json",
}


def run(command: list[str], *, env: dict[str, str] | None = None, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        check=True,
        env=env,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def validate() -> tuple[dict, dict]:
    m0_path = M0_DIR / "decision.json"
    m1_path = M1_DIR / "summary.json"
    if not m0_path.exists():
        raise FileNotFoundError(m0_path)
    if not m1_path.exists():
        raise FileNotFoundError(m1_path)
    m0 = json.loads(m0_path.read_text(encoding="utf-8"))
    m1 = json.loads(m1_path.read_text(encoding="utf-8"))
    if m0["status"] != "NATIVE_CLM_V0_M0_EXECUTION_SMOKE_PASS" or not m0["pass"]:
        raise RuntimeError("M0 is not a passing execution smoke")
    if m1.get("format") != "minicells.native-clm-v0.m1-summary.v1":
        raise RuntimeError("unexpected M1 summary format")
    if m1.get("scientific_decision") is not False:
        raise RuntimeError("M1 must not claim a formal scientific decision")
    if "final_checkpoint_sha256" not in m1:
        raise RuntimeError("M1 summary lacks checkpoint identity")
    return m0, m1


def stage_allowed() -> list[str]:
    allowed: list[Path] = []
    for name in M0_ALLOWED:
        path = M0_DIR / name
        if path.exists():
            allowed.append(path)
    for name in M1_ALLOWED:
        path = M1_DIR / name
        if path.exists():
            allowed.append(path)
    if not allowed:
        raise RuntimeError("no publishable artifacts found")

    run(["git", "reset"])
    run(["git", "add", "-f", "--", *[str(path) for path in allowed]])
    staged = run(["git", "diff", "--cached", "--name-only"], capture=True).splitlines()
    prefixes = (str(M0_DIR) + "/", str(M1_DIR) + "/")
    unexpected = [path for path in staged if not path.startswith(prefixes)]
    if unexpected:
        raise RuntimeError("unexpected staged files: " + ", ".join(unexpected))
    forbidden = [
        path
        for path in staged
        if path.endswith(".pt") or "checkpoint" in Path(path).name.lower()
    ]
    if forbidden:
        raise RuntimeError("binary checkpoints must not be Git-published: " + ", ".join(forbidden))
    return staged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()

    current = run(["git", "branch", "--show-current"], capture=True)
    if current != args.branch:
        raise RuntimeError(f"expected branch {args.branch}, got {current}")

    _, m1 = validate()
    staged = stage_allowed()
    print("Files to publish:")
    for path in staged:
        print(" ", path)

    if staged:
        run(["git", "config", "user.name", "MiniCells Research"])
        run(["git", "config", "user.email", "research@minicells.local"])
        run(["git", "commit", "-m", "research: publish Native CLM v0 M0/M1 results"])

    run(["git", "fetch", "origin"])
    run(["git", "rebase", f"origin/{args.branch}"])

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    askpass = Path("/tmp/minicells-native-clm-askpass.sh")
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

    commit = run(["git", "rev-parse", "HEAD"], capture=True)
    print(
        json.dumps(
            {
                "published": True,
                "branch": args.branch,
                "commit": commit,
                "m1_status": m1["status"],
                "m1_pass": m1["pass"],
                "checkpoint_sha256": m1["final_checkpoint_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
