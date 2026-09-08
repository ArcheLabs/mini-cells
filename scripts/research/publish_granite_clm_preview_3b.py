#!/usr/bin/env python3
"""Publish lightweight Granite-CLM-Preview-3B release evidence to GitHub."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

BRANCH = "codex/granite-clm-preview-3b-release"
OUTPUT = Path("artifacts/releases/granite-clm-preview-3b")
ALLOWED = {
    "clm_moe_manifest.json",
    "metrics.json",
    "parity_report.json",
    "provenance.json",
    "hf_publish.json",
    "CLM_PREVIEW.md",
    "RESULTS.md",
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


def stage_allowed() -> list[str]:
    paths = [OUTPUT / name for name in sorted(ALLOWED) if (OUTPUT / name).exists()]
    if not paths:
        raise RuntimeError("no Granite-CLM-Preview-3B release artifacts found")

    metrics_path = OUTPUT / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError("metrics.json is required")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("status") != "PASS":
        raise RuntimeError("refusing to publish GitHub release evidence for a non-PASS run")

    run(["git", "reset"])
    run(["git", "add", "-f", "--", *[str(path) for path in paths]])
    staged = run(["git", "diff", "--cached", "--name-only"], capture=True).splitlines()
    prefix = str(OUTPUT) + "/"
    unexpected = [path for path in staged if not path.startswith(prefix)]
    if unexpected:
        raise RuntimeError("unexpected staged files: " + ", ".join(unexpected))
    forbidden = [
        path for path in staged if path.endswith((".safetensors", ".bin", ".pt", ".pth"))
    ]
    if forbidden:
        raise RuntimeError(
            "model weights belong on Hugging Face, not Git: " + ", ".join(forbidden)
        )
    return staged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()

    current = run(["git", "branch", "--show-current"], capture=True)
    if current != args.branch:
        raise RuntimeError(f"expected branch {args.branch}, got {current}")

    staged = stage_allowed()
    if staged:
        run(["git", "config", "user.name", "MiniCells Research"])
        run(["git", "config", "user.email", "research@minicells.local"])
        run(["git", "commit", "-m", "artifacts: publish Granite CLM Preview 3B release evidence"])

    run(["git", "fetch", "origin"])
    run(["git", "rebase", f"origin/{args.branch}"])

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    askpass = Path("/tmp/minicells-granite-clm-preview-3b-askpass.sh")
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

    print(
        json.dumps(
            {
                "published": True,
                "branch": args.branch,
                "commit": run(["git", "rev-parse", "HEAD"], capture=True),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
