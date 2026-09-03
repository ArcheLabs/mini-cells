#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    REPO_ROOT
    / "research/validations/constructive-clm-005-scaffold-removal/protocol.json"
)
ARTIFACT_DIR = (
    REPO_ROOT
    / "artifacts/experiments/constructive-clm-005-scaffold-removal"
)
FORMAL_SEEDS = [90811, 90812, 90813]
VALID_STATUSES = {
    "LEARNED_CONTROL_PLANE_TRANSITION_SUPPORTED",
    "LEARNED_CONTROL_PLANE_TRANSITION_NOT_SUPPORTED",
}
REQUIRED_ARTIFACTS = [
    "decision.json",
    "gate-summary.csv",
    "controller-summary.csv",
    "stage-summary.csv",
    "RESULTS.md",
]


def run(
    *args: str,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        list(args),
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=capture,
        env=env,
    )
    return result.stdout.strip() if capture else ""


def verify() -> dict:
    for name in REQUIRED_ARTIFACTS:
        path = ARTIFACT_DIR / name
        if not path.exists():
            raise SystemExit(f"missing formal artifact: {path}")
    payload = json.loads((ARTIFACT_DIR / "decision.json").read_text())
    if payload.get("status") not in VALID_STATUSES:
        raise SystemExit(f"unexpected formal status: {payload.get('status')}")
    if payload.get("scientific_decision") is not True:
        raise SystemExit("decision is not a formal scientific decision")
    if payload.get("completed_seeds") != FORMAL_SEEDS:
        raise SystemExit(
            f"completed seeds mismatch: {payload.get('completed_seeds')}"
        )
    if payload.get("missing_seeds") != []:
        raise SystemExit(f"missing formal seeds: {payload.get('missing_seeds')}")
    expected_sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if payload.get("protocol_sha256") != expected_sha:
        raise SystemExit(
            "protocol hash mismatch: formal artifacts do not match current protocol"
        )
    if len(payload.get("results", [])) != len(FORMAL_SEEDS):
        raise SystemExit("formal result count mismatch")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--branch",
        default="codex/constructive-clm-005-endogenous-control",
    )
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()

    payload = verify()
    branch = run("git", "branch", "--show-current", capture=True)
    if branch != args.branch:
        raise SystemExit(f"expected branch {args.branch!r}, got {branch!r}")

    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            str(ARTIFACT_DIR.relative_to(REPO_ROOT) / "decision.json"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if tracked.returncode == 0:
        raise SystemExit(
            "CLM-005 canonical formal decision is already tracked; refusing duplicate publication"
        )

    run("git", "config", "user.name", "MiniCells Research")
    run("git", "config", "user.email", "research@minicells.local")
    run(
        "git",
        "add",
        "-f",
        "--",
        str(ARTIFACT_DIR.relative_to(REPO_ROOT)),
    )
    staged = run(
        "git", "diff", "--cached", "--name-only", capture=True
    ).splitlines()
    prefix = str(ARTIFACT_DIR.relative_to(REPO_ROOT)) + "/"
    unexpected = [path for path in staged if not path.startswith(prefix)]
    if unexpected:
        raise SystemExit(
            "refusing to commit unexpected staged paths: " + ", ".join(unexpected)
        )
    if not staged:
        raise SystemExit("no CLM-005 formal artifacts staged")
    run(
        "git",
        "commit",
        "-m",
        f"research: publish Constructive CLM-005 formal result ({payload['status']})",
    )

    run("git", "fetch", args.remote, args.branch)
    run("git", "rebase", f"{args.remote}/{args.branch}")

    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"missing token environment variable {args.token_env}")

    with tempfile.NamedTemporaryFile(
        "w", prefix="minicells-askpass-", delete=False
    ) as handle:
        askpass = Path(handle.name)
        handle.write(
            '#!/bin/sh\n'
            'case "$1" in\n'
            '  *Username*) echo "x-access-token" ;;\n'
            f'  *) echo "${args.token_env}" ;;\n'
            'esac\n'
        )
    askpass.chmod(askpass.stat().st_mode | stat.S_IXUSR)
    env = os.environ.copy()
    env["GIT_ASKPASS"] = str(askpass)
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        run(
            "git",
            "push",
            "https://github.com/ArcheLabs/mini-cells.git",
            f"HEAD:{args.branch}",
            env=env,
        )
    finally:
        askpass.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "published": True,
                "branch": args.branch,
                "commit": run("git", "rev-parse", "HEAD", capture=True),
                "status": payload["status"],
                "protocol_sha256": payload["protocol_sha256"],
                "formal_seeds": FORMAL_SEEDS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
