"""Git-publish lightweight Native CLM v0 M2-R0 update-invariant evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

DEFAULT_BRANCH = "codex/native-clm-v0-m2r0-update-invariant-audit"
DEFAULT_OUTPUT = Path("artifacts/experiments/native-clm-v0-m2r0-update-invariant-audit")


def run(command: list[str], *, env=None, capture=False, timeout=180) -> str:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        check=True,
        env=env,
        text=True,
        capture_output=capture,
        timeout=timeout,
    )
    return result.stdout.strip() if capture else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--github-token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()

    result_path = args.output_dir / "diagnostic-result.json"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("format") != "minicells.native-clm-v0.m2r0-update-invariant-audit.result.v1":
        raise RuntimeError("unexpected M2-R0 result format")
    allowed = {
        "INCONCLUSIVE_REFERENCE_FAILURE",
        "CURRENT_UPDATE_INVARIANT_HOLDS",
        "WEIGHT_DECAY_BREAKS_UPDATE_INVARIANT",
        "ADAMW_PRECONDITIONER_BREAKS_UPDATE_INVARIANT",
        "MIXED_ADAMW_UPDATE_INVARIANT_VIOLATION",
    }
    if result.get("classification") not in allowed:
        raise RuntimeError("unknown M2-R0 classification")
    if result.get("scientific_decision") is not False:
        raise RuntimeError("M2-R0 unexpectedly claims a scientific decision")
    if result.get("native_clm_training_milestone") is not False:
        raise RuntimeError("M2-R0 unexpectedly claims a training milestone")
    if result.get("new_formal_seeds_consumed") is not False:
        raise RuntimeError("M2-R0 unexpectedly consumed formal seeds")

    current = run(["git", "branch", "--show-current"], capture=True)
    if current != args.branch:
        raise RuntimeError(f"expected branch {args.branch}, got {current}")
    files = [
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md"}
    ]
    if not files:
        raise RuntimeError("no lightweight M2-R0 artifacts found")
    run(["git", "reset"])
    run(["git", "add", "-f", "--", *[str(path) for path in sorted(files)]])
    staged = run(["git", "diff", "--cached", "--name-only"], capture=True).splitlines()
    if any(path.endswith(".pt") for path in staged):
        raise RuntimeError("refusing to Git-publish checkpoints")
    if staged:
        run(["git", "config", "user.name", "MiniCells Research"])
        run(["git", "config", "user.email", "research@minicells.local"])
        run(["git", "commit", "-m", "research: publish M2-R0 update invariant audit"])

    refspec = f"+refs/heads/{args.branch}:refs/remotes/origin/{args.branch}"
    run(["git", "fetch", "--no-tags", "origin", refspec], timeout=180)
    try:
        run(["git", "rebase", f"origin/{args.branch}"], timeout=120)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "rebase", "--abort"], check=False)
        raise

    token = os.environ.get(args.github_token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.github_token_env}")
    askpass = Path("/tmp/minicells-m2r0-askpass.sh")
    askpass.write_text(
        '#!/bin/sh\ncase "$1" in\n  *Username*) echo "x-access-token" ;;\n  *) echo "$GITHUB_TOKEN" ;;\nesac\n',
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
            timeout=180,
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
                "classification": result["classification"],
                "scientific_decision": False,
                "native_clm_training_milestone": False,
                "new_formal_seeds_consumed": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
