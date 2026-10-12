"""Git-publish lightweight Native CLM v0 M3W-0 diagnostic evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

DEFAULT_BRANCH = "codex/native-clm-v0-m3w0-write-drift-restoration"
DEFAULT_OUTPUT = Path("artifacts/experiments/native-clm-v0-m3w0-write-drift-restoration")


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
    if result.get("format") != "minicells.native-clm-v0.m3w0-write-drift-restoration.result.v1":
        raise RuntimeError("unexpected M3W-0 result format")
    allowed = {
        "INCONCLUSIVE_IDENTITY",
        "ROOT_WRITE_DOMINANT_CHILDREN_CARRY_PLASTICITY",
        "ROOT_WRITE_DOMINANT_TRANSFER_GAP",
        "DESCENDANT_WRITE_DOMINANT",
        "DISTRIBUTED_WRITE_DRIFT",
    }
    if result.get("classification") not in allowed:
        raise RuntimeError("unknown M3W-0 classification")
    if result.get("scientific_decision") is not False:
        raise RuntimeError("M3W-0 unexpectedly claims a scientific decision")
    if result.get("native_clm_training") is not False:
        raise RuntimeError("M3W-0 unexpectedly claims Native CLM training")
    if result.get("new_formal_seeds_consumed") is not False:
        raise RuntimeError("M3W-0 unexpectedly consumed new formal seeds")

    current = run(["git", "branch", "--show-current"], capture=True)
    if current != args.branch:
        raise RuntimeError(f"expected branch {args.branch}, got {current}")
    files = [
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md"}
    ]
    if not files:
        raise RuntimeError("no lightweight M3W-0 artifacts found")
    run(["git", "reset"])
    run(["git", "add", "-f", "--", *[str(path) for path in sorted(files)]])
    staged = run(["git", "diff", "--cached", "--name-only"], capture=True).splitlines()
    if any(path.endswith(".pt") for path in staged):
        raise RuntimeError("refusing to Git-publish checkpoints")
    if staged:
        run(["git", "config", "user.name", "MiniCells Research"])
        run(["git", "config", "user.email", "research@minicells.local"])
        run(["git", "commit", "-m", "research: publish M3W-0 write-drift restoration diagnostic"])

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
    askpass = Path("/tmp/minicells-m3w0-askpass.sh")
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
                "native_clm_training": False,
                "new_formal_seeds_consumed": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
