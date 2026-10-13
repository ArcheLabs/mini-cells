#!/usr/bin/env python3
"""Publish lightweight Native CLM JAM demo results from a hosted GPU run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

BRANCH = "codex/native-clm-jam-demo-v0.1"
OUTPUT = Path("artifacts/demos/native-clm-jam-v0.1")
ALLOWED = {
    "benchmarks.json",
    "provenance.json",
    "QA_LOG.md",
    "training.csv",
    "HF_README.md",
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


def write_results() -> None:
    benchmarks = json.loads((OUTPUT / "benchmarks.json").read_text(encoding="utf-8"))
    provenance = json.loads((OUTPUT / "provenance.json").read_text(encoding="utf-8"))
    before = benchmarks["before"]
    after = benchmarks["after"]
    lines = [
        "# Native CLM v0 — JAM Learning Demo",
        "",
        "Status: **DEMO_COMPLETE**",
        "",
        "> This is an engineering demonstration, not a new formal continual-learning decision.",
        "",
        "## Model identity",
        "",
        f"- Base SHA-256: `{provenance['base_checkpoint_sha256']}`",
        f"- JAM SHA-256: `{provenance['final_checkpoint_sha256']}`",
        f"- Selected step: **{provenance['selected_step']}**",
        f"- Training rows: **{provenance['train_rows']}**",
        f"- Learner-invisible reasoning rows: **{provenance['reasoning_rows']}**",
        "",
        "## Before / after",
        "",
        "| Benchmark | Before | After |",
        "|---|---:|---:|",
        f"| JAM validation answer NLL | {before['validation']['answer_nll']:.4f} | {after['validation']['answer_nll']:.4f} |",
        f"| JAM factual token accuracy | {before['factual']['answer_token_accuracy']:.4f} | {after['factual']['answer_token_accuracy']:.4f} |",
        f"| JAM relational token accuracy | {before['relational']['answer_token_accuracy']:.4f} | {after['relational']['answer_token_accuracy']:.4f} |",
        f"| JAM misconception token accuracy | {before['misconceptions']['answer_token_accuracy']:.4f} | {after['misconceptions']['answer_token_accuracy']:.4f} |",
        f"| JAM reasoning answer NLL | {before['reasoning']['answer_nll']:.4f} | {after['reasoning']['answer_nll']:.4f} |",
        f"| JAM reasoning token accuracy | {before['reasoning']['answer_token_accuracy']:.4f} | {after['reasoning']['answer_token_accuracy']:.4f} |",
        f"| TinyStories validation perplexity | {before['base']['perplexity']:.4f} | {after['base']['perplexity']:.4f} |",
        "",
        "## Claim boundary",
        "",
        "The run supports the narrow demo claim that an already-trained Native CLM can acquire",
        "bounded JAM knowledge through post-training. It does not claim replay-free continual",
        "learning, general JAM reasoning, or superiority over ordinary fine-tuning.",
        "",
        "See `QA_LOG.md` for deterministic before/after generations.",
    ]
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_allowed() -> list[str]:
    paths = [OUTPUT / name for name in sorted(ALLOWED) if (OUTPUT / name).exists()]
    if not paths:
        raise RuntimeError("no JAM demo artifacts found")
    run(["git", "reset"])
    run(["git", "add", "-f", "--", *[str(path) for path in paths]])
    staged = run(["git", "diff", "--cached", "--name-only"], capture=True).splitlines()
    prefix = str(OUTPUT) + "/"
    unexpected = [path for path in staged if not path.startswith(prefix)]
    if unexpected:
        raise RuntimeError("unexpected staged files: " + ", ".join(unexpected))
    forbidden = [path for path in staged if path.endswith(".pt")]
    if forbidden:
        raise RuntimeError("model weights belong on Hugging Face, not Git: " + ", ".join(forbidden))
    return staged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()

    current = run(["git", "branch", "--show-current"], capture=True)
    if current != args.branch:
        raise RuntimeError(f"expected branch {args.branch}, got {current}")

    write_results()
    staged = stage_allowed()
    if staged:
        run(["git", "config", "user.name", "MiniCells Research"])
        run(["git", "config", "user.email", "research@minicells.local"])
        run(["git", "commit", "-m", "artifacts: publish Native CLM JAM demo results"])

    run(["git", "fetch", "origin"])
    run(["git", "rebase", f"origin/{args.branch}"])

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    askpass = Path("/tmp/minicells-native-clm-jam-askpass.sh")
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
