#!/usr/bin/env python3
"""HF-first publish Shadow Cell Validation 001 formal evidence, then Git metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from huggingface_hub import HfApi

DEFAULT_BRANCH = "codex/shadow-cell-validation-001"
DEFAULT_OUTPUT = Path(
    "artifacts/experiments/shadow-cell-validation-001-copy-on-write-functional-isolation"
)
DEFAULT_HF_REPO = "archelabsxyz/shadow-cell-validation-001"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, env=None, capture=False, timeout=240) -> str:
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
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--github-token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()

    decision_path = args.output_dir / "decision.json"
    if not decision_path.exists():
        raise FileNotFoundError(decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("format") != "minicells.shadow-cell-validation-001.result.v1":
        raise RuntimeError("unexpected Shadow Cell result format")
    if decision.get("phase") != "formal" or decision.get("scientific_decision") is not True:
        raise RuntimeError("publisher requires the frozen formal result")
    allowed = {
        "INCONCLUSIVE_BASE_TRAINING",
        "INCONCLUSIVE_PARENT_CONFLICT",
        "INCONCLUSIVE_DIRECT_PLASTICITY",
        "INCONCLUSIVE_GATE_CAPACITY",
        "INCONCLUSIVE_IDENTITY_CONTROL",
        "SHADOW_CELL_NOT_SUPPORTED",
        "ISOLATED_SHADOW_ADVANTAGE_NOT_SUPPORTED",
        "SHADOW_ISOLATION_SUPPORTED_MATURATION_NOT_NECESSARY",
        "SHADOW_CELL_CONTROLLED_MATURATION_SUPPORTED",
    }
    if decision.get("classification") not in allowed:
        raise RuntimeError("unknown Shadow Cell classification")
    if decision.get("independent_of_native_clm_m2_chain") is not True:
        raise RuntimeError("independence boundary missing")
    if decision.get("native_clm_m2_decision_modified") is not False:
        raise RuntimeError("Shadow validation must not modify Native CLM M2 decision")

    hf_token = os.environ.get(args.hf_token_env)
    if not hf_token:
        raise RuntimeError(f"missing environment variable {args.hf_token_env}")
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=args.hf_repo, repo_type="model", exist_ok=True, private=False)
    model_records: list[dict[str, object]] = []
    for seed in decision["seeds"]:
        seed_dir = args.output_dir / f"seed-{seed}"
        for local_name, remote_name in (
            ("base.pt", "base.pt"),
            ("candidates.pt", "candidates.pt"),
        ):
            path = seed_dir / local_name
            if not path.exists():
                raise FileNotFoundError(path)
            remote = f"formal/seed-{seed}/{remote_name}"
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=remote,
                repo_id=args.hf_repo,
                repo_type="model",
                commit_message=f"Shadow Cell Validation 001 seed {seed} {remote_name}",
            )
            model_records.append(
                {
                    "seed": int(seed),
                    "local_name": local_name,
                    "hf_path": remote,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    resolved_revision = api.model_info(args.hf_repo).sha
    publication = {
        "format": "minicells.shadow-cell-validation-001.publication.v1",
        "classification": decision["classification"],
        "protocol_sha256": decision["protocol_sha256"],
        "hf_repo": args.hf_repo,
        "hf_revision": resolved_revision,
        "hf_upload_status": "PUBLISHED",
        "model_artifacts": model_records,
    }
    (args.output_dir / "model-artifacts.json").write_text(
        json.dumps(publication, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    current = run(["git", "branch", "--show-current"], capture=True)
    if current != args.branch:
        raise RuntimeError(f"expected branch {args.branch}, got {current}")
    files = [
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md"}
    ]
    run(["git", "reset"])
    run(["git", "add", "-f", "--", *[str(path) for path in sorted(files)]])
    staged = run(["git", "diff", "--cached", "--name-only"], capture=True).splitlines()
    if any(path.endswith(".pt") for path in staged):
        raise RuntimeError("refusing to Git-publish model checkpoints")
    if staged:
        run(["git", "config", "user.name", "MiniCells Research"])
        run(["git", "config", "user.email", "research@minicells.local"])
        run(["git", "commit", "-m", "research: publish Shadow Cell Validation 001"])

    refspec = f"+refs/heads/{args.branch}:refs/remotes/origin/{args.branch}"
    run(["git", "fetch", "--no-tags", "origin", refspec])
    try:
        run(["git", "rebase", f"origin/{args.branch}"], timeout=180)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "rebase", "--abort"], check=False)
        raise

    github_token = os.environ.get(args.github_token_env)
    if not github_token:
        raise RuntimeError(f"missing environment variable {args.github_token_env}")
    askpass = Path("/tmp/minicells-shadow001-askpass.sh")
    askpass.write_text(
        '#!/bin/sh\ncase "$1" in\n  *Username*) echo "x-access-token" ;;\n  *) echo "$GITHUB_TOKEN" ;;\nesac\n',
        encoding="utf-8",
    )
    askpass.chmod(0o700)
    env = os.environ.copy()
    env["GITHUB_TOKEN"] = github_token
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
                "classification": decision["classification"],
                "scientific_decision": True,
                "independent_of_native_clm_m2_chain": True,
                "hf_repo": args.hf_repo,
                "hf_revision": resolved_revision,
                "hf_upload_status": "PUBLISHED",
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
