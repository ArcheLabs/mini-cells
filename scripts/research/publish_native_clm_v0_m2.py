"""Publish Native CLM v0 M2 evidence and checkpoints.

Binary M2 checkpoints go to the existing Hugging Face model repository. Git receives
only lightweight scientific evidence plus exact HF revision/path/SHA provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi

from minicells.native_clm_m2 import sha256_file


DEFAULT_BRANCH = "codex/native-clm-v0-m2-continual-language"
DEFAULT_OUTPUT = Path("artifacts/experiments/native-clm-v0-m2-continual-language")
DEFAULT_HF_REPO = "archelabsxyz/native-clm-v0"


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
    parser.add_argument("--checkpoint-provenance", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--github-token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()

    decision_path = args.output_dir / "decision.json"
    if not decision_path.exists():
        raise FileNotFoundError(decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("format") != "minicells.native-clm-v0.m2-decision.v1":
        raise RuntimeError("unexpected M2 decision format")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.checkpoint_provenance, args.output_dir / "m1-checkpoint-provenance.json")
    shutil.copy2(args.data_manifest, args.output_dir / "data-manifest.json")

    hf_token = os.environ.get(args.hf_token_env)
    if not hf_token:
        raise RuntimeError(f"missing environment variable {args.hf_token_env}")
    api = HfApi(token=hf_token)
    uploads = []
    for seed in decision["formal_seeds"]:
        for arm in ("protected", "unsafe"):
            local = args.output_dir / f"seed-{seed}" / arm / "final.pt"
            if not local.exists():
                raise FileNotFoundError(local)
            remote = f"m2/seed-{seed}/{arm}-final.pt"
            digest = sha256_file(local)
            print(
                f"Uploading M2 seed={seed} arm={arm} -> {args.hf_repo}/{remote} "
                f"({local.stat().st_size / 1024 / 1024:.1f} MiB)",
                flush=True,
            )
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=args.hf_repo,
                repo_type="model",
                commit_message=f"Native CLM v0 M2 seed {seed} {arm}",
            )
            uploads.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "path": remote,
                    "sha256": digest,
                    "bytes": local.stat().st_size,
                }
            )

    api.upload_file(
        path_or_fileobj=str(decision_path),
        path_in_repo="m2/decision.json",
        repo_id=args.hf_repo,
        repo_type="model",
        commit_message="Native CLM v0 M2 decision",
    )
    final_revision = api.model_info(args.hf_repo).sha
    model_artifacts = {
        "format": "minicells.native-clm-v0.m2-model-artifacts.v1",
        "repo_id": args.hf_repo,
        "resolved_revision_after_upload": final_revision,
        "files": uploads,
        "scientific_status": decision["status"],
    }
    (args.output_dir / "model-artifacts.json").write_text(
        json.dumps(model_artifacts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(model_artifacts, indent=2), flush=True)

    current = run(["git", "branch", "--show-current"], capture=True)
    if current != args.branch:
        raise RuntimeError(f"expected branch {args.branch}, got {current}")

    allowed = []
    for path in args.output_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md"}:
            allowed.append(path)
    if not allowed:
        raise RuntimeError("no lightweight M2 artifacts found")

    print("Staging lightweight M2 evidence (never .pt)...", flush=True)
    run(["git", "reset"])
    run(["git", "add", "-f", "--", *[str(path) for path in sorted(allowed)]])
    staged = run(["git", "diff", "--cached", "--name-only"], capture=True).splitlines()
    forbidden = [path for path in staged if path.endswith(".pt")]
    if forbidden:
        raise RuntimeError("refusing to Git-publish checkpoints: " + ", ".join(forbidden))

    if staged:
        run(["git", "config", "user.name", "MiniCells Research"])
        run(["git", "config", "user.email", "research@minicells.local"])
        run(["git", "commit", "-m", "research: publish Native CLM v0 M2 results"])

    print("Fetching only the M2 research branch...", flush=True)
    refspec = f"+refs/heads/{args.branch}:refs/remotes/origin/{args.branch}"
    run(["git", "fetch", "--no-tags", "origin", refspec], timeout=180)
    print("Rebasing M2 evidence commit...", flush=True)
    try:
        run(["git", "rebase", f"origin/{args.branch}"], timeout=120)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "rebase", "--abort"], check=False)
        raise

    github_token = os.environ.get(args.github_token_env)
    if not github_token:
        raise RuntimeError(f"missing environment variable {args.github_token_env}")
    askpass = Path("/tmp/minicells-m2-askpass.sh")
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
        print("Pushing lightweight M2 evidence...", flush=True)
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
                "m2_status": decision["status"],
                "hf_repo": args.hf_repo,
                "hf_revision": final_revision,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
