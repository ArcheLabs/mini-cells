"""Publish Native CLM v0 M3 checkpoints to Hugging Face and lightweight evidence to Git."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, auth_check
from huggingface_hub.errors import HfHubHTTPError

from minicells.native_clm_m2 import sha256_file

DEFAULT_BRANCH = "codex/native-clm-v0-m3-growth-restored-continual-language"
DEFAULT_OUTPUT = Path("artifacts/experiments/native-clm-v0-m3-growth-restored-continual-language")
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


def _checkpoint_records(output_dir: Path, formal_seeds: list[int]) -> list[dict]:
    records: list[dict] = []
    for seed in formal_seeds:
        for arm in ("fixed_protected", "growth_protected"):
            local = output_dir / f"seed-{seed}" / arm / "final.pt"
            if not local.exists():
                raise FileNotFoundError(local)
            records.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "local_path": str(local),
                    "path": f"m3/seed-{seed}/{arm}-final.pt",
                    "sha256": sha256_file(local),
                    "bytes": local.stat().st_size,
                    "uploaded": False,
                }
            )
    return records


def _publish_hf(
    *,
    output_dir: Path,
    decision_path: Path,
    decision: dict,
    repo_id: str,
    token: str | None,
    require_upload: bool,
) -> dict:
    records = _checkpoint_records(output_dir, [int(seed) for seed in decision["formal_seeds"]])
    result = {
        "format": "minicells.native-clm-v0.m3-model-artifacts.v1",
        "repo_id": repo_id,
        "hf_upload_status": "NOT_ATTEMPTED",
        "resolved_revision_after_upload": None,
        "files": records,
        "scientific_status": decision["status"],
        "error": None,
    }
    if not token:
        result["hf_upload_status"] = "SKIPPED_MISSING_TOKEN"
        result["error"] = "HF_TOKEN is missing"
        if require_upload:
            raise RuntimeError(result["error"])
        return result

    try:
        auth_check(repo_id, repo_type="model", token=token, write=True)
    except HfHubHTTPError as exc:
        result["hf_upload_status"] = "SKIPPED_WRITE_PERMISSION_DENIED"
        result["error"] = f"HF token lacks write permission for {repo_id}"
        if require_upload:
            raise RuntimeError(result["error"]) from exc
        return result

    api = HfApi(token=token)
    try:
        for record in records:
            local = Path(record["local_path"])
            print(
                f"Uploading M3 seed={record['seed']} arm={record['arm']} -> "
                f"{repo_id}/{record['path']} ({record['bytes'] / 1024 / 1024:.1f} MiB)",
                flush=True,
            )
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=record["path"],
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Native CLM v0 M3 seed {record['seed']} {record['arm']}",
            )
            record["uploaded"] = True
        api.upload_file(
            path_or_fileobj=str(decision_path),
            path_in_repo="m3/decision.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Native CLM v0 M3 decision",
        )
        result["resolved_revision_after_upload"] = api.model_info(repo_id).sha
        result["hf_upload_status"] = "PUBLISHED"
    except HfHubHTTPError as exc:
        result["hf_upload_status"] = "FAILED_DURING_UPLOAD"
        result["error"] = str(exc)
        if require_upload:
            raise RuntimeError("Hugging Face M3 checkpoint upload failed") from exc
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-provenance", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--github-token-env", default="GITHUB_TOKEN")
    parser.add_argument("--require-hf-upload", action="store_true")
    args = parser.parse_args()

    decision_path = args.output_dir / "decision.json"
    if not decision_path.exists():
        raise FileNotFoundError(decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("format") != "minicells.native-clm-v0.m3-decision.v1":
        raise RuntimeError("unexpected M3 decision format")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.checkpoint_provenance, args.output_dir / "m1-checkpoint-provenance.json")
    shutil.copy2(args.data_manifest, args.output_dir / "data-manifest.json")

    model_artifacts = _publish_hf(
        output_dir=args.output_dir,
        decision_path=decision_path,
        decision=decision,
        repo_id=args.hf_repo,
        token=os.environ.get(args.hf_token_env),
        require_upload=args.require_hf_upload,
    )
    (args.output_dir / "model-artifacts.json").write_text(
        json.dumps(model_artifacts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(model_artifacts, indent=2), flush=True)

    current = run(["git", "branch", "--show-current"], capture=True)
    if current != args.branch:
        raise RuntimeError(f"expected branch {args.branch}, got {current}")

    allowed = [
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md"}
    ]
    if not allowed:
        raise RuntimeError("no lightweight M3 artifacts found")

    run(["git", "reset"])
    run(["git", "add", "-f", "--", *[str(path) for path in sorted(allowed)]])
    staged = run(["git", "diff", "--cached", "--name-only"], capture=True).splitlines()
    forbidden = [path for path in staged if path.endswith(".pt")]
    if forbidden:
        raise RuntimeError("refusing to Git-publish checkpoints: " + ", ".join(forbidden))

    if staged:
        run(["git", "config", "user.name", "MiniCells Research"])
        run(["git", "config", "user.email", "research@minicells.local"])
        run(["git", "commit", "-m", "research: publish Native CLM v0 M3 results"])

    refspec = f"+refs/heads/{args.branch}:refs/remotes/origin/{args.branch}"
    print("Fetching only the M3 research branch...", flush=True)
    run(["git", "fetch", "--no-tags", "origin", refspec], timeout=180)
    print("Rebasing M3 evidence commit...", flush=True)
    try:
        run(["git", "rebase", f"origin/{args.branch}"], timeout=120)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "rebase", "--abort"], check=False)
        raise

    github_token = os.environ.get(args.github_token_env)
    if not github_token:
        raise RuntimeError(f"missing environment variable {args.github_token_env}")
    askpass = Path("/tmp/minicells-m3-askpass.sh")
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
        print("Pushing lightweight M3 evidence...", flush=True)
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
                "m3_status": decision["status"],
                "hf_repo": args.hf_repo,
                "hf_upload_status": model_artifacts["hf_upload_status"],
                "hf_revision": model_artifacts["resolved_revision_after_upload"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
