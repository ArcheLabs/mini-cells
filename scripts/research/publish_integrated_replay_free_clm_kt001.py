"""Publish KT001 seed evidence durably, including incomplete/failed seeds.

Completed seeds publish five arm summaries plus a seed decision. Incomplete seeds
may publish any already-produced arm summaries, failure records, phase diagnostics,
and checkpoints. Partial evidence is explicitly classified as incomplete and is
never accepted by the formal aggregator.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from huggingface_hub import HfApi, auth_check
from huggingface_hub.errors import HfHubHTTPError

from minicells.integrated_replay_free_clm_kt001 import SEED_REGISTRY_PATH, canonical_arm_map
from minicells.native_clm_m2 import sha256_file


DEFAULT_BRANCH = "codex/integrated-replay-free-clm-kill-test-001"
DEFAULT_OUTPUT = Path("artifacts/experiments/integrated-replay-free-clm-kill-test-001")
DEFAULT_HF_REPO = "archelabsxyz/native-clm-v0"


def run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    capture: bool = False,
    timeout: int = 180,
) -> str:
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


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_mode(seed: int) -> str:
    registry = _load(Path(SEED_REGISTRY_PATH))
    if seed in {int(value) for value in registry["development"]}:
        return "development"
    if seed in {int(value) for value in registry["formal"]}:
        return "formal"
    raise RuntimeError("KT001 publisher refuses unregistered seed")


def _verify_seed_evidence(seed_dir: Path, seed: int) -> dict[str, Any]:
    completed: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    missing: list[str] = []
    for arm in canonical_arm_map():
        arm_dir = seed_dir / arm
        summary_path = arm_dir / "arm-summary.json"
        failure_path = arm_dir / "failure.json"
        if summary_path.exists():
            summary = _load(summary_path)
            if int(summary.get("seed", -1)) != seed or summary.get("arm") != arm:
                raise RuntimeError(f"KT001 seed/arm identity mismatch: {summary_path}")
            completed[arm] = summary
        elif failure_path.exists():
            failure = _load(failure_path)
            if int(failure.get("seed", -1)) != seed or failure.get("arm") != arm:
                raise RuntimeError(f"KT001 failure identity mismatch: {failure_path}")
            failures[arm] = failure
        else:
            missing.append(arm)

    if not completed and not failures:
        raise RuntimeError("KT001 seed directory contains no publishable arm evidence")

    complete = len(completed) == len(canonical_arm_map()) and not failures and not missing
    decision = None
    decision_path = seed_dir / "seed-decision.json"
    if complete:
        if not decision_path.exists():
            raise FileNotFoundError(decision_path)
        decision = _load(decision_path)
        if int(decision.get("seed", -1)) != seed:
            raise RuntimeError("KT001 seed decision identity mismatch")
    elif decision_path.exists():
        raise RuntimeError("incomplete KT001 seed must not already contain a scientific seed decision")

    return {
        "complete": complete,
        "completed_arms": sorted(completed),
        "failed_arms": sorted(failures),
        "missing_arms": sorted(missing),
        "decision": decision,
    }


def _model_records(seed_dir: Path, seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for local in sorted(seed_dir.rglob("*.pt")):
        relative = local.relative_to(seed_dir)
        records.append(
            {
                "seed": seed,
                "local_path": str(local),
                "relative_path": str(relative),
                "path": f"kt001/seed-{seed}/{relative.as_posix()}",
                "sha256": sha256_file(local),
                "bytes": local.stat().st_size,
                "uploaded": False,
            }
        )
    return records


def _publish_hf(
    *,
    seed_dir: Path,
    seed: int,
    repo_id: str,
    token: str | None,
    require_upload: bool,
) -> dict[str, Any]:
    records = _model_records(seed_dir, seed)
    result = {
        "format": "minicells.kt001-model-artifacts.v1",
        "seed": seed,
        "repo_id": repo_id,
        "hf_upload_status": "NOT_ATTEMPTED",
        "resolved_revision_after_upload": None,
        "files": records,
        "error": None,
    }
    if not records:
        result["hf_upload_status"] = "NO_MODEL_ARTIFACTS"
        return result
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
            print(
                "Uploading KT001 seed={seed} -> {repo}/{path} ({mib:.1f} MiB)".format(
                    seed=seed,
                    repo=repo_id,
                    path=record["path"],
                    mib=record["bytes"] / 1024 / 1024,
                ),
                flush=True,
            )
            api.upload_file(
                path_or_fileobj=record["local_path"],
                path_in_repo=record["path"],
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"KT001 seed {seed} {record['relative_path']}",
            )
            record["uploaded"] = True
        result["resolved_revision_after_upload"] = api.model_info(repo_id).sha
        result["hf_upload_status"] = "PUBLISHED"
    except HfHubHTTPError as exc:
        result["hf_upload_status"] = "FAILED_DURING_UPLOAD"
        result["error"] = str(exc)
        if require_upload:
            raise RuntimeError("Hugging Face KT001 artifact upload failed") from exc
    return result


def _lightweight_files(seed_dir: Path) -> list[Path]:
    allowed_suffixes = {".json", ".jsonl", ".csv", ".md"}
    return [
        path
        for path in seed_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    ]


def _git_publish(seed_dir: Path, branch: str, github_token: str, seed: int) -> str:
    current = run(["git", "branch", "--show-current"], capture=True)
    if current != branch:
        raise RuntimeError(f"expected branch {branch}, got {current}")

    allowed = _lightweight_files(seed_dir)
    if not allowed:
        raise RuntimeError("no lightweight KT001 seed evidence found")
    run(["git", "reset"])
    run(["git", "add", "-f", "--", *[str(path) for path in sorted(allowed)]])
    staged = run(["git", "diff", "--cached", "--name-only"], capture=True).splitlines()
    forbidden = [path for path in staged if path.endswith(".pt")]
    if forbidden:
        raise RuntimeError("refusing to Git-publish KT001 checkpoints: " + ", ".join(forbidden))
    if staged:
        run(["git", "config", "user.name", "MiniCells Research"])
        run(["git", "config", "user.email", "research@minicells.local"])
        run(["git", "commit", "-m", f"research: publish KT001 seed {seed} evidence"])

    refspec = f"+refs/heads/{branch}:refs/remotes/origin/{branch}"
    run(["git", "fetch", "--no-tags", "origin", refspec], timeout=180)
    try:
        run(["git", "rebase", f"origin/{branch}"], timeout=120)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "rebase", "--abort"], check=False)
        raise

    askpass = Path("/tmp/minicells-kt001-askpass.sh")
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
                f"HEAD:{branch}",
            ],
            env=env,
            timeout=180,
        )
    finally:
        askpass.unlink(missing_ok=True)
    return run(["git", "rev-parse", "HEAD"], capture=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--checkpoint-provenance", type=Path)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--github-token-env", default="GITHUB_TOKEN")
    parser.add_argument("--require-hf-upload", action="store_true")
    args = parser.parse_args()

    mode = _seed_mode(args.seed)
    seed_dir = args.output_dir / f"seed-{args.seed}"
    evidence = _verify_seed_evidence(seed_dir, args.seed)

    if args.data_manifest is not None:
        shutil.copy2(args.data_manifest, seed_dir / "data-manifest.json")
    if args.checkpoint_provenance is not None:
        shutil.copy2(args.checkpoint_provenance, seed_dir / "m1-checkpoint-provenance.json")

    has_models = any(seed_dir.rglob("*.pt"))
    require_hf = bool(args.require_hf_upload or (mode == "formal" and has_models))
    model_artifacts = _publish_hf(
        seed_dir=seed_dir,
        seed=args.seed,
        repo_id=args.hf_repo,
        token=os.environ.get(args.hf_token_env),
        require_upload=require_hf,
    )
    (seed_dir / "model-artifacts.json").write_text(
        json.dumps(model_artifacts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if require_hf and model_artifacts["hf_upload_status"] != "PUBLISHED":
        raise RuntimeError("KT001 HF-first durability requirement was not satisfied")

    publication_state = {
        "format": "minicells.kt001-publication-state.v1",
        "seed": args.seed,
        "mode": mode,
        "complete": evidence["complete"],
        "completed_arms": evidence["completed_arms"],
        "failed_arms": evidence["failed_arms"],
        "missing_arms": evidence["missing_arms"],
        "scientific_aggregation_allowed": bool(evidence["complete"]),
        "classification": (
            evidence["decision"]["classification"]
            if evidence["decision"] is not None
            else "INCOMPLETE_FAILURE_EVIDENCE"
        ),
    }
    (seed_dir / "publication-state.json").write_text(
        json.dumps(publication_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    github_token = os.environ.get(args.github_token_env)
    if not github_token:
        raise RuntimeError(f"missing environment variable {args.github_token_env}")
    commit = _git_publish(seed_dir, args.branch, github_token, args.seed)
    print(
        json.dumps(
            {
                "published": True,
                "seed": args.seed,
                "mode": mode,
                "branch": args.branch,
                "commit": commit,
                "complete": evidence["complete"],
                "completed_arms": evidence["completed_arms"],
                "failed_arms": evidence["failed_arms"],
                "missing_arms": evidence["missing_arms"],
                "classification": publication_state["classification"],
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
