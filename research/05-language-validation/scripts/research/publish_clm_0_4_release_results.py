#!/usr/bin/env python3
"""Publish curated CLM-0.4 Release smoke or 30M evidence to Git."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from publish_experiment_results import DEFAULT_SECRET_NAME, push_results, repo_root


ALLOWED_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".png"}
EXCLUDED_NAMES = {"checkpoint.pt", "latest.pt", "tokenizer.json"}
EXCLUDED_PARTS = {"checkpoints", "base-corpus", "tokenizer"}
PROFILE_TARGETS = {
    "smoke-1m": (
        "artifacts/releases/clm-0.4/smoke-1m/latest",
        "kaggle/clm-0.4-release-1m-smoke-results",
    ),
    "release-30m": (
        "artifacts/releases/clm-0.4/30m/latest",
        "kaggle/clm-0.4-release-30m-results",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    args = parser.parse_args()

    root = repo_root()
    source = args.results.resolve()
    decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    if decision.get("release_track") != "release":
        raise RuntimeError("refusing non-Release result directory")
    profile = str(decision.get("profile"))
    if profile not in PROFILE_TARGETS:
        raise RuntimeError(f"unknown Release profile: {profile}")
    artifact_relative, default_branch = PROFILE_TARGETS[profile]
    branch = args.branch or default_branch
    destination = root / artifact_relative
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    copied = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if path.name in EXCLUDED_NAMES:
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append({
            "path": relative.as_posix(),
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
        })
    if not copied:
        raise RuntimeError("no Release evidence found")
    if profile == "smoke-1m":
        readiness = json.loads((source / "release-readiness.json").read_text(encoding="utf-8"))
        if readiness.get("status") != "READY_FOR_30M":
            raise RuntimeError("refusing to publish an incomplete 1M release smoke")
    metadata = {
        "format": "minicells.clm-0.4-release-publication.v1",
        "profile": profile,
        "status": decision.get("status"),
        "seed": decision.get("seed"),
        "transactions": decision.get("transactions"),
        "pipeline_sha256": decision.get("pipeline_sha256"),
        "source_fingerprint_sha256": decision.get("source_fingerprint_sha256"),
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "kaggle_script_version_id": args.kaggle_script_version_id,
        "source_results": str(source),
        "branch": branch,
        "excluded": ["model checkpoints", "raw token shards", "tokenizer payload"],
        "files": copied,
    }
    (destination / "publication-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared Release evidence: {destination.relative_to(root)}")
    print(f"Profile: {profile}")
    print(f"Files: {len(copied)}")
    if args.push:
        push_results(root, destination, f"clm-0.4-release-{profile}", branch, args.secret_name)
    else:
        print("Not pushed. Re-run with --push after inspecting curated evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
