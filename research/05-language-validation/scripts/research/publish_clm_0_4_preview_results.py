#!/usr/bin/env python3
"""Publish curated CLM-0.4 Preview telemetry and visualizations to Git."""

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
DEFAULT_ARTIFACT = "artifacts/previews/clm-0.4-preview/latest"
DEFAULT_BRANCH = "kaggle/clm-0.4-preview-results"


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
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    args = parser.parse_args()

    root = repo_root()
    source = args.results.resolve()
    decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    if decision.get("release_track") != "preview":
        raise RuntimeError("refusing non-Preview result directory")
    destination = root / DEFAULT_ARTIFACT
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
        raise RuntimeError("no Preview telemetry found")
    metadata = {
        "format": "minicells.clm-0.4-preview-publication.v1",
        "status": decision.get("status"),
        "seed": decision.get("seed"),
        "transactions": decision.get("transactions"),
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "kaggle_script_version_id": args.kaggle_script_version_id,
        "source_results": str(source),
        "branch": args.branch,
        "excluded": ["model checkpoints", "raw 30M token shards", "tokenizer payload"],
        "files": copied,
    }
    (destination / "publication-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared Preview evidence: {destination.relative_to(root)}")
    print(f"Files: {len(copied)}")
    if args.push:
        push_results(root, destination, "clm-0.4-preview", args.branch, args.secret_name)
    else:
        print("Not pushed. Re-run with --push after inspecting curated evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
