#!/usr/bin/env python3
"""Publish curated CLM-0.4-mini calibration evidence from Kaggle.

The publisher accepts an external results directory, copies only analysis-sized
text/table/image evidence into Git, excludes checkpoints and raw data, then can
push a dedicated result branch using the existing Kaggle GitHub-token flow.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from publish_experiment_results import (
    DEFAULT_SECRET_NAME,
    push_results,
    repo_root,
)


MODES = {
    "clm-0.4-mini-m1-v1-calibration": {
        "seed": 90401,
        "artifact": "artifacts/experiments/clm-0.4-mini-m1-v1-calibration/dev-90401",
        "branch": "kaggle/clm-0.4-mini-m1-v1-calibration-90401-results",
        "formats": {
            None,
            "minicells.clm-0.4-mini.m1-calibration.v1",
        },
        "protocol": "research/validations/clm-0.4-mini-language-validation/protocol.json",
    },
    "clm-0.4-mini-m1-v2-calibration": {
        "seed": 90402,
        "artifact": "artifacts/experiments/clm-0.4-mini-m1-v2-calibration/dev-90402",
        "branch": "kaggle/clm-0.4-mini-m1-v2-90402-results",
        "formats": {
            "minicells.clm-0.4-mini.m1-v2-calibration.v1",
        },
        "protocol": "research/validations/clm-0.4-mini-m1-v2-language-validation/protocol.json",
    },
}

ALLOWED_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".png"}
EXCLUDED_NAMES = {
    "checkpoint.pt",
    "tokenizer.json",
}
EXCLUDED_PARTS = {
    "checkpoints",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _curated_files(source: Path) -> list[Path]:
    result = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if path.name in EXCLUDED_NAMES:
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        result.append(path)
    return sorted(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    args = parser.parse_args()

    root = repo_root()
    spec = MODES[args.mode]
    source = args.results.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    decision_path = source / "decision.json"
    summary_path = source / "summary.json"
    if not decision_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("calibration decision.json/summary.json required")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if decision.get("scientific_decision") is not False:
        raise RuntimeError("refusing calibration output marked as scientific")
    if bool(decision.get("formal_seeds_observed")):
        raise RuntimeError("refusing result that reports formal seed observation")
    expected_seed = int(spec["seed"])
    observed_seed = summary.get("seed")
    if decision.get("development_seed_observed"):
        if int(observed_seed) != expected_seed:
            raise RuntimeError(
                f"development result seed {observed_seed!r} != expected {expected_seed}"
            )
    decision_format = decision.get("format")
    if decision_format not in spec["formats"]:
        raise RuntimeError(
            f"unexpected decision format for {args.mode}: {decision_format!r}"
        )

    destination = root / str(spec["artifact"])
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    files = _curated_files(source)
    if not files:
        raise RuntimeError("no curated calibration evidence found")
    copied = []
    for path in files:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(
            {
                "path": relative.as_posix(),
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )
    protocol_source = root / str(spec["protocol"])
    shutil.copy2(protocol_source, destination / "protocol.json")
    copied.append(
        {
            "path": "protocol.json",
            "bytes": (destination / "protocol.json").stat().st_size,
            "sha256": _sha256(destination / "protocol.json"),
        }
    )
    metadata = {
        "format": "minicells.clm-calibration-publication.v1",
        "mode": args.mode,
        "development_seed": expected_seed,
        "status": decision.get("status"),
        "scientific_decision": False,
        "formal_seeds_observed": False,
        "source_results": str(source),
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "kaggle_script_version_id": args.kaggle_script_version_id,
        "excluded": [
            "model checkpoints",
            "30M-token raw/base shards",
            "tokenizer binary/json payload",
            "regenerable checkpoint directories",
        ],
        "files": copied,
    }
    (destination / "publication-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared curated evidence: {destination.relative_to(root)}")
    print(f"Files: {len(copied)}")
    if args.push:
        push_results(
            root,
            destination,
            args.mode,
            args.branch or str(spec["branch"]),
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push after inspecting curated evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
