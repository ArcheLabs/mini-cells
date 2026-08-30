#!/usr/bin/env python3
"""Publish curated Core Validation 003 formal results."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from publish_experiment_results import DEFAULT_SECRET_NAME, push_results, repo_root

SOURCE = Path("results/core-validation-003-dependency-scoped-transactional-learning")
DEST = Path("artifacts/experiments/core-validation-003-dependency-scoped-transactional-learning")
BRANCH = "kaggle/core-validation-003-dependency-scoped-transactional-learning-results"
FORMAT = "minicells.core-validation.dependency-scoped-transactional-learning.v1"
FILES = (
    "raw.json", "decision.json", "transaction-records.csv", "seed-summary.csv",
    "gate-summary.csv", "scope-safety-frontier.png", "granularity-scope-acceptance.png",
    "cost-per-accepted-update.png", "transactional-tradeoff.png",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    args = parser.parse_args()

    root = repo_root()
    source = root / SOURCE
    destination = root / DEST
    raw = json.loads((source / "raw.json").read_text(encoding="utf-8"))
    decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    if raw.get("format") != FORMAT or decision.get("format") != FORMAT:
        raise RuntimeError("unexpected Core Validation 003 format")
    if raw.get("mode") != "formal" or decision.get("scientific_decision") is not True:
        raise RuntimeError("refusing to publish a non-formal result")
    if raw.get("provenance", {}).get("tracked_tree_dirty") is not False:
        raise RuntimeError("refusing publication from dirty tracked provenance")
    allowed = {
        "DEPENDENCY_SCOPED_TRANSACTIONAL_LEARNING_SUPPORTED",
        "DEPENDENCY_SCOPED_TRANSACTIONAL_LEARNING_NOT_SUPPORTED",
    }
    if decision.get("status") not in allowed:
        raise RuntimeError("unexpected Core Validation 003 status")
    frozen = set(raw.get("research_transition", {}).get("frozen_prior_outcomes", []))
    if frozen != {
        "WRITE_ADDRESSABILITY_NOT_SUPPORTED",
        "SPARSE_WRITE_ASSEMBLY_NOT_SUPPORTED",
        "ORACLE_SPARSE_ASSEMBLY_NOT_SUPPORTED",
    }:
        raise RuntimeError("003 must preserve frozen 002-series outcomes")

    missing = [name for name in FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing Core Validation 003 outputs: {missing}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in FILES:
        shutil.copy2(source / name, destination / name)
    shutil.copy2(root / "research/validations/core-003-dependency-scoped-transactional-learning/protocol.json", destination / "protocol.json")
    (destination / "RESULTS.md").write_text(
        "# Core Validation 003 Results\n\n"
        f"- Status: `{decision.get('status')}`\n"
        f"- Shared supported granularities: `{decision.get('shared_supported_granularities')}`\n"
        "- Hidden full-history validation is evaluator-only.\n",
        encoding="utf-8",
    )
    print(f"Prepared {destination.relative_to(root)}")
    if args.push:
        push_results(root, destination, "core-validation-003", args.branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
