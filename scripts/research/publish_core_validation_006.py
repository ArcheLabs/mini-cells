#!/usr/bin/env python3
"""Publish Core Validation 006 formal outputs into canonical artifacts."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "core-validation-006-real-representation-continual-plasticity"
DEFAULT_ARTIFACTS = (
    ROOT / "artifacts" / "experiments" / "core-validation-006-real-representation-continual-plasticity"
)
PROTOCOL = (
    ROOT
    / "research"
    / "validations"
    / "core-006-real-representation-continual-plasticity"
    / "protocol.json"
)
EXCLUDE = {"frozen-hidden.pt", "frozen-hidden-smoke.pt"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = p.parse_args()
    decision_path = args.results / "decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError(decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("mode") != "formal" or not decision.get("scientific_decision"):
        raise RuntimeError("only formal scientific outputs may be published")
    if args.artifacts.exists():
        shutil.rmtree(args.artifacts)
    args.artifacts.mkdir(parents=True)
    for src in args.results.iterdir():
        if src.name in EXCLUDE:
            continue
        if src.is_file():
            shutil.copy2(src, args.artifacts / src.name)
    shutil.copy2(PROTOCOL, args.artifacts / "protocol.json")
    print(args.artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
