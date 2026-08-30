#!/usr/bin/env python3
"""Check research catalog paths, IDs, stages, and frozen validation outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    catalog = yaml.safe_load((ROOT / "research/catalog.yaml").read_text(encoding="utf-8"))
    stages = set(catalog["stages"])
    entries = catalog["entries"]
    ids = [entry["id"] for entry in entries]
    assert len(ids) == len(set(ids)), "duplicate catalog IDs"

    checked = 0
    for entry in entries:
        assert entry["stage"] in stages, f"invalid stage: {entry['id']}"
        for field in ("report", "protocol", "artifacts", "notebook", "implementation"):
            value = entry.get(field)
            if value:
                assert (ROOT / value).exists(), f"missing {field} for {entry['id']}: {value}"
        if entry["kind"] != "core-validation":
            continue
        decision_path = ROOT / entry["artifacts"] / "decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        actual = decision.get("status")
        assert actual == entry["outcome"], (
            f"outcome mismatch for {entry['id']}: catalog={entry['outcome']} decision={actual}"
        )
        checked += 1
    print(f"research catalog: OK ({len(entries)} entries, {checked} formal outcomes checked)")


if __name__ == "__main__":
    main()
