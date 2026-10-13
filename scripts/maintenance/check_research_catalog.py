#!/usr/bin/env python3
"""Check research catalog paths, IDs, stages, and frozen validation outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    catalog = yaml.safe_load((ROOT / "research/governance/catalog.yaml").read_text(encoding="utf-8"))
    assert catalog["schema"] == "minicells.research-project-catalog.v1"
    projects = catalog["projects"]
    project_ids = [project["id"] for project in projects]
    assert len(project_ids) == len(set(project_ids)), "duplicate research project IDs"
    project_roots = set()
    for project in projects:
        root = ROOT / project["directory"]
        assert root.is_dir() and (root / "README.md").is_file(), f"incomplete project: {project['id']}"
        project_roots.add(root.resolve())
        for stage in project.get("historical_context", []):
            assert (root / "context" / stage).is_dir(), f"missing project context: {project['id']}/{stage}"

    registry = json.loads((ROOT / "research/governance/registry.yaml").read_text(encoding="utf-8"))
    for experiment in registry["experiments"]:
        directory = ROOT / experiment["directory"]
        assert directory.is_dir(), f"missing experiment directory: {experiment['id']}"
        assert any(root in directory.resolve().parents for root in project_roots), (
            f"experiment has no project owner: {experiment['id']}"
        )

    # Keep checking historical frozen outcomes against their original decisions.
    history = yaml.safe_load((ROOT / "research/governance/historical-stage-catalog.yaml").read_text(encoding="utf-8"))
    stages = set(history["stages"])
    checked = 0
    ids = [entry["id"] for entry in history["entries"]]
    assert len(ids) == len(set(ids)), "duplicate historical catalog IDs"
    for entry in history["entries"]:
        assert entry["stage"] in stages, f"invalid historical stage: {entry['id']}"
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
    print(f"research catalog: OK ({len(projects)} projects, {len(registry['experiments'])} experiments, {checked} frozen outcomes)")

if __name__ == "__main__":
    main()
