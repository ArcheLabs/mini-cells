#!/usr/bin/env python3
"""Check repository architecture without executing scientific experiments."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"architecture: ERROR: {message}")


def main() -> None:
    require(not (ROOT / "research/minicells").exists(), "research/minicells still exists")
    require((ROOT / "src/minicells").is_dir(), "src/minicells is missing")
    for path in (ROOT / "src/minicells").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("research"):
                require(False, f"reusable package imports research: {path}")
            if isinstance(node, ast.Import):
                require(
                    not any(alias.name.startswith("research") for alias in node.names),
                    f"reusable package imports research: {path}",
                )
    require(
        not list((ROOT / "docs").glob("experiment-*.md")),
        "top-level historical experiment docs remain",
    )
    require(
        not list((ROOT / "scripts").glob("run_core_validation_*.py")),
        "top-level Core Validation runners remain",
    )
    project_root = ROOT / "research/projects"
    projects = sorted(path for path in project_root.iterdir() if path.is_dir())
    required = {"cellular-computing", "continual-learning", "cellular-language-models", "safe-model-evolution"}
    require({path.name for path in projects} == required, "research project roots differ from the project catalog")
    for project in projects:
        require((project / "README.md").is_file(), f"missing project README: {project.name}")
        require((project / "experiments").is_dir(), f"missing experiment units: {project.name}")
    obsolete_roots = ("experiments", "kaggle", "notebooks", "research/01-foundations",
                      "research/02-self-organization", "research/03-routing-and-growth",
                      "research/04-continual-learning", "research/05-language-validation",
                      "research/06-native-clm", "research/07-safe-model-evolution",
                      "research/08-hybrid-clm")
    for obsolete in obsolete_roots:
        require(not (ROOT / obsolete).exists(), f"obsolete root path remains: {obsolete}")
    for directory, names, files in os.walk(ROOT, topdown=True, followlinks=False):
        names[:] = [name for name in names if name not in {".git", ".venv", "node_modules"}]
        for name in names + files:
            require(not (Path(directory) / name).is_symlink(), "repository contains a symlink")
    notebooks = list(project_root.rglob("*.ipynb")) + list((ROOT / "docs/tutorials").rglob("*.ipynb"))
    for path in notebooks:
        json.loads(path.read_text(encoding="utf-8"))
    print(f"architecture: OK ({len(notebooks)} project-owned notebooks indexed)")


if __name__ == "__main__":
    main()
