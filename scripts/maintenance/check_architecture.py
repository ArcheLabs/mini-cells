#!/usr/bin/env python3
"""Check repository architecture without executing scientific experiments."""

from __future__ import annotations

import ast
import json
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
    for stage in (
        "01-foundations",
        "02-self-organization",
        "03-routing-and-growth",
        "04-continual-learning-core",
        "05-language-validation",
    ):
        require(
            (ROOT / "research/notebooks" / stage).is_dir(),
            f"missing notebook stage {stage}",
        )
    notebooks = list((ROOT / "research/notebooks").rglob("*.ipynb"))
    for path in notebooks:
        json.loads(path.read_text(encoding="utf-8"))
    print(f"architecture: OK ({len(notebooks)} notebooks indexed)")


if __name__ == "__main__":
    main()
