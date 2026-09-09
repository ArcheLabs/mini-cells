#!/usr/bin/env python3
"""Check the HybridCLM release notebook's safety and orchestration contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FORBIDDEN = ("GITHUB_TOKEN", "github_pat_", "ghp_", "PYPI_API_TOKEN", "print(token)")
REQUIRED = ("check_release_identity.py", "export_hybrid_clm_mutation.py", "prepare_hybrid_clm_release.py", "validate_hybrid_clm_publication.py", "render_hybrid_clm_release_assets.py")


def check(path: Path) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError("notebook metadata is invalid")
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    missing = [name for name in REQUIRED if name not in source]
    if missing:
        raise ValueError(f"notebook does not invoke canonical release scripts: {missing}")
    if "RELEASE_TAG = 'v0.2.0a1'" not in source or "PUBLISH = False" not in source:
        raise ValueError("notebook release identity or publication latch is missing")
    for token in FORBIDDEN:
        if token in source:
            raise ValueError(f"forbidden credential pattern in notebook: {token}")
    if "formal seed" not in source.lower() or "formal" not in source.lower():
        raise ValueError("notebook is missing the formal boundary statement")
    return {"status": "NOTEBOOK_CONTRACT_VALID", "path": str(path), "cells": len(notebook["cells"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.path), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
