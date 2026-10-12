#!/usr/bin/env python3
"""Thin repository CLI for immutable Experiment 005 evidence publication."""
from __future__ import annotations

import runpy
from pathlib import Path


ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "pyproject.toml").is_file())
ENTRY = (ROOT / "research/08-hybrid-clm/experiments/005-compositional-cell-graph/"
         "implementation/exp005/publish.py")
runpy.run_path(str(ENTRY), run_name="__main__")
