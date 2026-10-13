#!/usr/bin/env python3
"""Thin repository CLI for the HybridCLM-owned Experiment 005 analyzer."""
from __future__ import annotations

import runpy
from pathlib import Path


ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "pyproject.toml").is_file())
ENTRY = (ROOT / "research/projects/cellular-language-models/experiments/005-compositional-cell-graph/"
         "implementation/exp005/analyze_experiment_005.py")
runpy.run_path(str(ENTRY), run_name="__main__")
