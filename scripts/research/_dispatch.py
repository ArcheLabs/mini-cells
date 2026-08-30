"""Explicit registry for unified research entrypoints."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = {
    "run": {
        "core-validation-001": "run_core_validation_001.py",
        "core-validation-002": "run_core_validation_002.py",
        "core-validation-002b": "run_core_validation_002b.py",
        "core-validation-002c": "run_core_validation_002c.py",
        "core-validation-003": "run_core_validation_003.py",
        "core-validation-004": "run_core_validation_004.py",
        "clm-0.4-mini-m0": "run_clm_0_4_mini_m0.py",
    },
    "report": {
        "core-validation-001": "report_core_validation_001.py",
        "core-validation-001b": "report_core_validation_001b.py",
        "core-validation-002": "report_core_validation_002.py",
        "core-validation-002b": "report_core_validation_002b.py",
        "core-validation-002c": "report_core_validation_002c.py",
        "core-validation-003": "report_core_validation_003.py",
        "core-validation-004": "report_core_validation_004.py",
        "clm-0.4-mini-m0": "report_clm_0_4_mini_m0.py",
    },
    "publish": {
        "core-validation-001": "publish_core_validation_001.py",
        "core-validation-001b": "publish_core_validation_001b.py",
        "core-validation-002": "publish_core_validation_002.py",
        "core-validation-002b": "publish_core_validation_002b.py",
        "core-validation-002c": "publish_core_validation_002c.py",
        "core-validation-003": "publish_core_validation_003.py",
        "core-validation-004": "publish_core_validation_004.py",
    },
}


def dispatch(kind: str, argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in REGISTRY[kind]:
        available = ", ".join(sorted(REGISTRY[kind]))
        raise SystemExit(f"usage: {kind}.py <experiment-id> [args...]\navailable: {available}")
    experiment_id = args.pop(0)
    script = Path(__file__).with_name(REGISTRY[kind][experiment_id])
    sys.argv = [str(script), *args]
    runpy.run_path(str(script), run_name="__main__")
