"""Explicit registry for unified research/product entrypoints."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = next(_root for _root in Path(__file__).resolve().parents if (_root / ".git").exists() or (_root / "pyproject.toml").exists())
REGISTRY = {
    "run": {
        "core-validation-001": "research/04-continual-learning/scripts/research/run_core_validation_001.py",
        "routing-002": "research/08-hybrid-clm/experiments/002-cell-internal-router/implementation/cell_internal_routing_002/train.py",
        "core-validation-002": "research/04-continual-learning/scripts/research/run_core_validation_002.py",
        "core-validation-002b": "research/04-continual-learning/scripts/research/run_core_validation_002b.py",
        "core-validation-002c": "research/04-continual-learning/scripts/research/run_core_validation_002c.py",
        "core-validation-003": "research/04-continual-learning/scripts/research/run_core_validation_003.py",
        "core-validation-004": "research/04-continual-learning/scripts/research/run_core_validation_004.py",
        "clm-0.4-mini-m0": "research/05-language-validation/scripts/research/run_clm_0_4_mini_m0.py",
        "clm-0.4-mini-m1": "research/05-language-validation/scripts/research/run_clm_0_4_mini_m1.py",
        "clm-0.4-mini-calibration": "research/05-language-validation/scripts/research/run_clm_0_4_mini_calibration.py",
        "clm-0.4-mini-m1-v2-calibration": "research/05-language-validation/scripts/research/run_clm_0_4_mini_v2_calibration.py",
        "clm-0.4-preview": "research/05-language-validation/scripts/research/run_clm_0_4_preview.py",
        "clm-0.4-release": "research/05-language-validation/scripts/research/run_clm_0_4_release.py",
    },
    "report": {
        "core-validation-001": "research/04-continual-learning/scripts/research/report_core_validation_001.py",
        "core-validation-001b": "research/04-continual-learning/scripts/research/report_core_validation_001b.py",
        "routing-002": "research/08-hybrid-clm/experiments/002-cell-internal-router/implementation/cell_internal_routing_002/report.py",
        "core-validation-002": "research/04-continual-learning/scripts/research/report_core_validation_002.py",
        "core-validation-002b": "research/04-continual-learning/scripts/research/report_core_validation_002b.py",
        "core-validation-002c": "research/04-continual-learning/scripts/research/report_core_validation_002c.py",
        "core-validation-003": "research/04-continual-learning/scripts/research/report_core_validation_003.py",
        "core-validation-004": "research/04-continual-learning/scripts/research/report_core_validation_004.py",
        "clm-0.4-mini-m0": "research/05-language-validation/scripts/research/report_clm_0_4_mini_m0.py",
        "clm-0.4-mini-m1": "research/05-language-validation/scripts/research/report_clm_0_4_mini_m1.py",
        "clm-0.4-mini-calibration": "research/05-language-validation/scripts/research/report_clm_0_4_mini_calibration.py",
        "clm-0.4-mini-m1-v2-calibration": "research/05-language-validation/scripts/research/report_clm_0_4_mini_v2_calibration.py",
        "clm-0.4-preview": "research/05-language-validation/scripts/research/report_clm_0_4_preview.py",
        "clm-0.4-release": "research/05-language-validation/scripts/research/report_clm_0_4_release.py",
    },
    "publish": {
        "core-validation-001": "research/04-continual-learning/scripts/research/publish_core_validation_001.py",
        "routing-002": "research/08-hybrid-clm/experiments/002-cell-internal-router/implementation/cell_internal_routing_002/publish.py",
        "core-validation-001b": "research/04-continual-learning/scripts/research/publish_core_validation_001b.py",
        "core-validation-002": "research/04-continual-learning/scripts/research/publish_core_validation_002.py",
        "core-validation-002b": "research/04-continual-learning/scripts/research/publish_core_validation_002b.py",
        "core-validation-002c": "research/04-continual-learning/scripts/research/publish_core_validation_002c.py",
        "core-validation-003": "research/04-continual-learning/scripts/research/publish_core_validation_003.py",
        "core-validation-004": "research/04-continual-learning/scripts/research/publish_core_validation_004.py",
        "clm-0.4-mini-m1-v1-calibration": "research/05-language-validation/scripts/research/publish_clm_0_4_mini_calibration_results.py",
        "clm-0.4-mini-m1-v2-calibration": "research/05-language-validation/scripts/research/publish_clm_0_4_mini_calibration_results.py",
        "clm-0.4-preview": "research/05-language-validation/scripts/research/publish_clm_0_4_preview_results.py",
        "clm-0.4-release": "research/05-language-validation/scripts/research/publish_clm_0_4_release_results.py",
    },
}


def dispatch(kind: str, argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in REGISTRY[kind]:
        available = ", ".join(sorted(REGISTRY[kind]))
        raise SystemExit(
            f"usage: {kind}.py <experiment-id> [args...]\navailable: {available}"
        )
    experiment_id = args.pop(0)
    script = (ROOT / REGISTRY[kind][experiment_id]).resolve()
    if kind == "publish" and script.name == "publish_clm_0_4_mini_calibration_results.py":
        args.insert(0, experiment_id)
    sys.argv = [str(script), *args]
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(script), run_name="__main__")
