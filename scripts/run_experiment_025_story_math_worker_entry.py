#!/usr/bin/env python3
"""Stable CLI entrypoint for the Experiment-025 GPU worker.

The original worker predates the one-shot Kaggle orchestrator and its ``parser``
helper returns a parsed Namespace while ``main`` expects an ArgumentParser.  Keep
the frozen training implementation untouched and adapt only the CLI boundary.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "run_experiment_025_story_math_worker.py"


def _load_worker():
    spec = importlib.util.spec_from_file_location("experiment_025_worker_impl", TARGET)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment-025 worker: {TARGET}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser(module) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run one Experiment-025 GPU arm")
    result.add_argument("--arm", choices=("llm", "clm"), required=True)
    result.add_argument("--story-cache-dir", type=Path, required=True)
    result.add_argument("--math-cache-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--shift-tokens", type=int, default=module.SHIFT_TOKENS)
    result.add_argument("--max-wall-hours", type=float, default=module.DEFAULT_MAX_WALL_HOURS)
    result.add_argument("--reset", action="store_true")
    return result


def main() -> int:
    module = _load_worker()
    module.parser = lambda: _parser(module)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
