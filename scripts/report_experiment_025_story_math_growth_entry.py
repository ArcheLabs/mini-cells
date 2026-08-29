#!/usr/bin/env python3
"""Stable CLI entrypoint for Experiment-025 reporting."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "report_experiment_025_story_math_growth.py"


def _load_report():
    spec = importlib.util.spec_from_file_location("experiment_025_report_impl", TARGET)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment-025 report: {TARGET}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Report Experiment 025")
    result.add_argument("--results-dir", type=Path, required=True)
    return result


def main() -> int:
    module = _load_report()
    module.parser = _parser
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
