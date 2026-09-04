from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = (
    ROOT
    / "scripts"
    / "research"
    / "functional_boundary_oracle_001"
    / "kaggle_preflight.py"
)


def _load_preflight_module():
    spec = importlib.util.spec_from_file_location("functional_boundary_oracle_kaggle_preflight", PREFLIGHT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_accepts_canonical_and_legacy_memory_flags():
    module = _load_preflight_module()
    parser = module._build_parser()

    canonical = parser.parse_args(["--minimum-free-mb", "12345"])
    legacy = parser.parse_args(["--min-free-mib", "12345"])

    assert canonical.minimum_free_mb == 12345
    assert legacy.minimum_free_mb == 12345
