from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "datasets" / "jam-knowledge-v0.1"
SCRIPT_DIR = ROOT / "scripts" / "research" / "jam_knowledge_v0_1"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_registry_valid() -> None:
    validator = load_module(SCRIPT_DIR / "validate_dataset.py", "jam_validate")
    result = validator.validate(DATASET)
    assert result == {
        "status": "JAM_KNOWLEDGE_V0_1_VALID",
        "concepts": 180,
        "reasoning_holdout": 50,
        "relations": 66,
        "misconceptions": 49,
    }


def test_deterministic_materialization(tmp_path: Path) -> None:
    builder = load_module(SCRIPT_DIR / "build_dataset.py", "jam_build")
    validator = load_module(SCRIPT_DIR / "validate_dataset.py", "jam_validate_materialized")
    target = tmp_path / "jam-knowledge-v0.1"
    shutil.copytree(DATASET, target, ignore=shutil.ignore_patterns("generated"))
    counts = builder.build(target)
    assert counts == {
        "concepts": 180,
        "train": 409,
        "validation": 180,
        "factual": 180,
        "relational": 66,
        "misconceptions": 49,
        "reasoning": 50,
    }
    assert validator.validate(target)["status"] == "JAM_KNOWLEDGE_V0_1_VALID"
