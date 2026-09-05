#!/usr/bin/env python3
"""Assemble two PCU branch artifacts for functional runtime composition."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minicells.pcu_kill_001.registry import CellRegistry, merge_registries, validate_fork_artifacts  # noqa: E402


def _registry_path(value: Path) -> Path:
    return value / "CELL_REGISTRY.json" if value.is_dir() else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--branch-a", type=Path, required=True)
    parser.add_argument("--branch-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = CellRegistry.load(str(_registry_path(args.base)))
    branch_a = CellRegistry.load(str(_registry_path(args.branch_a)))
    branch_b = CellRegistry.load(str(_registry_path(args.branch_b)))
    validate_fork_artifacts(branch_a)
    validate_fork_artifacts(branch_b)
    merged = merge_registries(base, branch_a, branch_b)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save(str(args.output / "CELL_REGISTRY.json"))
    (args.output / "MERGE_MANIFEST.json").write_text(
        '{\n  "schema": "minicells.pcu-kill-001.merge-manifest.v2",\n  "operation": "functional_cell_delta_sum",\n  "runtime": "compose_cellular_experts",\n  "tensor_averaging": false,\n  "artifact_bindings_validated": true,\n  "registry_sha256": "' + merged.content_hash() + '"\n}\n',
        encoding="utf-8",
    )
    print(merged.content_hash())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
