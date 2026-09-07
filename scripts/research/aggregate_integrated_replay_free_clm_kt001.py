"""Aggregate KT001 arm artifacts into per-seed and formal decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from minicells.integrated_replay_free_clm_kt001 import SEED_REGISTRY_PATH, canonical_arm_map
from minicells.integrated_replay_free_clm_kt001_aggregate import (
    aggregate_formal_seed_decisions,
    compare_seed,
)


EXPERIMENT_DIR = Path(
    "research/experiments/04-continual-learning-core/"
    "integrated-replay-free-clm-kill-test-001"
)
DECISION_PATH = EXPERIMENT_DIR / "DECISION.json"
IMPLEMENTATION_LOCK = EXPERIMENT_DIR / "IMPLEMENTATION_LOCK.json"
DEFAULT_OUTPUT = Path("artifacts/experiments/integrated-replay-free-clm-kill-test-001")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_seed(output_dir: Path, seed: int, decision: dict[str, Any]) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    arm_dirs: dict[str, Path] = {}
    for arm in canonical_arm_map():
        arm_dir = output_dir / f"seed-{seed}" / arm
        summary_path = arm_dir / "arm-summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = _load(summary_path)
        if int(summary.get("seed", -1)) != seed or summary.get("arm") != arm:
            raise RuntimeError(f"KT001 summary identity mismatch: seed={seed} arm={arm}")
        arms[arm] = summary
        arm_dirs[arm] = arm_dir

    result = compare_seed(arms, arm_dirs=arm_dirs, decision=decision)
    result["provenance"] = {
        "decision_sha256": _sha256(DECISION_PATH),
        "seed_registry_sha256": _sha256(Path(SEED_REGISTRY_PATH)),
    }
    seed_dir = output_dir / f"seed-{seed}"
    (seed_dir / "seed-decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _require_formal_lock() -> dict[str, Any]:
    if not IMPLEMENTATION_LOCK.exists():
        raise RuntimeError("KT001 formal aggregation requires IMPLEMENTATION_LOCK.json")
    lock = _load(IMPLEMENTATION_LOCK)
    if lock.get("status") != "SEALED_FOR_FORMAL_EXECUTION":
        raise RuntimeError("KT001 implementation is not sealed for formal aggregation")
    return lock


def aggregate_formal(output_dir: Path, decision: dict[str, Any]) -> dict[str, Any]:
    _require_formal_lock()
    registry = _load(Path(SEED_REGISTRY_PATH))
    formal_seeds = [int(seed) for seed in registry["formal"]]
    per_seed = [aggregate_seed(output_dir, seed, decision) for seed in formal_seeds]
    result = aggregate_formal_seed_decisions(per_seed, decision_protocol=decision)
    result["provenance"] = {
        "decision_sha256": _sha256(DECISION_PATH),
        "seed_registry_sha256": _sha256(Path(SEED_REGISTRY_PATH)),
        "implementation_lock_sha256": _sha256(IMPLEMENTATION_LOCK),
    }
    target = output_dir / "FORMAL_DECISION.json"
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()

    decision = _load(DECISION_PATH)
    if decision.get("format") != "minicells.kt001-decision.v1":
        raise RuntimeError("unexpected KT001 decision protocol")

    if args.formal:
        if args.seed is not None:
            raise RuntimeError("--formal aggregates the frozen formal registry; omit --seed")
        result = aggregate_formal(args.output_dir, decision)
    else:
        registry = _load(Path(SEED_REGISTRY_PATH))
        seed = int(args.seed if args.seed is not None else registry["development"][0])
        if seed not in {int(value) for value in registry["development"]}:
            raise RuntimeError("development aggregation requires a registered development seed")
        result = aggregate_seed(args.output_dir, seed, decision)

    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
