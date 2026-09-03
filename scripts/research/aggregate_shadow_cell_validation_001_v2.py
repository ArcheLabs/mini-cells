#!/usr/bin/env python3
"""Aggregate only the preregistered formal Shadow v2 seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

FORMAL_SEEDS = (95311, 95312, 95313)
DEVELOPMENT_SEED = 95301
TAXONOMY = (
    "SHADOW_MATURATION_SUPPORTED",
    "SHADOW_MATURATION_ORACLE_ONLY",
    "SHADOW_ISOLATION_SUPPORTED_MATURATION_NOT_NECESSARY",
    "SHADOW_MATURATION_NOT_SUPPORTED",
    "INCONCLUSIVE",
    "INVALID",
)


def _phase_pass(row: dict, threshold: float) -> bool:
    frontier = row.get("maturity_frontier", [])
    selected = row.get("selected_maturity")
    if selected is None:
        return False
    point = next((item for item in frontier if item.get("maturity") == selected), None)
    return point is not None and float(point.get("old_regression", float("inf"))) <= threshold and float(point.get("new_gain", float("-inf"))) >= 0.01


def classify(results: list[dict], *, max_old_regression: float = 0.2) -> str:
    if any(not result.get("validity", {}).get("required_arms_completed", False) or not result.get("validity", {}).get("m0_identity_passes", False) for result in results):
        return "INVALID"
    arm_passes = _arm_passes(results, max_old_regression=max_old_regression)
    for arm in ("shadow_full", "shadow_oracle", "shadow_sketch", "task_id_shadow"):
        arm_passes[arm] = all(_phase_pass(result["arms"][arm][phase], max_old_regression) for result in results for phase in ("B", "C", "D"))
    if not arm_passes["task_id_shadow"]:
        return "SHADOW_MATURATION_NOT_SUPPORTED"
    if not arm_passes["shadow_oracle"]:
        return "SHADOW_MATURATION_NOT_SUPPORTED"
    if arm_passes["shadow_sketch"] and not arm_passes["shadow_full"]:
        return "SHADOW_MATURATION_SUPPORTED"
    if arm_passes["shadow_oracle"] and not arm_passes["shadow_sketch"]:
        return "SHADOW_MATURATION_ORACLE_ONLY"
    if arm_passes["shadow_full"]:
        return "SHADOW_ISOLATION_SUPPORTED_MATURATION_NOT_NECESSARY"
    return "INCONCLUSIVE"


def _arm_passes(results: list[dict], *, max_old_regression: float) -> dict[str, bool]:
    return {
        arm: all(
            _phase_pass(result["arms"][arm][phase], max_old_regression)
            for result in results
            for phase in ("B", "C", "D")
        )
        for arm in ("shadow_full", "shadow_oracle", "shadow_sketch", "task_id_shadow")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.results_root / f"seed-{seed}" / "result.json" for seed in FORMAL_SEEDS]
    dev_path = args.results_root / f"seed-{DEVELOPMENT_SEED}" / "result.json"
    if dev_path.exists():
        raise SystemExit("formal aggregation rejects development seed 95301")
    if not all(path.is_file() for path in paths):
        payload = {"status": "INCOMPLETE", "scientific_decision": False, "formal_seeds": list(FORMAL_SEEDS), "missing": [str(path) for path in paths if not path.is_file()]}
    else:
        results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        hashes = {result.get("protocol_sha256") for result in results}
        if len(hashes) != 1:
            payload = {"status": "INVALID", "scientific_decision": False, "reason": "formal results have mixed protocol hashes"}
        else:
            payload = {"status": classify(results), "scientific_decision": True, "formal_seeds": list(FORMAL_SEEDS), "protocol_sha256": hashes.pop(), "arm_passes": _arm_passes(results, max_old_regression=0.2)}
    args.results_root.mkdir(parents=True, exist_ok=True)
    (args.results_root / "aggregate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
