#!/usr/bin/env python3
"""Validate and aggregate the preregistered formal Shadow v2 seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

FORMAL_SEEDS = (95311, 95312, 95313)
VALIDATION_ID = "shadow-cell-validation-001-v2-developmental-maturation"
MATURITY_GRID = (0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0)
PHASES = ("B", "C", "D")
SHADOW_ARMS = ("shadow_full", "shadow_oracle", "shadow_sketch", "task_id_shadow")
INPUT_ONLY_ARMS = ("shadow_full", "shadow_oracle", "shadow_sketch")
DEFAULT_PROTOCOL = Path("research/validations") / VALIDATION_ID / "protocol.json"


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _phase_pass(row: dict[str, Any], thresholds: dict[str, float]) -> bool:
    frontier = row.get("maturity_frontier", [])
    selected = row.get("selected_maturity")
    if selected is None or [float(item.get("maturity")) for item in frontier] != list(MATURITY_GRID):
        return False
    point = next((item for item in frontier if float(item.get("maturity")) == float(selected)), None)
    if point is None:
        return False
    return (
        float(point.get("old_regression", float("inf"))) <= float(thresholds["max_old_regression"])
        and float(point.get("new_gain_normalized", float("-inf"))) >= float(thresholds["min_normalized_new_gain"])
        and float(point.get("accuracy_gain", float("-inf"))) >= float(thresholds["minimum_accuracy_gain"])
    )


def _hypervolume(frontier: list[dict[str, Any]]) -> float:
    points = sorted(
        (max(0.0, float(row["old_regression"])), max(0.0, float(row["new_gain_normalized"])))
        for row in frontier
    )
    best = 0.0
    area = 0.0
    previous_x = 0.0
    for x, y in points:
        if x > previous_x:
            area += (x - previous_x) * best
            previous_x = x
        best = max(best, y)
    return area


def _controls_pass(result: dict[str, Any], thresholds: dict[str, float]) -> bool:
    direct = result.get("controls", {}).get("direct_interp", {})
    shuffled = result.get("controls", {}).get("shuffled_gate", {})
    if any(phase not in direct for phase in PHASES):
        return False
    for arm in INPUT_ONLY_ARMS:
        if arm not in shuffled:
            return False
        for phase in PHASES:
            row = result["arms"][arm][phase]
            shuffled_row = shuffled[arm].get(phase)
            if shuffled_row is None or row.get("selected_maturity") is None:
                return False
            if float(shuffled_row.get("old_regression", 0.0)) - float(row.get("old_regression", 0.0)) < float(thresholds["shuffled_gate_advantage"]):
                return False
    return True


def validate_seed(result: dict[str, Any], protocol_sha: str) -> tuple[bool, str | None]:
    if result.get("validation_id") != VALIDATION_ID:
        return False, "validation_id_mismatch"
    if int(result.get("seed", -1)) not in FORMAL_SEEDS:
        return False, "unregistered_seed"
    if result.get("phase") != "formal" or result.get("protocol_sha256") != protocol_sha:
        return False, "phase_or_protocol_hash_mismatch"
    validity = result.get("validity", {})
    structural = (
        "formal_seed_registered", "protocol_hash_matches_locked_protocol",
        "canonical_checkpoint_hash_matches", "formal_dataset_hash_matches",
        "finite_results",
    )
    if any(validity.get(key) is not True for key in structural):
        return False, "validity_gate_failed"
    if result.get("status", "").startswith("INCONCLUSIVE_"):
        return True, result["status"]
    if result.get("status") != "COMPLETE":
        return False, "seed_not_complete"
    required = (
        "base_capability_passes", "same_mature_parent_passes", "accepted_immutable",
        "m0_identity_passes", "no_learner_historical_replay", "gate_capacity_passes",
        "direct_plasticity_passes", "required_arms_completed", "required_controls_completed",
        "all_maturity_values_evaluated", "copy_on_write_artifacts_complete",
    )
    if any(validity.get(key) is not True for key in required):
        return False, "validity_gate_failed"
    return True, None


def classify(results: list[dict[str, Any]], thresholds: dict[str, float]) -> str:
    if not all(result["validity"]["base_capability_passes"] for result in results):
        return "INCONCLUSIVE_BASE_CAPABILITY"
    if not all(result["validity"]["same_mature_parent_passes"] for result in results):
        return "INCONCLUSIVE_PARENT_CONFLICT"
    if not all(result["validity"]["direct_plasticity_passes"] for result in results):
        return "INCONCLUSIVE_DIRECT_PLASTICITY"
    if not all(result["validity"]["gate_capacity_passes"] for result in results):
        return "INCONCLUSIVE_GATE_CAPACITY"
    if not all(_controls_pass(result, thresholds) for result in results):
        return "INCONCLUSIVE_IDENTITY_CONTROL"
    arm_passes = {
        arm: all(_phase_pass(result["arms"][arm][phase], thresholds) for result in results for phase in PHASES)
        for arm in SHADOW_ARMS
    }
    if not arm_passes["task_id_shadow"] or not arm_passes["shadow_oracle"]:
        return "SHADOW_MATURATION_NOT_SUPPORTED"
    conditional_hv = [_hypervolume(result["arms"]["shadow_oracle"][phase]["maturity_frontier"]) for result in results for phase in PHASES]
    interp_hv = [_hypervolume(result["controls"]["direct_interp"][phase]["maturity_frontier"]) for result in results for phase in PHASES]
    if any(candidate <= baseline + float(thresholds["direct_interp_hypervolume_tolerance"]) for candidate, baseline in zip(conditional_hv, interp_hv)):
        return "ISOLATED_SHADOW_ADVANTAGE_NOT_SUPPORTED"
    if arm_passes["shadow_sketch"]:
        return "SHADOW_MATURATION_SUPPORTED"
    return "SHADOW_MATURATION_ORACLE_ONLY"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--thresholds", type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol
    protocol_sha = args.protocol_sha256 or hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    thresholds = {"max_old_regression": 0.2, "min_normalized_new_gain": 0.9, "minimum_accuracy_gain": 0.0, "shuffled_gate_advantage": 0.1, "direct_interp_hypervolume_tolerance": 1e-6}
    if args.thresholds:
        thresholds.update(json.loads(args.thresholds.read_text(encoding="utf-8")))
    paths = [args.results_root / f"seed-{seed}" / "result.json" for seed in FORMAL_SEEDS]
    if not all(path.is_file() for path in paths):
        payload = {"status": "INCOMPLETE", "scientific_decision": False, "formal_seeds": list(FORMAL_SEEDS), "missing": [str(path) for path in paths if not path.is_file()], "protocol_sha256": protocol_sha}
    else:
        results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        validity = [validate_seed(result, protocol_sha) for result in results]
        if not all(ok for ok, _ in validity) or not all(_finite(result) for result in results):
            payload = {"status": "INVALID", "scientific_decision": False, "formal_seeds": list(FORMAL_SEEDS), "protocol_sha256": protocol_sha, "seed_errors": {str(seed): reason for seed, (_, reason) in zip(FORMAL_SEEDS, validity) if reason}}
        else:
            classification = classify(results, thresholds)
            payload = {"status": classification, "scientific_decision": True, "formal_seeds": list(FORMAL_SEEDS), "protocol_sha256": protocol_sha, "arm_passes": {arm: all(_phase_pass(result["arms"][arm][phase], thresholds) for result in results for phase in PHASES) for arm in SHADOW_ARMS}, "seed_results": results}
    args.results_root.mkdir(parents=True, exist_ok=True)
    (args.results_root / "aggregate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
