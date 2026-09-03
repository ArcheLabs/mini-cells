#!/usr/bin/env python3
"""Fail-closed aggregation for Shadow Cell Validation 001 v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

FORMAL_SEEDS = (95311, 95312, 95313)
VALIDATION_ID = "shadow-cell-validation-001-v2-developmental-maturation"
MATURITY_GRID = (0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0)
PHASES = ("B", "C", "D")
SHADOW_ARMS = ("shadow_full", "shadow_oracle", "shadow_sketch", "task_id_shadow")
INPUT_ONLY_ARMS = ("shadow_full", "shadow_oracle", "shadow_sketch")
TAXONOMY = (
    "INCONCLUSIVE_BASE_CAPABILITY", "INCONCLUSIVE_PARENT_CONFLICT",
    "INCONCLUSIVE_DIRECT_PLASTICITY", "INCONCLUSIVE_GATE_CAPACITY",
    "INCONCLUSIVE_IDENTITY_CONTROL", "SHADOW_MATURATION_NOT_SUPPORTED",
    "ISOLATED_SHADOW_ADVANTAGE_NOT_SUPPORTED",
    "SHADOW_ISOLATION_SUPPORTED_MATURATION_NOT_NECESSARY",
    "SHADOW_MATURATION_ORACLE_ONLY", "SHADOW_MATURATION_SUPPORTED", "INVALID",
)
ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = ROOT / "research/validations" / VALIDATION_ID
DEFAULT_PROTOCOL = VALIDATION_DIR / "protocol.json"
DEFAULT_LOCK = VALIDATION_DIR / "protocol-lock.json"
IMPLEMENTATION_FILES = (
    "scripts/research/run_shadow_cell_validation_001_v2.py",
    "scripts/research/aggregate_shadow_cell_validation_001_v2.py",
    "scripts/research/publish_shadow_cell_validation_001_v2.py",
    "scripts/research/report_shadow_cell_validation_001_v2.py",
    "src/minicells/shadow_maturation.py",
    f"research/validations/{VALIDATION_ID}/protocol.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_manifest(root: Path = ROOT) -> tuple[dict[str, str], str]:
    values = {name: sha256_file(root / name) for name in IMPLEMENTATION_FILES}
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return values, hashlib.sha256(canonical).hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _protocol_and_lock(protocol_path: Path = DEFAULT_PROTOCOL, lock_path: Path = DEFAULT_LOCK) -> tuple[dict[str, Any], dict[str, Any], str]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    protocol_sha = sha256_file(protocol_path)
    if protocol.get("validation_id") != VALIDATION_ID or lock.get("validation_id") != VALIDATION_ID:
        raise ValueError("validation id mismatch")
    if lock.get("status") != "FROZEN":
        raise ValueError("protocol-lock.json is not FROZEN")
    if lock.get("protocol_sha256") != protocol_sha:
        raise ValueError("protocol SHA-256 does not match protocol-lock.json")
    locked_files = lock.get("implementation_files")
    if not isinstance(locked_files, dict) or set(locked_files) != set(IMPLEMENTATION_FILES):
        raise ValueError("implementation manifest is missing from protocol-lock.json")
    actual_files, actual_manifest = implementation_manifest()
    if locked_files != actual_files or lock.get("implementation_manifest_sha256") != actual_manifest:
        raise ValueError("implementation manifest does not match protocol-lock.json")
    if tuple(protocol.get("decision_taxonomy", ())) != TAXONOMY:
        raise ValueError("protocol decision taxonomy drift")
    return protocol, lock, protocol_sha


def _thresholds(protocol: dict[str, Any]) -> dict[str, float]:
    values = protocol.get("thresholds")
    if not isinstance(values, dict):
        raise ValueError("protocol thresholds are missing")
    required = ("max_old_regression", "min_normalized_new_gain", "minimum_accuracy_gain",
                "minimum_shadow_over_interp_gain", "minimum_correct_over_shuffled_gain",
                "max_mean_forgetting", "shadow_sketch_utility_ratio",
                "shadow_sketch_regression_difference", "full_oracle_utility_tolerance")
    if any(key not in values for key in required):
        raise ValueError("protocol is missing a registered scientific threshold")
    return {key: float(value) for key, value in values.items() if isinstance(value, (int, float))}


def _frontier(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("maturity_frontier")
    return value if isinstance(value, list) else []


def _phase_pass(row: dict[str, Any], thresholds: dict[str, float]) -> bool:
    frontier = _frontier(row)
    if [float(item.get("maturity", float("nan"))) for item in frontier] != list(MATURITY_GRID):
        return False
    selected = row.get("selected_maturity")
    if selected is None:
        return False
    point = next((item for item in frontier if float(item.get("maturity")) == float(selected)), None)
    return point is not None and float(point.get("old_regression", float("inf"))) <= thresholds["max_old_regression"] and float(point.get("new_gain_normalized", float("-inf"))) >= thresholds["min_normalized_new_gain"] and float(point.get("accuracy_gain", float("-inf"))) >= thresholds["minimum_accuracy_gain"]


def _safe_gain(frontier: Iterable[dict[str, Any]], epsilon: float) -> float:
    return max((float(row.get("new_gain_normalized", float("-inf"))) for row in frontier if float(row.get("old_regression", float("inf"))) <= epsilon), default=float("-inf"))


def _safe_point(frontier: Iterable[dict[str, Any]], epsilon: float) -> dict[str, Any] | None:
    eligible = [row for row in frontier if float(row.get("old_regression", float("inf"))) <= epsilon]
    if not eligible:
        return None
    return max(eligible, key=lambda row: float(row.get("new_gain_normalized", float("-inf"))))


def _arm_pass(result: dict[str, Any], arm: str, thresholds: dict[str, float]) -> bool:
    return all(_phase_pass(result.get("arms", {}).get(arm, {}).get(phase, {}), thresholds) for phase in PHASES)


def _controls_pass(result: dict[str, Any], thresholds: dict[str, float]) -> bool:
    controls = result.get("controls", {})
    direct = controls.get("direct_interp", {})
    shuffled = controls.get("shuffled_gate", {})
    if any(phase not in direct or len(_frontier(direct[phase])) != len(MATURITY_GRID) for phase in PHASES):
        return False
    for arm in INPUT_ONLY_ARMS:
        for phase in PHASES:
            correct = result["arms"][arm][phase]
            shuffled_row = shuffled.get(arm, {}).get(phase, {})
            if len(_frontier(shuffled_row)) != len(MATURITY_GRID):
                return False
            if _safe_gain(_frontier(correct), thresholds["max_old_regression"]) - _safe_gain(_frontier(shuffled_row), thresholds["max_old_regression"]) < thresholds["minimum_correct_over_shuffled_gain"]:
                return False
    return True


def validate_seed(result: dict[str, Any], protocol_sha: str, lock: dict[str, Any] | None = None) -> tuple[bool, str | None]:
    if result.get("validation_id") != VALIDATION_ID:
        return False, "validation_id_mismatch"
    seed = int(result.get("seed", -1))
    if seed not in FORMAL_SEEDS:
        return False, "unregistered_seed"
    if result.get("phase") != "formal" or result.get("protocol_sha256") != protocol_sha:
        return False, "phase_or_protocol_hash_mismatch"
    if lock is not None:
        if result.get("implementation_manifest_sha256") != lock.get("implementation_manifest_sha256"):
            return False, "implementation_manifest_mismatch"
        if result.get("checkpoint_sha256") != lock.get("canonical_checkpoint_sha256"):
            return False, "checkpoint_hash_mismatch"
        expected = lock.get("formal_dataset_sha256", {}).get(str(seed))
        if result.get("dataset_sha256") != expected:
            return False, "dataset_hash_mismatch"
    validity = result.get("validity", {})
    for key in ("formal_seed_registered", "protocol_hash_matches_locked_protocol", "canonical_checkpoint_hash_matches", "formal_dataset_hash_matches", "finite_results"):
        if validity.get(key) is not True:
            return False, "infrastructure_validity_failed"
    if not _finite(result):
        return False, "non_finite_result"
    prereq = {name for name in TAXONOMY if name.startswith("INCONCLUSIVE_")}
    if result.get("status") in prereq:
        return True, result["status"]
    if result.get("status") != "COMPLETE":
        return False, "seed_not_complete"
    required = ("accepted_immutable", "m0_identity_passes", "no_learner_historical_replay", "gate_capacity_passes", "direct_plasticity_passes", "required_arms_completed", "required_controls_completed", "all_maturity_values_evaluated", "copy_on_write_artifacts_complete", "absolute_retention_passes")
    if any(validity.get(key) is not True for key in required):
        return False, "scientific_validity_gate_failed"
    if any(arm not in result.get("arms", {}) or any(phase not in result["arms"][arm] for phase in PHASES) for arm in SHADOW_ARMS):
        return False, "required_result_structure_missing"
    controls = result.get("controls", {})
    if any(phase not in controls.get("direct_interp", {}) for phase in PHASES) or "shuffled_gate" not in controls:
        return False, "required_control_structure_missing"
    return True, None


def classify(results: list[dict[str, Any]], thresholds: dict[str, float] | None = None) -> str:
    """Apply the frozen all-seed decision precedence."""
    thresholds = thresholds or {"max_old_regression": 0.2, "min_normalized_new_gain": 0.9, "minimum_accuracy_gain": 0.0, "minimum_shadow_over_interp_gain": 0.1, "minimum_correct_over_shuffled_gain": 0.1, "max_mean_forgetting": 0.15, "shadow_sketch_utility_ratio": 0.9, "shadow_sketch_regression_difference": 0.03, "full_oracle_utility_tolerance": 0.1}
    prereq = (("INCONCLUSIVE_BASE_CAPABILITY", "base_capability_passes"), ("INCONCLUSIVE_PARENT_CONFLICT", "same_mature_parent_passes"), ("INCONCLUSIVE_DIRECT_PLASTICITY", "direct_plasticity_passes"), ("INCONCLUSIVE_GATE_CAPACITY", "gate_capacity_passes"))
    for status, key in prereq:
        if any(result.get("status") == status or not result.get("validity", {}).get(key, True) for result in results):
            return status
    if not all(_arm_pass(result, "task_id_shadow", thresholds) for result in results):
        return "SHADOW_MATURATION_NOT_SUPPORTED"
    if not all(_arm_pass(result, "shadow_oracle", thresholds) for result in results):
        return "SHADOW_MATURATION_NOT_SUPPORTED"
    for result in results:
        for phase in PHASES:
            oracle_gain = _safe_gain(_frontier(result["arms"]["shadow_oracle"][phase]), thresholds["max_old_regression"])
            interp_gain = _safe_gain(_frontier(result["controls"]["direct_interp"][phase]), thresholds["max_old_regression"])
            if oracle_gain - interp_gain < thresholds["minimum_shadow_over_interp_gain"]:
                return "ISOLATED_SHADOW_ADVANTAGE_NOT_SUPPORTED"
    if not all(_controls_pass(result, thresholds) for result in results):
        return "INCONCLUSIVE_IDENTITY_CONTROL"
    full_passes = all(_arm_pass(result, "shadow_full", thresholds) for result in results)
    full_approx = all(abs(_safe_gain(_frontier(result["arms"]["shadow_full"][phase]), thresholds["max_old_regression"]) - _safe_gain(_frontier(result["arms"]["shadow_oracle"][phase]), thresholds["max_old_regression"])) <= thresholds["full_oracle_utility_tolerance"] for result in results for phase in PHASES)
    if full_passes and full_approx:
        return "SHADOW_ISOLATION_SUPPORTED_MATURATION_NOT_NECESSARY"
    for result in results:
        for phase in PHASES:
            oracle = _safe_point(_frontier(result["arms"]["shadow_oracle"][phase]), thresholds["max_old_regression"])
            sketch = _safe_point(_frontier(result["arms"]["shadow_sketch"][phase]), thresholds["max_old_regression"])
            if oracle is None or sketch is None:
                return "SHADOW_MATURATION_ORACLE_ONLY"
            oracle_gain = _safe_gain(_frontier(result["arms"]["shadow_oracle"][phase]), thresholds["max_old_regression"])
            sketch_gain = _safe_gain(_frontier(result["arms"]["shadow_sketch"][phase]), thresholds["max_old_regression"])
            if oracle_gain <= 0.0 or sketch_gain / oracle_gain < thresholds["shadow_sketch_utility_ratio"] or abs(float(sketch["old_regression"]) - float(oracle["old_regression"])) > thresholds["shadow_sketch_regression_difference"]:
                return "SHADOW_MATURATION_ORACLE_ONLY"
    return "SHADOW_MATURATION_SUPPORTED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        protocol, lock, protocol_sha = _protocol_and_lock()
        thresholds = _thresholds(protocol)
        paths = [args.results_root / f"seed-{seed}" / "result.json" for seed in FORMAL_SEEDS]
        if not all(path.is_file() for path in paths):
            payload = {"status": "FORMAL_INCOMPLETE", "scientific_decision": False, "formal_seeds": list(FORMAL_SEEDS), "completed_seeds": [seed for seed, path in zip(FORMAL_SEEDS, paths) if path.is_file()], "missing_seeds": [seed for seed, path in zip(FORMAL_SEEDS, paths) if not path.is_file()], "protocol_sha256": protocol_sha}
        else:
            results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            checks = [validate_seed(result, protocol_sha, lock) for result in results]
            if not all(ok for ok, _ in checks):
                payload = {"status": "INVALID", "scientific_decision": False, "formal_seeds": list(FORMAL_SEEDS), "protocol_sha256": protocol_sha, "seed_errors": {str(seed): reason for seed, (_, reason) in zip(FORMAL_SEEDS, checks) if reason}}
            else:
                classification = classify(results, thresholds)
                payload = {"status": classification, "scientific_decision": classification != "INVALID" and not classification.startswith("INCONCLUSIVE_"), "formal_seeds": list(FORMAL_SEEDS), "protocol_sha256": protocol_sha, "implementation_manifest_sha256": lock.get("implementation_manifest_sha256"), "checkpoint_sha256": lock.get("canonical_checkpoint_sha256"), "formal_dataset_sha256": lock.get("formal_dataset_sha256"), "seed_results": results}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        payload = {"status": "INVALID", "scientific_decision": False, "formal_seeds": list(FORMAL_SEEDS), "error": str(exc)}
    args.results_root.mkdir(parents=True, exist_ok=True)
    (args.results_root / "aggregate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
