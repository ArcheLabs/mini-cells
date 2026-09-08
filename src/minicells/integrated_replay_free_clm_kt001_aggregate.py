"""Decision and aggregation logic for Integrated Replay-Free CLM Kill Test 001."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .integrated_replay_free_clm_kt001 import canonical_arm_map
from .native_clm_m2 import sha256_file
from .native_clm_m2r0 import summarize_invariant_rows


REQUIRED_ARMS = tuple(canonical_arm_map())
PHASES = ("B", "C", "D")


def _loss(summary: dict[str, Any], stage: str, domain: str) -> float:
    return float(summary["evaluation_matrix"][stage][domain]["loss"])


def phase_gains(summary: dict[str, Any]) -> dict[str, float]:
    before = {"B": "initial", "C": "after_B", "D": "after_C"}
    return {
        phase: (
            _loss(summary, before[phase], phase) - _loss(summary, f"after_{phase}", phase)
        )
        / max(_loss(summary, before[phase], phase), 1e-12)
        for phase in PHASES
    }


def forgetting(summary: dict[str, Any]) -> dict[str, float]:
    references = {
        "A": _loss(summary, "initial", "A"),
        "B": _loss(summary, "after_B", "B"),
        "C": _loss(summary, "after_C", "C"),
    }
    return {
        domain: max(
            0.0,
            _loss(summary, "after_D", domain) / max(reference, 1e-12) - 1.0,
        )
        for domain, reference in references.items()
    }


def a_regression(summary: dict[str, Any]) -> float:
    return max(
        0.0,
        _loss(summary, "after_D", "A") / max(_loss(summary, "initial", "A"), 1e-12) - 1.0,
    )


def mean_forgetting(summary: dict[str, Any]) -> float:
    values = forgetting(summary).values()
    return float(sum(values) / 3.0)


def max_active_fraction(summary: dict[str, Any]) -> float:
    return max(
        float(metrics["active_fraction_vs_dense"])
        for stage in summary["evaluation_matrix"].values()
        for metrics in stage.values()
    )


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _invariant_summary(arm_dir: Path) -> dict[str, Any]:
    path = arm_dir / "realized-update-invariant.jsonl"
    if not path.exists():
        return {
            "file_exists": False,
            "audited_cell_updates": 0,
            "violation_ratio_p95": float("inf"),
            "violation_ratio_max": float("inf"),
        }
    rows = _load_jsonl(path)
    summary = summarize_invariant_rows(rows)
    summary["file_exists"] = True
    summary["sha256"] = sha256_file(path)
    summary["bytes"] = path.stat().st_size
    return summary


def _shadow_events(summary: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for phase_summary in summary.get("phase_summaries", []):
        event = phase_summary.get("shadow_event")
        if event is not None:
            events.append(event)
    return events


def _boundary_checkpoint_gate(summary: dict[str, Any]) -> bool:
    if not summary["arm_switches"]["historical_address_read"]:
        return True
    for phase_summary in summary.get("phase_summaries", []):
        boundary = phase_summary.get("boundary_checkpoints", {})
        if set(boundary) != {"pre_shadow", "post_shadow"}:
            return False
        for record in boundary.values():
            path = Path(record["path"])
            if not path.exists():
                return False
            if path.stat().st_size != int(record["bytes"]):
                return False
            if sha256_file(path) != record["sha256"]:
                return False
    return True


def _address_gate(summary: dict[str, Any], decision: dict[str, Any]) -> bool:
    if not summary["arm_switches"]["historical_address_read"]:
        return True
    threshold = decision["source_thresholds"]["reused"]
    address = summary.get("structural_final", {}).get("address_state", {})
    return (
        int(address.get("maximum_rank", 10**9)) <= int(threshold["maximum_address_state_rank"])
        and int(address.get("maximum_bytes_per_cell", 10**18))
        <= int(threshold["maximum_address_state_bytes_per_cell"])
        and bool(address.get("bootstrap_complete"))
        and bool(address.get("bootstrap_access_released"))
    )


def _birth_gate(summary: dict[str, Any], decision: dict[str, Any]) -> bool:
    if not summary["arm_switches"]["historical_address_read"]:
        return True
    reused = decision["source_thresholds"]["reused"]
    specific = decision["kt001_specific"]
    events = _shadow_events(summary)
    return (
        len(events) == len(PHASES) * int(specific["forced_shadow_expansions_per_phase"])
        and int(summary["structural_final"]["cell_count"])
        == int(specific["expected_final_cell_count_for_address_arms"])
        and all(bool(event.get("forced_by_protocol")) for event in events)
        and all(not bool(event.get("trigger_uses_evaluation_metrics")) for event in events)
        and all(
            float(event["birth_logits_max_abs_drift"])
            <= float(reused["maximum_birth_logits_max_abs_drift"])
            for event in events
        )
        and all(
            float(event["birth_logits_mse"]) <= float(reused["maximum_birth_logits_mse"])
            for event in events
        )
        and all(float(event["birth_root_topk_match"]) == 1.0 for event in events)
        and all(
            float(event["birth_root_prob_max_abs_drift"])
            <= float(reused["maximum_birth_root_prob_drift"])
            for event in events
        )
    )


def _bootstrap_gate(summary: dict[str, Any]) -> bool:
    if not summary["arm_switches"]["historical_address_read"]:
        return True
    bootstrap = summary.get("bootstrap") or {}
    return (
        bootstrap.get("parameter_sha256_before") == bootstrap.get("parameter_sha256_after")
        and bootstrap.get("bootstrap_access_released_before_continual_start") is True
        and bootstrap.get("bootstrap_path_retained_by_model") is False
    )


def _replay_gate(summary: dict[str, Any]) -> bool:
    if summary["arm"] == "matched_replay_oracle":
        if int(summary.get("learner_replay_bytes", 0)) <= 0:
            return False
        audit = summary.get("replay_audit", {})
        return set(audit) == set(PHASES) and all(
            abs(float(audit[phase]["replay_example_fraction"]) - 0.5) <= 1e-12
            for phase in PHASES
        )
    return int(summary.get("learner_replay_bytes", -1)) == 0


def _provenance_gate(arms: dict[str, dict[str, Any]]) -> bool:
    provenance = [summary.get("provenance", {}) for summary in arms.values()]
    required = (
        "protocol_sha256",
        "seed_registry_sha256",
        "canonical_m3l2_protocol_sha256",
        "data_manifest_sha256",
        "seed",
    )
    return all(record and all(key in record for key in required) for record in provenance) and all(
        len({record[key] for record in provenance}) == 1 for key in required
    )


def _matched_config_gate(arms: dict[str, dict[str, Any]]) -> bool:
    keys = ("training_config", "growth_config", "address_config", "runner_config")
    return all(
        len({json.dumps(summary[key], sort_keys=True) for summary in arms.values()}) == 1
        for key in keys
    )


def _mechanics_gates(
    arms: dict[str, dict[str, Any]],
    arm_dirs: dict[str, Path],
    decision: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, dict[str, Any]]]:
    reused = decision["source_thresholds"]["reused"]
    invariant_threshold = decision["source_thresholds"]["r0b_update_invariant"]
    invariant = {name: _invariant_summary(arm_dirs[name]) for name in REQUIRED_ARMS}
    safe_names = [
        name
        for name, config in canonical_arm_map().items()
        if config.realized_update_write_safety
    ]
    gates = {
        "all_five_arms_present": set(arms) == set(REQUIRED_ARMS),
        "same_seed_checkpoint_and_provenance": (
            len({int(summary["seed"]) for summary in arms.values()}) == 1
            and len({summary["parent_checkpoint_sha256"] for summary in arms.values()}) == 1
            and _provenance_gate(arms)
        ),
        "matched_configs": _matched_config_gate(arms),
        "zero_replay_boundary_and_oracle_isolation": all(
            _replay_gate(summary) for summary in arms.values()
        ),
        "bootstrap_invariants": all(_bootstrap_gate(summary) for summary in arms.values()),
        "forced_birth_invariants": all(_birth_gate(summary, decision) for summary in arms.values()),
        "boundary_checkpoints": all(_boundary_checkpoint_gate(summary) for summary in arms.values()),
        "rank32_address_state_bounded": all(_address_gate(summary, decision) for summary in arms.values()),
        "sparse_compute": all(
            max_active_fraction(summary) <= float(reused["maximum_active_fraction_vs_dense"])
            for summary in arms.values()
        ),
        "realized_update_invariant": all(
            bool(invariant[name].get("file_exists"))
            and int(invariant[name].get("audited_cell_updates", 0)) > 0
            and float(invariant[name]["violation_ratio_max"])
            <= float(invariant_threshold["maximum_violation_ratio"])
            and float(invariant[name]["violation_ratio_p95"])
            <= float(invariant_threshold["maximum_p95_violation_ratio"])
            for name in safe_names
        ),
        "unsafe_interference_exposed": a_regression(arms["unsafe"])
        >= float(decision["kt001_specific"]["minimum_unsafe_A_regression"]),
    }
    return gates, invariant


def compare_seed(
    arms: dict[str, dict[str, Any]],
    *,
    arm_dirs: dict[str, Path],
    decision: dict[str, Any],
) -> dict[str, Any]:
    if set(arms) != set(REQUIRED_ARMS) or set(arm_dirs) != set(REQUIRED_ARMS):
        raise ValueError("KT001 seed aggregation requires exactly five arms")

    metrics = {
        name: {
            "phase_gains": phase_gains(summary),
            "forgetting": forgetting(summary),
            "A_regression": a_regression(summary),
            "mean_forgetting": mean_forgetting(summary),
            "max_active_fraction_vs_dense": max_active_fraction(summary),
        }
        for name, summary in arms.items()
    }
    mechanics, invariant = _mechanics_gates(arms, arm_dirs, decision)

    oracle = metrics["matched_replay_oracle"]
    full = metrics["full_no_replay"]
    write = metrics["write_transaction_only"]
    read = metrics["read_history_only"]
    specific = decision["kt001_specific"]
    reused = decision["source_thresholds"]["reused"]

    oracle_gates = {
        "oracle_phase_plasticity": all(
            oracle["phase_gains"][phase]
            >= float(specific["oracle_minimum_phase_gain_each_B_C_D"])
            for phase in PHASES
        ),
        "oracle_A_retention": oracle["A_regression"]
        <= float(specific["oracle_maximum_A_regression"]),
        "oracle_mean_forgetting": oracle["mean_forgetting"]
        <= float(specific["oracle_maximum_mean_forgetting"]),
    }

    ratios = {
        phase: full["phase_gains"][phase] / max(oracle["phase_gains"][phase], 1e-12)
        for phase in PHASES
    }
    scientific = {
        "full_phase_plasticity": all(
            full["phase_gains"][phase] >= float(reused["minimum_phase_gain_each_B_C_D"])
            for phase in PHASES
        ),
        "full_absolute_A_retention": full["A_regression"]
        <= float(reused["maximum_full_A_regression"]),
        "full_mean_forgetting": full["mean_forgetting"]
        <= float(reused["maximum_full_mean_forgetting"]),
        "full_advantage_vs_write_only": (
            write["A_regression"] - full["A_regression"]
            >= float(specific["minimum_full_A_retention_advantage_vs_write_transaction_only"])
        ),
        "full_advantage_vs_read_only": (
            read["A_regression"] - full["A_regression"]
            >= float(specific["minimum_full_A_retention_advantage_vs_read_history_only"])
        ),
        "full_phase_gain_ratio_vs_replay": all(
            ratios[phase] >= float(specific["minimum_full_phase_gain_ratio_vs_replay_each"])
            for phase in PHASES
        ),
        "full_geometric_mean_gain_ratio_vs_replay": _geometric_mean(list(ratios.values()))
        >= float(specific["minimum_full_geometric_mean_phase_gain_ratio_vs_replay"]),
    }

    if not all(mechanics.values()):
        classification = "INCONCLUSIVE_MECHANICS"
    elif not all(oracle_gates.values()):
        classification = "INCONCLUSIVE_ORACLE"
    elif all(scientific.values()):
        classification = "PASS"
    else:
        classification = "VALID_FAIL"

    return {
        "format": "minicells.kt001-seed-decision.v1",
        "seed": int(next(iter(arms.values()))["seed"]),
        "classification": classification,
        "mechanics_gates": mechanics,
        "oracle_gates": oracle_gates,
        "scientific_gates": scientific,
        "phase_gain_ratios_full_vs_replay": ratios,
        "geometric_mean_phase_gain_ratio_full_vs_replay": _geometric_mean(list(ratios.values())),
        "metrics": metrics,
        "realized_update_invariants": invariant,
    }


def aggregate_formal_seed_decisions(
    decisions: list[dict[str, Any]],
    *,
    decision_protocol: dict[str, Any],
) -> dict[str, Any]:
    expected = int(decision_protocol["kt001_specific"]["formal_support_requires_seed_passes"])
    if len(decisions) != expected:
        raise ValueError(f"KT001 formal aggregate requires exactly {expected} seed decisions")
    seeds = [int(item["seed"]) for item in decisions]
    if len(set(seeds)) != len(seeds):
        raise ValueError("KT001 formal aggregate contains duplicate seeds")

    classifications = [item["classification"] for item in decisions]
    if classifications.count("PASS") == expected:
        status = decision_protocol["positive_status"]
        scientific_decision: bool | None = True
    elif classifications.count("VALID_FAIL") == int(
        decision_protocol["kt001_specific"]["formal_negative_requires_valid_seed_failures"]
    ):
        status = decision_protocol["negative_status"]
        scientific_decision = False
    else:
        status = decision_protocol["mixed_status"]
        scientific_decision = None

    return {
        "format": "minicells.kt001-formal-decision.v1",
        "status": status,
        "scientific_decision": scientific_decision,
        "formal_seeds": sorted(seeds),
        "classifications": {str(item["seed"]): item["classification"] for item in decisions},
        "per_seed": decisions,
    }
