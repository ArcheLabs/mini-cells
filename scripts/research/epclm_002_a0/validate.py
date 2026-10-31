"""Static protocol and emitted-result checks for EPCLM-002-A0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL = ROOT / "research/governance/validations/epclm-002-a0-persistent-state-recall/protocol.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol(protocol_path: Path = PROTOCOL) -> list[str]:
    errors: list[str] = []
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot load protocol: {exc}"]
    if protocol.get("format") != "minicells.epclm-002-a0-persistent-state-recall.v1":
        errors.append("unexpected protocol format")
    if protocol.get("status") != "FROZEN":
        errors.append("protocol must remain frozen; outcomes belong in separate result artifacts")
    task = protocol.get("task", {})
    if task.get("key_bits") != 16 or task.get("value_bits") != 16:
        errors.append("key/value widths differ from the frozen 16-bit task")
    if task.get("facts_per_episode") != [1, 2, 4, 8, 16, 32]:
        errors.append("fact-count ladder differs from the frozen protocol")
    if task.get("distractors_per_episode") != [0, 16, 64]:
        errors.append("distractor ladder differs from the frozen protocol")
    training_seeds = protocol.get("training", {}).get("formal_seeds", [])
    evaluation_seeds = task.get("splits", {}).get("formal_evaluation_seeds", [])
    if len(training_seeds) != 3 or len(set(training_seeds)) != 3:
        errors.append("exactly three distinct formal model seeds must be registered")
    if len(evaluation_seeds) != 3 or len(set(evaluation_seeds)) != 3:
        errors.append("exactly three distinct formal evaluation seeds must be registered")
    if set(training_seeds) & set(evaluation_seeds):
        errors.append("training and evaluation seeds must be disjoint")
    expected_pairs = [
        {"model_seed": model_seed, "evaluation_seed": evaluation_seed}
        for model_seed, evaluation_seed in zip(training_seeds, evaluation_seeds)
    ]
    if task.get("splits", {}).get("formal_seed_pairing") != expected_pairs:
        errors.append("formal model/evaluation seed pairing is inconsistent")
    if task.get("splits", {}).get("formal_episodes_per_condition_per_model_seed", 0) < 1000:
        errors.append("formal evaluation must use at least 1000 episodes per condition and model seed")
    if protocol.get("thresholds", {}).get("learned_cells_accuracy_at_8_facts_0_distractors_each_seed", 0) != 0.95:
        errors.append("learned-cell primary threshold was changed")
    if protocol.get("thresholds", {}).get("direct_address_min_accuracy") != 0.99:
        errors.append("direct-address positive-control threshold was changed")
    if protocol.get("thresholds", {}).get("maximum_accuracy_drop_at_8_facts_64_distractors") != 0.05:
        errors.append("distractor degradation threshold was changed")
    if protocol.get("models", {}).get("direct_address", "").find("engineering reference only") < 0:
        errors.append("direct-address must remain labelled as an engineering reference")
    if protocol.get("models", {}).get("learned_cells", {}).get("candidate_scoring", "").find("All valid latent cells") < 0:
        errors.append("learned-cell full-scan limitation must stay explicit")
    if protocol.get("models", {}).get("learned_cells", {}).get("state_bytes") != protocol.get("models", {}).get("flat_recurrent", {}).get("state_bytes"):
        errors.append("flat and learned-cell state budgets must match")
    return errors


def validate_result(result_path: Path, protocol_path: Path = PROTOCOL) -> dict[str, Any]:
    errors = validate_protocol(protocol_path)
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [*errors, f"cannot load result: {exc}"]}
    if result.get("format") != "minicells.epclm-002-a0.condition-result.v1":
        errors.append("unexpected result format")
    if result.get("protocol_sha256") != sha256(protocol_path):
        errors.append("result was produced against a different protocol hash")
    records = result.get("records", [])
    if len(records) != result.get("episodes"):
        errors.append("episode record count does not match run summary")
    for index, row in enumerate(records):
        if not row.get("writer_terminated_before_reader"):
            errors.append(f"episode {index}: Writer was not proven terminated")
        if not row.get("raw_history_removed_before_reader"):
            errors.append(f"episode {index}: raw writer input was not removed")
        if row.get("reader_received_raw_history"):
            errors.append(f"episode {index}: Reader received raw history")
        if not row.get("reader_raw_root_denied"):
            errors.append(f"episode {index}: forbidden raw path guard was not active")
        if not row.get("reader_filesystem_allowlist_enforced"):
            errors.append(f"episode {index}: filesystem allowlist was not enforced")
        if not row.get("reader_child_process_and_network_guard_enforced"):
            errors.append(f"episode {index}: process/network guard was not enforced")
        if not row.get("reader_data_allowlist_ok"):
            errors.append(f"episode {index}: Reader data-file allowlist mismatch")
        if not row.get("state_unchanged"):
            errors.append(f"episode {index}: persistent state changed during read")
        predictions = row.get("predictions", [])
        targets = row.get("targets", [])
        if len(predictions) != len(targets) or not targets:
            errors.append(f"episode {index}: prediction/target counts are inconsistent")
        elif row.get("exact_matches") != sum(prediction == target for prediction, target in zip(predictions, targets)):
            errors.append(f"episode {index}: exact-match count does not match predictions")
    metrics = result.get("metrics", {})
    if not metrics.get("isolation_audit_pass"):
        errors.append("condition-level context-removal/isolation audit failed")
    if not metrics.get("no_state_reader_abstained"):
        errors.append("no-state negative control failed to abstain")
    if not metrics.get("no_state_reader_allowlist_pass"):
        errors.append("no-state Reader allowlist audit failed")
    if not metrics.get("wrong_episode_state_audit_pass"):
        errors.append("wrong-episode Reader allowlist audit failed")
    if metrics.get("wrong_episode_state_accuracy", 1.0) > 0.05:
        errors.append("wrong-episode state negative control exceeded its accuracy limit")
    if result.get("backend") == "direct-address" and result.get("scientific_role") != "engineering_reference":
        errors.append("direct-address result is mislabelled as learned evidence")
    if result.get("backend") == "direct-address" and metrics.get("exact_match_accuracy", 0) < 0.99:
        errors.append("direct-address positive control fell below its registered accuracy threshold")
    if result.get("formal_run"):
        if result.get("status") != "FORMAL_CONDITION":
            errors.append("formal run is not labelled as a formal condition")
        if result.get("seed") not in protocol.get("task", {}).get("splits", {}).get("formal_evaluation_seeds", []):
            errors.append("formal result uses an unregistered evaluation seed")
        if result.get("episodes", 0) < protocol.get("task", {}).get("splits", {}).get("formal_episodes_per_condition_per_model_seed", 1000):
            errors.append("formal result has too few episodes")
        if result.get("backend") != "direct-address":
            pairing = protocol.get("task", {}).get("splits", {}).get("formal_seed_pairing", [])
            expected = next((row["model_seed"] for row in pairing if row["evaluation_seed"] == result.get("seed")), None)
            if result.get("training_seed") != expected:
                errors.append("formal model/evaluation seed pairing does not match the protocol")
    elif result.get("backend") == "direct-address" and result.get("status") != "PARTIAL":
        errors.append("direct-address-only results must remain PARTIAL for the overall experiment")
    elif result.get("backend") != "direct-address" and result.get("status") != "PILOT":
        errors.append("single-condition learned results must remain PILOT until formally aggregated")
    return {"valid": not errors, "errors": errors, "protocol_sha256": sha256(protocol_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path, nargs="?")
    args = parser.parse_args()
    protocol_errors = validate_protocol()
    if protocol_errors:
        print(json.dumps({"valid": False, "errors": protocol_errors}, indent=2))
        raise SystemExit(1)
    if args.result:
        report = validate_result(args.result)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["valid"] else 1)
    print(json.dumps({"valid": True, "protocol_sha256": sha256(PROTOCOL)}, indent=2))


if __name__ == "__main__":
    main()
