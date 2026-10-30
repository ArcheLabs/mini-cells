"""Frozen formal-condition orchestrator for EPCLM-002-A0."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from .runner import PROTOCOL, run_condition
from .validate import sha256, validate_protocol, validate_result


def run_matrix(
    *,
    backend: str,
    output: Path,
    checkpoint_dir: Path | None,
    episodes_per_condition: int,
    formal: bool,
    max_runtime_seconds: int = 7200,
) -> dict[str, Any]:
    if not 1 <= max_runtime_seconds <= 7200:
        raise ValueError("max_runtime_seconds must be between 1 and 7200")
    protocol_errors = validate_protocol(PROTOCOL)
    if protocol_errors:
        raise ValueError("protocol validation failed: " + "; ".join(protocol_errors))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    required_episodes = int(protocol["task"]["splits"]["formal_episodes_per_condition_per_model_seed"])
    if not formal:
        raise ValueError("formal matrix execution requires --formal")
    if episodes_per_condition < required_episodes:
        raise ValueError(f"formal matrix requires at least {required_episodes} episodes per condition")
    if backend not in {"direct-address", "learned-cells", "flat-recurrent"}:
        raise ValueError("unsupported backend")

    formal_model_seeds = [int(value) for value in protocol["training"]["formal_seeds"]]
    evaluation_seeds = [int(value) for value in protocol["task"]["splits"]["formal_evaluation_seeds"]]
    if backend == "direct-address":
        run_pairs = [(None, evaluation_seed, None) for evaluation_seed in evaluation_seeds]
    else:
        if checkpoint_dir is None:
            raise ValueError("learned formal matrix requires --checkpoint-dir")
        run_pairs = []
        for model_seed, evaluation_seed in zip(formal_model_seeds, evaluation_seeds):
            checkpoint = checkpoint_dir / f"{backend}-seed-{model_seed}.pt"
            metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
            if not checkpoint.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"missing formal checkpoint or metadata: {checkpoint}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("training_seed") != model_seed or metadata.get("training_seed_role") != "formal":
                raise ValueError(f"checkpoint seed metadata mismatch: {checkpoint}")
            if metadata.get("protocol_sha256") != sha256(PROTOCOL):
                raise ValueError(f"checkpoint protocol hash mismatch: {checkpoint}")
            run_pairs.append((model_seed, evaluation_seed, checkpoint))

    conditions = [
        (fact_count, distractor_count, 0)
        for fact_count in protocol["task"]["facts_per_episode"]
        for distractor_count in protocol["task"]["distractors_per_episode"]
    ]
    conditions.extend((int(row["facts"]), int(row["distractors"]), int(row["updates"])) for row in protocol["task"]["overwrite_conditions"])
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max_runtime_seconds
    results: list[dict[str, Any]] = []
    timed_out = False
    timed_out_condition: Path | None = None
    for model_seed, evaluation_seed, checkpoint in run_pairs:
        seed_label = f"model-seed-{model_seed}" if model_seed is not None else f"evaluation-seed-{evaluation_seed}"
        for fact_count, distractor_count, overwrite_count in conditions:
            remaining = deadline - time.monotonic()
            if remaining < 1:
                timed_out = True
                break
            suffix = f"facts-{fact_count}-distractors-{distractor_count}-updates-{overwrite_count}"
            condition_output = output / backend / seed_label / suffix
            print(
                f"Starting {backend} {seed_label} {suffix} ({episodes_per_condition} episodes)",
                flush=True,
            )
            try:
                result = run_condition(
                    backend=backend,
                    episodes=episodes_per_condition,
                    seed=evaluation_seed,
                    fact_count=fact_count,
                    distractor_count=distractor_count,
                    overwrite_count=overwrite_count,
                    negative_control_count=16,
                    checkpoint=checkpoint,
                    formal_run=True,
                    max_runtime_seconds=max(1, min(7200, int(remaining))),
                    output=condition_output,
                )
            except TimeoutError:
                timed_out = True
                timed_out_condition = condition_output / "partial-status.json"
                break
            validation = validate_result(condition_output / "result.json")
            results.append(
                {
                    "model_seed": model_seed,
                    "evaluation_seed": evaluation_seed,
                    "fact_count": fact_count,
                    "distractor_count": distractor_count,
                    "overwrite_count": overwrite_count,
                    "result_path": str(condition_output / "result.json"),
                    "result_valid": validation["valid"],
                    "validation_errors": validation["errors"],
                    "metrics": result["metrics"],
                }
            )
        if timed_out:
            break
    expected = len(run_pairs) * len(conditions)
    complete = len(results) == expected and all(row["result_valid"] for row in results)
    decision = {
        "format": "minicells.epclm-002-a0-formal-matrix.v1",
        "experiment_id": protocol["experiment_id"],
        "status": "FORMAL_MATRIX_COMPLETE" if complete else (
            "FORMAL_MATRIX_INCOMPLETE_TIMEOUT" if timed_out else "INVALID"
        ),
        "scientific_decision": complete,
        "backend": backend,
        "protocol_sha256": sha256(PROTOCOL),
        "episodes_per_condition": episodes_per_condition,
        "expected_condition_runs": expected,
        "completed_condition_runs": len(results),
        "h1": _h1_summary(backend, results) if complete else {"status": "INCONCLUSIVE"},
        "h2": _h2_summary(backend, results, formal_model_seeds) if complete else {"status": "INCONCLUSIVE"},
        "h3": _h3_summary(backend, results) if complete else {"status": "INCONCLUSIVE"},
        "resume_supported": True,
        "partial_condition_status_path": (
            str(timed_out_condition) if timed_out_condition is not None and timed_out_condition.is_file() else None
        ),
        "timeout_seconds": max_runtime_seconds,
        "results": results,
        "interpretation_limit": "A formal matrix result is limited to this protocol's synthetic episode distribution and registered model implementations.",
    }
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    columns = [
        "model_seed", "evaluation_seed", "fact_count", "distractor_count", "overwrite_count",
        "result_valid", "exact_match_accuracy", "no_memory_accuracy", "isolation_audit_pass",
        "mean_logical_state_bytes", "mean_serialized_state_bytes", "mean_writer_time_ms",
        "mean_reader_time_ms", "mean_read_cells", "max_candidate_cells", "full_scan",
    ]
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    **{key: row.get(key) for key in columns if key not in row},
                    **{key: row.get(key) for key in columns if key in row},
                    **{key: row["metrics"].get(key) for key in columns if key not in row},
                }
            )
    (output / "summary.md").write_text(
        "# EPCLM-002-A0 formal matrix summary\n\n"
        f"Status: **{decision['status']}**\n\n"
        f"Backend: `{backend}`; runs: {len(results)}/{expected}; episodes per condition: {episodes_per_condition}.\n\n"
        f"H1: **{decision['h1']['status']}**  \nH2: **{decision['h2']['status']}**  \nH3: **{decision['h3']['status']}**\n\n"
        "See `decision.json` for registered gates and per-condition results.\n",
        encoding="utf-8",
    )
    return decision


def _h1_summary(backend: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if backend != "direct-address":
        return {"status": "ENGINEERING_REFERENCE_MATRIX_REQUIRED_SEPARATELY"}
    passed = bool(rows) and all(
        row["result_valid"]
        and row["metrics"]["exact_match_accuracy"] >= 0.99
        and row["metrics"]["isolation_audit_pass"]
        for row in rows
    )
    return {"status": "PASS" if passed else "FAIL", "role": "engineering_reference_only"}


def _h2_summary(backend: str, rows: list[dict[str, Any]], model_seeds: list[int]) -> dict[str, Any]:
    if backend == "direct-address":
        return {"status": "NOT_EVIDENCE_FOR_LEARNED_STATE"}
    by_key = {
        (row["model_seed"], row["fact_count"], row["distractor_count"], row["overwrite_count"]): row
        for row in rows
    }
    gate_results: dict[str, bool] = {}
    primary: dict[str, float] = {}
    degradation: dict[str, float] = {}
    overwrite_accuracy: dict[str, float] = {}
    unchanged_accuracy: dict[str, float] = {}
    stale_rate: dict[str, float] = {}
    overwrite_by_seed: dict[str, list[float]] = {}
    unchanged_by_seed: dict[str, list[float]] = {}
    stale_by_seed: dict[str, list[float]] = {}
    for seed in model_seeds:
        clean = by_key.get((seed, 8, 0, 0))
        noisy = by_key.get((seed, 8, 64, 0))
        update_conditions = [by_key.get((seed, 8, 0, 2)), by_key.get((seed, 8, 64, 2))]
        if not clean or not noisy or any(not row for row in update_conditions):
            continue
        clean_accuracy = float(clean["metrics"]["exact_match_accuracy"])
        noisy_accuracy = float(noisy["metrics"]["exact_match_accuracy"])
        overwrite_by_seed[str(seed)] = [float(row["metrics"]["mean_overwritten_accuracy"] or 0.0) for row in update_conditions]
        unchanged_by_seed[str(seed)] = [float(row["metrics"]["mean_unchanged_accuracy"] or 0.0) for row in update_conditions]
        stale_by_seed[str(seed)] = [float(row["metrics"]["mean_stale_old_value_rate"] or 0.0) for row in update_conditions]
        overwrite_accuracy[str(seed)] = min(overwrite_by_seed[str(seed)])
        unchanged_accuracy[str(seed)] = min(unchanged_by_seed[str(seed)])
        stale_rate[str(seed)] = max(stale_by_seed[str(seed)])
        primary[str(seed)] = clean_accuracy
        degradation[str(seed)] = clean_accuracy - noisy_accuracy
    gate_results["all_formal_seeds_present"] = set(primary) == {str(seed) for seed in model_seeds}
    gate_results["primary_recall"] = bool(primary) and all(value >= 0.95 for value in primary.values())
    gate_results["distractor_degradation"] = bool(degradation) and all(value <= 0.05 for value in degradation.values())
    gate_results["overwrite_accuracy"] = bool(overwrite_accuracy) and all(value >= 0.90 for value in overwrite_accuracy.values())
    gate_results["unchanged_accuracy"] = bool(unchanged_accuracy) and all(value >= 0.95 for value in unchanged_accuracy.values())
    gate_results["stale_old_value_rate"] = bool(stale_rate) and all(value <= 0.10 for value in stale_rate.values())
    gate_results["context_removal"] = bool(rows) and all(
        row["result_valid"] and row["metrics"]["isolation_audit_pass"] for row in rows
    )
    supported = all(gate_results.values())
    return {
        "status": "PASS" if supported else "FAIL" if gate_results["all_formal_seeds_present"] else "INCONCLUSIVE",
        "gates": gate_results,
        "primary_accuracy_by_seed": primary,
        "drop_from_0_to_64_distractors_by_seed": degradation,
        "overwrite_accuracy_by_seed": overwrite_accuracy,
        "unchanged_accuracy_by_seed": unchanged_accuracy,
        "stale_old_value_rate_by_seed": stale_rate,
    }


def _h3_summary(backend: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if backend == "learned-cells":
        return {"status": "FAIL", "reason": "registered candidate scores all valid Cells (full_scan=true)"}
    if backend == "flat-recurrent":
        return {"status": "NOT_APPLICABLE", "reason": "single-vector baseline has no Cell candidate router"}
    return {"status": "NOT_TESTED_BY_ENGINEERING_REFERENCE"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("direct-address", "learned-cells", "flat-recurrent"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--episodes-per-condition", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal", action="store_true", help="explicitly authorize the frozen formal matrix")
    parser.add_argument("--max-runtime-seconds", type=int, default=7200)
    args = parser.parse_args()
    result = run_matrix(
        backend=args.backend,
        output=args.output,
        checkpoint_dir=args.checkpoint_dir,
        episodes_per_condition=args.episodes_per_condition,
        formal=args.formal,
        max_runtime_seconds=args.max_runtime_seconds,
    )
    print(json.dumps({"status": result["status"], "h1": result["h1"], "h2": result["h2"], "h3": result["h3"]}, sort_keys=True))


if __name__ == "__main__":
    main()
