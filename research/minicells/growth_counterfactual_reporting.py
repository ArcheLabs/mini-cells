"""Formal aggregation and decisions for CLM-0.3c counterfactual mitosis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FORMAL_REPLICATES = (0, 1, 2)
FORMAL_CANDIDATES_PER_REPLICATE = 12
FORMAL_BIRTHS_PER_REPLICATE = 13  # 12 probes + one independent confirmation birth
FORMAL_RHO_THRESHOLD = 0.30
FORMAL_PPL_RATIO_THRESHOLD = 0.995
FORMAL_DECISION_TOKENS = 1_500_000
FORMAL_PROBE_TOKENS = 100_000
FORMAL_CONFIRM_TOKENS = 500_000
FORMAL_EVAL_BATCHES = 32
FORMAL_CALIBRATION_BATCHES = 16
FORMAL_BOOTSTRAP_SAMPLES = 2_000


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_identity(identity: dict[str, Any], replicate: int) -> None:
    expected = {
        "replicate_seed": (55031, 55032, 55033)[replicate],
        "decision_tokens": FORMAL_DECISION_TOKENS,
        "probe_tokens": FORMAL_PROBE_TOKENS,
        "confirm_tokens": FORMAL_CONFIRM_TOKENS,
        "eval_batches": FORMAL_EVAL_BATCHES,
        "calibration_batches": FORMAL_CALIBRATION_BATCHES,
        "bootstrap_samples": FORMAL_BOOTSTRAP_SAMPLES,
        "balance_weight": 0.0,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise RuntimeError(
                f"r{replicate} formal semantic mismatch for {key}: {identity.get(key)!r} != {value!r}"
            )
    probe_hash = identity.get("probe_validation_schedule_sha256")
    confirm_hash = identity.get("confirm_validation_schedule_sha256")
    if not probe_hash or not confirm_hash or probe_hash == confirm_hash:
        raise RuntimeError(f"r{replicate} does not have two distinct validation holdouts")
    if identity.get("tracked_tree_dirty") is not False:
        raise RuntimeError(f"r{replicate} was not executed from a clean tracked tree")


def aggregate_counterfactual_results(
    output_root: str | Path,
    *,
    formal_gpu_experiment_run: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    summaries: list[dict[str, Any]] = []
    commits: set[str] = set()
    trees: set[str] = set()

    for replicate in FORMAL_REPLICATES:
        directory = root / f"r{replicate}-counterfactual"
        required = [
            directory / "run-provenance.json",
            directory / "replicate-result.json",
            directory / "probe-results.json",
            directory / "growth-equivalence.json",
            directory / "split-regret.csv",
            directory / "policy-decision.json",
            directory / "confirm-candidate.json",
            directory / "confirm-control.json",
            directory / "events.jsonl",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"CLM-0.3c artifacts missing for replicate {replicate}: {missing}")
        identity = _read(directory / "run-provenance.json")
        result = _read(directory / "replicate-result.json")
        probes = _read(directory / "probe-results.json")
        parity = _read(directory / "growth-equivalence.json")
        if formal_gpu_experiment_run:
            _validate_identity(identity, replicate)
        if len(probes) != FORMAL_CANDIDATES_PER_REPLICATE:
            raise RuntimeError(f"r{replicate} does not contain 12 candidate probes")
        if len(parity) != FORMAL_CANDIDATES_PER_REPLICATE:
            raise RuntimeError(f"r{replicate} does not contain 12 probe parity records")
        if int(result.get("births_checked", -1)) != FORMAL_BIRTHS_PER_REPLICATE:
            raise RuntimeError(f"r{replicate} confirmation birth evidence is incomplete")
        if not result.get("code_commit") or not result.get("code_tree_sha"):
            raise RuntimeError(f"r{replicate} is missing immutable code provenance")
        if identity.get("code_commit") != result.get("code_commit"):
            raise RuntimeError(f"r{replicate} run identity and result commit differ")
        if identity.get("code_tree_sha") != result.get("code_tree_sha"):
            raise RuntimeError(f"r{replicate} run identity and result tree differ")
        commits.add(str(result["code_commit"]))
        trees.add(str(result["code_tree_sha"]))

        best_realized = max(probes, key=lambda row: float(row["relative_improvement"]))
        analytic_top = min(probes, key=lambda row: int(row["analytic_rank"]))
        realized_rank_of_analytic_top = 1 + sorted(
            probes,
            key=lambda row: float(row["relative_improvement"]),
            reverse=True,
        ).index(analytic_top)
        selected = result["policy"]
        confirm = result["confirm"]
        summaries.append({
            "replicate": replicate,
            "code_commit": result["code_commit"],
            "code_tree_sha": result["code_tree_sha"],
            "births_checked": int(result["births_checked"]),
            "births_equivalent": int(result["births_equivalent"]),
            "spearman_rho": float(selected["spearman_split_regret_vs_probe_utility"]),
            "analytic_top_expert": str(analytic_top["expert_id"]),
            "analytic_top_realized_rank": realized_rank_of_analytic_top,
            "realized_probe_best_expert": str(best_realized["expert_id"]),
            "realized_probe_best_relative_improvement": float(best_realized["relative_improvement"]),
            "policy_action": str(selected["action"]),
            "policy_selected_expert": str(selected["selected_expert"]),
            "policy_probe_ci95_low": float(selected["probe_ci95_low"]),
            "policy_probe_ci95_high": float(selected["probe_ci95_high"]),
            "decision_calibrated": bool(result["decision_calibrated"]),
            "decision_inconclusive": bool(result["decision_inconclusive"]),
            "confirm_relative_improvement": float(confirm["relative_improvement"]),
            "confirm_ci95_low": float(confirm["ci95_low"]),
            "confirm_ci95_high": float(confirm["ci95_high"]),
            "confirm_ppl_ratio": float(confirm["ppl_ratio"]),
            "confirmed_positive_capacity_value": bool(result["confirmed_positive_capacity_value"]),
            "practical_growth_pass": bool(result["practical_growth_pass"]),
        })

    if formal_gpu_experiment_run and (len(commits) != 1 or len(trees) != 1):
        raise RuntimeError(
            f"formal CLM-0.3c matrix mixed code provenance: commits={sorted(commits)}, trees={sorted(trees)}"
        )

    total_births = sum(row["births_checked"] for row in summaries)
    equivalent_births = sum(row["births_equivalent"] for row in summaries)
    predictive = sum(row["spearman_rho"] >= FORMAL_RHO_THRESHOLD for row in summaries)
    calibrated = sum(row["decision_calibrated"] for row in summaries)
    positive_capacity = sum(row["confirmed_positive_capacity_value"] for row in summaries)
    practical = sum(row["practical_growth_pass"] for row in summaries)
    analytic_top3 = sum(row["analytic_top_realized_rank"] <= 3 for row in summaries)

    decision = {
        "format": "minicells.clm-0.3c-counterfactual-mitosis.decision.v1",
        "formal_gpu_experiment_run": bool(formal_gpu_experiment_run),
        "training_code_commit": next(iter(commits)) if len(commits) == 1 else None,
        "training_code_tree_sha": next(iter(trees)) if len(trees) == 1 else None,
        "growth_equivalence": {
            "status": (
                "CLM_COUNTERFACTUAL_GROWTH_EQUIVALENCE"
                if total_births == 39 and equivalent_births == 39
                else "CLM_COUNTERFACTUAL_GROWTH_EQUIVALENCE_FAILURE"
            ),
            "births_checked": total_births,
            "births_equivalent": equivalent_births,
        },
        "split_regret_prediction": {
            "status": (
                "CLM_SPLIT_REGRET_PREDICTIVE_SIGNAL"
                if predictive >= 2
                else "NO_SPLIT_REGRET_PREDICTIVE_SIGNAL"
            ),
            "rho_threshold": FORMAL_RHO_THRESHOLD,
            "replicates_passed": predictive,
            "analytic_top3_realized_hits": analytic_top3,
        },
        "counterfactual_decision": {
            "status": (
                "CLM_COUNTERFACTUAL_DECISION_SIGNAL"
                if calibrated >= 2
                else "NO_COUNTERFACTUAL_DECISION_SIGNAL"
            ),
            "replicates_calibrated": calibrated,
        },
        "capacity_value": {
            "status": (
                "CLM_COUNTERFACTUAL_CAPACITY_VALUE_SIGNAL"
                if positive_capacity >= 2
                else "NO_COUNTERFACTUAL_CAPACITY_VALUE_SIGNAL"
            ),
            "replicates_positive_ci95": positive_capacity,
        },
        "practical_growth": {
            "status": (
                "CLM_COUNTERFACTUAL_PRACTICAL_GROWTH_SIGNAL"
                if practical >= 2
                else "NO_COUNTERFACTUAL_PRACTICAL_GROWTH_SIGNAL"
            ),
            "ppl_ratio_threshold": FORMAL_PPL_RATIO_THRESHOLD,
            "replicates_passed": practical,
        },
    }
    (root / "replicate-summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"summaries": summaries, "decision": decision}
