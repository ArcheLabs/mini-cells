"""Formal aggregation for CLM-0.3d probationary mitosis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .growth_probationary import (
    FORMAL_CONDITIONS,
    FORMAL_HORIZONS,
    PRACTICAL_PPL_RATIO_THRESHOLD,
    SHORTLIST_K,
    STORY_RETENTION_RATIO_THRESHOLD,
)

FORMAL_REPLICATES = (0, 1, 2)
FORMAL_BIRTHS_PER_CONDITION = 12
FORMAL_TOTAL_BIRTHS = 72
FORMAL_DECISION_TOKENS = 1_500_000
FORMAL_EVAL_BATCHES = 32
FORMAL_BOOTSTRAP_SAMPLES = 2_000


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_identity(identity: dict[str, Any], replicate: int) -> None:
    expected = {
        "replicate_seed": (55031, 55032, 55033)[replicate],
        "decision_tokens": FORMAL_DECISION_TOKENS,
        "probation_horizons": list(FORMAL_HORIZONS),
        "shortlist_k": SHORTLIST_K,
        "eval_batches": FORMAL_EVAL_BATCHES,
        "bootstrap_samples": FORMAL_BOOTSTRAP_SAMPLES,
        "practical_ppl_ratio_threshold": PRACTICAL_PPL_RATIO_THRESHOLD,
        "story_retention_ratio_threshold": STORY_RETENTION_RATIO_THRESHOLD,
        "balance_weight": 0.0,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise RuntimeError(
                f"r{replicate} formal semantic mismatch for {key}: "
                f"{identity.get(key)!r} != {value!r}"
            )
    if identity.get("tracked_tree_dirty") is not False:
        raise RuntimeError(f"r{replicate} was not executed from a clean tracked tree")
    for condition in FORMAL_CONDITIONS:
        hashes = identity.get("conditions", {}).get(condition, {})
        if not hashes.get("future_schedule_sha256"):
            raise RuntimeError(f"r{replicate} {condition} future schedule hash missing")
        holdout_a = hashes.get("holdout_a_sha256")
        holdout_b = hashes.get("holdout_b_sha256")
        if not holdout_a or not holdout_b or holdout_a == holdout_b:
            raise RuntimeError(f"r{replicate} {condition} does not have disjoint holdouts")


def aggregate_probationary_results(
    output_root: str | Path,
    *,
    formal_gpu_experiment_run: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    summaries: list[dict[str, Any]] = []
    commits: set[str] = set()
    trees: set[str] = set()

    for replicate in FORMAL_REPLICATES:
        directory = root / f"r{replicate}-probationary"
        required = [
            directory / "run-provenance.json",
            directory / "replicate-result.json",
            directory / "events.jsonl",
            directory / "trunk-history.json",
        ]
        for condition in FORMAL_CONDITIONS:
            cdir = directory / condition
            required.extend([
                cdir / "control-trajectory.json",
                cdir / "initial-shadow-results.json",
                cdir / "shortlist.json",
                cdir / "probation-trajectories.json",
                cdir / "probation-decisions.json",
                cdir / "promotion-decision.json",
                cdir / "growth-equivalence.json",
                cdir / "final-control.json",
            ])
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"CLM-0.3d artifacts missing for r{replicate}: {missing}")

        identity = _read(directory / "run-provenance.json")
        result = _read(directory / "replicate-result.json")
        if formal_gpu_experiment_run:
            _validate_identity(identity, replicate)
        if not result.get("code_commit") or not result.get("code_tree_sha"):
            raise RuntimeError(f"r{replicate} missing immutable code provenance")
        if result.get("code_commit") != identity.get("code_commit"):
            raise RuntimeError(f"r{replicate} run identity and result commit differ")
        if result.get("code_tree_sha") != identity.get("code_tree_sha"):
            raise RuntimeError(f"r{replicate} run identity and result tree differ")
        commits.add(str(result["code_commit"]))
        trees.add(str(result["code_tree_sha"]))

        condition_rows: dict[str, dict[str, Any]] = {}
        for condition in FORMAL_CONDITIONS:
            cdir = directory / condition
            parity = _read(cdir / "growth-equivalence.json")
            promotion = _read(cdir / "promotion-decision.json")
            initial = _read(cdir / "initial-shadow-results.json")
            shortlist = _read(cdir / "shortlist.json")
            if len(parity) != FORMAL_BIRTHS_PER_CONDITION:
                raise RuntimeError(f"r{replicate} {condition} must contain 12 parity records")
            if len(initial) != FORMAL_BIRTHS_PER_CONDITION:
                raise RuntimeError(f"r{replicate} {condition} must contain 12 initial shadows")
            if len(shortlist.get("experts", [])) != SHORTLIST_K:
                raise RuntimeError(
                    f"r{replicate} {condition} shortlist must contain {SHORTLIST_K} experts"
                )
            absorption = None
            if condition == "story_arithmetic_shift":
                path = cdir / "absorption-diagnostic.json"
                if not path.exists():
                    raise RuntimeError(f"r{replicate} shift absorption diagnostic missing")
                absorption = _read(path)
            condition_rows[condition] = {
                "births_checked": len(parity),
                "births_equivalent": sum(bool(row.get("equivalent")) for row in parity),
                "action": str(promotion.get("action")),
                "selected_expert": promotion.get("selected_expert"),
                "probe_accepted": bool(promotion.get("probe_accepted")),
                "independent_confirmed": bool(promotion.get("independent_confirmed")),
                "maturation_rescue": bool(promotion.get("maturation_rescue")),
                "final_ppl_ratio": promotion.get("final_ppl_ratio"),
                "absorbable_without_mitosis": (
                    bool(absorption.get("absorbable_without_mitosis")) if absorption else None
                ),
            }

        summaries.append({
            "replicate": replicate,
            "code_commit": result["code_commit"],
            "code_tree_sha": result["code_tree_sha"],
            "conditions": condition_rows,
        })

    if formal_gpu_experiment_run and (len(commits) != 1 or len(trees) != 1):
        raise RuntimeError(
            f"formal CLM-0.3d matrix mixed provenance: commits={sorted(commits)}, "
            f"trees={sorted(trees)}"
        )

    total_births = sum(
        row["conditions"][condition]["births_checked"]
        for row in summaries
        for condition in FORMAL_CONDITIONS
    )
    equivalent_births = sum(
        row["conditions"][condition]["births_equivalent"]
        for row in summaries
        for condition in FORMAL_CONDITIONS
    )
    stationary_rejects = sum(
        not row["conditions"]["stationary_story"]["independent_confirmed"]
        for row in summaries
    )
    stationary_promotions = 3 - stationary_rejects
    shift_promotions = sum(
        row["conditions"]["story_arithmetic_shift"]["independent_confirmed"]
        for row in summaries
    )
    shift_absorbable = sum(
        row["conditions"]["story_arithmetic_shift"]["absorbable_without_mitosis"]
        for row in summaries
    )
    maturation = sum(
        row["conditions"]["story_arithmetic_shift"]["maturation_rescue"]
        for row in summaries
    )

    equivalence_ok = (
        total_births == FORMAL_TOTAL_BIRTHS
        and equivalent_births == FORMAL_TOTAL_BIRTHS
    )
    stationary_ok = stationary_rejects >= 2
    shift_ok = shift_promotions >= 2
    if equivalence_ok and stationary_ok and shift_ok:
        overall_status = "CLM_PROBATIONARY_MITOSIS_SIGNAL"
    elif stationary_promotions >= 2:
        overall_status = "CLM_STATIONARY_OVERGROWTH"
    elif stationary_ok and shift_promotions < 2 and shift_absorbable >= 2:
        overall_status = "CLM_SHIFT_ABSORBED_WITHOUT_MITOSIS"
    else:
        overall_status = "CLM_PROBATIONARY_MITOSIS_NOT_CONFIRMED"

    decision = {
        "format": "minicells.clm-0.3d-probationary-mitosis.decision.v1",
        "formal_gpu_experiment_run": bool(formal_gpu_experiment_run),
        "training_code_commit": next(iter(commits)) if len(commits) == 1 else None,
        "training_code_tree_sha": next(iter(trees)) if len(trees) == 1 else None,
        "growth_equivalence": {
            "status": (
                "CLM_PROBATIONARY_GROWTH_EQUIVALENCE"
                if equivalence_ok
                else "CLM_PROBATIONARY_GROWTH_EQUIVALENCE_FAILURE"
            ),
            "births_checked": total_births,
            "births_equivalent": equivalent_births,
        },
        "stationary_specificity": {
            "status": (
                "CLM_STATIONARY_REJECTION_SIGNAL"
                if stationary_ok
                else "CLM_STATIONARY_OVERGROWTH"
            ),
            "replicates_rejected": stationary_rejects,
            "replicates_promoted": stationary_promotions,
        },
        "shift_sensitivity": {
            "status": (
                "CLM_SHIFT_PROMOTION_SIGNAL"
                if shift_ok
                else "NO_CLM_SHIFT_PROMOTION_SIGNAL"
            ),
            "replicates_promoted": shift_promotions,
            "replicates_absorbable_without_mitosis": shift_absorbable,
        },
        "maturation": {
            "status": (
                "CLM_LINEAGE_MATURATION_SIGNAL"
                if maturation >= 2
                else "NO_CLM_LINEAGE_MATURATION_SIGNAL"
            ),
            "replicates_rescued_after_100k_inconclusive": maturation,
        },
        "overall": {"status": overall_status},
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
