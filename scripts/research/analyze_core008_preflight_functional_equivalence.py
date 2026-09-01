#!/usr/bin/env python3
"""Artifact-only bridge audit between Core 007 and Core 008.

This is deliberately NOT a scientific decision runner. It asks whether the
published Core 007 confirmation artifacts are sufficient to support the claim
that oracle/deploy routing disagreements are functionally equivalent, and
quantifies the scale of the Cell-ablation signal that can be recovered without
rehydrating the lost Kaggle cache.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE007 = ROOT / "artifacts" / "experiments" / "core-validation-007-functional-boundary-discovery" / "confirmation"
OUTDIR = ROOT / "artifacts" / "experiments" / "core-008-preflight-functional-equivalence"
SEEDS = (80721, 80722)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    p = (len(xs) - 1) * q
    lo = math.floor(p)
    hi = math.ceil(p)
    if lo == hi:
        return xs[lo]
    w = p - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def main() -> int:
    gate_rows = _rows(CORE007 / "gate-summary.csv")
    causal_rows = _rows(CORE007 / "causal-load.csv")

    causal_by_seed: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in causal_rows:
        causal_by_seed[int(row["seed"])].append(row)

    seed_results: list[dict[str, object]] = []
    for gate in gate_rows:
        seed = int(gate["seed"])
        if seed not in SEEDS:
            continue

        oracle_nll = _f(gate, "oracle_eval_nll")
        deploy_nll = _f(gate, "deploy_eval_nll")
        absolute_nll_gap = abs(deploy_nll - oracle_nll)
        relative_nll_gap = absolute_nll_gap / max(abs(oracle_nll), 1e-12)
        eval_agreement = _f(gate, "eval_routing_agreement")
        evaluated_causal = [
            abs(_f(row, "causal_delta_nll"))
            for row in causal_by_seed[seed]
            if int(float(row.get("eval_sequences", "0"))) > 0
        ]
        max_causal_ratio = max(evaluated_causal, default=0.0) / max(abs(oracle_nll), 1e-12)

        seed_results.append(
            {
                "seed": seed,
                "oracle_eval_nll": oracle_nll,
                "deploy_eval_nll": deploy_nll,
                "absolute_oracle_deploy_nll_gap": absolute_nll_gap,
                "absolute_relative_oracle_deploy_nll_gap": relative_nll_gap,
                "registered_one_sided_deploy_relative_nll_gap": _f(gate, "deploy_relative_nll_gap"),
                "eval_mode_agreement": eval_agreement,
                "eval_mode_disagreement": 1.0 - eval_agreement,
                "cumulative_new_gain": _f(gate, "cumulative_new_gain"),
                "evaluated_causal_cells": len(evaluated_causal),
                "abs_causal_delta_nll_mean": statistics.fmean(evaluated_causal) if evaluated_causal else 0.0,
                "abs_causal_delta_nll_median": statistics.median(evaluated_causal) if evaluated_causal else 0.0,
                "abs_causal_delta_nll_p95": _quantile(evaluated_causal, 0.95),
                "abs_causal_delta_nll_max": max(evaluated_causal, default=0.0),
                "max_abs_causal_delta_over_eval_nll": max_causal_ratio,
            }
        )

    # This is only a proxy because causal-load.csv contains per-Cell ablations,
    # not per-route counterfactual swaps. The exact bridge uses a stronger local
    # normalization against the foundation path after deterministic rehydration.
    weak_effect_proxy = bool(seed_results) and all(
        float(r["max_abs_causal_delta_over_eval_nll"]) <= 1e-3 for r in seed_results
    )

    result = {
        "format": "minicells.core008-preflight.functional-equivalence-artifact-audit.v1",
        "scientific_decision": False,
        "source": "published Core 007 confirmation artifacts only",
        "completed_seeds": [int(r["seed"]) for r in seed_results],
        "seed_results": seed_results,
        "artifact_sufficiency": {
            "can_measure_eval_route_disagreement_from_gate_summary": True,
            "can_measure_oracle_deploy_whole_model_nll_gap": True,
            "can_measure_per_cell_ablation_scale": True,
            "can_compute_pairwise_cell_output_distance_Eij": False,
            "can_compute_mode_swap_logit_KL": False,
            "can_compute_normalized_local_functional_regret": False,
            "missing_for_exact_bridge": [
                "final candidate Cell matrices A (or equivalent Cell-output state)",
                "per-evaluation-sequence projected z / hidden state",
                "LM-head/logit state sufficient for counterfactual swap evaluation"
            ]
        },
        "diagnostic_conclusion": {
            "functional_equivalence_established": False,
            "weak_cell_effect_proxy_present": weak_effect_proxy,
            "weak_effect_proxy_definition": "max evaluated per-Cell |ablation delta NLL| / whole-model eval NLL <= 1e-3 in each completed seed",
            "interpretation": (
                "Near-zero oracle/deploy whole-model NLL gap cannot establish mode functional equivalence from the published artifacts. "
                "The artifact-only causal scale can identify a weak-effect confound proxy, but exact equivalence requires route-level "
                "counterfactual Cell-output/logit evaluation after deterministic rehydration."
            )
        },
        "recomputation": {
            "original_lost_kaggle_hidden_cache_required": False,
            "fresh_gpu_rehydration_required_for_exact_bridge": True,
            "reason": (
                "Core 007 deterministically re-selects pinned model/dataset inputs and verifies the frozen manifest; frozen-hidden.pt is a cache and is regenerated when absent."
            )
        }
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "artifact-audit.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
