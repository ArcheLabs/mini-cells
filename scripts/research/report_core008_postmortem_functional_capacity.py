#!/usr/bin/env python3
"""Aggregate Core 008 postmortem functional-capacity diagnostics."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "core-008-postmortem-functional-capacity"
PROTOCOL = ROOT / "research" / "validations" / "core-008-postmortem-functional-capacity" / "protocol.json"
SEEDS = (80821, 80822, 80823)


def _load() -> list[dict[str, Any]]:
    runs = []
    for seed in SEEDS:
        path = RESULTS / "seeds" / f"seed-{seed}.json"
        if path.is_file():
            runs.append(json.loads(path.read_text(encoding="utf-8")))
    return runs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    runs = _load()
    missing = [s for s in SEEDS if s not in {int(r["seed"]) for r in runs}]

    per, pca, sparse, fac, seeds = [], [], [], [], []
    for run in runs:
        seed = int(run["seed"])
        per.extend({"seed": seed, **r} for r in run["per_write_svd"])
        pca.extend({"seed": seed, **r} for r in run["global_pca"])
        sparse.extend({"seed": seed, **r} for r in run["pca_sparsity"])
        fac.extend({"seed": seed, **r} for r in run["factorized_dictionary"])
        ref = run["core008_reference"]
        per16 = next(r for r in run["per_write_svd"] if r["partition"] == "eval" and int(r["rank"]) == 16)
        pca32 = next(r for r in run["global_pca"] if r["partition"] == "eval" and int(r["dimension"]) == 32)
        best_fac = min((r for r in run["factorized_dictionary"] if r["partition"] == "eval"), key=lambda r: float(r["median_local_action_residual"]))
        seeds.append({
            "seed": seed,
            "classification": run["classification"],
            "per_write_rank16_eval_action_residual": per16["median_local_action_residual"],
            "pca32_eval_action_residual": pca32["median_local_action_residual"],
            "best_factorized_atom_rank": best_fac["atom_rank"],
            "best_factorized_eval_action_residual": best_fac["median_local_action_residual"],
            "core008_adaptive_oracle_action_residual": ref["adaptive_oracle_local_action_residual"],
            "core008_adaptive_deploy_action_residual": ref["adaptive_deploy_local_action_residual"],
        })

    _write_csv(RESULTS / "per-write-rank-summary.csv", per)
    _write_csv(RESULTS / "global-pca-summary.csv", pca)
    _write_csv(RESULTS / "pca-sparsity-summary.csv", sparse)
    _write_csv(RESULTS / "factorized-dictionary-summary.csv", fac)
    _write_csv(RESULTS / "seed-summary.csv", seeds)

    classifications = [s["classification"] for s in seeds]
    overall = classifications[0] if classifications and len(set(classifications)) == 1 else "MIXED_CAPACITY_EVIDENCE"
    if missing:
        overall = "POSTMORTEM_INCOMPLETE"
    decision = {
        "format": "minicells.core008-postmortem.functional-capacity-decision.v1",
        "status": overall,
        "scientific_decision": False,
        "source_core008_status_changed": False,
        "completed_seeds": sorted(int(r["seed"]) for r in runs),
        "missing_seeds": missing,
        "seed_classifications": {str(s["seed"]): s["classification"] for s in seeds},
        "interpretation_reference_only": float(protocol["interpretation_reference"]["core008_target_local_action_residual"]),
    }
    if seeds:
        decision["median_across_seeds"] = {
            "per_write_rank16_eval_action_residual": _median([float(s["per_write_rank16_eval_action_residual"]) for s in seeds]),
            "pca32_eval_action_residual": _median([float(s["pca32_eval_action_residual"]) for s in seeds]),
            "best_factorized_eval_action_residual": _median([float(s["best_factorized_eval_action_residual"]) for s in seeds]),
            "core008_adaptive_oracle_action_residual": _median([float(s["core008_adaptive_oracle_action_residual"]) for s in seeds]),
        }
    (RESULTS / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Core 008 Postmortem — Functional Capacity Decomposition",
        "",
        f"- Status: `{overall}`",
        "- Scientific decision: `false` (diagnostic bridge on already-observed Core 008 seeds)",
        f"- Completed seeds: `{decision['completed_seeds']}`",
        f"- Missing seeds: `{missing}`",
        "",
        "## Seed summary",
        "",
        "| seed | classification | rank-16 per-write | PCA-32 | best factorized | Core008 oracle |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for s in seeds:
        lines.append(
            f"| {s['seed']} | {s['classification']} | {float(s['per_write_rank16_eval_action_residual']):.4f} | "
            f"{float(s['pca32_eval_action_residual']):.4f} | {float(s['best_factorized_eval_action_residual']):.4f} "
            f"(r={s['best_factorized_atom_rank']}) | {float(s['core008_adaptive_oracle_action_residual']):.4f} |"
        )
    lines.extend([
        "",
        "## Reading the result",
        "",
        "Per-write SVD measures intrinsic rank. PCA-32 measures a shared linear subspace without parameter-budget matching. The factorized dictionary keeps the same 32-rank-unit budget as Core 008 but removes online allocation, certificate, and deployable routing constraints.",
        "",
        "The 0.35 value is carried over only as an interpretive reference from Core 008; it is not a new confirmatory gate.",
        "",
    ])
    (RESULTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
