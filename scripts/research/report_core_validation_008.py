#!/usr/bin/env python3
"""Aggregate frozen Core Validation 008 formal seeds and emit the decision."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-008-certified-functional-atoms"
PROTOCOL_PATH = VALIDATION / "protocol.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-008-certified-functional-atoms"
_EPS = 1e-12


def _le_ratio(a: float, ratio: float, b: float) -> bool:
    return a <= ratio * max(b, _EPS)


def _seed_gates(run: dict, gates: dict) -> dict[str, bool]:
    v = run["variant_results"]
    a = v["adaptive_atoms"]
    r1 = v["rank1_atoms"]
    mono = v["monolithic_certified"]
    rank1_comp = (
        (
            _le_ratio(a["median_eval_deploy_local_action_residual"], 0.85, r1["median_eval_deploy_local_action_residual"])
            or _le_ratio(a["spawned_atoms_per_train_sequence"], 0.75, r1["spawned_atoms_per_train_sequence"])
        )
        and _le_ratio(a["median_eval_deploy_local_action_residual"], 1.10, r1["median_eval_deploy_local_action_residual"])
        and _le_ratio(a["spawned_atoms_per_train_sequence"], 1.10, r1["spawned_atoms_per_train_sequence"])
    )
    mono_comp = (
        _le_ratio(a["unresolved_write_fraction"], 0.80, mono["unresolved_write_fraction"])
        or _le_ratio(a["median_eval_deploy_local_action_residual"], 0.85, mono["median_eval_deploy_local_action_residual"])
    )
    return {
        "oracle_action": a["median_eval_oracle_local_action_residual"] <= gates["adaptive_maximum_median_oracle_local_action_residual"],
        "deploy_action": a["median_eval_deploy_local_action_residual"] <= gates["adaptive_maximum_median_deploy_local_action_residual"],
        "unresolved": a["unresolved_write_fraction"] <= gates["adaptive_maximum_unresolved_write_fraction"],
        "reuse": a["online_reuse_fraction"] >= gates["adaptive_minimum_reuse_fraction"],
        "growth": a["spawned_atoms_per_train_sequence"] <= gates["adaptive_maximum_spawned_atoms_per_train_sequence"],
        "budget": a["factor_budget_fraction"] <= gates["adaptive_maximum_factor_budget_fraction"],
        "hidden_history_drift": a["p95_hidden_history_drift"] <= gates["adaptive_maximum_p95_hidden_history_drift"],
        "certificate_constraint": max(x["maximum_certificate_constraint_violation"] for x in v.values()) <= gates["maximum_certificate_constraint_violation"],
        "rank1_comparative": rank1_comp,
        "monolithic_comparative": mono_comp,
    }


def main() -> int:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    out = DEFAULT_OUT
    seeds = list(map(int, protocol["replication"]["formal_seeds"]))
    runs, missing = [], []
    for seed in seeds:
        path = out / "seeds" / f"seed-{seed}.json"
        if not path.is_file():
            missing.append(seed)
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        if str(run.get("data_manifest_sha256")) != str(protocol["data"]["expected_manifest_sha256"]):
            raise RuntimeError(f"seed {seed}: data identity mismatch")
        runs.append(run)

    out.mkdir(parents=True, exist_ok=True)
    gate_rows, variant_rows = [], []
    for run in runs:
        seed = int(run["seed"])
        per = _seed_gates(run, protocol["confirmation_gates"])
        gate_rows.append({"seed": seed, "pass": all(per.values()), **per})
        for name, row in run["variant_results"].items():
            variant_rows.append({
                "seed": seed,
                "variant": name,
                "atom_count": row["atom_count"],
                "total_rank_units": row["total_rank_units"],
                "factor_budget_fraction": row["factor_budget_fraction"],
                "online_reuse_fraction": row["online_reuse_fraction"],
                "spawned_atoms_per_train_sequence": row["spawned_atoms_per_train_sequence"],
                "unresolved_write_fraction": row["unresolved_write_fraction"],
                "median_eval_oracle_local_action_residual": row["median_eval_oracle_local_action_residual"],
                "median_eval_deploy_local_action_residual": row["median_eval_deploy_local_action_residual"],
                "p95_hidden_history_drift": row["p95_hidden_history_drift"],
                "maximum_certificate_constraint_violation": row["maximum_certificate_constraint_violation"],
                "median_coefficient_cosine": row["median_coefficient_cosine"],
                "median_active_set_overlap": row["median_active_set_overlap"],
            })

    with (out / "gate-summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(gate_rows[0]) if gate_rows else ["seed", "pass"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(gate_rows)
    with (out / "variant-summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(variant_rows[0]) if variant_rows else ["seed", "variant"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(variant_rows)

    complete = not missing and len(runs) == len(seeds)
    passed = complete and all(bool(r["pass"]) for r in gate_rows)
    if not complete:
        status, scientific = "CORE008_CONFIRMATION_INCOMPLETE", False
    else:
        status = protocol["confirmation_gates"]["positive_status"] if passed else protocol["confirmation_gates"]["negative_status"]
        scientific = True
    decision = {
        "format": "minicells.core-validation.certified-functional-atoms-decision.v1",
        "experiment_id": "core-validation-008",
        "protocol_version": protocol["protocol_version"],
        "scientific_decision": scientific,
        "status": status,
        "supported": passed if complete else None,
        "completed_seeds": [int(r["seed"]) for r in runs],
        "missing_seeds": missing,
        "passed_seeds": sum(bool(r["pass"]) for r in gate_rows),
        "total_formal_seeds": len(seeds),
        "data_manifest_sha256": protocol["data"]["expected_manifest_sha256"],
        "gate_rows": gate_rows,
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Core Validation 008 — Certified Adaptive Functional Atoms", "",
        f"- Status: `{status}`", f"- Scientific decision: `{str(scientific).lower()}`",
        f"- Completed formal seeds: `{decision['completed_seeds']}`", f"- Missing formal seeds: `{missing}`", "",
        "## What is being tested", "",
        "The certificate mechanism is held fixed. The experiment compares a monolithic certified transform with rank-1, rank-2, rank-4 and adaptive-rank sparse functional atoms under the same conceptual 4096-scalar factor budget. Primary gates use normalized write/action geometry rather than raw whole-model NLL.", "",
        "## Seed gates", "",
        "| seed | pass | oracle | deploy | unresolved | reuse | growth | drift | cert | rank1 cmp | mono cmp |",
        "|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in gate_rows:
        lines.append("| {seed} | {pass} | {oracle_action} | {deploy_action} | {unresolved} | {reuse} | {growth} | {hidden_history_drift} | {certificate_constraint} | {rank1_comparative} | {monolithic_comparative} |".format(**r))
    lines += ["", "## Interpretation boundary", "", "A positive result supports the functional-atom mechanism only in frozen Pythia representations with linear projected write transforms. A negative result blocks the current compositional write-demand/certificate geometry rather than disproving continual learning in general.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
