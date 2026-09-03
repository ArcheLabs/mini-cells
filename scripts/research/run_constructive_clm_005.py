#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minicells.constructive_clm_005 import endogenous_control_smoke, run_seed

DEVELOPMENT_ONLY_SEEDS = (601, 602, 603)
EXCLUDED_DIAGNOSTIC_SEEDS = (90711, 90712, 90713)
FORMAL_SEEDS = (90811, 90812, 90813)
PROTOCOL = (
    REPO_ROOT
    / "research/validations/constructive-clm-005-scaffold-removal/protocol.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts/experiments/constructive-clm-005-scaffold-removal"
)


def _protocol_sha() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def _safe_run(seed: int) -> dict[str, Any]:
    try:
        return run_seed(int(seed))
    except Exception as exc:
        return {
            "seed": int(seed),
            "pass": False,
            "execution_error": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "controller_diagnostics": {},
            "structural_bridge": {},
            "acquisition": {},
            "mutation": {},
            "final_composition": {},
            "shared_substrate": {},
        }


def _write_results(output: Path, payload: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    gate_names = sorted(
        {
            gate
            for result in payload["results"]
            for gate in result.get("gates", {})
        }
    )
    with (output / "gate-summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "pass", *gate_names])
        writer.writeheader()
        for result in payload["results"]:
            writer.writerow(
                {
                    "seed": result["seed"],
                    "pass": result.get("pass", False),
                    **{
                        gate: result.get("gates", {}).get(gate, False)
                        for gate in gate_names
                    },
                }
            )

    controller_rows: list[dict[str, Any]] = []
    for result in payload["results"]:
        diagnostics = result.get("controller_diagnostics", {})
        if not diagnostics:
            continue
        controller_rows.append(
            {
                "seed": result["seed"],
                "controller_state_sha256": result.get("controller_state_sha256", ""),
                "router_meta_accuracy": diagnostics["router"]["heldout_meta_accuracy"],
                "growth_meta_accuracy": diagnostics["growth"]["heldout_meta_accuracy"],
                "write_meta_accuracy": diagnostics["write"]["heldout_meta_accuracy"],
                "formal_data_used": diagnostics["meta_training_uses_formal_seed_data"],
                "hidden_ids_used_as_targets": diagnostics["meta_training_uses_hidden_ids_as_targets"],
            }
        )
    if controller_rows:
        with (output / "controller-summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(controller_rows[0]))
            writer.writeheader()
            writer.writerows(controller_rows)

    stage_rows: list[dict[str, Any]] = []
    for result in payload["results"]:
        if not result.get("mutation"):
            continue
        mutation = result["mutation"]
        final = result["final_composition"]
        shared = result["shared_substrate"]
        stage_rows.append(
            {
                "seed": result["seed"],
                "final_cells": mutation["final_cells"],
                "spawned_children": mutation["conflict_spawn_count"],
                "safe_commits": mutation["safe_commit_count"],
                "conflict_write_rejections": mutation["conflict_write_rejection_count"],
                "child_reuse_hits": mutation["child_reuse_hits"],
                "child_reuse_trials": mutation["child_reuse_trials"],
                "history_mse": mutation["maximum_historical_composition_mse"],
                "unsafe_reuse_mse": mutation["mean_unsafe_reuse_historical_mse"],
                "simultaneous_mse": final["simultaneous"]["mean_mse"],
                "sequential_mse": final["sequential"]["mean_mse"],
                "simultaneous_route": final["simultaneous"]["exact_route_sequence_accuracy"],
                "sequential_route": final["sequential"]["exact_route_sequence_accuracy"],
                "max_execution_fraction": max(
                    final["simultaneous"]["cell_execution_fraction_vs_dense"],
                    final["sequential"]["cell_execution_fraction_vs_dense"],
                ),
                "shared_fit_mse": shared["safe_fit_mse"],
                "shared_history_mse": shared["safe_historical_composition_mse"],
                "shared_unsafe_history_mse": shared["unsafe_historical_composition_mse"],
            }
        )
    if stage_rows:
        with (output / "stage-summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0]))
            writer.writeheader()
            writer.writerows(stage_rows)

    lines = [
        "# Constructive CLM-005 — Scaffold Removal / Endogenous Control",
        "",
        f"- Status: `{payload['status']}`",
        f"- Scientific decision: `{payload['scientific_decision']}`",
        f"- Protocol SHA-256: `{payload['protocol_sha256']}`",
        f"- Completed seeds: `{payload['completed_seeds']}`",
        f"- Missing seeds: `{payload['missing_seeds']}`",
        "",
        "| seed | pass | router meta | growth meta | write meta | cells | children | final sim MSE | final seq MSE | history MSE | unsafe reuse MSE | max exec frac |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        if not result.get("mutation"):
            lines.append(
                f"| {result['seed']} | False | - | - | - | - | - | - | - | - | - | - |"
            )
            continue
        diag = result["controller_diagnostics"]
        mut = result["mutation"]
        final = result["final_composition"]
        lines.append(
            f"| {result['seed']} | {result['pass']} | "
            f"{diag['router']['heldout_meta_accuracy']:.4f} | "
            f"{diag['growth']['heldout_meta_accuracy']:.4f} | "
            f"{diag['write']['heldout_meta_accuracy']:.4f} | "
            f"{mut['final_cells']} | {mut['conflict_spawn_count']} | "
            f"{final['simultaneous']['mean_mse']:.3e} | "
            f"{final['sequential']['mean_mse']:.3e} | "
            f"{mut['maximum_historical_composition_mse']:.3e} | "
            f"{mut['mean_unsafe_reuse_historical_mse']:.3e} | "
            f"{max(final['simultaneous']['cell_execution_fraction_vs_dense'], final['sequential']['cell_execution_fraction_vs_dense']):.4f} |"
        )
    lines += [
        "",
        "A positive result supports a learned control-plane transition over the already-supported Cell substrate: learned pairwise routing plus learned write/grow decisions preserve the registered protected, bounded-growth and compositional invariants. The Core-005 certificate/projector remains a fixed safety primitive, and this result does not by itself establish an LLM-scale endogenous CLM.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    selected = sum(bool(flag) for flag in (args.formal, args.smoke, bool(args.seed)))
    if selected > 1:
        raise SystemExit("use exactly one of --formal, --smoke, or --seed")

    if args.smoke:
        print(
            json.dumps(
                endogenous_control_smoke(DEVELOPMENT_ONLY_SEEDS[0]),
                indent=2,
                allow_nan=False,
            )
        )
        return

    if args.formal:
        seeds = list(FORMAL_SEEDS)
    else:
        seeds = list(args.seed or [DEVELOPMENT_ONLY_SEEDS[0]])
        forbidden = sorted(
            set(seeds) & (set(FORMAL_SEEDS) | set(EXCLUDED_DIAGNOSTIC_SEEDS))
        )
        if forbidden:
            raise SystemExit(
                f"reserved/excluded seeds {forbidden} cannot be development seeds; use --formal only for the frozen formal set"
            )

    results = [_safe_run(seed) for seed in seeds]
    completed = sorted(int(result["seed"]) for result in results)
    missing = [seed for seed in FORMAL_SEEDS if seed not in completed]
    is_formal_set = not missing and set(completed) == set(FORMAL_SEEDS)
    passed = all(bool(result.get("pass")) for result in results)

    if is_formal_set:
        status = (
            "LEARNED_CONTROL_PLANE_TRANSITION_SUPPORTED"
            if passed
            else "LEARNED_CONTROL_PLANE_TRANSITION_NOT_SUPPORTED"
        )
        scientific_decision = True
    else:
        status = "DEVELOPMENT_RUN"
        scientific_decision = False

    payload = {
        "format": "minicells.constructive-clm-005.decision.v1",
        "experiment_id": "constructive-clm-005",
        "protocol_sha256": _protocol_sha(),
        "development_only_seeds": list(DEVELOPMENT_ONLY_SEEDS),
        "excluded_diagnostic_seeds": list(EXCLUDED_DIAGNOSTIC_SEEDS),
        "formal_seeds": list(FORMAL_SEEDS),
        "completed_seeds": completed,
        "missing_seeds": missing,
        "status": status,
        "scientific_decision": scientific_decision,
        "results": results,
    }
    _write_results(args.output_dir, payload)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "status",
                    "scientific_decision",
                    "completed_seeds",
                    "missing_seeds",
                    "protocol_sha256",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
