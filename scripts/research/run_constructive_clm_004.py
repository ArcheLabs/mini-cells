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

from minicells.constructive_clm_004 import model_level_smoke, run_seed

DEVELOPMENT_ONLY_SEEDS = (501, 502, 503)
FORMAL_SEEDS = (90611, 90612, 90613)
PROTOCOL = REPO_ROOT / "research/validations/constructive-clm-004-model-level-multicell-computation/protocol.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/experiments/constructive-clm-004-model-level-multicell-computation"


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
            "structural_bridge": {},
            "acquisition": {},
            "simultaneous": {},
            "sequential": {},
            "protected_mutation": {},
        }


def _write_results(output: Path, payload: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    gate_names = sorted(
        {gate for result in payload["results"] for gate in result.get("gates", {})}
    )
    with (output / "gate-summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "pass", *gate_names])
        writer.writeheader()
        for result in payload["results"]:
            writer.writerow(
                {
                    "seed": result["seed"],
                    "pass": result.get("pass", False),
                    **{gate: result.get("gates", {}).get(gate, False) for gate in gate_names},
                }
            )

    composition_rows: list[dict[str, Any]] = []
    for result in payload["results"]:
        for mode in ("simultaneous", "sequential"):
            row = result.get(mode)
            if not row:
                continue
            composition_rows.append(
                {
                    "seed": result["seed"],
                    "mode": mode,
                    "mean_mse": row["mean_mse"],
                    "max_mse": row["max_mse"],
                    "route_accuracy": row["exact_route_sequence_accuracy"],
                    "mean_active_cells": row["mean_active_cells"],
                    "total_cells": row["total_cells"],
                    "execution_fraction_vs_dense": row["cell_execution_fraction_vs_dense"],
                    "single_cell_baseline_mse": row["single_cell_baseline_mse"],
                    "wrong_semantics_baseline_mse": row["wrong_semantics_baseline_mse"],
                    "dense_all_cells_baseline_mse": row["dense_all_cells_baseline_mse"],
                }
            )
    if composition_rows:
        with (output / "composition-summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(composition_rows[0]))
            writer.writeheader()
            writer.writerows(composition_rows)

    mutation_rows = [
        {"seed": result["seed"], **result["protected_mutation"]}
        for result in payload["results"]
        if result.get("protected_mutation")
    ]
    if mutation_rows:
        with (output / "mutation-summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mutation_rows[0]))
            writer.writeheader()
            writer.writerows(mutation_rows)

    lines = [
        "# Constructive CLM-004 — Model-Level Multi-Cell Computation",
        "",
        f"- Status: `{payload['status']}`",
        f"- Scientific decision: `{payload['scientific_decision']}`",
        f"- Protocol SHA-256: `{payload['protocol_sha256']}`",
        f"- Completed seeds: `{payload['completed_seeds']}`",
        f"- Missing seeds: `{payload['missing_seeds']}`",
        "",
        "| seed | pass | sim MSE | seq MSE | sim route | seq route | seq order effect | exec fraction | protected hist MSE | unsafe hist MSE |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        if not result.get("simultaneous") or not result.get("sequential"):
            lines.append(f"| {result['seed']} | False | - | - | - | - | - | - | - | - |")
            continue
        sim = result["simultaneous"]
        seq = result["sequential"]
        mutation = result["protected_mutation"]
        lines.append(
            f"| {result['seed']} | {result['pass']} | "
            f"{sim['mean_mse']:.3e} | {seq['mean_mse']:.3e} | "
            f"{sim['exact_route_sequence_accuracy']:.4f} | {seq['exact_route_sequence_accuracy']:.4f} | "
            f"{seq['mean_true_order_effect_mse']:.3e} | "
            f"{max(sim['cell_execution_fraction_vs_dense'], seq['cell_execution_fraction_vs_dense']):.4f} | "
            f"{mutation['safe_historical_composition_mse']:.3e} | "
            f"{mutation['unsafe_historical_composition_mse']:.3e} |"
        )
    lines += [
        "",
        "A positive result supports the registered controlled claim that learned route-addressed Cell operators can compose at model level with sparse execution and preserve a replay-free protected-mutation invariant. It does not establish natural-language or fully endogenous routing/growth.",
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
        print(json.dumps(model_level_smoke(DEVELOPMENT_ONLY_SEEDS[0]), indent=2, allow_nan=False))
        return

    if args.formal:
        seeds = list(FORMAL_SEEDS)
    else:
        seeds = list(args.seed or [DEVELOPMENT_ONLY_SEEDS[0]])
        forbidden = sorted(set(seeds) & set(FORMAL_SEEDS))
        if forbidden:
            raise SystemExit(
                f"formal seeds {forbidden} are frozen; use --formal to run the registered decision"
            )

    results = [_safe_run(seed) for seed in seeds]
    completed = sorted(int(result["seed"]) for result in results)
    missing = [seed for seed in FORMAL_SEEDS if seed not in completed]
    is_formal_set = not missing and set(completed) == set(FORMAL_SEEDS)
    passed = all(bool(result.get("pass")) for result in results)

    if is_formal_set:
        status = (
            "MODEL_LEVEL_MULTICELL_COMPUTATION_SUPPORTED"
            if passed
            else "MODEL_LEVEL_MULTICELL_COMPUTATION_NOT_SUPPORTED"
        )
        scientific_decision = True
    else:
        status = "DEVELOPMENT_RUN"
        scientific_decision = False

    payload = {
        "format": "minicells.constructive-clm-004.decision.v1",
        "experiment_id": "constructive-clm-004",
        "protocol_sha256": _protocol_sha(),
        "development_only_seeds": list(DEVELOPMENT_ONLY_SEEDS),
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
            {key: payload[key] for key in ("status", "scientific_decision", "completed_seeds", "missing_seeds")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
