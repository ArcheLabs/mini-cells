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

from minicells.constructive_clm_003 import VARIANTS, protection_only_smoke, run_seed

DEVELOPMENT_ONLY_SEEDS = (401, 402, 403)
FORMAL_SEEDS = (90511, 90512, 90513)
PROTOCOL = (
    REPO_ROOT
    / "research/validations/constructive-clm-003-protected-growing-cells/protocol.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts/experiments/constructive-clm-003-protected-growing-cells"
)


def _protocol_sha() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def _safe_run(seed: int) -> dict[str, Any]:
    try:
        return run_seed(int(seed))
    except Exception as exc:  # Formal negative results must still serialize.
        return {
            "seed": int(seed),
            "pass": False,
            "execution_error": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "variants": {},
            "structural_bridge": {},
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
    with (output / "gate-summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["seed", "pass", *gate_names]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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

    variant_rows: list[dict[str, Any]] = []
    for result in payload["results"]:
        for variant in VARIANTS:
            row = result.get("variants", {}).get(variant)
            if row is None:
                continue
            variant_rows.append(
                {
                    "seed": result["seed"],
                    "variant": variant,
                    "final_cells": row["final_cells"],
                    "child_count": row["child_count"],
                    "acquisition_gain": row["acquisition_gain"],
                    "accepted_acquisition": row["accepted_acquisition"],
                    "rejected_acquisition": row["rejected_acquisition"],
                    "growth_rescues": row["growth_rescues"],
                    "final_historical_regression_mse": row[
                        "final_historical_regression_mse"
                    ],
                    "cumulative_positive_historical_regression": row[
                        "cumulative_positive_historical_regression"
                    ],
                    "replay_accesses": row["replay_accesses"],
                    "exact_mode_route_accuracy": row["route"]["exact_mode_accuracy"],
                    "tail_child_route_accuracy": row["tail_child_route_accuracy"],
                    "tail_spawns": row["tail_spawns"],
                    "final_behavior_mse": row["final_behavior"]["mse"],
                    "compression": row["write_transaction_to_cell_compression"],
                }
            )
    if variant_rows:
        with (output / "variant-summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(variant_rows[0]))
            writer.writeheader()
            writer.writerows(variant_rows)

    lines = [
        "# Constructive CLM-003 — Protected Learned/Growing Cells",
        "",
        f"- Status: `{payload['status']}`",
        f"- Scientific decision: `{payload['scientific_decision']}`",
        f"- Protocol SHA-256: `{payload['protocol_sha256']}`",
        f"- Completed seeds: `{payload['completed_seeds']}`",
        f"- Missing seeds: `{payload['missing_seeds']}`",
        "",
        "| seed | pass | roots | cert cells | cert gain | cert history MSE | cert replay | route acc | replay gain |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        if not result.get("variants"):
            lines.append(
                f"| {result['seed']} | False | - | - | - | - | - | - | - |"
            )
            continue
        cert = result["variants"]["certificate_growth"]
        replay = result["variants"]["replay_growth_oracle"]
        lines.append(
            f"| {result['seed']} | {result['pass']} | "
            f"{result['structural_bridge']['root_cells']} | {cert['final_cells']} | "
            f"{cert['acquisition_gain']:.6f} | "
            f"{cert['final_historical_regression_mse']:.3e} | "
            f"{cert['replay_accesses']} | "
            f"{cert['route']['exact_mode_accuracy']:.4f} | "
            f"{replay['acquisition_gain']:.6f} |"
        )
    lines += [
        "",
        "A positive CLM-003 result is an integration result: the Core-005 certificate is reused inside learned/root-routed Cell lineages. It is not a new proof of the certificate principle and does not establish language-scale or fully endogenous growth.",
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
        smoke = protection_only_smoke(DEVELOPMENT_ONLY_SEEDS[0])
        print(json.dumps(smoke, indent=2, allow_nan=False))
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
            "PROTECTED_GROWING_CELL_INTEGRATION_SUPPORTED"
            if passed
            else "PROTECTED_GROWING_CELL_INTEGRATION_NOT_SUPPORTED"
        )
        scientific_decision = True
    else:
        status = "DEVELOPMENT_RUN"
        scientific_decision = False

    payload = {
        "format": "minicells.constructive-clm-003.decision.v1",
        "experiment_id": "constructive-clm-003",
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
            {
                key: payload[key]
                for key in (
                    "status",
                    "scientific_decision",
                    "completed_seeds",
                    "missing_seeds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
