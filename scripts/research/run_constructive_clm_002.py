#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minicells.constructive_clm_002 import run_seed


OBSERVED_EXCLUDED_SEEDS = (301, 302, 303, 90311, 90312, 90313)
FORMAL_SEEDS = (90411, 90412, 90413)


def _protocol_sha() -> str:
    path = REPO_ROOT / "research/validations/constructive-clm-002-long-horizon-growth-law/protocol.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_results(output: Path, payload: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    gate_rows: list[dict] = []
    curve_rows: list[dict] = []
    for result in payload["results"]:
        gate_rows.append(
            {
                "seed": result["seed"],
                "pass": result["pass"],
                "cell_growth_exponent": result["cell_growth_exponent"],
                "latent_growth_exponent": result["latent_growth_exponent"],
                "final_cells": result["checkpoints"][-1]["cells"] if result["checkpoints"] else 0,
                "final_true_factors": result["checkpoints"][-1]["true_factors"] if result["checkpoints"] else 0,
                "final_K_over_N": result["checkpoints"][-1]["cell_to_transaction_ratio"] if result["checkpoints"] else 0.0,
                "late_spawn_rate": result["window_metrics"][-1]["spawn_rate"] if result["window_metrics"] else 0.0,
                "late_reuse_rate": result["window_metrics"][-1]["reuse_rate"] if result["window_metrics"] else 0.0,
                "last_spawn_step": result["last_spawn_step"],
                "transaction_to_cell_compression": result["transaction_to_cell_compression"],
            }
        )
        for checkpoint in result["checkpoints"]:
            curve_rows.append(
                {
                    "seed": result["seed"],
                    "transactions": checkpoint["transactions"],
                    "true_factors": checkpoint["true_factors"],
                    "cells": checkpoint["cells"],
                    "tracking_error": checkpoint["tracking_error"],
                    "K_over_N": checkpoint["cell_to_transaction_ratio"],
                    "pair_mse": checkpoint["evaluation"]["pair"]["mse"],
                    "triple_mse": checkpoint["evaluation"]["triple"]["mse"],
                    "pair_route_recall": checkpoint["evaluation"]["pair"]["route_recall"],
                    "triple_route_recall": checkpoint["evaluation"]["triple"]["route_recall"],
                    "key_cosine": checkpoint["evaluation"]["alignment"]["mean_matched_key_cosine"],
                    "effect_cosine": checkpoint["evaluation"]["alignment"]["mean_matched_effect_cosine"],
                }
            )

    if gate_rows:
        with (output / "gate-summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(gate_rows[0]))
            writer.writeheader()
            writer.writerows(gate_rows)
    if curve_rows:
        with (output / "growth-curves.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
            writer.writeheader()
            writer.writerows(curve_rows)

    lines = [
        "# Constructive CLM-002 — Long-Horizon Structure-Tracking Growth Law",
        "",
        f"- Status: `{payload['status']}`",
        f"- Scientific decision: `{payload['scientific_decision']}`",
        f"- Protocol SHA-256: `{payload['protocol_sha256']}`",
        f"- Completed seeds: `{payload['completed_seeds']}`",
        f"- Missing seeds: `{payload['missing_seeds']}`",
        "",
        "| seed | pass | growth exponent | oracle exponent | final Cells | final K/N | late spawn | late reuse | compression |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        final = result["checkpoints"][-1] if result["checkpoints"] else None
        late = result["window_metrics"][-1] if result["window_metrics"] else None
        lines.append(
            f"| {result['seed']} | {result['pass']} | {result['cell_growth_exponent']:.4f} | "
            f"{result['latent_growth_exponent']:.4f} | {final['cells'] if final else 0} | "
            f"{final['cell_to_transaction_ratio'] if final else 0.0:.6f} | "
            f"{late['spawn_rate'] if late else 0.0:.6f} | "
            f"{late['reuse_rate'] if late else 0.0:.6f} | "
            f"{result['transaction_to_cell_compression']:.2f}x |"
        )
    lines += [
        "",
        "A positive result is finite-horizon evidence only. It means learned Cell state tracks the registered sublinearly growing latent vocabulary across N=256..4096 while retention and composition remain usable. It is not an asymptotic proof of K(N)=o(N), a language-scale result, or a learned growth-policy result.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="artifacts/experiments/constructive-clm-002-long-horizon-growth-law",
    )
    args = parser.parse_args()

    if args.formal and args.seed:
        raise SystemExit("use either --formal or --seed, not both")
    if args.formal:
        seeds = list(FORMAL_SEEDS)
    else:
        seeds = list(args.seed or [301])
        forbidden = sorted(set(seeds) & set(FORMAL_SEEDS))
        if forbidden:
            raise SystemExit(
                f"formal seeds {forbidden} are frozen; use --formal to run the registered decision"
            )

    results = [run_seed(seed) for seed in seeds]
    completed = sorted(int(result["seed"]) for result in results)
    missing = [seed for seed in FORMAL_SEEDS if seed not in completed]
    all_formal = not missing and set(completed) == set(FORMAL_SEEDS)
    passed = all(bool(result["pass"]) for result in results)

    if all_formal:
        status = (
            "LONG_HORIZON_STRUCTURE_TRACKING_GROWTH_SUPPORTED"
            if passed
            else "LONG_HORIZON_STRUCTURE_TRACKING_GROWTH_NOT_SUPPORTED"
        )
        scientific_decision = True
    else:
        status = "DEVELOPMENT_RUN"
        scientific_decision = False

    payload = {
        "format": "minicells.constructive-clm-002.decision.v1",
        "experiment_id": "constructive-clm-002",
        "protocol_sha256": _protocol_sha(),
        "observed_excluded_seeds": list(OBSERVED_EXCLUDED_SEEDS),
        "formal_seeds": list(FORMAL_SEEDS),
        "completed_seeds": completed,
        "missing_seeds": missing,
        "status": status,
        "scientific_decision": scientific_decision,
        "results": results,
    }
    _write_results(REPO_ROOT / args.output_dir, payload)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in ("status", "scientific_decision", "completed_seeds", "missing_seeds")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
