#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from minicells.constructive_clm_001 import run_seed


FORMAL_SEEDS = (90111, 90112, 90113)


def _protocol_sha(repo_root: Path) -> str:
    path = (
        repo_root
        / "research/validations/constructive-clm-001-learned-coordinate-formation/protocol.json"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_results(output: Path, payload: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = []
    for result in payload["results"]:
        rows.append(
            {
                "seed": result["seed"],
                "pass": result["pass"],
                "active_cells": result["active_cells"],
                "late_spawns": result["late_spawns"],
                "compression": result["independent_memory_compression"],
                "singleton_mse": result["heldout_singleton_mse"],
                "pair_mse": result["heldout_pair_mse"],
                "single_route_recall": result["heldout_single_route_recall"],
                "pair_route_recall": result["heldout_pair_route_recall"],
                "value_cosine": result["alignment"]["mean_best_value_cosine"],
                "key_cosine": result["alignment"]["mean_matched_key_cosine"],
                "covered_factors": result["alignment"]["covered_factors"],
                "shuffled_pair_mse": result["shuffled_address_pair_mse"],
            }
        )
    if rows:
        with (output / "gate-summary.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Constructive CLM-001 — Learned Coordinate Formation",
        "",
        f"- Status: `{payload['status']}`",
        f"- Scientific decision: `{payload['scientific_decision']}`",
        f"- Protocol SHA-256: `{payload['protocol_sha256']}`",
        f"- Completed seeds: `{payload['completed_seeds']}`",
        f"- Missing seeds: `{payload['missing_seeds']}`",
        "",
        "## Seed summary",
        "",
        "| seed | pass | cells | late spawns | pair MSE | pair route recall | shuffled pair MSE |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        lines.append(
            f"| {result['seed']} | {result['pass']} | {result['active_cells']} | "
            f"{result['late_spawns']} | {result['heldout_pair_mse']:.6f} | "
            f"{result['heldout_pair_route_recall']:.4f} | "
            f"{result['shuffled_address_pair_mse']:.6f} |"
        )
    lines += [
        "",
        "This experiment tests learned Cell-coordinate/read-key formation only. "
        "It does not re-test Core 005 certificates, language-scale transfer, or "
        "endogenous foundation plasticity.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="artifacts/experiments/constructive-clm-001-learned-coordinate-formation",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if args.formal and args.seed:
        raise SystemExit("use either --formal or --seed, not both")

    seeds = list(FORMAL_SEEDS if args.formal else (args.seed or [1001]))
    results = [run_seed(seed) for seed in seeds]
    completed = sorted(int(result["seed"]) for result in results)
    missing = [seed for seed in FORMAL_SEEDS if seed not in completed]
    all_formal = not missing and set(completed) == set(FORMAL_SEEDS)
    passed = all(bool(result["pass"]) for result in results)

    if all_formal:
        status = (
            "LEARNED_COORDINATE_FORMATION_SUPPORTED"
            if passed
            else "LEARNED_COORDINATE_FORMATION_NOT_SUPPORTED"
        )
        scientific_decision = True
    else:
        status = "PARTIAL_RUN"
        scientific_decision = False

    payload = {
        "format": "minicells.constructive-clm-001.decision.v1",
        "experiment_id": "constructive-clm-001",
        "protocol_sha256": _protocol_sha(repo_root),
        "formal_seeds": list(FORMAL_SEEDS),
        "completed_seeds": completed,
        "missing_seeds": missing,
        "status": status,
        "scientific_decision": scientific_decision,
        "results": results,
    }
    _write_results(repo_root / args.output_dir, payload)
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
