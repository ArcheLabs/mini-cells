#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minicells.constructive_clm_001b import run_seed


DEVELOPMENT_ONLY_SEEDS = tuple(range(201, 211))
FORMAL_SEEDS = (90211, 90212, 90213)


def _protocol_sha() -> str:
    path = (
        REPO_ROOT
        / "research/validations/constructive-clm-001b-latent-superposition/protocol.json"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return 1e30
    if isinstance(value, dict):
        return {key: _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_strict_json_value(item) for item in value]
    return value


def _write_results(output: Path, payload: dict) -> None:
    payload = _strict_json_value(payload)
    output.mkdir(parents=True, exist_ok=True)
    (output / "decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    rows = []
    for result in payload["results"]:
        rows.append(
            {
                "seed": result["seed"],
                "pass": result["pass"],
                "cells": result["active_cells"],
                "prototypes": result["prototype_count"],
                "pair_mse": result["heldout_pair"]["mse"],
                "triple_mse": result["heldout_triple"]["mse"],
                "pair_recall": result["heldout_pair"]["route_recall"],
                "triple_recall": result["heldout_triple"]["route_recall"],
                "key_cosine": result["alignment"]["mean_matched_key_cosine"],
                "effect_cosine": result["alignment"]["mean_matched_effect_cosine"],
                "late_similarity": result["late_checkpoint_similarity_min"],
                "transaction_baseline_pair_mse": result["transaction_memory_pair_mse"],
                "shuffled_effect_pair_mse": result["shuffled_effect_pair_mse"],
            }
        )
    if rows:
        with (output / "gate-summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Constructive CLM-001B — Latent Coordinate Discovery under Superposition",
        "",
        f"- Status: `{payload['status']}`",
        f"- Scientific decision: `{payload['scientific_decision']}`",
        f"- Protocol SHA-256: `{payload['protocol_sha256']}`",
        f"- Completed seeds: `{payload['completed_seeds']}`",
        f"- Missing seeds: `{payload['missing_seeds']}`",
        "",
        "| seed | pass | cells | prototypes | pair MSE | triple MSE | pair recall | triple recall |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        lines.append(
            f"| {result['seed']} | {result['pass']} | {result['active_cells']} | "
            f"{result['prototype_count']} | {result['heldout_pair']['mse']:.8f} | "
            f"{result['heldout_triple']['mse']:.8f} | "
            f"{result['heldout_pair']['route_recall']:.4f} | "
            f"{result['heldout_triple']['route_recall']:.4f} |"
        )
    lines += [
        "",
        "001B contains no singleton training transactions. A positive result applies only to the "
        "registered pair-superposition family and does not establish arbitrary blind source separation.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="artifacts/experiments/constructive-clm-001b-latent-superposition",
    )
    args = parser.parse_args()

    if args.formal and args.seed:
        raise SystemExit("use either --formal or --seed, not both")
    if args.formal:
        seeds = list(FORMAL_SEEDS)
    else:
        seeds = list(args.seed or [201])
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
            "LATENT_COORDINATE_DISCOVERY_UNDER_SUPERPOSITION_SUPPORTED"
            if passed
            else "LATENT_COORDINATE_DISCOVERY_UNDER_SUPERPOSITION_NOT_SUPPORTED"
        )
        scientific_decision = True
    else:
        status = "DEVELOPMENT_RUN"
        scientific_decision = False

    payload = {
        "format": "minicells.constructive-clm-001b.decision.v1",
        "experiment_id": "constructive-clm-001b",
        "protocol_sha256": _protocol_sha(),
        "development_only_seeds": list(DEVELOPMENT_ONLY_SEEDS),
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
