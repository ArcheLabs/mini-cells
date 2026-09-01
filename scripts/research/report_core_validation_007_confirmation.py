#!/usr/bin/env python3
"""Aggregate resumable Core Validation 007 confirmation seed checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from minicells.real_representation_007_experiment import summarize_confirmation

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery"
AMENDMENT = VALIDATION / "confirmation-protocol-v1.1.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-007-functional-boundary-discovery"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    amendment = _load_json(AMENDMENT)
    amendment_sha = _sha256(AMENDMENT)
    expected = [int(x) for x in amendment["confirmation_seeds"]]
    phase = args.out / "confirmation"
    seeds_dir = phase / "seeds"
    failures_dir = phase / "failures"
    phase.mkdir(parents=True, exist_ok=True)

    checkpoints: dict[int, dict[str, Any]] = {}
    failures: dict[int, dict[str, Any]] = {}
    for seed in expected:
        path = seeds_dir / f"seed-{seed}.json"
        if path.is_file():
            payload = _load_json(path)
            if payload.get("complete") is not True:
                raise RuntimeError(f"seed {seed} checkpoint is not complete")
            if payload.get("seed") != seed:
                raise RuntimeError(f"seed identity mismatch in {path}")
            if payload.get("confirmation_protocol_sha256") != amendment_sha:
                raise RuntimeError(f"seed {seed} was produced by another confirmation amendment")
            if payload.get("base_protocol_sha256") != amendment["base_discovery_protocol_sha256"]:
                raise RuntimeError(f"seed {seed} base protocol mismatch")
            if payload.get("data_manifest_sha256") != amendment["expected_data_manifest_sha256"]:
                raise RuntimeError(f"seed {seed} data manifest mismatch")
            if payload.get("winner") != amendment["winner"]:
                raise RuntimeError(f"seed {seed} winner mismatch")
            checkpoints[seed] = payload
        failure = failures_dir / f"seed-{seed}.json"
        if failure.is_file() and seed not in checkpoints:
            failures[seed] = _load_json(failure)

    completed = [s for s in expected if s in checkpoints]
    failed = [s for s in expected if s in failures]
    pending = [s for s in expected if s not in checkpoints and s not in failures]
    scientific_hashes = {
        str(checkpoints[s].get("scientific_code_sha256")) for s in completed
    }
    if len(scientific_hashes) > 1:
        raise RuntimeError("completed seeds were produced by different scientific code identities")

    runs = [checkpoints[s]["run"] for s in expected if s in checkpoints]
    if len(completed) == len(expected):
        decision = summarize_confirmation(
            runs,
            winner=str(amendment["winner"]),
            positive_status=str(amendment["positive_status"]),
            negative_status=str(amendment["negative_status"]),
        )
        decision.update(
            {
                "confirmation_protocol_version": amendment["protocol_version"],
                "confirmation_protocol_sha256": amendment_sha,
                "completed_seeds": completed,
                "failed_seeds": [],
                "pending_seeds": [],
            }
        )
    else:
        decision = {
            "status": str(amendment["allowed_partial_status"]),
            "scientific_decision": False,
            "pass": None,
            "winner": amendment["winner"],
            "confirmation_protocol_version": amendment["protocol_version"],
            "confirmation_protocol_sha256": amendment_sha,
            "completed_seeds": completed,
            "failed_seeds": failed,
            "pending_seeds": pending,
            "passed_seeds_so_far": sum(bool(r["pass"]) for r in runs),
            "total_required_seeds": len(expected),
            "reason": "Scientific decision is forbidden until all amended confirmation seeds have complete matching checkpoints.",
        }

    aggregate = {
        "format": "minicells.core-validation.functional-boundary-confirmation-aggregate.v1",
        "experiment_id": "core-validation-007",
        "phase": "confirmation",
        "base_protocol_sha256": amendment["base_discovery_protocol_sha256"],
        "confirmation_protocol_sha256": amendment_sha,
        "data_manifest_sha256": amendment["expected_data_manifest_sha256"],
        "winner": amendment["winner"],
        "scientific_code_sha256": next(iter(scientific_hashes), None),
        "runs": runs,
        "failures": [failures[s] for s in failed],
        "decision": decision,
    }
    (phase / "raw.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (phase / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    gate_rows: list[dict[str, Any]] = []
    tx_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    routing_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    for run in runs:
        seed = int(run["seed"])
        gate_rows.append(
            {
                "seed": seed,
                "pass": run["pass"],
                **run["candidate"],
                **{f"gate_{k}": v for k, v in run["gates"].items()},
            }
        )
        tx_rows.extend({"seed": seed, **r} for r in run["records"])
        split_rows.extend({"seed": seed, **r} for r in run["split_records"])
        routing_rows.extend({"seed": seed, **r} for r in run["routing_records"])
        rank_rows.extend({"seed": seed, **r} for r in run["rank_records"])
        causal_rows.extend({"seed": seed, **r} for r in run["causal_records"])
        mode_rows.extend({"seed": seed, **r} for r in run["mode_metrics"])

    pd.DataFrame(gate_rows).to_csv(phase / "gate-summary.csv", index=False)
    pd.DataFrame(tx_rows).to_csv(phase / "transaction-records.csv", index=False)
    pd.DataFrame(split_rows).to_csv(phase / "split-records.csv", index=False)
    pd.DataFrame(routing_rows).to_csv(phase / "routing-records.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(phase / "rank-trajectory.csv", index=False)
    pd.DataFrame(causal_rows).to_csv(phase / "causal-load.csv", index=False)
    pd.DataFrame(mode_rows).to_csv(phase / "mode-metrics.csv", index=False)

    lines = [
        "# Core Validation 007 Confirmation v1.1",
        "",
        f"- Status: `{decision['status']}`",
        f"- Scientific decision: `{decision['scientific_decision']}`",
        f"- Winner: `{amendment['winner']}`",
        f"- Completed: `{completed}`",
        f"- Failed: `{failed}`",
        f"- Pending: `{pending}`",
        "",
        "The original 80711/80712/80713 confirmation set is retired by the infrastructure amendment and is not merged into this decision.",
        "",
    ]
    if gate_rows:
        lines.extend(["## Completed seed gates", "", pd.DataFrame(gate_rows).to_markdown(index=False), ""])
    (phase / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
