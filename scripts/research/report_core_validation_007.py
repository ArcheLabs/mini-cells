#!/usr/bin/env python3
"""Generate Core Validation 007 discovery/confirmation reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "core-validation-007-functional-boundary-discovery"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def _decision_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload["decision"],
        "format": payload["format"],
        "experiment_id": payload["experiment_id"],
        "protocol_version": payload["protocol_version"],
        "phase": payload["phase"],
        "protocol_sha256": payload["protocol_sha256"],
        "parent_experiment": payload["parent_experiment"],
        "data_manifest_sha256": payload["data_manifest_sha256"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "provenance": payload["provenance"],
    }


def _discovery(payload: dict[str, Any], out: Path) -> None:
    candidate_rows, pairs, routing, modes = [], [], [], []
    for run in payload["runs"]:
        seed = run["seed"]
        candidate_rows.extend({"seed": seed, **r} for r in run["candidate_rows"])
        pairs.extend({"seed": seed, **r} for r in run["pair_diagnostics"])
        routing.extend({"seed": seed, **r} for r in run["routing_records"])
        modes.extend({"seed": seed, **r} for r in run["mode_metrics"])
    candidates = pd.DataFrame(candidate_rows)
    pair_df = pd.DataFrame(pairs)
    routing_df = pd.DataFrame(routing)
    mode_df = pd.DataFrame(modes)
    candidates.to_csv(out / "candidate-boundaries.csv", index=False)
    pair_df.to_csv(out / "pair-diagnostics.csv", index=False)
    routing_df.to_csv(out / "routing-records.csv", index=False)
    mode_df.to_csv(out / "mode-metrics.csv", index=False)
    pd.DataFrame(payload["decision"]["candidate_summary"]).to_csv(
        out / "candidate-summary.csv", index=False
    )

    if not pair_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        size = 20 + 180 * pair_df["interference"] / max(float(pair_df["interference"].max()), 1e-12)
        ax.scatter(pair_df["activation_overlap"], pair_df["write_overlap"], s=size)
        ax.set_xlabel("activation-subspace overlap")
        ax.set_ylabel("write-demand overlap")
        ax.set_title("Core 007 discovery: bubble size = direct interference")
        fig.tight_layout()
        fig.savefig(out / "activation-write-interference.png", dpi=180)
        plt.close(fig)
    summary = pd.DataFrame(payload["decision"]["candidate_summary"])
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(summary["candidate"], summary["mean_selection_score"])
        ax.set_ylabel("frozen discovery score")
        ax.set_title("Core 007 functional-boundary candidates")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(out / "candidate-score.png", dpi=180)
        plt.close(fig)

    lines = [
        "# Core Validation 007 Discovery",
        "",
        "Discovery is mechanism selection only; it is not a scientific confirmation.",
        "",
        f"- Status: `{payload['decision']['status']}`",
        f"- Provisional winner: `{payload['decision']['provisional_winner']}`",
        f"- Winner meets routing floor: `{payload['decision']['winner_meets_routing_floor']}`",
        "",
        "## Candidate summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No candidate rows.",
        "",
        "Confirmation seeds remain unopened until `winner-lock.json` is committed.",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _confirmation(payload: dict[str, Any], out: Path) -> None:
    gates, tx, splits, routing, ranks, causal, modes = [], [], [], [], [], [], []
    for run in payload["runs"]:
        seed = run["seed"]
        gates.append({"seed": seed, "pass": run["pass"], **run["candidate"], **{f"gate_{k}": v for k, v in run["gates"].items()}})
        tx.extend({"seed": seed, **r} for r in run["records"])
        splits.extend({"seed": seed, **r} for r in run["split_records"])
        routing.extend({"seed": seed, **r} for r in run["routing_records"])
        ranks.extend({"seed": seed, **r} for r in run["rank_records"])
        causal.extend({"seed": seed, **r} for r in run["causal_records"])
        modes.extend({"seed": seed, **r} for r in run["mode_metrics"])
    gate_df = pd.DataFrame(gates)
    tx_df = pd.DataFrame(tx)
    split_df = pd.DataFrame(splits)
    routing_df = pd.DataFrame(routing)
    rank_df = pd.DataFrame(ranks)
    causal_df = pd.DataFrame(causal)
    mode_df = pd.DataFrame(modes)
    gate_df.to_csv(out / "gate-summary.csv", index=False)
    tx_df.to_csv(out / "transaction-records.csv", index=False)
    split_df.to_csv(out / "split-records.csv", index=False)
    routing_df.to_csv(out / "routing-records.csv", index=False)
    rank_df.to_csv(out / "rank-trajectory.csv", index=False)
    causal_df.to_csv(out / "causal-load.csv", index=False)
    mode_df.to_csv(out / "mode-metrics.csv", index=False)

    if not split_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(split_df))
        ax.scatter(list(x), split_df["conflict_before"], label="before")
        ax.scatter(list(x), split_df["conflict_after"], label="after")
        ax.set_xlabel("functional mitosis event")
        ax.set_ylabel("certificate conflict fraction")
        ax.set_title("Core 007 functional split conflict relief")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "split-conflict-relief.png", dpi=180)
        plt.close(fig)
    if not rank_df.empty:
        grouped = rank_df.groupby("transaction", as_index=False).agg(
            participation_rank=("participation_rank", "median"),
            dependency_tokens=("dependency_tokens", "median"),
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(grouped["transaction"], grouped["participation_rank"])
        ax.set_xlabel("transaction")
        ax.set_ylabel("median Cell participation rank")
        ax.set_title("Core 007 functional-rank trajectory")
        fig.tight_layout()
        fig.savefig(out / "rank-growth.png", dpi=180)
        plt.close(fig)
    if not routing_df.empty:
        grouped = routing_df.groupby("transaction", as_index=False).agg(agreement=("agreement", "mean"))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(grouped["transaction"], grouped["agreement"])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("transaction")
        ax.set_ylabel("z-router / oracle mode agreement")
        ax.set_title("Core 007 deployable routing identifiability")
        fig.tight_layout()
        fig.savefig(out / "routing-agreement.png", dpi=180)
        plt.close(fig)

    lines = [
        "# Core Validation 007 Confirmation",
        "",
        f"- Status: `{payload['decision']['status']}`",
        f"- Winner: `{payload['decision']['winner']}`",
        f"- Passed seeds: `{payload['decision']['passed_seeds']}/{payload['decision']['total_seeds']}`",
        "",
        "## Gate summary",
        "",
        gate_df.to_markdown(index=False) if not gate_df.empty else "No confirmation rows.",
        "",
        "## Interpretation",
        "",
        "The oracle functional router is an upper bound only. A positive scientific decision additionally requires the inference-visible z-only router to remain close to the oracle under the frozen gates.",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = args.out / args.phase
    raw = out / "raw.json"
    if not raw.is_file():
        raise FileNotFoundError(raw)
    payload = json.loads(raw.read_text(encoding="utf-8"))
    if payload.get("phase") != args.phase:
        raise RuntimeError("phase mismatch between command and raw payload")
    out.mkdir(parents=True, exist_ok=True)
    decision = _decision_envelope(payload)
    (out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.phase == "discovery":
        _discovery(payload, out)
    else:
        _confirmation(payload, out)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
