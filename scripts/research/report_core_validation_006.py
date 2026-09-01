#!/usr/bin/env python3
"""Report Core Validation 006 outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "core-validation-006-real-representation-continual-plasticity"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def _flatten(payload: dict[str, Any]):
    tx, rank, split, checkpoint, eval_rows, causal, gates = [], [], [], [], [], [], []
    for run in payload["runs"]:
        seed = run["seed"]
        g = run["gate_summary"]
        gates.append(
            {
                "seed": seed,
                "pass": g["pass"],
                "registered_regression_ratio_vs_unsafe": g[
                    "registered_regression_ratio_vs_unsafe"
                ],
                "gain_ratio_vs_replay": g["gain_ratio_vs_replay"],
                "midstream_energy_rank_fraction": g["midstream_energy_rank_fraction"],
                "midstream_reuse_ratio": g["midstream_reuse_ratio"],
                "median_split_conflict_reduction": g[
                    "median_split_conflict_reduction"
                ],
                "spawned_fraction_of_addresses": g["spawned_fraction_of_addresses"],
                "causal_nonzero_cells": g["causal_nonzero_cells"],
                **{f"gate_{k}": v for k, v in g["gates"].items()},
            }
        )
        for variant, result in run["variants"].items():
            for row in result["records"]:
                tx.append({"seed": seed, "variant": variant, **row})
            for row in result["rank_records"]:
                rank.append({"seed": seed, **row})
            for row in result["split_records"]:
                split.append({"seed": seed, "variant": variant, **row})
        checkpoint.extend(run["checkpoint_records"])
        eval_rows.extend(run.get("eval_records", []))
        causal.extend(run["causal_records"])
    return (
        pd.DataFrame(tx),
        pd.DataFrame(rank),
        pd.DataFrame(split),
        pd.DataFrame(checkpoint),
        pd.DataFrame(eval_rows),
        pd.DataFrame(causal),
        pd.DataFrame(gates),
    )


def _plot_rank_growth(rank: pd.DataFrame, dest: Path) -> None:
    rows = rank[rank["variant"] == "certificate_mitosis"].copy()
    if rows.empty:
        return
    grouped = rows.groupby("transaction", as_index=False).agg(
        median_energy_rank=("energy_rank_99", "median"),
        median_participation_rank=("participation_rank", "median"),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped["transaction"], grouped["median_energy_rank"], label="99% energy rank")
    ax.plot(
        grouped["transaction"],
        grouped["median_participation_rank"],
        label="participation rank",
    )
    ax.set_xlabel("Transaction")
    ax.set_ylabel("Median effective rank across active Cells")
    ax.set_title("Core 006: real-representation rank growth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest, dpi=180)
    plt.close(fig)


def _plot_dependency_rank_causal(causal: pd.DataFrame, dest: Path) -> None:
    rows = causal[causal["variant"] == "certificate_mitosis"].copy()
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    size = 30 + 120 * rows["causal_delta_nll"].abs() / max(
        float(rows["causal_delta_nll"].abs().max()), 1e-8
    )
    ax.scatter(rows["dependency_tokens"], rows["participation_rank"], s=size)
    for _, row in rows.iterrows():
        ax.annotate(
            str(int(row["cell_id"])),
            (row["dependency_tokens"], row["participation_rank"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Dependency tokens")
    ax.set_ylabel("Participation rank")
    ax.set_title("Dependency load × functional rank (bubble = causal ablation magnitude)")
    fig.tight_layout()
    fig.savefig(dest, dpi=180)
    plt.close(fig)


def _plot_stability_plasticity(tx: pd.DataFrame, dest: Path) -> None:
    if tx.empty:
        return
    grouped = tx.groupby(["seed", "variant"], as_index=False).agg(
        gain=("relative_new_gain", lambda x: sum(max(float(v), 0.0) for v in x)),
        regression=("checkpoint_positive_regression", "max"),
    )
    summary = grouped.groupby("variant", as_index=False).agg(
        gain=("gain", "mean"), regression=("regression", "mean")
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(summary["regression"], summary["gain"], s=80)
    for _, row in summary.iterrows():
        ax.annotate(
            row["variant"],
            (row["regression"], row["gain"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Final positive registered-history regression")
    ax.set_ylabel("Cumulative positive new-learning gain")
    ax.set_title("Core 006: stability–plasticity frontier")
    fig.tight_layout()
    fig.savefig(dest, dpi=180)
    plt.close(fig)


def _plot_split_relief(split: pd.DataFrame, dest: Path) -> None:
    rows = split[split["variant"] == "certificate_mitosis"].copy()
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(rows))
    ax.scatter(list(x), rows["conflict_before"], label="before split")
    ax.scatter(list(x), rows["conflict_after"], label="after split")
    ax.set_xlabel("Mitosis event")
    ax.set_ylabel("Certificate conflict fraction")
    ax.set_title("Dependency-partitioned mitosis conflict relief")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest, dpi=180)
    plt.close(fig)


def _write_results_md(
    payload: dict[str, Any], gates: pd.DataFrame, causal: pd.DataFrame, dest: Path
) -> None:
    d = payload["decision"]
    lines = [
        "# Core Validation 006 Results",
        "",
        f"- Mode: `{payload['mode']}`",
        f"- Status: `{d['status']}`",
        f"- Protocol SHA-256: `{payload['protocol_sha256']}`",
        f"- Data manifest SHA-256: `{payload['data_manifest_sha256']}`",
        "",
    ]
    if payload["mode"] == "formal":
        lines.extend(
            [
                f"- Passed seeds: `{d['passed_seeds']}/{d['total_seeds']}`",
                "",
                "## Gate summary",
                "",
                gates.to_markdown(index=False),
                "",
            ]
        )
    growth_causal = causal[causal["variant"] == "certificate_mitosis"]
    if not growth_causal.empty:
        corr = growth_causal[
            ["dependency_tokens", "participation_rank", "causal_delta_nll"]
        ].corr(method="spearman")
        lines.extend(
            [
                "## Dependency / rank / causal diagnostics",
                "",
                "Spearman correlation matrix (diagnostic only):",
                "",
                corr.to_markdown(),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "A formal positive result is limited to frozen Pythia representations, "
            "fixed routing, linear writable Cells, and the registered-history protocol. "
            "It does not establish safe nonlinear foundation updates or autonomous router drift.",
            "",
        ]
    )
    dest.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    raw = args.out / "raw.json"
    if not raw.is_file():
        raise FileNotFoundError(raw)
    payload = json.loads(raw.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    tx, rank, split, checkpoint, eval_rows, causal, gates = _flatten(payload)
    tx.to_csv(args.out / "transaction-records.csv", index=False)
    rank.to_csv(args.out / "rank-trajectory.csv", index=False)
    split.to_csv(args.out / "split-records.csv", index=False)
    checkpoint.to_csv(args.out / "checkpoint-regression.csv", index=False)
    eval_rows.to_csv(args.out / "heldout-source-nll.csv", index=False)
    causal.to_csv(args.out / "causal-load.csv", index=False)
    gates.to_csv(args.out / "gate-summary.csv", index=False)

    decision = {
        **payload["decision"],
        "format": payload["format"],
        "experiment_id": payload["experiment_id"],
        "protocol_version": payload["protocol_version"],
        "mode": payload["mode"],
        "protocol_sha256": payload["protocol_sha256"],
        "parent_experiment": payload["parent_experiment"],
        "data_manifest_sha256": payload["data_manifest_sha256"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "provenance": payload["provenance"],
    }
    (args.out / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_rank_growth(rank, args.out / "rank-growth.png")
    _plot_dependency_rank_causal(causal, args.out / "dependency-rank-causal.png")
    _plot_stability_plasticity(tx, args.out / "stability-plasticity.png")
    _plot_split_relief(split, args.out / "split-conflict-relief.png")
    _write_results_md(payload, gates, causal, args.out / "RESULTS.md")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
