#!/usr/bin/env python3
"""Report Core Validation 003 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "core-validation-003-dependency-scoped-transactional-learning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _transaction_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in payload["runs"]:
        for granularity_run in run["granularities"]:
            granularity = int(granularity_run["granularity"])
            for variant, variant_run in granularity_run["variants"].items():
                for record in variant_run["records"]:
                    row = {
                        "seed": run["seed"],
                        "granularity": granularity,
                        "variant": variant,
                        **record,
                    }
                    row["touched_experts"] = json.dumps(row["touched_experts"])
                    rows.append(row)
    return pd.DataFrame(rows)


def _summary_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in payload["runs"]:
        for granularity_run in run["granularities"]:
            granularity = int(granularity_run["granularity"])
            pretraining = granularity_run["pretraining"]
            for variant, variant_run in granularity_run["variants"].items():
                rows.append(
                    {
                        "seed": run["seed"],
                        "granularity": granularity,
                        "variant": variant,
                        **pretraining,
                        **variant_run["summary"],
                    }
                )
    return pd.DataFrame(rows)


def _gate_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in payload["runs"]:
        for granularity_run in run["granularities"]:
            gate = granularity_run["gate_summary"]
            tx = gate["local_tx_summary"]
            stress = gate["router_drift_stress_summary"]
            rows.append(
                {
                    "seed": run["seed"],
                    "granularity": gate["granularity"],
                    "pass": gate["pass"],
                    "base_normalized_mse": gate["base_normalized_mse"],
                    "regression_damage_ratio_vs_local_always":
                        gate["regression_damage_ratio_vs_local_always"],
                    "committed_gain_ratio_vs_local_always":
                        gate["committed_gain_ratio_vs_local_always"],
                    "dependency_ratio_vs_coarsest": gate["dependency_ratio_vs_coarsest"],
                    **{f"gate_{k}": v for k, v in gate["gates"].items()},
                    **{f"tx_{k}": v for k, v in tx.items()},
                    **{f"stress_{k}": v for k, v in stress.items()},
                }
            )
    return pd.DataFrame(rows)


def _plot_scope_safety(summary: pd.DataFrame, destination: Path) -> None:
    rows = summary[
        summary["variant"].isin(["local_tx_frozen", "local_tx_router_drift"])
    ].copy()
    if rows.empty:
        return
    grouped = (
        rows.groupby(["variant", "granularity"], as_index=False)
        .agg(
            coverage=("mean_dependency_coverage", "mean"),
            false_safe=("false_safe_rate", "mean"),
        )
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    for variant, group in grouped.groupby("variant"):
        ax.plot(
            group["coverage"],
            group["false_safe"],
            marker="o",
            label=variant,
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"g={int(row['granularity'])}",
                (row["coverage"], row["false_safe"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.axvline(0.20, linestyle="--", linewidth=1, label="coverage gate")
    ax.axhline(0.01, linestyle="--", linewidth=1, label="false-safe gate")
    ax.set_xlabel("Mean dependency coverage")
    ax.set_ylabel("False-safe rate")
    ax.set_title("Core Validation 003: local validation scope vs hidden global safety")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_granularity_cost(summary: pd.DataFrame, destination: Path) -> None:
    rows = summary[summary["variant"] == "local_tx_frozen"].copy()
    if rows.empty:
        return
    grouped = rows.groupby("granularity", as_index=False).agg(
        coverage=("mean_dependency_coverage", "mean"),
        cost=("normalized_state_validation_cost_per_accepted_update", "mean"),
        acceptance=("acceptance_rate", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped["granularity"], grouped["coverage"], marker="o", label="dependency coverage")
    ax.plot(grouped["granularity"], grouped["acceptance"], marker="o", label="acceptance rate")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Granularity g")
    ax.set_ylabel("Fraction")
    ax.set_title("Granularity: validation scope and accepted learning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped["granularity"], grouped["cost"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Granularity g")
    ax.set_ylabel("Normalized state + validation cost / accepted update")
    ax.set_title("Cost proxy per accepted transaction")
    fig.tight_layout()
    fig.savefig(destination.with_name("cost-per-accepted-update.png"), dpi=180)
    plt.close(fig)


def _plot_transactional_tradeoff(gates: pd.DataFrame, destination: Path) -> None:
    if gates.empty:
        return
    grouped = gates.groupby("granularity", as_index=False).agg(
        damage_ratio=("regression_damage_ratio_vs_local_always", "mean"),
        gain_ratio=("committed_gain_ratio_vs_local_always", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(grouped["damage_ratio"], grouped["gain_ratio"], s=55)
    for _, row in grouped.iterrows():
        ax.annotate(
            f"g={int(row['granularity'])}",
            (row["damage_ratio"], row["gain_ratio"]),
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.axvline(0.70, linestyle="--", linewidth=1, label="damage-ratio gate")
    ax.axhline(0.60, linestyle="--", linewidth=1, label="gain-ratio gate")
    ax.set_xlabel("Cumulative regression damage / local-always")
    ax.set_ylabel("Committed new-learning gain / local-always")
    ax.set_title("Transactional safety-learning tradeoff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    raw_path = args.out / "raw.json"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    transactions = _transaction_frame(payload)
    summary = _summary_frame(payload)
    gates = _gate_frame(payload)
    transactions.to_csv(args.out / "transaction-records.csv", index=False)
    summary.to_csv(args.out / "seed-summary.csv", index=False)
    gates.to_csv(args.out / "gate-summary.csv", index=False)

    decision = {
        **payload["decision"],
        "format": payload["format"],
        "experiment_id": payload["experiment_id"],
        "mode": payload["mode"],
        "protocol_sha256": payload["protocol_sha256"],
        "research_transition": payload["research_transition"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "provenance": payload["provenance"],
    }
    (args.out / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _plot_scope_safety(summary, args.out / "scope-safety-frontier.png")
    _plot_granularity_cost(summary, args.out / "granularity-scope-acceptance.png")
    _plot_transactional_tradeoff(gates, args.out / "transactional-tradeoff.png")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
