#!/usr/bin/env python3
"""Report Core Validation 005 formal/smoke outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "core-validation-005-subspace-certified-mitosis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _records(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in payload["runs"]:
        for variant, variant_run in run["variants"].items():
            for record in variant_run["records"]:
                rows.append({"seed": run["seed"], "variant": variant, **record})
    return pd.DataFrame(rows)


def _summary(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in payload["runs"]:
        for variant, variant_run in run["variants"].items():
            rows.append({"seed": run["seed"], "variant": variant, **variant_run["summary"]})
    return pd.DataFrame(rows)


def _gates(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in payload["runs"]:
        gate = run["gate_summary"]
        growth = gate["variant_summaries"]["certificate_growth"]
        wrong = gate["variant_summaries"]["wrong_certificate"]
        rows.append(
            {
                "seed": run["seed"],
                "pass": gate["pass"],
                "regression_damage_ratio_vs_unsafe": gate[
                    "regression_damage_ratio_vs_unsafe"
                ],
                "committed_gain_ratio_vs_unsafe": gate["committed_gain_ratio_vs_unsafe"],
                **{f"gate_{name}": value for name, value in gate["gates"].items()},
                **{f"growth_{name}": value for name, value in growth.items()},
                "wrong_false_safe_count": wrong["false_safe_count"],
                "wrong_regression": wrong["cumulative_positive_global_regression"],
            }
        )
    return pd.DataFrame(rows)


def _plot_frontier(summary: pd.DataFrame, dest: Path) -> None:
    if summary.empty:
        return
    grouped = summary.groupby("variant", as_index=False).agg(
        gain=("cumulative_committed_new_gain", "mean"),
        damage=("cumulative_positive_global_regression", "mean"),
        spawn=("spawned_cells_per_effective_commit", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(grouped["damage"], grouped["gain"], s=70)
    for _, row in grouped.iterrows():
        ax.annotate(
            row["variant"],
            (row["damage"], row["gain"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Cumulative positive historical regression")
    ax.set_ylabel("Cumulative committed new-learning gain")
    ax.set_title("Core Validation 005: stability–plasticity frontier")
    fig.tight_layout()
    fig.savefig(dest, dpi=180)
    plt.close(fig)


def _plot_certificate(gates: pd.DataFrame, dest: Path) -> None:
    if gates.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(gates))
    ax.bar([i - 0.2 for i in x], gates["growth_false_safe_count"], width=0.4, label="candidate false-safe")
    ax.bar([i + 0.2 for i in x], gates["wrong_false_safe_count"], width=0.4, label="wrong-Q false-safe")
    ax.set_xticks(list(x), gates["seed"].astype(str))
    ax.set_xlabel("Formal seed")
    ax.set_ylabel("Count")
    ax.set_title("Certificate geometry causal control")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest, dpi=180)
    plt.close(fig)


def _plot_growth(gates: pd.DataFrame, dest: Path) -> None:
    if gates.empty:
        return
    metrics = [
        "growth_growth_rescue_rate",
        "growth_child_reuse_acceptance_rate",
        "growth_spawned_cells_per_effective_commit",
    ]
    values = [gates[name].mean() for name in metrics]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(metrics, values)
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("Fraction / ratio")
    ax.set_title("Growth rescue, reuse, and boundedness")
    fig.tight_layout()
    fig.savefig(dest, dpi=180)
    plt.close(fig)


def _results_markdown(payload: dict[str, Any], gates: pd.DataFrame) -> str:
    decision = payload["decision"]
    lines = [
        "# Core Validation 005 Results",
        "",
        f"- Status: `{decision['status']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Passed seeds: `{decision.get('passed_seeds')}/{decision.get('total_seeds')}`",
        "- Learner replay access: forbidden by protocol and instrumented as zero in the primary candidate.",
    ]
    if not gates.empty:
        lines.extend(
            [
                f"- Candidate false-safe counts: `{gates['growth_false_safe_count'].tolist()}`",
                f"- Certificate/full-history decision mismatches: `{gates['growth_decision_mismatch_count'].tolist()}`",
                f"- Candidate gain ratios vs unsafe: `{gates['committed_gain_ratio_vs_unsafe'].round(6).tolist()}`",
                f"- Candidate regression-damage ratios vs unsafe: `{gates['regression_damage_ratio_vs_unsafe'].round(6).tolist()}`",
                f"- Growth rescue rates: `{gates['growth_growth_rescue_rate'].round(6).tolist()}`",
                f"- Child reuse rates: `{gates['growth_child_reuse_acceptance_rate'].round(6).tolist()}`",
                f"- Wrong-certificate false-safe counts: `{gates['wrong_false_safe_count'].tolist()}`",
            ]
        )
    lines.extend(
        [
            "",
            "This decision applies only to the frozen linear-writable, fixed-feature, explicit-routing synthetic setting described by the protocol.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    raw = args.out / "raw.json"
    if not raw.is_file():
        raise FileNotFoundError(raw)
    payload = json.loads(raw.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    records = _records(payload)
    summary = _summary(payload)
    gates = _gates(payload)
    records.to_csv(args.out / "transaction-records.csv", index=False)
    summary.to_csv(args.out / "seed-summary.csv", index=False)
    gates.to_csv(args.out / "gate-summary.csv", index=False)
    decision = {
        **payload["decision"],
        "format": payload["format"],
        "experiment_id": payload["experiment_id"],
        "protocol_version": payload["protocol_version"],
        "mode": payload["mode"],
        "protocol_sha256": payload["protocol_sha256"],
        "parent_experiment": payload["parent_experiment"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "provenance": payload["provenance"],
    }
    (args.out / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out / "RESULTS.md").write_text(
        _results_markdown(payload, gates), encoding="utf-8"
    )
    _plot_frontier(summary, args.out / "stability-plasticity-frontier.png")
    _plot_certificate(gates, args.out / "certificate-causal-control.png")
    _plot_growth(gates, args.out / "growth-recovery.png")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
