#!/usr/bin/env python3
"""Report Core Validation 002 results and mechanism diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "core-validation-002-write-addressability"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _record_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in payload["runs"]:
        seed = int(run["seed"])
        for record in run["records"]:
            rows.append({"seed": seed, **record})
    return pd.DataFrame(rows)


def _summary_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in payload["runs"]:
        for variant, summary in run["summary"].items():
            rows.append(
                {
                    "seed": run["seed"],
                    "variant": variant,
                    "superposition_load": run["superposition_load"],
                    "recovery_load": run["recovery_load"],
                    "parameter_count": (
                        run["parameter_counts"]["sparse"]
                        if variant
                        in {
                            "inferred_address",
                            "oracle_address",
                            "permuted_address",
                            "global_write",
                        }
                        else run["parameter_counts"][variant]
                    ),
                    **summary,
                    **{f"gate_{key}": value for key, value in run["gates"].items()},
                }
            )
    return pd.DataFrame(rows)


def _plot_update_leakage(frame: pd.DataFrame, destination: Path) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    grouped = frame.groupby(["seed", "variant"], as_index=False).agg(
        update_error=("update_error", "median"),
        write_leakage=("write_leakage", "median"),
    )
    for variant, rows in grouped.groupby("variant"):
        ax.scatter(rows["update_error"], rows["write_leakage"], label=variant, s=45)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Median normalized Update Error U (lower is better)")
    ax.set_ylabel("Median Write Leakage L (lower is better)")
    ax.set_title("Core Validation 002: update locality")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_sequential_leakage(frame: pd.DataFrame, destination: Path) -> None:
    if frame.empty:
        return
    selected = frame[
        frame["variant"].isin(
            ["inferred_address", "global_write", "permuted_address"]
        )
    ]
    if selected.empty:
        return
    mean = selected.groupby(["edit_index", "variant"], as_index=False)["write_leakage"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant, rows in mean.groupby("variant"):
        ax.plot(rows["edit_index"], rows["write_leakage"].clip(lower=1e-12), label=variant)
    ax.set_yscale("log")
    ax.set_xlabel("Sequential edit")
    ax.set_ylabel("Mean Write Leakage L")
    ax.set_title("Leakage across continual edits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_mechanistic(frame: pd.DataFrame, destination: Path) -> None:
    rows = frame[
        (frame["variant"] == "inferred_address")
        & frame["leakage_proxy"].notna()
        & frame["write_leakage"].notna()
    ].copy()
    rows = rows[(rows["leakage_proxy"] > 0) & (rows["write_leakage"] > 0)]
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(rows["leakage_proxy"], rows["write_leakage"], alpha=0.7, s=24)
    lower = max(min(rows["leakage_proxy"].min(), rows["write_leakage"].min()), 1e-12)
    upper = max(rows["leakage_proxy"].max(), rows["write_leakage"].max())
    if math.isfinite(lower) and math.isfinite(upper) and upper > lower:
        ax.plot([lower, upper], [lower, upper], linestyle="--", label="y = x reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Off-support activation proxy Q")
    ax.set_ylabel("Observed Write Leakage L")
    ax.set_title("Mechanistic prediction: representation leakage -> write leakage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _sweep_frame(payload: dict[str, Any]) -> pd.DataFrame:
    sweep = payload.get("diagnostic_sweep")
    if not sweep:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for condition in sweep:
        for variant, summary in condition["summary"].items():
            rows.append(
                {
                    "condition": condition["condition"],
                    "seed": condition["seed"],
                    "num_features": condition["num_features"],
                    "active_features": condition["active_features"],
                    "latent_dim": condition["latent_dim"],
                    "latent_topk": condition["latent_topk"],
                    "superposition_load": condition["superposition_load"],
                    "recovery_load": condition["recovery_load"],
                    "variant": variant,
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def _plot_recovery_sweep(frame: pd.DataFrame, destination: Path) -> None:
    rows = frame[frame["variant"].isin(["inferred_address", "global_write"])]
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, group in rows.groupby("variant"):
        ordered = group.sort_values("recovery_load")
        ax.plot(
            ordered["recovery_load"],
            ordered["median_write_leakage"].clip(lower=1e-12),
            marker="o",
            label=variant,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Recovery Load rho = k log(F/k) / d")
    ax.set_ylabel("Median Write Leakage L")
    ax.set_title("Descriptive capacity diagnostic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    raw_path = args.out / "raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    records = _record_frame(payload)
    summaries = _summary_frame(payload)
    sweep = _sweep_frame(payload)
    records.to_csv(args.out / "edit-records.csv", index=False)
    summaries.to_csv(args.out / "seed-summary.csv", index=False)
    if not sweep.empty:
        sweep.to_csv(args.out / "recovery-sweep.csv", index=False)

    decision = {
        **payload["decision"],
        "format": payload["format"],
        "experiment_id": payload["experiment_id"],
        "mode": payload["mode"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "provenance": payload["provenance"],
        "oracle_exact_zero_check": payload["oracle_exact_zero_check"],
    }
    (args.out / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_update_leakage(records, args.out / "update-error-vs-write-leakage.png")
    _plot_sequential_leakage(records, args.out / "sequential-write-leakage.png")
    _plot_mechanistic(records, args.out / "mechanistic-leakage-prediction.png")
    if not sweep.empty:
        _plot_recovery_sweep(sweep, args.out / "recovery-load-sweep.png")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
