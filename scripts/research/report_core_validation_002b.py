#!/usr/bin/env python3
"""Report Core Validation 002B results and sparse-assembly diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "core-validation-002b-sparse-write-assembly"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _record_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in payload["runs"]:
        for record in run["records"]:
            row = {"seed": run["seed"], **record}
            if isinstance(row.get("selected_support"), list):
                row["selected_support"] = json.dumps(row["selected_support"])
            rows.append(row)
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
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def _gate_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in payload["runs"]:
        gates = dict(run["gates"])
        matched = gates.pop("matched_global", {})
        rows.append(
            {
                "seed": run["seed"],
                **gates,
                **{f"matched_{key}": value for key, value in matched.items()},
                "sparse_base_normalized_mse": run["pretraining"]["base_normalized_mse"]["sparse"],
                "dense_base_normalized_mse": run["pretraining"]["base_normalized_mse"]["dense"],
                "moe_base_normalized_mse": run["pretraining"]["base_normalized_mse"]["moe"],
                **{f"oracle_{key}": value for key, value in run["oracle_latent_sanity"].items()},
            }
        )
    return pd.DataFrame(rows)


def _plot_width_update(summary: pd.DataFrame, destination: Path) -> None:
    rows = summary[summary["variant_kind"] == "assembly"].copy()
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed, group in rows.groupby("seed"):
        ordered = group.sort_values("address_width")
        ax.plot(ordered["address_width"], ordered["median_update_error"], marker="o", label=str(seed))
    ax.axhline(0.10, linestyle="--", linewidth=1, label="002B U gate")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sparse write address width r")
    ax.set_ylabel("Median Update Error U")
    ax.set_title("Core Validation 002B: update fidelity vs address width")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_u_l(summary: pd.DataFrame, destination: Path) -> None:
    rows = summary[summary["variant_kind"].isin(["assembly", "global_ridge"])].copy()
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for variant, group in rows.groupby("variant"):
        ax.scatter(group["median_update_error"], group["median_write_leakage"], s=45, label=variant)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Median Update Error U")
    ax.set_ylabel("Median Write Leakage L")
    ax.set_title("Sparse assemblies vs full-writer ridge curve")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_geometry(summary: pd.DataFrame, destination: Path) -> None:
    rows = summary[summary["variant_kind"] == "assembly"].copy()
    if rows.empty:
        return
    grouped = rows.groupby("address_width", as_index=False).agg(
        fit_error=("median_assembly_fit_error", "mean"),
        context_variance=("median_assembly_context_ratio_variance", "mean"),
        off_support=("median_assembly_off_support_energy_ratio", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped["address_width"], grouped["fit_error"], marker="o", label="assembly fit error")
    ax.plot(grouped["address_width"], grouped["context_variance"], marker="o", label="context ratio variance")
    ax.plot(grouped["address_width"], grouped["off_support"], marker="o", label="off-support energy ratio")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sparse write address width r")
    ax.set_ylabel("Evaluator-only geometry metric")
    ax.set_title("Representation geometry vs address width")
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
    summary = _summary_frame(payload)
    gates = _gate_frame(payload)
    records.to_csv(args.out / "edit-records.csv", index=False)
    summary.to_csv(args.out / "seed-summary.csv", index=False)
    gates.to_csv(args.out / "gate-summary.csv", index=False)

    decision = {
        **payload["decision"],
        "format": payload["format"],
        "experiment_id": payload["experiment_id"],
        "mode": payload["mode"],
        "protocol_sha256": payload["protocol_sha256"],
        "parent_experiment": payload["parent_experiment"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "provenance": payload["provenance"],
    }
    (args.out / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_width_update(summary, args.out / "update-error-vs-address-width.png")
    _plot_u_l(summary, args.out / "matched-update-leakage-frontier.png")
    _plot_geometry(summary, args.out / "representation-geometry-vs-width.png")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
