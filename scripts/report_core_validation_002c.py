#!/usr/bin/env python3
"""Report Core Validation 002C oracle tomography results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "core-validation-002c-oracle-tomography"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _width(name: str) -> int | None:
    return int(name.removeprefix("sparse_r")) if name.startswith("sparse_r") else None


def _seed_summary(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in payload["runs"]:
        seed = int(run["seed"])
        for name, summary in run["summaries"].items():
            rows.append(
                {
                    "seed": seed,
                    "decoder": name,
                    "width": _width(name),
                    "sparse_base_normalized_mse": run["pretraining"]["base_normalized_mse"],
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def _feature_metrics(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in payload["runs"]:
        seed = int(run["seed"])
        width1 = run["feature_metrics"]["sparse_r1"]["affected_fit_error"]
        for name, metrics in run["feature_metrics"].items():
            width = _width(name)
            count = len(metrics["affected_fit_error"])
            for feature in range(count):
                fit = float(metrics["affected_fit_error"][feature])
                base_fit = float(width1[feature])
                rows.append(
                    {
                        "seed": seed,
                        "decoder": name,
                        "width": width,
                        "feature": feature,
                        "affected_fit_error": fit,
                        "off_support_leakage": float(metrics["off_support_leakage"][feature]),
                        "unconditional_normalized_mse": float(metrics["unconditional_normalized_mse"][feature]),
                        "context_ratio_variance": float(metrics["context_ratio_variance"][feature]),
                        "positive_examples": float(metrics["positive_examples"][feature]),
                        "relative_fit_error_vs_width1": (
                            fit / max(base_fit, 1e-12) if width is not None else None
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _gate_summary(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in payload["runs"]:
        gates = run["gates"]
        row: dict[str, Any] = {
            "seed": run["seed"],
            "base_quality": gates["base_quality"],
            "sparse_assembly_present": gates["sparse_assembly_present"],
            "representation_regime": gates["representation_regime"],
            "width1_median_affected_fit_error": gates["width1_median_affected_fit_error"],
            "best_sparse_width": gates["best_sparse_width"],
            "best_sparse_median_affected_fit_error": gates["best_sparse_median_affected_fit_error"],
            "best_sparse_median_off_support_leakage": gates["best_sparse_median_off_support_leakage"],
            "dense_linear_reference_passes_thresholds": gates["dense_linear_reference_passes_thresholds"],
            "pass": gates["pass"],
        }
        for candidate in gates["candidate_widths"]:
            width = int(candidate["width"])
            row[f"r{width}_pass"] = candidate["pass"]
            row[f"r{width}_relative_fit"] = candidate["median_relative_fit_error_vs_width1"]
            row[f"r{width}_improvement_fraction"] = candidate["feature_improvement_fraction"]
        rows.append(row)
    return pd.DataFrame(rows)


def _set_width_axis(ax: Any, values: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(values)
    ax.set_xticklabels([str(value) for value in values])


def _plot_fit(summary: pd.DataFrame, destination: Path) -> None:
    sparse = summary[summary["width"].notna()].copy()
    if sparse.empty:
        return
    widths = sorted(int(value) for value in sparse["width"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed, rows in sparse.groupby("seed"):
        ordered = rows.sort_values("width")
        ax.plot(ordered["width"], ordered["median_affected_fit_error"], marker="o", label=f"seed {seed}")
    dense = summary[summary["decoder"] == "dense_linear"]
    if not dense.empty:
        ax.axhline(dense["median_affected_fit_error"].median(), linestyle="--", label="dense-linear median")
    _set_width_axis(ax, widths)
    ax.set_yscale("log")
    ax.set_xlabel("Oracle sparse address width r")
    ax.set_ylabel("Median held-out affected fit error")
    ax.set_title("Core Validation 002C: oracle fidelity vs width")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_leakage(summary: pd.DataFrame, destination: Path) -> None:
    sparse = summary[summary["width"].notna()].copy()
    if sparse.empty:
        return
    widths = sorted(int(value) for value in sparse["width"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed, rows in sparse.groupby("seed"):
        ordered = rows.sort_values("width")
        ax.plot(ordered["width"], ordered["median_off_support_leakage"].clip(lower=1e-12), marker="o", label=f"seed {seed}")
    dense = summary[summary["decoder"] == "dense_linear"]
    if not dense.empty:
        ax.axhline(dense["median_off_support_leakage"].median(), linestyle="--", label="dense-linear median")
    _set_width_axis(ax, widths)
    ax.set_yscale("log")
    ax.set_xlabel("Oracle sparse address width r")
    ax.set_ylabel("Median held-out off-support leakage")
    ax.set_title("Core Validation 002C: oracle locality vs width")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_improvement(features: pd.DataFrame, destination: Path) -> None:
    sparse = features[(features["width"].notna()) & (features["width"] > 1)].copy()
    if sparse.empty:
        return
    widths = sorted(int(value) for value in sparse["width"].unique())
    grouped = sparse.groupby(["seed", "width"], as_index=False).agg(
        median_relative_fit=("relative_fit_error_vs_width1", "median"),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed, rows in grouped.groupby("seed"):
        ordered = rows.sort_values("width")
        ax.plot(ordered["width"], ordered["median_relative_fit"], marker="o", label=f"seed {seed}")
    ax.axhline(0.60, linestyle="--", label="pre-registered 0.60 threshold")
    _set_width_axis(ax, widths)
    ax.set_xlabel("Oracle sparse address width r")
    ax.set_ylabel("Median U_repr(r) / U_repr(1)")
    ax.set_title("Feature fidelity improvement from wider oracle assemblies")
    ax.legend(fontsize=8)
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
    seed_summary = _seed_summary(payload)
    features = _feature_metrics(payload)
    gates = _gate_summary(payload)
    seed_summary.to_csv(args.out / "seed-summary.csv", index=False)
    features.to_csv(args.out / "feature-metrics.csv", index=False)
    gates.to_csv(args.out / "gate-summary.csv", index=False)

    decision = {
        **payload["decision"],
        "format": payload["format"],
        "experiment_id": payload["experiment_id"],
        "mode": payload["mode"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "protocol_sha256": payload["protocol_sha256"],
        "parent_experiments": payload["parent_experiments"],
        "provenance": payload["provenance"],
    }
    (args.out / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_fit(seed_summary, args.out / "oracle-fit-vs-width.png")
    _plot_leakage(seed_summary, args.out / "oracle-leakage-vs-width.png")
    _plot_improvement(features, args.out / "featurewise-improvement-vs-width.png")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
