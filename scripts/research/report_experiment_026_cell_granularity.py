"""Aggregate and report formal Experiment 026 results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

GRANULARITIES = (1, 2, 4, 8)
FORMAT = "minicells.cell-granularity-30m.v2"
DIFFERENTIATION_MIN_GAIN_DELTA = 0.05
PERFORMANCE_NLL_RATIO_MAX = 1.02
STABILITY_MIN_COSINE = 0.80


def parser() -> argparse.Namespace:
    result = argparse.ArgumentParser(description="Report Experiment 026")
    result.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "experiment-026-cell-granularity",
    )
    return result.parse_args()


def _json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load(results: Path):
    trajectories = []
    cells = []
    tissues = []
    summaries = {}
    for granularity in GRANULARITIES:
        arm = results / f"g{granularity}"
        metrics_path = arm / "metrics.csv"
        cell_path = arm / "cell-diagnostics.csv"
        tissue_path = arm / "tissue-diagnostics.csv"
        summary_path = arm / "worker-summary.json"
        for path in (metrics_path, cell_path, tissue_path, summary_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        trajectory = pd.read_csv(metrics_path)
        cell = pd.read_csv(cell_path)
        tissue = pd.read_csv(tissue_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("complete") is not True:
            raise RuntimeError(f"G={granularity} is incomplete")
        trajectories.append(trajectory)
        cells.append(cell)
        tissues.append(tissue)
        summaries[granularity] = summary
    return (
        pd.concat(trajectories, ignore_index=True),
        pd.concat(cells, ignore_index=True),
        pd.concat(tissues, ignore_index=True),
        summaries,
    )


def _with_specialization_gain(trajectory: pd.DataFrame) -> pd.DataFrame:
    output = trajectory.copy()
    output["specialization_gain"] = 0.0
    for granularity in GRANULARITIES:
        mask = output["granularity"] == granularity
        rows = output[mask]
        age_zero = rows[rows["continuation_tokens"] == 0]
        if len(age_zero) != 1:
            raise RuntimeError(f"G={granularity} requires exactly one age-zero row")
        baseline = float(age_zero.iloc[0]["median_cell_specialization"])
        output.loc[mask, "specialization_gain"] = (
            output.loc[mask, "median_cell_specialization"] - baseline
        )
    return output


def _plot_performance(trajectory: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for granularity in GRANULARITIES:
        rows = trajectory[trajectory["granularity"] == granularity].sort_values(
            "continuation_tokens"
        )
        ax.plot(
            rows["continuation_tokens"] / 1e6,
            rows["balanced_nll"],
            marker="o",
            label=f"G={granularity}",
        )
    ax.set_xlabel("Continuation tokens (M)")
    ax.set_ylabel("Balanced four-domain NLL")
    ax.set_title("Experiment 026 — Performance by cell granularity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_differentiation(trajectory: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for granularity in GRANULARITIES:
        rows = trajectory[trajectory["granularity"] == granularity].sort_values(
            "continuation_tokens"
        )
        ax.plot(
            rows["continuation_tokens"] / 1e6,
            rows["specialization_gain"],
            marker="o",
            label=f"G={granularity}",
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Continuation tokens (M)")
    ax.set_ylabel("Median specialization gain from age zero")
    ax.set_title("Experiment 026 — Functional differentiation gain")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_frontier(final: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.scatter(final["balanced_nll"], final["specialization_gain"], s=70)
    for row in final.to_dict(orient="records"):
        ax.annotate(
            f"G={int(row['granularity'])}",
            (float(row["balanced_nll"]), float(row["specialization_gain"])),
            xytext=(6, 6),
            textcoords="offset points",
        )
    ax.set_xlabel("Final balanced four-domain NLL (lower is better)")
    ax.set_ylabel("Final specialization gain from age zero")
    ax.set_title("Experiment 026 — Performance / differentiation frontier")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parser()
    results = args.results_dir
    worker_summary = json.loads(
        (results / "worker-summary.json").read_text(encoding="utf-8")
    )
    if worker_summary.get("complete") is not True:
        raise RuntimeError(
            "refusing formal report: Experiment 026 workers are incomplete"
        )
    protocol = json.loads((results / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT:
        raise RuntimeError(
            f"unexpected protocol format: {protocol.get('format')!r}"
        )

    trajectory, cells, tissues, summaries = _load(results)
    trajectory = _with_specialization_gain(trajectory)
    trajectory.to_csv(results / "granularity-trajectory.csv", index=False)
    cells.to_csv(results / "cell-diagnostics.csv", index=False)
    tissues.to_csv(results / "tissue-diagnostics.csv", index=False)

    final_rows = []
    for granularity in GRANULARITIES:
        rows = trajectory[trajectory["granularity"] == granularity]
        final_token = int(rows["continuation_tokens"].max())
        final_row = rows[rows["continuation_tokens"] == final_token].iloc[-1]
        age_zero_row = rows[rows["continuation_tokens"] == 0].iloc[0]
        summary = summaries[granularity]
        final_rows.append(
            {
                **final_row.to_dict(),
                "age_zero_median_cell_specialization": float(
                    age_zero_row["median_cell_specialization"]
                ),
                "elapsed_seconds": float(summary["elapsed_seconds"]),
                "peak_vram_bytes": int(summary["peak_vram_bytes"]),
                "age_zero_max_abs_diff": float(
                    summary["age_zero_parity"]["max_abs_diff"]
                ),
            }
        )

    final = pd.DataFrame(final_rows).sort_values("granularity")
    baseline = final[final["granularity"] == 1].iloc[0]
    baseline_nll = float(baseline["balanced_nll"])
    baseline_gain = float(baseline["specialization_gain"])

    decision_rows = []
    for row in final.to_dict(orient="records"):
        granularity = int(row["granularity"])
        gain = float(row["specialization_gain"])
        gain_delta = gain - baseline_gain
        nll_ratio = float(row["balanced_nll"]) / max(baseline_nll, 1e-12)
        stability = float(row["mean_profile_stability_vs_age0"])
        qualifies = bool(
            granularity > 1
            and gain_delta >= DIFFERENTIATION_MIN_GAIN_DELTA
            and nll_ratio <= PERFORMANCE_NLL_RATIO_MAX
            and stability >= STABILITY_MIN_COSINE
        )
        decision_rows.append(
            {
                "granularity": granularity,
                "age_zero_median_specialization": float(
                    row["age_zero_median_cell_specialization"]
                ),
                "final_median_specialization": float(
                    row["median_cell_specialization"]
                ),
                "specialization_gain": gain,
                "specialization_gain_delta_vs_g1": gain_delta,
                "balanced_nll_ratio_vs_g1": nll_ratio,
                "profile_stability_vs_age0": stability,
                "plasticity_std": float(row.get("plasticity_std", 0.0)),
                "qualifies": qualifies,
            }
        )

    decision_frame = pd.DataFrame(decision_rows)[
        [
            "granularity",
            "specialization_gain_delta_vs_g1",
            "balanced_nll_ratio_vs_g1",
            "qualifies",
        ]
    ]
    final = final.merge(decision_frame, on="granularity", how="left")
    final.to_csv(results / "granularity-final.csv", index=False)

    qualifying = [row for row in decision_rows if row["qualifies"]]
    qualifying.sort(
        key=lambda row: (
            -float(row["specialization_gain_delta_vs_g1"]),
            float(row["balanced_nll_ratio_vs_g1"]),
            int(row["granularity"]),
        )
    )
    selected = qualifying[0] if qualifying else None
    structure_counts = {int(row["stored_parameters"]) for row in final_rows}
    parameter_parity = len(structure_counts) == 1
    age_zero_max = max(
        float(summaries[g]["age_zero_parity"]["max_abs_diff"])
        for g in GRANULARITIES
    )
    age_zero_valid = age_zero_max <= 5e-4
    status = (
        "GRANULARITY_DIFFERENTIATION_SIGNAL"
        if selected is not None and parameter_parity and age_zero_valid
        else "NO_GRANULARITY_DIFFERENTIATION_SIGNAL"
    )

    decision = {
        "format": FORMAT,
        "status": status,
        "baseline_granularity": 1,
        "selected_granularity": (
            int(selected["granularity"]) if selected is not None else None
        ),
        "parameter_parity": parameter_parity,
        "stored_parameters": sorted(structure_counts),
        "max_age_zero_logit_abs_diff": age_zero_max,
        "age_zero_valid": age_zero_valid,
        "thresholds": {
            "differentiation_min_gain_delta": DIFFERENTIATION_MIN_GAIN_DELTA,
            "performance_nll_ratio_max": PERFORMANCE_NLL_RATIO_MAX,
            "stability_min_cosine": STABILITY_MIN_COSINE,
        },
        "arms": decision_rows,
        "interpretation": (
            "A positive status means at least one finer granularity produced a "
            "preregistered increase in specialization gain beyond its own age-zero "
            "measurement baseline, under the same normalized local-plasticity rule, "
            "without more than 2% balanced-NLL degradation. It does not establish "
            "autonomous mitosis or full NCA self-organization."
        ),
    }
    _json_write(results / "decision.json", decision)

    _plot_performance(trajectory, results / "performance-by-granularity.png")
    _plot_differentiation(
        trajectory,
        results / "differentiation-by-granularity.png",
    )
    _plot_frontier(final, results / "granularity-frontier.png")

    print(json.dumps(decision, indent=2, sort_keys=True))
    if not parameter_parity:
        raise RuntimeError(
            "Experiment 026 invalid: parameter count differs across granularity arms"
        )
    if not age_zero_valid:
        raise RuntimeError("Experiment 026 invalid: age-zero function parity failed")
    if not math.isfinite(baseline_nll):
        raise RuntimeError("Experiment 026 invalid: non-finite baseline NLL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
