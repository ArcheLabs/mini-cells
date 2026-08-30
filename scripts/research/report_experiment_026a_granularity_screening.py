#!/usr/bin/env python3
"""Aggregate Experiment 026a and decide whether the longer 026 run is warranted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GRANULARITIES = (1, 4, 8)
FORMAT = "minicells.cell-granularity-screening-30m.v1"
MIN_GAIN_DELTA = 0.02
MAX_NLL_RATIO = 1.03


def parser() -> argparse.Namespace:
    value = argparse.ArgumentParser(description="Report Experiment 026a screening")
    value.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "experiment-026a-granularity-screening",
    )
    return value.parse_args()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_arm(results: Path, granularity: int) -> tuple[pd.DataFrame, dict[str, object]]:
    arm = results / f"g{granularity}"
    metrics = pd.read_csv(arm / "metrics.csv").sort_values("continuation_tokens")
    summary = json.loads((arm / "worker-summary.json").read_text(encoding="utf-8"))
    if summary.get("complete") is not True:
        raise RuntimeError(f"G={granularity} is incomplete")
    if len(metrics) < 2 or int(metrics.iloc[0]["continuation_tokens"]) != 0:
        raise RuntimeError(f"G={granularity} lacks an age-zero checkpoint")
    return metrics, summary


def main() -> int:
    args = parser()
    results = args.results_dir
    protocol = json.loads((results / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT:
        raise RuntimeError(f"unexpected screening protocol: {protocol.get('format')!r}")

    trajectories: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    summaries: dict[int, dict[str, object]] = {}
    for granularity in GRANULARITIES:
        metrics, summary = _load_arm(results, granularity)
        summaries[granularity] = summary
        age_zero = float(metrics.iloc[0]["median_cell_specialization"])
        metrics = metrics.copy()
        metrics["specialization_gain"] = metrics["median_cell_specialization"] - age_zero
        trajectories.append(metrics)
        final = metrics.iloc[-1]
        rows.append(
            {
                "granularity": granularity,
                "age_zero_specialization": age_zero,
                "final_specialization": float(final["median_cell_specialization"]),
                "specialization_gain": float(final["specialization_gain"]),
                "balanced_nll": float(final["balanced_nll"]),
                "plasticity_std": float(final.get("plasticity_std", 0.0)),
                "profile_stability": float(final["mean_profile_stability_vs_age0"]),
                "stored_parameters": int(final["stored_parameters"]),
                "elapsed_seconds": float(summary["elapsed_seconds"]),
                "peak_vram_bytes": int(summary["peak_vram_bytes"]),
                "age_zero_max_abs_diff": float(summary["age_zero_parity"]["max_abs_diff"]),
            }
        )

    trajectory = pd.concat(trajectories, ignore_index=True)
    final = pd.DataFrame(rows).sort_values("granularity")
    baseline = final[final["granularity"] == 1].iloc[0]
    baseline_gain = float(baseline["specialization_gain"])
    baseline_nll = float(baseline["balanced_nll"])

    decisions = []
    for row in final.to_dict(orient="records"):
        granularity = int(row["granularity"])
        gain_delta = float(row["specialization_gain"]) - baseline_gain
        nll_ratio = float(row["balanced_nll"]) / max(baseline_nll, 1e-12)
        local_adaptation = granularity == 1 or float(row["plasticity_std"]) > 1e-6
        qualifies = bool(
            granularity > 1
            and gain_delta >= MIN_GAIN_DELTA
            and nll_ratio <= MAX_NLL_RATIO
            and local_adaptation
        )
        decisions.append(
            {
                "granularity": granularity,
                "specialization_gain_delta_vs_g1": gain_delta,
                "balanced_nll_ratio_vs_g1": nll_ratio,
                "local_plasticity_variance_observed": local_adaptation,
                "qualifies_for_confirmation": qualifies,
            }
        )

    parameter_parity = len({int(row["stored_parameters"]) for row in rows}) == 1
    age_zero_valid = max(float(row["age_zero_max_abs_diff"]) for row in rows) <= 5e-4
    candidates = [row for row in decisions if row["qualifies_for_confirmation"]]
    candidates.sort(key=lambda row: (-float(row["specialization_gain_delta_vs_g1"]), int(row["granularity"])))
    selected = candidates[0] if candidates else None
    status = (
        "PROCEED_TO_026_CONFIRMATION"
        if selected is not None and parameter_parity and age_zero_valid
        else "DO_NOT_PROCEED_TO_026_CONFIRMATION"
    )

    decision = {
        "format": FORMAT,
        "status": status,
        "selected_granularity": int(selected["granularity"]) if selected else None,
        "parameter_parity": parameter_parity,
        "age_zero_function_parity": age_zero_valid,
        "thresholds": {
            "minimum_specialization_gain_delta": MIN_GAIN_DELTA,
            "maximum_balanced_nll_ratio": MAX_NLL_RATIO,
            "require_nonzero_local_plasticity_variance": True,
        },
        "arms": decisions,
        "interpretation": (
            "This is a screening gate only. PROCEED means the observed mechanism is strong enough "
            "to justify the longer Experiment 026 confirmation budget; it is not a confirmatory claim."
        ),
    }

    trajectory.to_csv(results / "screening-trajectory.csv", index=False)
    final.to_csv(results / "screening-final.csv", index=False)
    _write_json(results / "decision.json", decision)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for granularity in GRANULARITIES:
        arm = trajectory[trajectory["granularity"] == granularity].sort_values("continuation_tokens")
        ax.plot(arm["continuation_tokens"] / 1e6, arm["specialization_gain"], marker="o", label=f"G={granularity}")
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Continuation tokens (M)")
    ax.set_ylabel("Median specialization gain from age zero")
    ax.set_title("Experiment 026a — Granularity screening")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results / "screening-specialization.png", dpi=180)
    plt.close(fig)

    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
