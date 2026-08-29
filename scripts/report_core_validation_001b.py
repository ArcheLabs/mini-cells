#!/usr/bin/env python3
"""Report Core Validation 001b residual-memorization diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "core-validation-001b-residual-memorization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _flatten_run(run: dict[str, object]) -> dict[str, object]:
    coupling = run["late_coupling"]
    gates = run["gates"]
    return {
        "task": run["task"],
        "seed": run["seed"],
        "source_control_valid": run.get("source_control_valid"),
        "early_seen_minus_unseen_gap_at_k0": run[
            "early_seen_minus_unseen_gap_at_k0"
        ],
        "frequency_ranking": ",".join(str(v) for v in run["frequency_ranking"]),
        "exclusion_accuracy_correlation": coupling[
            "exclusion_accuracy_correlation"
        ],
        "mean_absolute_old_heldout_gap": coupling["mean_absolute_gap"],
        "maximum_positive_old_heldout_gap": coupling["maximum_positive_gap"],
        "maximum_absolute_old_heldout_gap": coupling["maximum_absolute_gap"],
        "dc_only_old_accuracy": coupling["endpoint_left_accuracy"],
        "dc_only_heldout_accuracy": coupling["endpoint_right_accuracy"],
        "old_exclusion_auc_mean": coupling["left_auc_mean"],
        "heldout_exclusion_auc_mean": coupling["right_auc_mean"],
        "parent_preconditions": gates["parent_preconditions"],
        "assay_sensitivity": gates["assay_sensitivity"],
        "synchronized_decay": gates["synchronized_decay"],
        "no_material_membership_advantage": gates[
            "no_material_membership_advantage"
        ],
        "dc_endpoint_destroyed": gates["dc_endpoint_destroyed"],
        "pass": gates["pass"],
    }


def _sweep_rows(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        for phase, sweep in (("early", run["early_sweep"]), ("late", run["late_sweep"])):
            for item in sweep:
                base = {
                    "task": run["task"],
                    "seed": run["seed"],
                    "phase": phase,
                    "k": item["k"],
                    "selected_frequency_pairs": ",".join(
                        str(v) for v in item["selected_frequency_pairs"]
                    ),
                }
                for intervention in ("excluded", "restricted"):
                    for partition, metrics in item[intervention].items():
                        rows.append(
                            {
                                **base,
                                "intervention": intervention,
                                "partition": partition,
                                "accuracy": metrics["accuracy"],
                                "nll": metrics["nll"],
                            }
                        )
    return rows


def _oracle_rows(oracle: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in oracle["sweep"]:
        for intervention in ("excluded", "restricted"):
            for partition, metrics in item[intervention].items():
                rows.append(
                    {
                        "seed": oracle["seed"],
                        "k": item["k"],
                        "selected_frequency_pairs": ",".join(
                            str(v) for v in item["selected_frequency_pairs"]
                        ),
                        "intervention": intervention,
                        "partition": partition,
                        "accuracy": metrics["accuracy"],
                        "nll": metrics["nll"],
                    }
                )
    return rows


def _plot_exclusion(sweep: pd.DataFrame, destination: Path) -> None:
    primary = sweep[
        (sweep["task"] == "modular_addition")
        & (sweep["phase"] == "late")
        & (sweep["intervention"] == "excluded")
        & (sweep["partition"].isin(["old", "heldout"]))
    ]
    if primary.empty:
        return
    seeds = sorted(primary["seed"].unique())
    fig, axes = plt.subplots(len(seeds), 1, figsize=(8, 3.5 * len(seeds)), sharex=True)
    if len(seeds) == 1:
        axes = [axes]
    for ax, seed in zip(axes, seeds):
        frame = primary[primary["seed"] == seed]
        for partition in ("old", "heldout"):
            part = frame[frame["partition"] == partition].sort_values("k")
            ax.plot(part["k"], part["accuracy"], marker="o", label=partition)
        ax.axhline(1 / 31, linestyle="--", linewidth=1, label="chance" if seed == seeds[0] else None)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"seed {seed}: cumulative exclusion")
        ax.legend()
    axes[-1].set_xlabel("Top Fourier pairs removed (k)")
    fig.suptitle("Core Validation 001b: old vs heldout degradation")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_gap(sweep: pd.DataFrame, destination: Path) -> None:
    primary = sweep[
        (sweep["task"] == "modular_addition")
        & (sweep["phase"] == "late")
        & (sweep["intervention"] == "excluded")
        & (sweep["partition"].isin(["old", "heldout"]))
    ]
    if primary.empty:
        return
    pivot = primary.pivot_table(
        index=["seed", "k"], columns="partition", values="accuracy"
    ).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed, frame in pivot.groupby("seed"):
        frame = frame.sort_values("k")
        ax.plot(frame["k"], frame["old"] - frame["heldout"], marker="o", label=f"seed {seed}")
    ax.axhline(0.0, linewidth=1)
    ax.axhline(0.10, linestyle="--", linewidth=1, label="preregistered +0.10 limit")
    ax.set_xlabel("Top Fourier pairs removed (k)")
    ax.set_ylabel("Old accuracy - heldout accuracy")
    ax.set_title("Membership-specific advantage under cumulative exclusion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_oracle(frame: pd.DataFrame, destination: Path) -> None:
    frame = frame[
        (frame["intervention"] == "excluded")
        & (frame["partition"].isin(["seen", "heldout"]))
    ]
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for partition in ("seen", "heldout"):
        part = frame[frame["partition"] == partition].sort_values("k")
        ax.plot(part["k"], part["accuracy"], marker="o", label=partition)
    ax.axhline(1 / 31, linestyle="--", linewidth=1, label="chance")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Top Fourier pairs removed (k)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Oracle assay validity: seen vs heldout degradation")
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

    runs = pd.DataFrame([_flatten_run(run) for run in payload["runs"]])
    runs.to_csv(args.out / "runs.csv", index=False)
    sweep = pd.DataFrame(_sweep_rows(payload["runs"]))
    sweep.to_csv(args.out / "frequency-sweep.csv", index=False)
    oracle = pd.DataFrame(_oracle_rows(payload["oracle"]))
    oracle.to_csv(args.out / "oracle-frequency-sweep.csv", index=False)

    decision = {
        **payload["decision"],
        "format": payload["format"],
        "experiment_id": payload["experiment_id"],
        "parent_experiment": payload["parent_experiment"],
        "mode": payload["mode"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "provenance": payload["provenance"],
        "parent_training_provenance": payload["parent_training_provenance"],
        "parent_decision": payload["parent_decision"],
        "oracle": {
            "seed": payload["oracle"]["seed"],
            "coupling": payload["oracle"]["coupling"],
            "gates": payload["oracle"]["gates"],
        },
    }
    (args.out / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_exclusion(sweep, args.out / "frequency-exclusion-trajectories.png")
    _plot_gap(sweep, args.out / "membership-gap-trajectories.png")
    _plot_oracle(oracle, args.out / "oracle-exclusion-trajectory.png")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
