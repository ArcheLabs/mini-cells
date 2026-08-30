#!/usr/bin/env python3
"""Report Core Validation 001 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "core-validation-001-knowledge-subsumption"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _flatten_run(run: dict[str, object]) -> dict[str, object]:
    early = run["early"]
    late = run["late"]
    mech = run["mechanistic"]
    gates = run["gates"]
    return {
        "task": run["task"],
        "seed": run["seed"],
        "parameter_count": run["parameter_count"],
        "total_steps": run["total_steps"],
        "control_valid": run.get("control_valid"),
        "early_seen_accuracy": early["seen_accuracy"],
        "early_unseen_accuracy": early["unseen_accuracy"],
        "late_old_accuracy": late["old"]["accuracy"],
        "late_current_accuracy": late["current"]["accuracy"],
        "late_heldout_accuracy": late["heldout"]["accuracy"],
        "key_frequency_pairs": ",".join(str(v) for v in mech["key_frequency_pairs"]),
        "early_fourier_concentration": mech["early_fourier_concentration"],
        "late_fourier_concentration": mech["late_fourier_concentration"],
        "fourier_concentration_gain": mech["fourier_concentration_gain"],
        "early_excluded_seen_accuracy": mech["early"]["seen"]["excluded"]["accuracy"],
        "late_restricted_old_accuracy": mech["late"]["old"]["restricted"]["accuracy"],
        "late_restricted_heldout_accuracy": mech["late"]["heldout"]["restricted"]["accuracy"],
        "late_excluded_old_accuracy": mech["late"]["old"]["excluded"]["accuracy"],
        "late_excluded_heldout_accuracy": mech["late"]["heldout"]["excluded"]["accuracy"],
        "early_path_reuse": mech["early_path_reuse"],
        "late_path_reuse": mech["late_path_reuse"],
        "path_reuse_gain": mech["path_reuse_gain"],
        "early_memorization_gate": gates["early_memorization"],
        "late_generalization_gate": gates["late_generalization"],
        "generalizing_circuit_gate": gates["generalizing_circuit"],
        "memorization_cleanup_gate": gates["memorization_cleanup"],
        "pass": gates["pass"],
    }


def _plot_fourier(frame: pd.DataFrame, destination: Path) -> None:
    if frame.empty:
        return
    labels = [f"{row.task}\nseed {row.seed}" for row in frame.itertuples()]
    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    ax.bar(
        [value - width / 2 for value in x],
        frame["early_fourier_concentration"],
        width,
        label="early",
    )
    ax.bar(
        [value + width / 2 for value in x],
        frame["late_fourier_concentration"],
        width,
        label="late",
    )
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Energy in late key Fourier pairs")
    ax.set_title("Core Validation 001: Fourier circuit concentration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_interventions(frame: pd.DataFrame, destination: Path) -> None:
    if frame.empty:
        return
    primary = frame[frame["task"] == "modular_addition"]
    if primary.empty:
        return
    labels = [f"seed {row.seed}" for row in primary.itertuples()]
    x = list(range(len(labels)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 2), 5))
    ax.bar(
        [value - width for value in x],
        primary["late_heldout_accuracy"],
        width,
        label="full heldout",
    )
    ax.bar(
        x,
        primary["late_restricted_heldout_accuracy"],
        width,
        label="restricted heldout",
    )
    ax.bar(
        [value + width for value in x],
        primary["late_excluded_heldout_accuracy"],
        width,
        label="excluded heldout",
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Generalizing circuit intervention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_path_reuse(frame: pd.DataFrame, destination: Path) -> None:
    if frame.empty:
        return
    labels = [f"{row.task}\nseed {row.seed}" for row in frame.itertuples()]
    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    ax.bar([value - width / 2 for value in x], frame["early_path_reuse"], width, label="early")
    ax.bar([value + width / 2 for value in x], frame["late_path_reuse"], width, label="late")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Mean pairwise causal-cell Jaccard")
    ax.set_title("Secondary diagnostic: fixed-cell path reuse")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    raw_path = args.out / "raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    payload = json.loads(raw_path.read_text())
    frame = pd.DataFrame([_flatten_run(run) for run in payload["runs"]])
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "runs.csv", index=False)
    decision = {
        **payload["decision"],
        "format": payload["format"],
        "experiment_id": payload["experiment_id"],
        "mode": payload["mode"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "provenance": payload["provenance"],
        "oracle_reference": payload.get("oracle_reference"),
    }
    (args.out / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    _plot_fourier(frame, args.out / "fourier-circuit-concentration.png")
    _plot_interventions(frame, args.out / "fourier-circuit-interventions.png")
    _plot_path_reuse(frame, args.out / "causal-path-reuse.png")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
