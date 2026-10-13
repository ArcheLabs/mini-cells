#!/usr/bin/env python3
"""Generate the v2 developmental phase diagram from machine results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.results / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    figures = args.results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if plt is not None:
        for phase in ("B", "C", "D"):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for arm, phases in payload.get("arms", {}).items():
                row = phases.get(phase, {})
                frontier = row.get("maturity_frontier", [])
                if not frontier:
                    continue
                x = [float(item["maturity"]) for item in frontier]
                axes[0].plot(x, [float(item["new_gain"]) for item in frontier], marker="o", label=arm)
                axes[1].plot(x, [float(item["old_regression"]) for item in frontier], marker="o", label=arm)
            axes[0].set(xlabel="Shadow maturity m", ylabel="New-domain gain", title=f"Phase {phase}: capability")
            axes[1].set(xlabel="Shadow maturity m", ylabel="Old-domain regression", title=f"Phase {phase}: retention")
            axes[1].axhline(float(payload["thresholds"]["max_old_regression"]), color="black", linestyle="--")
            axes[0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(figures / f"developmental-frontier-{phase}.png", dpi=160)
            plt.close(fig)
    rows = []
    for arm, phases in payload.get("arms", {}).items():
        selected_rows = []
        gains = []
        selected = []
        replay = 0
        for phase in ("B", "C", "D"):
            item = phases.get(phase, {})
            frontier = item.get("maturity_frontier", [])
            value = item.get("selected_maturity")
            selected.append("N/A" if value is None else str(value))
            replay += int(item.get("historical_examples_seen_by_candidate_trainer", 0))
            match = next((row for row in frontier if row.get("maturity") == value), None)
            if match:
                selected_rows.append(match)
                gains.append(float(match["new_gain"]))
        rows.append({
            "Arm": arm,
            "A Regression": max((float(row["old_regression"]) for row in selected_rows), default=0.0),
            "Mean Forgetting": (sum(float(row["old_regression"]) for row in selected_rows) / len(selected_rows)) if selected_rows else 0.0,
            "Plasticity": "N/A",
            "B Gain": gains[0] if len(gains) > 0 else "N/A",
            "C Gain": gains[1] if len(gains) > 1 else "N/A",
            "D Gain": gains[2] if len(gains) > 2 else "N/A",
            "Raw Replay": replay,
            "Selected m": ", ".join(selected),
        })
    if rows:
        with (args.results / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        lines = ["# Shadow Cell Validation 001 v2 report", "", f"Status: `{payload.get('status')}`", "", "| " + " | ".join(rows[0]) + " |", "|" + "|".join("---" for _ in rows[0]) + "|"]
        lines.extend("| " + " | ".join(str(row[key]) for key in rows[0]) + " |" for row in rows)
        (args.results / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.results / "phase-diagram-manifest.json").write_text(
        json.dumps({"source": str(result_path), "phases": ["B", "C", "D"], "maturity_grid": payload.get("maturity_grid")}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "FIGURES_WRITTEN" if plt is not None else "REPORT_WRITTEN_NO_MATPLOTLIB", "output": str(figures)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
