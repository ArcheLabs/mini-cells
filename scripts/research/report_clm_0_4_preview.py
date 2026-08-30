#!/usr/bin/env python3
"""Regenerate CLM-0.4 Preview public dashboard and visualizations from telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minicells.clm04mini.preview import render_preview_visualizations


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "clm-0.4-preview"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    decision = json.loads((args.results / "decision.json").read_text(encoding="utf-8"))
    dashboard = json.loads((args.results / "dashboard.json").read_text(encoding="utf-8"))
    visualizations = render_preview_visualizations(args.results)
    dashboard["visualizations"] = visualizations
    (args.results / "dashboard.json").write_text(
        json.dumps(dashboard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    model = dashboard["model"]
    learning = dashboard["learning"]
    safety = dashboard["safety"]
    growth = dashboard["growth"]
    base = dashboard["base_capability"]
    lines = [
        "# CLM-0.4 Preview",
        "",
        f"> **{decision['status']}** — product Preview telemetry, not a formal scientific decision.",
        "",
        "## Snapshot",
        "",
        f"- Parameters: `{model['parameter_count']}`",
        f"- Cells: `{model['total_cells']}` = `{model['base_cells']}` base + `{model['private_cells']}` grown",
        f"- Transactions: `{learning['transactions']}`",
        f"- Acceptance: `{learning['acceptance_rate']:.6f}`",
        f"- Growth rescue: `{learning['growth_rescue_rate']:.6f}`",
        f"- Private reuse acceptance: `{learning['private_reuse_acceptance_rate']:.6f}`",
        f"- Base Math answer exact: `{base['math_teacher_forced_answer_exact']}`",
        f"- Base Story answer exact: `{base['story_teacher_forced_answer_exact']}`",
        "",
        "## Safety / locality",
        "",
        f"- False-safe rate: `{safety['false_safe_rate']}`",
        f"- Maximum structural escape: `{safety['maximum_structural_escape_rate']}`",
        f"- Mean direct dependency coverage: `{safety['mean_direct_dependency_coverage']}`",
        f"- Final protected token accuracy: `{safety['final_protected_token_accuracy']}`",
        f"- Growth parameter overhead: `{growth['growth_parameter_overhead_ratio']}`",
        "",
        "## Public data",
        "",
        "- `dashboard.json` — current product-facing snapshot",
        "- `telemetry/timeline.csv` — longitudinal market/research metrics",
        "- `telemetry/transactions.jsonl` — full transaction outcomes",
        "- `telemetry/cell-snapshots.jsonl` — Cell lifecycle snapshots",
        "- `telemetry/cell-registry-final.jsonl` — final Cell registry",
        "- `PUBLIC_METRICS.md` — human-readable public metrics",
        "",
        "## Visualizations",
        "",
        *[f"- `{path}`" for path in visualizations],
        "",
    ]
    path = args.results / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
