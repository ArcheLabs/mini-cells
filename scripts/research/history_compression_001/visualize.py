from __future__ import annotations

import csv
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts" / "experiments" / "history-compression-001"
OUT = ARTIFACTS / "visualization"


def _load_rows() -> list[dict[str, Any]]:
    path = ARTIFACTS / "summary.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {
        "seed": int,
        "history_prompt_count": int,
        "heldout_nll_gain": float,
        "history_evaluation_mean_kl": float,
        "history_evaluation_top1_identity": float,
        "delta_l2_norm": float,
        "expert_index": int,
        "group_index": int,
        "target_router_topk_identity": float,
    }
    for row in rows:
        for key, caster in numeric.items():
            if row.get(key) not in (None, ""):
                row[key] = caster(row[key])
    return rows


def _group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)
    summary: list[dict[str, Any]] = []
    for mode, items in grouped.items():
        summary.append(
            {
                "mode": mode,
                "history_prompt_count": int(items[0]["history_prompt_count"]),
                "pass_count": sum(1 for item in items if item["status"] == "PASS"),
                "seed_count": len(items),
                "median_gain": statistics.median(
                    float(item["heldout_nll_gain"]) for item in items
                ),
                "median_kl": statistics.median(
                    float(item["history_evaluation_mean_kl"]) for item in items
                ),
                "median_top1": statistics.median(
                    float(item["history_evaluation_top1_identity"]) for item in items
                ),
                "coordinates": sorted(
                    {
                        (int(item["expert_index"]), int(item["group_index"]))
                        for item in items
                    }
                ),
            }
        )
    return sorted(summary, key=lambda row: int(row["history_prompt_count"]))


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if high <= low:
        return (start + end) / 2
    ratio = (value - low) / (high - low)
    return start + ratio * (end - start)


def _svg(summary: list[dict[str, Any]], decision: dict[str, Any]) -> str:
    width, height = 1200, 760
    left = 110
    panel_w = 480
    panel_h = 245
    gap_x = 90
    gap_y = 70
    x_values = [float(row["history_prompt_count"]) for row in summary]
    x_low, x_high = min(x_values, default=0.0), max(x_values, default=32.0)

    def esc(value: Any) -> str:
        return html.escape(str(value))

    def panel(
        x0: int,
        y0: int,
        title: str,
        values: list[float],
        y_min: float,
        y_max: float,
        formatter,
        threshold: float | None = None,
    ) -> str:
        parts = [
            f'<text x="{x0}" y="{y0 - 18}" font-size="20" font-weight="600">{esc(title)}</text>',
            f'<line x1="{x0}" y1="{y0 + panel_h}" x2="{x0 + panel_w}" y2="{y0 + panel_h}" stroke="#888"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + panel_h}" stroke="#888"/>',
        ]
        for tick in range(5):
            frac = tick / 4
            value = y_max - frac * (y_max - y_min)
            y = y0 + frac * panel_h
            parts.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_w}" y2="{y:.1f}" stroke="#ddd" stroke-dasharray="3 5"/>'
            )
            parts.append(
                f'<text x="{x0 - 12}" y="{y + 5:.1f}" font-size="13" text-anchor="end" fill="#555">{esc(formatter(value))}</text>'
            )
        if threshold is not None and y_min <= threshold <= y_max:
            ty = _scale(threshold, y_min, y_max, y0 + panel_h, y0)
            parts.append(
                f'<line x1="{x0}" y1="{ty:.1f}" x2="{x0 + panel_w}" y2="{ty:.1f}" stroke="#666" stroke-width="2" stroke-dasharray="8 6"/>'
            )
            parts.append(
                f'<text x="{x0 + panel_w - 4}" y="{ty - 6:.1f}" font-size="12" text-anchor="end" fill="#555">gate {esc(formatter(threshold))}</text>'
            )
        points: list[str] = []
        for row, value in zip(summary, values):
            x = _scale(
                float(row["history_prompt_count"]), x_low, x_high, x0 + 18, x0 + panel_w - 18
            )
            y = _scale(value, y_min, y_max, y0 + panel_h, y0)
            points.append(f"{x:.1f},{y:.1f}")
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#444"/><text x="{x:.1f}" y="{y - 10:.1f}" font-size="12" text-anchor="middle">{esc(formatter(value))}</text>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{y0 + panel_h + 22}" font-size="13" text-anchor="middle" fill="#555">{int(row["history_prompt_count"])} prompts</text>'
            )
        if len(points) > 1:
            parts.insert(
                3,
                f'<polyline points="{" ".join(points)}" fill="none" stroke="#444" stroke-width="2"/>',
            )
        return "".join(parts)

    gains = [float(row["median_gain"]) for row in summary]
    kls = [float(row["median_kl"]) for row in summary]
    top1 = [float(row["median_top1"]) for row in summary]
    pass_rates = [
        100.0 * float(row["pass_count"]) / max(float(row["seed_count"]), 1.0)
        for row in summary
    ]
    gain_max = max([13.0, *gains])
    kl_max = max([0.055, *kls])

    decision_status = decision.get("status", "PENDING")
    minimum = decision.get("minimum_observed_supported_history_prompts")
    subtitle = (
        f"Decision: {decision_status}"
        + (f" · minimum observed supported budget: {minimum}" if minimum is not None else "")
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#222}</style>',
        '<text x="50" y="46" font-size="28" font-weight="700">History Compression 001</text>',
        f'<text x="50" y="74" font-size="15" fill="#555">{esc(subtitle)}</text>',
        panel(left, 125, "Supported formal seeds", pass_rates, 0.0, 100.0, lambda v: f"{v:.0f}%", threshold=66.6667),
        panel(left + panel_w + gap_x, 125, "Median held-out NLL gain", gains, 0.0, gain_max, lambda v: f"{v:.2f}", threshold=0.5),
        panel(left, 125 + panel_h + gap_y, "Median withheld-history KL", kls, 0.0, kl_max, lambda v: f"{v:.4f}", threshold=0.05),
        panel(left + panel_w + gap_x, 125 + panel_h + gap_y, "Median withheld-history Top-1 identity", top1, 0.0, 1.0, lambda v: f"{v:.3f}", threshold=0.96875),
        '<text x="50" y="735" font-size="13" fill="#666">Coordinates by mode: ',
    ]
    coord_text = " · ".join(
        f"{row['mode']}={','.join(f'E{e}/G{g}' for e, g in row['coordinates'])}"
        for row in summary
    )
    parts.append(f'<tspan>{esc(coord_text)}</tspan></text>')
    parts.append("</svg>")
    return "".join(parts)


def _markdown(summary: list[dict[str, Any]], decision: dict[str, Any]) -> str:
    lines = [
        "# History Compression 001 — Result Summary",
        "",
        f"Status: **{decision.get('status', 'PENDING')}**",
        "",
        "| Mode | History prompts | Pass | Median heldout gain | Median eval KL | Median Top-1 | Coordinates |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        coords = ", ".join(f"E{e}/G{g}" for e, g in row["coordinates"])
        lines.append(
            "| {mode} | {count} | {passed}/{total} | {gain:.6f} | {kl:.8f} | {top1:.5f} | {coords} |".format(
                mode=row["mode"],
                count=row["history_prompt_count"],
                passed=row["pass_count"],
                total=row["seed_count"],
                gain=row["median_gain"],
                kl=row["median_kl"],
                top1=row["median_top1"],
                coords=coords,
            )
        )
    lines.extend(
        [
            "",
            "The plot and this table are derived from durable per-mode `result.json` files. They are views, not the scientific source of truth.",
            "",
        ]
    )
    return "\n".join(lines)


def visualize() -> list[Path]:
    rows = _load_rows()
    if not rows:
        return []
    decision_path = ARTIFACTS / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.is_file() else {}
    summary = _group(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / "history-compression-summary.svg"
    md_path = OUT / "summary.md"
    svg_path.write_text(_svg(summary, decision), encoding="utf-8")
    md_path.write_text(_markdown(summary, decision), encoding="utf-8")
    return [svg_path, md_path]


def main() -> int:
    paths = visualize()
    print(json.dumps({"visualizations": [str(path.relative_to(ROOT)) for path in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
