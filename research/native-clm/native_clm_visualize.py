from __future__ import annotations

import html
import json
import math
from pathlib import Path


PALETTE = {
    "T1": "#374151",
    "C0": "#9ca3af",
    "C1": "#6b7280",
    "M4": "#2563eb",
    "N1": "#059669",
    "X1": "#7c3aed",
    "X2": "#c026d3",
    "X3": "#dc2626",
    "X4": "#ea580c",
}


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _write(path: Path, body: str) -> None:
    path.write_text(body + "\n", encoding="utf-8")


def _svg_start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{_esc(title)}</title>',
        f'<desc id="desc">{_esc(subtitle)}</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="40" y="38" font-family="system-ui,sans-serif" font-size="24" font-weight="700" fill="#111827">{_esc(title)}</text>',
        f'<text x="40" y="62" font-family="system-ui,sans-serif" font-size="13" fill="#6b7280">{_esc(subtitle)}</text>',
    ]


def _color(model_id: str) -> str:
    return PALETTE.get(model_id, "#64748b")


def write_final_ppl_svg(rows: list[dict[str, object]], path: Path) -> None:
    selected = [row for row in rows if str(row["id"]) in {"T1", "C0", "C1", "M4", "N1", "X1", "X2", "X3", "X4"}]
    selected.sort(key=lambda row: float(row["validation_ppl_10m"]))
    width = 980
    row_h = 42
    top = 92
    left = 120
    right = 80
    plot_w = width - left - right
    height = top + row_h * len(selected) + 70
    max_value = max(float(row["validation_ppl_10m"]) for row in selected) * 1.08
    lines = _svg_start(width, height, "Native CLM final quality", "Validation PPL at 10M training tokens; lower is better. Bars start at zero.")
    for tick in range(0, 5):
        value = max_value * tick / 4
        x = left + plot_w * tick / 4
        lines.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{height-48}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{x:.1f}" y="{height-25}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#6b7280">{value:.1f}</text>')
    for index, row in enumerate(selected):
        model_id = str(row["id"])
        value = float(row["validation_ppl_10m"])
        y = top + index * row_h
        bar_w = plot_w * value / max_value
        lines.append(f'<text x="{left-12}" y="{y+22}" text-anchor="end" font-family="system-ui,sans-serif" font-size="14" font-weight="600" fill="#111827">{_esc(model_id)}</text>')
        lines.append(f'<rect x="{left}" y="{y+7}" width="{bar_w:.1f}" height="22" rx="4" fill="{_color(model_id)}"/>')
        lines.append(f'<text x="{left+bar_w+8:.1f}" y="{y+23}" font-family="ui-monospace,SFMono-Regular,monospace" font-size="13" fill="#111827">{value:.4f}</text>')
    lines.append('</svg>')
    _write(path, "\n".join(lines))


def write_learning_curves_svg(curves: dict[str, list[dict[str, object]]], path: Path) -> None:
    order = [model_id for model_id in ("T1", "C1", "M4", "N1", "X1", "X2", "X3", "X4") if model_id in curves]
    all_rows = [row for model_id in order for row in curves[model_id]]
    min_x = min(math.log10(float(row["consumed_tokens"])) for row in all_rows)
    max_x = max(math.log10(float(row["consumed_tokens"])) for row in all_rows)
    min_y = min(float(row["validation_nll"]) for row in all_rows)
    max_y = max(float(row["validation_nll"]) for row in all_rows)
    y_pad = max(0.08, (max_y - min_y) * 0.08)
    min_y -= y_pad
    max_y += y_pad
    width, height = 980, 610
    left, right, top, bottom = 80, 220, 90, 70
    plot_w, plot_h = width-left-right, height-top-bottom
    lines = _svg_start(width, height, "Native CLM learning curves", "Validation NLL versus consumed training tokens. X axis is logarithmic; lower is better.")
    for tick in range(5):
        y_val = min_y + (max_y-min_y)*tick/4
        y = top + plot_h - plot_h*(y_val-min_y)/(max_y-min_y)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="system-ui,sans-serif" font-size="12" fill="#6b7280">{y_val:.2f}</text>')
    token_ticks = [1_000_000, 2_500_000, 5_000_000, 7_500_000, 10_000_000]
    for token in token_ticks:
        x_log = math.log10(token)
        x = left + plot_w*(x_log-min_x)/(max_x-min_x)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#f3f4f6"/>')
        label = f'{token/1_000_000:g}M'
        lines.append(f'<text x="{x:.1f}" y="{top+plot_h+28}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#6b7280">{label}</text>')
    for model_id in order:
        points = []
        for row in curves[model_id]:
            x_log = math.log10(float(row["consumed_tokens"]))
            x = left + plot_w*(x_log-min_x)/(max_x-min_x)
            y_val = float(row["validation_nll"])
            y = top + plot_h - plot_h*(y_val-min_y)/(max_y-min_y)
            points.append((x, y))
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        lines.append(f'<polyline points="{poly}" fill="none" stroke="{_color(model_id)}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in points:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.3" fill="{_color(model_id)}"/>')
    legend_x = left + plot_w + 30
    for index, model_id in enumerate(order):
        y = top + 18 + index*28
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+24}" y2="{y}" stroke="{_color(model_id)}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+34}" y="{y+4}" font-family="system-ui,sans-serif" font-size="13" fill="#111827">{_esc(model_id)}</text>')
    lines.append('</svg>')
    _write(path, "\n".join(lines))


def write_quality_compute_svg(rows: list[dict[str, object]], path: Path) -> None:
    selected = [row for row in rows if str(row["id"]) in {"T1", "C1", "M4", "N1", "X1", "X2", "X3", "X4"}]
    t1_flops = next(float(row["train_flops_estimate"]) for row in selected if str(row["id"]) == "T1")
    points = [(row, float(row["train_flops_estimate"])/t1_flops, float(row["validation_ppl_10m"])) for row in selected]
    min_x = 0.8
    max_x = max(x for _, x, _ in points) * 1.12
    min_y = min(y for _, _, y in points) * 0.94
    max_y = max(y for _, _, y in points) * 1.05
    width, height = 980, 600
    left, right, top, bottom = 90, 80, 90, 75
    plot_w, plot_h = width-left-right, height-top-bottom
    lines = _svg_start(width, height, "Quality / compute frontier", "Final validation PPL versus estimated training FLOPs normalized to T1. Lower-left is better.")
    for tick in range(5):
        x_val = min_x + (max_x-min_x)*tick/4
        x = left + plot_w*(x_val-min_x)/(max_x-min_x)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#f3f4f6"/>')
        lines.append(f'<text x="{x:.1f}" y="{top+plot_h+30}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#6b7280">{x_val:.1f}×</text>')
        y_val = min_y + (max_y-min_y)*tick/4
        y = top + plot_h - plot_h*(y_val-min_y)/(max_y-min_y)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="system-ui,sans-serif" font-size="12" fill="#6b7280">{y_val:.2f}</text>')
    for row, x_val, y_val in points:
        model_id = str(row["id"])
        x = left + plot_w*(x_val-min_x)/(max_x-min_x)
        y = top + plot_h - plot_h*(y_val-min_y)/(max_y-min_y)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{_color(model_id)}" stroke="white" stroke-width="2"/>')
        lines.append(f'<text x="{x+10:.1f}" y="{y-9:.1f}" font-family="system-ui,sans-serif" font-size="13" font-weight="600" fill="#111827">{_esc(model_id)}</text>')
    lines.append(f'<text x="{left+plot_w/2:.1f}" y="{height-18}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="13" fill="#374151">Estimated training FLOPs / T1</text>')
    lines.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18 {top+plot_h/2:.1f})" text-anchor="middle" font-family="system-ui,sans-serif" font-size="13" fill="#374151">Validation PPL @ 10M</text>')
    lines.append('</svg>')
    _write(path, "\n".join(lines))


def write_phase_c_visualizations(leaderboard: list[dict[str, object]], curves: dict[str, list[dict[str, object]]], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "final_ppl": "phase-c-final-ppl.svg",
        "learning_curves": "phase-c-learning-curves.svg",
        "quality_compute": "phase-c-quality-compute.svg",
    }
    write_final_ppl_svg(leaderboard, output_dir / files["final_ppl"])
    write_learning_curves_svg(curves, output_dir / files["learning_curves"])
    write_quality_compute_svg(leaderboard, output_dir / files["quality_compute"])
    manifest = {
        "format": "minicells.native-clm-phase-c-visualizations.v1",
        "files": files,
        "notes": {
            "final_ppl": "10M validation PPL; lower is better; zero-based bars.",
            "learning_curves": "Validation NLL at frozen checkpoints; logarithmic token x-axis.",
            "quality_compute": "Estimated training FLOPs normalized to T1 versus final validation PPL; lower-left is better.",
        },
    }
    (output_dir / "phase-c-visualizations.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
