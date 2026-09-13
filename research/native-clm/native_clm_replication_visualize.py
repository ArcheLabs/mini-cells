from __future__ import annotations

import html
import math
from pathlib import Path


PALETTE = {
    "T1": "#374151",
    "M4": "#2563eb",
    "X3": "#dc2626",
    "X4": "#ea580c",
}


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _header(width: int, height: int, title: str, subtitle: str) -> list[str]:
    display_title = title if "Native CLM" in title else f"Native CLM — {title}"
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{_esc(display_title)}</title>',
        f'<desc id="desc">{_esc(subtitle)}</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="40" y="38" font-family="system-ui,sans-serif" font-size="24" font-weight="700" fill="#111827">{_esc(display_title)}</text>',
        f'<text x="40" y="62" font-family="system-ui,sans-serif" font-size="13" fill="#6b7280">{_esc(subtitle)}</text>',
    ]


def _color(model_id: str) -> str:
    return PALETTE.get(model_id, "#64748b")


def write_seed_final_ppl(rows: list[dict[str, object]], seed: int, path: Path) -> None:
    selected = sorted(rows, key=lambda r: float(r["validation_ppl_10m"]))
    width, left, right, top, row_h = 920, 120, 80, 92, 48
    height = top + len(selected) * row_h + 70
    plot_w = width - left - right
    max_v = max(float(r["validation_ppl_10m"]) for r in selected) * 1.08
    lines = _header(width, height, f"seed {seed} final PPL",
                    "Validation PPL at 10M tokens; lower is better. Bars start at zero.")
    for tick in range(5):
        value = max_v * tick / 4
        x = left + plot_w * tick / 4
        lines.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{height-48}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{x:.1f}" y="{height-24}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#6b7280">{value:.1f}</text>')
    for idx, row in enumerate(selected):
        model_id = str(row["id"])
        value = float(row["validation_ppl_10m"])
        y = top + idx * row_h
        bar_w = plot_w * value / max_v
        lines.append(f'<text x="{left-12}" y="{y+25}" text-anchor="end" font-family="system-ui,sans-serif" font-size="14" font-weight="600" fill="#111827">{_esc(model_id)}</text>')
        lines.append(f'<rect x="{left}" y="{y+8}" width="{bar_w:.1f}" height="24" rx="4" fill="{_color(model_id)}"/>')
        lines.append(f'<text x="{left+bar_w+8:.1f}" y="{y+25}" font-family="ui-monospace,SFMono-Regular,monospace" font-size="13" fill="#111827">{value:.4f}</text>')
    lines.append("</svg>")
    _write(path, lines)


def write_seed_learning_curves(curves: dict[str, list[dict[str, object]]], seed: int, path: Path) -> None:
    order = [m for m in ("T1", "M4", "X3", "X4") if m in curves]
    all_rows = [row for model_id in order for row in curves[model_id]]
    min_x = min(math.log10(float(row["consumed_tokens"])) for row in all_rows)
    max_x = max(math.log10(float(row["consumed_tokens"])) for row in all_rows)
    min_y = min(float(row["validation_nll"]) for row in all_rows)
    max_y = max(float(row["validation_nll"]) for row in all_rows)
    pad = max(0.06, (max_y - min_y) * 0.08)
    min_y, max_y = min_y - pad, max_y + pad
    width, height = 940, 600
    left, right, top, bottom = 78, 180, 90, 70
    plot_w, plot_h = width-left-right, height-top-bottom
    lines = _header(width, height, f"seed {seed} learning curves",
                    "Validation NLL versus consumed tokens; logarithmic token axis; lower is better.")
    for tick in range(5):
        yv = min_y + (max_y-min_y)*tick/4
        y = top + plot_h - plot_h*(yv-min_y)/(max_y-min_y)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="system-ui,sans-serif" font-size="12" fill="#6b7280">{yv:.2f}</text>')
    for token in (1_000_000, 2_500_000, 5_000_000, 7_500_000, 10_000_000):
        xlog = math.log10(token)
        x = left + plot_w*(xlog-min_x)/(max_x-min_x)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#f3f4f6"/>')
        lines.append(f'<text x="{x:.1f}" y="{top+plot_h+28}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#6b7280">{token/1_000_000:g}M</text>')
    for model_id in order:
        pts = []
        for row in curves[model_id]:
            xlog = math.log10(float(row["consumed_tokens"]))
            x = left + plot_w*(xlog-min_x)/(max_x-min_x)
            yv = float(row["validation_nll"])
            y = top + plot_h - plot_h*(yv-min_y)/(max_y-min_y)
            pts.append((x, y))
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{_color(model_id)}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in pts:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{_color(model_id)}"/>')
    lx = left + plot_w + 26
    for idx, model_id in enumerate(order):
        y = top + 20 + idx*30
        lines.append(f'<line x1="{lx}" y1="{y}" x2="{lx+24}" y2="{y}" stroke="{_color(model_id)}" stroke-width="3"/>')
        lines.append(f'<text x="{lx+34}" y="{y+4}" font-family="system-ui,sans-serif" font-size="13" fill="#111827">{model_id}</text>')
    lines.append("</svg>")
    _write(path, lines)


def write_seed_quality_compute(rows: list[dict[str, object]], seed: int, path: Path) -> None:
    t1 = next(float(r["train_flops_estimate"]) for r in rows if str(r["id"]) == "T1")
    pts = [(r, float(r["train_flops_estimate"])/t1, float(r["validation_ppl_10m"])) for r in rows]
    min_x, max_x = 0.85, max(x for _,x,_ in pts)*1.10
    min_y, max_y = min(y for _,_,y in pts)*0.97, max(y for _,_,y in pts)*1.03
    width, height = 920, 590
    left, right, top, bottom = 90, 70, 90, 75
    pw, ph = width-left-right, height-top-bottom
    lines = _header(width, height, f"seed {seed} quality / compute",
                    "Final PPL versus estimated train FLOPs normalized to same-seed T1; lower-left is better.")
    for tick in range(5):
        xv = min_x + (max_x-min_x)*tick/4
        x = left + pw*(xv-min_x)/(max_x-min_x)
        yv = min_y + (max_y-min_y)*tick/4
        y = top + ph - ph*(yv-min_y)/(max_y-min_y)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+ph}" stroke="#f3f4f6"/>')
        lines.append(f'<text x="{x:.1f}" y="{top+ph+30}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#6b7280">{xv:.1f}×</text>')
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="system-ui,sans-serif" font-size="12" fill="#6b7280">{yv:.2f}</text>')
    for row, xv, yv in pts:
        model_id = str(row["id"])
        x = left + pw*(xv-min_x)/(max_x-min_x)
        y = top + ph - ph*(yv-min_y)/(max_y-min_y)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{_color(model_id)}" stroke="white" stroke-width="2"/>')
        lines.append(f'<text x="{x+10:.1f}" y="{y-9:.1f}" font-family="system-ui,sans-serif" font-size="13" font-weight="600" fill="#111827">{model_id}</text>')
    lines.append(f'<text x="{left+pw/2:.1f}" y="{height-18}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="13" fill="#374151">Estimated train FLOPs / same-seed T1</text>')
    lines.append("</svg>")
    _write(path, lines)


def write_cross_seed_ppl(per_seed: dict[int, list[dict[str, object]]], path: Path) -> None:
    model_ids = ("T1", "M4", "X3", "X4")
    values = {
        model_id: [
            float(next(row["validation_ppl_10m"] for row in per_seed[seed] if str(row["id"]) == model_id))
            for seed in sorted(per_seed)
        ]
        for model_id in model_ids
    }
    allv = [v for seq in values.values() for v in seq]
    min_y, max_y = min(allv)*0.97, max(allv)*1.03
    width, height = 900, 590
    left, right, top, bottom = 90, 60, 90, 80
    pw, ph = width-left-right, height-top-bottom
    lines = _header(width, height, "cross-seed final PPL",
                    "Individual development-seed results with mean and min–max range; lower is better.")
    for tick in range(5):
        yv = min_y + (max_y-min_y)*tick/4
        y = top + ph - ph*(yv-min_y)/(max_y-min_y)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="system-ui,sans-serif" font-size="12" fill="#6b7280">{yv:.2f}</text>')
    step = pw / len(model_ids)
    offsets = (-8, 8)
    for idx, model_id in enumerate(model_ids):
        x = left + step*(idx+0.5)
        seq = values[model_id]
        mean = sum(seq)/len(seq)
        lo, hi = min(seq), max(seq)
        ymean = top + ph - ph*(mean-min_y)/(max_y-min_y)
        ylo = top + ph - ph*(lo-min_y)/(max_y-min_y)
        yhi = top + ph - ph*(hi-min_y)/(max_y-min_y)
        lines.append(f'<line x1="{x:.1f}" y1="{yhi:.1f}" x2="{x:.1f}" y2="{ylo:.1f}" stroke="{_color(model_id)}" stroke-width="3"/>')
        lines.append(f'<line x1="{x-9:.1f}" y1="{ymean:.1f}" x2="{x+9:.1f}" y2="{ymean:.1f}" stroke="{_color(model_id)}" stroke-width="5"/>')
        for j, (seed, value) in enumerate(zip(sorted(per_seed), seq)):
            y = top + ph - ph*(value-min_y)/(max_y-min_y)
            lines.append(f'<circle cx="{x+offsets[j]:.1f}" cy="{y:.1f}" r="5" fill="{_color(model_id)}" opacity="0.75"/>')
            lines.append(f'<text x="{x+offsets[j]+7:.1f}" y="{y-7:.1f}" font-family="system-ui,sans-serif" font-size="10" fill="#6b7280">{seed}</text>')
        lines.append(f'<text x="{x:.1f}" y="{top+ph+34}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="600" fill="#111827">{model_id}</text>')
        lines.append(f'<text x="{x:.1f}" y="{top+ph+54}" text-anchor="middle" font-family="ui-monospace,SFMono-Regular,monospace" font-size="11" fill="#6b7280">mean {mean:.4f}</text>')
    lines.append("</svg>")
    _write(path, lines)


def write_paired_delta(per_seed: dict[int, list[dict[str, object]]], path: Path) -> None:
    model_ids = ("M4", "X3", "X4")
    seeds = sorted(per_seed)
    deltas: dict[str, list[float]] = {}
    for model_id in model_ids:
        vals = []
        for seed in seeds:
            rows = per_seed[seed]
            t1 = float(next(r["validation_ppl_10m"] for r in rows if str(r["id"]) == "T1"))
            model = float(next(r["validation_ppl_10m"] for r in rows if str(r["id"]) == model_id))
            vals.append(model - t1)
        deltas[model_id] = vals
    allv = [0.0] + [v for seq in deltas.values() for v in seq]
    min_y, max_y = min(allv)-0.08, max(allv)+0.08
    if max_y-min_y < 0.2:
        max_y = min_y + 0.2
    width, height = 900, 560
    left, right, top, bottom = 90, 60, 90, 80
    pw, ph = width-left-right, height-top-bottom
    lines = _header(width, height, "paired ΔPPL versus T1",
                    "Each point is model PPL minus same-seed T1 PPL. Negative values favor the CLM candidate.")
    y0 = top + ph - ph*(0-min_y)/(max_y-min_y)
    lines.append(f'<line x1="{left}" y1="{y0:.1f}" x2="{left+pw}" y2="{y0:.1f}" stroke="#111827" stroke-width="1.5"/>')
    for tick in range(5):
        yv = min_y + (max_y-min_y)*tick/4
        y = top + ph - ph*(yv-min_y)/(max_y-min_y)
        lines.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="system-ui,sans-serif" font-size="12" fill="#6b7280">{yv:+.2f}</text>')
    step = pw/len(model_ids)
    for idx, model_id in enumerate(model_ids):
        x = left + step*(idx+0.5)
        seq = deltas[model_id]
        mean = sum(seq)/len(seq)
        for j, (seed, value) in enumerate(zip(seeds, seq)):
            y = top + ph - ph*(value-min_y)/(max_y-min_y)
            xo = x + (-8 if j == 0 else 8)
            lines.append(f'<circle cx="{xo:.1f}" cy="{y:.1f}" r="6" fill="{_color(model_id)}"/>')
            lines.append(f'<text x="{xo+8:.1f}" y="{y-8:.1f}" font-family="system-ui,sans-serif" font-size="10" fill="#6b7280">{seed}</text>')
        ym = top + ph - ph*(mean-min_y)/(max_y-min_y)
        lines.append(f'<line x1="{x-18:.1f}" y1="{ym:.1f}" x2="{x+18:.1f}" y2="{ym:.1f}" stroke="{_color(model_id)}" stroke-width="4"/>')
        lines.append(f'<text x="{x:.1f}" y="{top+ph+34}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="600" fill="#111827">{model_id}</text>')
        lines.append(f'<text x="{x:.1f}" y="{top+ph+54}" text-anchor="middle" font-family="ui-monospace,SFMono-Regular,monospace" font-size="11" fill="#6b7280">mean {mean:+.4f}</text>')
    lines.append("</svg>")
    _write(path, lines)


def write_seed_visualizations(rows, curves, seed: int, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "final_ppl": f"replication-seed-{seed}-final-ppl.svg",
        "learning_curves": f"replication-seed-{seed}-learning-curves.svg",
        "quality_compute": f"replication-seed-{seed}-quality-compute.svg",
    }
    write_seed_final_ppl(rows, seed, output_dir/files["final_ppl"])
    write_seed_learning_curves(curves, seed, output_dir/files["learning_curves"])
    write_seed_quality_compute(rows, seed, output_dir/files["quality_compute"])
    return files
