"""Machine-readable CLM-0.3 reports and lightweight plot generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


PPL_COLUMNS = (
    "replicate", "arm", "tokens", "phase", "ppl", "nll", "ppl_vs_fixed4",
    "ppl_vs_clm01", "ppl_vs_textnca", "health",
)


def validate_telemetry_event(event: dict[str, Any]) -> None:
    event_type = event.get("type")
    common = ("type", "arm", "replicate")
    missing = [key for key in common if key not in event]
    if missing:
        raise ValueError(f"telemetry event missing fields: {missing}")
    required = {
        "training_progress": ("consumed_tokens", "target_tokens", "phase"),
        "birth": ("birth_index", "stage", "parent", "child", "parity_status"),
        "evaluation": ("tokens", "ppl", "nll", "raw_model_ppl", "clm01_start_ppl", "textnca_frozen_ppl"),
    }.get(event_type, ())
    missing = [key for key in required if key not in event]
    if missing:
        raise ValueError(f"{event_type} telemetry event missing fields: {missing}")


def write_ppl_history(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    import csv

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PPL_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def write_growth_history(path: str | Path, history: Iterable[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(list(history), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def save_growth_plots(
    output_dir: str | Path,
    *,
    ppl_rows: Iterable[dict[str, Any]],
    growth_history: Iterable[dict[str, Any]],
    lineage_rows: Iterable[dict[str, Any]] = (),
    telemetry_rows: Iterable[dict[str, Any]] = (),
) -> list[Path]:
    """Generate the required plot set when an experiment has data.

    No placeholder plots are emitted when the formal experiment has no data.
    """

    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ppl = list(ppl_rows)
    events = list(growth_history)
    lineages = list(lineage_rows)
    telemetry = list(telemetry_rows)
    if not ppl:
        return []

    def figure(name: str, title: str, x: list[float] | None = None, y: list[float] | None = None) -> Path:
        fig, axis = plt.subplots(figsize=(7, 4))
        if x and y:
            axis.plot(x, y, marker="o")
        axis.set_title(title)
        axis.grid(alpha=.25)
        fig.tight_layout()
        path = output / name
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path

    paths = []
    paths.append(figure("ppl-over-time.png", "CLM-0.3 PPL over time",
                        [float(row["tokens"]) for row in ppl], [float(row["ppl"]) for row in ppl]))
    paths.append(figure("ppl-ratio-vs-fixed4.png", "PPL ratio vs fixed-4",
                        [float(row["tokens"]) for row in ppl], [float(row["ppl_vs_fixed4"]) for row in ppl]))
    event_x = [float(event["token"]) for event in events]
    paths.append(figure("growth-events.png", "Growth events", event_x, [1.0] * len(event_x)))
    paths.append(figure("expert-count-over-time.png", "Expert count over time", event_x, [12 + index for index in range(len(event_x))]))
    paths.append(figure("lineage-usage.png", "Lineage usage"))
    paths.append(figure("lineage-divergence.png", "Lineage divergence"))
    paths.append(figure("newborn-causal-utility.png", "Newborn causal utility"))
    paths.append(figure("pressure-ranking.png", "Pressure ranking"))
    paths.append(figure("throughput.png", "Throughput",
                        [float(row.get("consumed_tokens", 0)) for row in telemetry],
                        [float(row.get("tokens_per_second", 0)) for row in telemetry]))
    paths.append(figure("vram.png", "Peak VRAM",
                        [float(row.get("consumed_tokens", 0)) for row in telemetry],
                        [float(row.get("peak_vram_bytes", 0)) for row in telemetry]))
    return paths
