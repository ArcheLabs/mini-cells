"""Publication-ready visualizations for the CLM-0.3 public release benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    for suffix in ("png", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
        generated.append(path.name)
    plt.close(fig)
    return generated


def _annotate_bars(axis: plt.Axes, bars: Any, *, formatter) -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.annotate(
            formatter(value),
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def figure_language_foundation(
    historical: dict[str, Any], bridge: dict[str, Any], output_dir: Path
) -> list[str]:
    ratio_006 = float(
        historical["experiment_006"]["ppl_ratio_textnca_over_transformer"]
    )
    ratio_007 = float(
        historical["experiment_007"]["ppl_ratio_textnca_over_transformer"]
    )
    dense = bridge["arms"]["textnca_continuation"]
    clm = bridge["arms"]["clm_fixed4"]
    ratio_bridge = float(clm["final_ppl"]) / float(dense["final_ppl"])
    labels = [
        "TextNCA / Transformer\n1.17M · 10M tokens",
        "TextNCA / Transformer\n~30M · 100M tokens",
        "CLM fixed / TextNCA\nrelease bridge · +1M",
    ]
    values = [ratio_006, ratio_007, ratio_bridge]

    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    bars = axis.bar(labels, values)
    axis.axhline(1.0, linewidth=1.0, linestyle="--")
    axis.set_ylabel("Validation PPL ratio (1.0 = matched baseline)")
    axis.set_title("CLM-0.3 language-model quality chain")
    lower = min(0.97, min(values) - 0.01)
    upper = max(1.06, max(values) + 0.02)
    axis.set_ylim(lower, upper)
    axis.grid(axis="y", alpha=0.22)
    _annotate_bars(axis, bars, formatter=lambda value: f"{value:.4f}×")
    axis.text(
        0.0,
        -0.19,
        "Each bar uses its own matched baseline; ratios from different experiments are not multiplied.",
        transform=axis.transAxes,
        fontsize=9,
        va="top",
    )
    fig.tight_layout()
    return _save(fig, output_dir, "figure-1-language-quality")


def figure_bridge_curve(bridge: dict[str, Any], output_dir: Path) -> list[str]:
    fig, axis = plt.subplots(figsize=(7.8, 4.8))
    for arm, label in (
        ("textnca_continuation", "TextNCA continuation"),
        ("clm_fixed4", "CLM fixed4"),
    ):
        checkpoints = sorted(
            bridge["arms"][arm]["checkpoints"],
            key=lambda row: int(row["consumed_tokens"]),
        )
        x = [int(row["consumed_tokens"]) / 1_000_000.0 for row in checkpoints]
        y = [float(row["validation_ppl"]) for row in checkpoints]
        axis.plot(x, y, marker="o", label=label)
    axis.set_xlabel("Continuation tokens (millions)")
    axis.set_ylabel("Validation PPL")
    axis.set_title("Same trained TextNCA, with and without CLM machinery")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, output_dir, "figure-2-machinery-bridge")


def figure_developmental_selectivity(
    capability: dict[str, Any], output_dir: Path
) -> list[str]:
    stationary_rate = 100.0 * (
        1.0 - float(capability["stationary_rejected"]) / float(capability["stationary_total"])
    )
    shift_rate = 100.0 * float(capability["shift_promoted"]) / float(capability["shift_total"])
    labels = ["Stationary\nTinyStories", "Capability shift\nStory + Arithmetic"]
    values = [stationary_rate, shift_rate]

    fig, axis = plt.subplots(figsize=(6.8, 4.8))
    bars = axis.bar(labels, values)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Persistent promotion rate")
    axis.set_title("Probation grows only when future utility survives selection")
    axis.grid(axis="y", alpha=0.22)
    _annotate_bars(axis, bars, formatter=lambda value: f"{value:.0f}%")
    gains = [
        float(row["ppl_improvement_percent"])
        for row in capability.get("promoted_replicates", [])
    ]
    if gains:
        gain_text = "Independent shift gains: " + ", ".join(f"{value:.2f}%" for value in gains)
        axis.text(0.0, -0.18, gain_text, transform=axis.transAxes, fontsize=9, va="top")
    fig.tight_layout()
    return _save(fig, output_dir, "figure-3-developmental-selectivity")


def figure_reference_cost(
    decision: dict[str, Any], bridge: dict[str, Any], output_dir: Path
) -> list[str]:
    runtime = decision["reference_runtime"]
    quality = decision["language_quality"]
    dense = bridge["arms"]["textnca_continuation"]
    clm = bridge["arms"]["clm_fixed4"]
    dense_parameters = dense["parameters"]
    clm_parameters = clm["parameters"]
    values = [
        float(quality["final_ppl_ratio_clm_over_textnca"]),
        float(clm_parameters["active_parameter_proxy"])
        / float(dense_parameters["active_parameter_proxy"]),
        float(clm_parameters["total_parameters"])
        / float(dense_parameters["total_parameters"]),
        float(runtime["inference_time_per_token_ratio_clm_over_textnca"]),
        float(runtime["train_time_per_token_ratio_clm_over_textnca"]),
        float(runtime["inference_vram_ratio_clm_over_textnca"]),
    ]
    labels = [
        "Final PPL",
        "Active-param proxy",
        "Stored params",
        "Inference time/token",
        "Training time/token",
        "Inference VRAM",
    ]

    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    y = list(range(len(labels)))
    bars = axis.barh(y, values)
    axis.axvline(1.0, linewidth=1.0, linestyle="--")
    axis.set_yticks(y, labels=labels)
    axis.invert_yaxis()
    axis.set_xlabel("CLM / TextNCA ratio (1.0 = parity)")
    axis.set_title("CLM-0.3 reference quality and runtime cost")
    axis.grid(axis="x", alpha=0.22)
    for bar, value in zip(bars, values):
        axis.annotate(
            f"{value:.2f}×",
            xy=(value, bar.get_y() + bar.get_height() / 2.0),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    axis.text(
        0.0,
        -0.13,
        "Active parameters are a structural proxy, not measured FLOPs. Runtime uses the current sparse_dispatch reference backend.",
        transform=axis.transAxes,
        fontsize=9,
        va="top",
    )
    fig.tight_layout()
    return _save(fig, output_dir, "figure-4-reference-cost")


def save_release_figures(
    *,
    historical: dict[str, Any],
    bridge: dict[str, Any],
    capability: dict[str, Any],
    decision: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    generated: list[str] = []
    generated += figure_language_foundation(historical, bridge, output_dir)
    generated += figure_bridge_curve(bridge, output_dir)
    generated += figure_developmental_selectivity(capability, output_dir)
    generated += figure_reference_cost(decision, bridge, output_dir)
    return generated
