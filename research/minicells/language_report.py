from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(frame: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for name, group in frame.groupby("model"):
        group = group.sort_values("consumed_tokens")
        axis.plot(group["consumed_tokens"], group["validation_nll"], marker="o", label=name)
    axis.set_xlabel("Consumed training tokens")
    axis.set_ylabel("Validation NLL")
    axis.set_title("Experiment 005 — Validation loss scaling")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(fig, path)


def save_ppl_scaling(frame: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for name, group in frame.groupby("model"):
        group = group.sort_values("consumed_tokens")
        axis.plot(group["consumed_tokens"], group["validation_ppl"], marker="o", label=name)
    axis.set_xlabel("Consumed training tokens")
    axis.set_ylabel("Validation perplexity")
    axis.set_title("Perplexity at 125K / 250K / 500K")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(fig, path)


def save_relative_gap(frame: pd.DataFrame, transformer_name: str, path: Path) -> None:
    pivot = frame.pivot(index="consumed_tokens", columns="model", values="validation_ppl").sort_index()
    if transformer_name not in pivot:
        raise ValueError(f"missing transformer baseline {transformer_name!r}")
    baseline = pivot[transformer_name]
    fig, axis = plt.subplots(figsize=(8, 5))
    for name in pivot.columns:
        if name == transformer_name:
            continue
        ratio = pivot[name] / baseline
        axis.plot(pivot.index, ratio, marker="o", label=f"{name} / Transformer")
    axis.axhline(1.0, linewidth=1, linestyle="--")
    axis.axhline(1.25, linewidth=1, linestyle=":", label="GREEN threshold 1.25×")
    axis.axhline(1.60, linewidth=1, linestyle=":", label="YELLOW ceiling 1.60×")
    axis.set_xlabel("Consumed training tokens")
    axis.set_ylabel("Relative validation PPL")
    axis.set_title("Relative gap to parameter-matched Transformer")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(fig, path)


def save_learning_slopes(model_summary: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(model_summary["model"], model_summary["learning_slope_alpha"])
    axis.set_ylabel("Estimated alpha in log(NLL) ~ -alpha log(tokens)")
    axis.set_title("Early learning slope")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    _save(fig, path)


def save_throughput(model_summary: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(model_summary["model"], model_summary["tokens_per_second"])
    axis.set_ylabel("Training tokens / second")
    axis.set_title("Kaggle training throughput")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    _save(fig, path)


def save_consumer_summary(
    model_summary: pd.DataFrame,
    decision: dict[str, object],
    path: Path,
) -> None:
    columns = [
        "model",
        "parameters",
        "ppl_125k",
        "ppl_250k",
        "ppl_500k",
        "learning_slope_alpha",
        "tokens_per_second",
    ]
    table = model_summary[columns].copy()
    for field in ("ppl_125k", "ppl_250k", "ppl_500k"):
        table[field] = table[field].map(lambda value: f"{value:.1f}")
    table["learning_slope_alpha"] = table["learning_slope_alpha"].map(lambda value: f"{value:.3f}")
    table["tokens_per_second"] = table["tokens_per_second"].map(lambda value: f"{value:.0f}")
    table["parameters"] = table["parameters"].map(lambda value: f"{int(value):,}")

    fig, axis = plt.subplots(figsize=(11, 4.6))
    axis.axis("off")
    title = (
        f"Experiment 005 — {decision['status']}\n"
        f"{decision['diagnosis']} | MiniTextNCA+/Transformer @500K = "
        f"{decision['candidate']['ppl_ratio_500k']:.3f}×"
    )
    axis.set_title(title, fontsize=13, pad=16)
    display_columns = [
        "Model",
        "Parameters",
        "PPL 125K",
        "PPL 250K",
        "PPL 500K",
        "alpha",
        "tok/s",
    ]
    rendered = axis.table(
        cellText=table.to_numpy(),
        colLabels=display_columns,
        cellLoc="center",
        loc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(9)
    rendered.scale(1.0, 1.6)
    _save(fig, path)


def write_generation_progression(generations: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Experiment 005 Generation Progression",
        "",
        "Fixed prompts use temperature 0.8, top-k 40, 32 new tokens and deterministic seeds.",
        "These samples are qualitative evidence only; perplexity remains the quantitative metric.",
        "",
    ]
    frame = pd.DataFrame(generations).sort_values(["consumed_tokens", "model", "prompt"])
    for consumed, token_group in frame.groupby("consumed_tokens"):
        lines.extend([f"## {int(consumed):,} consumed tokens", ""])
        for model, model_group in token_group.groupby("model"):
            lines.extend([f"### {model}", ""])
            for row in model_group.itertuples(index=False):
                lines.extend(
                    [
                        f"**Prompt:** `{row.prompt}`",
                        "",
                        str(row.text).replace("\n", " "),
                        "",
                    ]
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_configs(configs: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(configs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
