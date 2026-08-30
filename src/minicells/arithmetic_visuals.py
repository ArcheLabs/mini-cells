from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .arithmetic_tasks import ArithmeticExample, arithmetic_batch
from .continual_learning import exact_logits
from .vocab import CharVocab


def save_learning_curves(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame["generation"], frame["old_token_accuracy"], label="Echo retention")
    ax.plot(frame["generation"], frame["train_answer_accuracy"], label="Arithmetic train")
    ax.plot(frame["generation"], frame["heldout_answer_accuracy"], label="Arithmetic held-out")
    ax.axhline(0.95, linestyle="--", linewidth=1, label="Echo gate 95%")
    ax.axhline(0.50, linestyle=":", linewidth=1, label="Held-out gate 50%")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Native continual learning")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_retention_capability(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(summary["old_token_accuracy"], summary["heldout_answer_accuracy"], s=70)
    for row in summary.itertuples(index=False):
        ax.annotate(row.name, (row.old_token_accuracy, row.heldout_answer_accuracy),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axvline(0.95, linestyle="--", linewidth=1)
    ax.axhline(0.50, linestyle="--", linewidth=1)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Echo retention")
    ax.set_ylabel("Held-out arithmetic accuracy")
    ax.set_title("Retention vs new capability")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _operation_grid(
    flat: torch.Tensor,
    vocab: CharVocab,
    examples: list[ArithmeticExample],
    operation: str,
) -> np.ndarray:
    selected = [item for item in examples if item.operation == operation]
    batch = arithmetic_batch(vocab, selected)
    pred = exact_logits(flat, batch.input_ids).argmax(dim=-1)
    grid = np.full((10, 10), np.nan, dtype=float)
    for row, item in enumerate(selected):
        answer_position = len(item.expression) - 1
        expected = vocab.token_to_id[str(item.answer)]
        grid[item.left, item.right] = float(int(pred[row, answer_position]) == expected)
    return grid


def save_operation_heatmap(
    flat: torch.Tensor,
    vocab: CharVocab,
    examples: list[ArithmeticExample],
    operation: str,
    path: Path,
) -> None:
    grid = _operation_grid(flat, vocab, examples, operation)
    masked = np.ma.masked_invalid(grid)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(masked, vmin=0.0, vmax=1.0, interpolation="nearest", cmap="RdYlGn")
    for left in range(10):
        for right in range(10):
            if np.isnan(grid[left, right]):
                continue
            ax.text(right, left, "✓" if grid[left, right] else "×",
                    ha="center", va="center", fontsize=10)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Right operand")
    ax.set_ylabel("Left operand")
    ax.set_title("Addition" if operation == "add" else "Subtraction")
    fig.colorbar(image, ax=ax, ticks=[0, 1], label="Incorrect / correct")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_capability_summary(metrics: dict[str, float], path: Path) -> None:
    labels = ["Echo retention", "Train arithmetic", "Held-out arithmetic", "Addition", "Subtraction"]
    values = [
        metrics["old_token_accuracy"],
        metrics["train_answer_accuracy"],
        metrics["heldout_answer_accuracy"],
        metrics["addition_answer_accuracy"],
        metrics["subtraction_answer_accuracy"],
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Tiny Arithmetic Capability — final selected model")
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.1%}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
