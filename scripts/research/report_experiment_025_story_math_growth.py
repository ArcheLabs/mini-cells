#!/usr/bin/env python3
"""Aggregate Experiment 025 and render internal/public figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "research"))

from minicells.story_math_shift_30m import (  # noqa: E402
    EXPECTED_SOURCE_TOKENS,
    FORMAT,
    SOURCE_007_CHECKPOINTS,
    pareto_crossover,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Report Experiment 025")
    result.add_argument("--results-dir", type=Path, required=True)
    return result.parse_args()


def _json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_rows(path: Path) -> list[dict[str, object]]:
    return pd.read_csv(path).to_dict(orient="records")


def _metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in (
        "shift_tokens",
        "story_ppl",
        "math_exact_answer_accuracy",
        "program_cells",
        "total_parameters",
        "active_parameter_proxy",
    ):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.sort_values("shift_tokens").drop_duplicates("shift_tokens", keep="last")


def _panel_a(results: Path, llm: pd.DataFrame, clm: pd.DataFrame) -> dict[str, object]:
    historical = pd.read_csv(ROOT / SOURCE_007_CHECKPOINTS)
    source = historical.loc[historical["model"] == "minicells-30m-v0"].copy()
    fresh_path = results / "llm" / "pretrain" / "transformer-30m-checkpoints.csv"
    if fresh_path.is_file():
        transformer = pd.read_csv(fresh_path)
    else:
        transformer = historical.loc[historical["model"] == "transformer-30m"].copy()

    source["series"] = "CLM source (TextNCA)"
    transformer["series"] = "LLM (Transformer)"
    panel = pd.concat(
        [
            transformer[["consumed_tokens", "validation_ppl", "series"]],
            source[["consumed_tokens", "validation_ppl", "series"]],
        ],
        ignore_index=True,
    ).sort_values(["series", "consumed_tokens"])
    panel.to_csv(results / "panel-a-starting-comparability.csv", index=False)

    fig, axis = plt.subplots(figsize=(8.4, 5.0))
    for label, group in panel.groupby("series"):
        ordered = group.sort_values("consumed_tokens")
        axis.plot(
            ordered["consumed_tokens"] / 1e6,
            ordered["validation_ppl"],
            marker="o",
            label=label,
        )
    axis.axvline(100, linestyle="--", linewidth=1)
    axis.text(100, axis.get_ylim()[0], "  CLM upcycle", va="bottom", fontsize=9)
    axis.set_xlabel("Story pretraining tokens (M)")
    axis.set_ylabel("TinyStories validation PPL (lower is better)")
    axis.set_title("Panel A — Starting language-model comparability")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(results / "panel-a-starting-comparability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    llm_start = llm.loc[llm["shift_tokens"] == 0].iloc[-1]
    clm_start = clm.loc[clm["shift_tokens"] == 0].iloc[-1]
    ratio = float(clm_start["story_ppl"] / llm_start["story_ppl"])
    symmetric_ratio = max(ratio, 1.0 / max(ratio, 1e-12))
    return {
        "llm_start_story_ppl": float(llm_start["story_ppl"]),
        "clm_start_story_ppl": float(clm_start["story_ppl"]),
        "clm_over_llm_story_ppl_ratio": ratio,
        "symmetric_story_ppl_ratio": symmetric_ratio,
        "within_3_percent": symmetric_ratio <= 1.03,
        "scope": "internal starting-condition check; not a public superiority claim",
    }


def _public_performance(
    results: Path,
    llm: pd.DataFrame,
    clm: pd.DataFrame,
    crossover: dict[str, object] | None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.0), sharex=True)
    for label, frame in (("LLM", llm), ("CLM", clm)):
        axes[0].plot(
            frame["shift_tokens"] / 1e6,
            frame["math_exact_answer_accuracy"] * 100,
            marker="o",
            label=label,
        )
        axes[1].plot(
            frame["shift_tokens"] / 1e6,
            frame["story_ppl"],
            marker="o",
            label=label,
        )
    axes[0].set_ylabel("Arithmetic exact-answer accuracy (%)")
    axes[0].set_title("Learning a new capability")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_ylabel("Story validation PPL (lower is better)")
    axes[1].set_xlabel("Math-heavy continual-learning tokens (M)")
    axes[1].set_title("Retaining old knowledge")
    axes[1].grid(alpha=0.25)
    if crossover is not None:
        x = float(crossover["shift_tokens"]) / 1e6
        for axis in axes:
            axis.axvline(x, linestyle="--", linewidth=1.2)
        axes[0].text(x, axes[0].get_ylim()[1], " Pareto crossover", va="top", fontsize=9)
    fig.suptitle("Experiment 025 — Fixed LLM vs Growing CLM")
    fig.tight_layout()
    fig.savefig(results / "story-math-performance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _retention_index(frame: pd.DataFrame) -> np.ndarray:
    start = float(frame.iloc[0]["story_ppl"])
    return 100.0 * start / frame["story_ppl"].to_numpy(dtype=float)


def _growth_timeline(
    results: Path,
    llm: pd.DataFrame,
    clm: pd.DataFrame,
    events: list[dict[str, object]],
    crossover: dict[str, object] | None,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 9.6), sharex=True)
    for label, frame in (("LLM", llm), ("CLM", clm)):
        x = frame["shift_tokens"] / 1e6
        axes[0].plot(x, frame["math_exact_answer_accuracy"] * 100, marker="o", label=label)
        axes[1].plot(x, _retention_index(frame), marker="o", label=label)
    axes[0].set_ylabel("Math capability (%)")
    axes[0].set_title("New capability learned")
    axes[0].legend()
    axes[1].axhline(100.0, linewidth=1, linestyle="--")
    axes[1].set_ylabel("Story retention index")
    axes[1].set_title("Old knowledge retained (100 = shift start)")

    x = clm["shift_tokens"].to_numpy(dtype=float) / 1e6
    cells = clm["program_cells"].to_numpy(dtype=float)
    axes[2].step(x, cells, where="post", linewidth=2, label="CLM program cells")
    axes[2].scatter(x, cells, s=28)
    axes[2].set_ylabel("Persistent program cells")
    axes[2].set_xlabel("Math-heavy continual-learning tokens (M)")
    axes[2].set_title("The model grows")

    for event in events:
        if event.get("type") != "growth_decision":
            continue
        decision_x = float(event["decision_shift_tokens"]) / 1e6
        outcome = str(event.get("outcome", ""))
        axes[2].axvline(decision_x, alpha=0.3, linewidth=1)
        axes[2].text(
            decision_x,
            axes[2].get_ylim()[0],
            f" proposal\n{outcome.lower()}",
            va="bottom",
            fontsize=8,
        )
        if outcome == "PROMOTE" and event.get("probation_end_shift_tokens") is not None:
            birth_x = float(event["probation_end_shift_tokens"]) / 1e6
            axes[2].axvline(birth_x, linestyle="--", linewidth=1.2)
            child = event.get("birth", {}).get("child", "new cell")
            axes[2].text(birth_x, axes[2].get_ylim()[1], f" {child} promoted", va="top", fontsize=8)

    if crossover is not None:
        cross_x = float(crossover["shift_tokens"]) / 1e6
        for axis in axes:
            axis.axvline(cross_x, linestyle=":", linewidth=1.5)
        axes[0].text(cross_x, axes[0].get_ylim()[1], " crossover", va="top", fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.22)
    fig.suptitle("MiniCells — learning math without rewriting its whole history")
    fig.tight_layout()
    fig.savefig(results / "growth-timeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _growth_animation(
    results: Path,
    llm: pd.DataFrame,
    clm: pd.DataFrame,
    events: list[dict[str, object]],
    crossover: dict[str, object] | None,
) -> str | None:
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception:
        return None

    shared = sorted(set(llm["shift_tokens"].astype(int)) & set(clm["shift_tokens"].astype(int)))
    if not shared:
        return None
    llm = llm.set_index(llm["shift_tokens"].astype(int), drop=False)
    clm = clm.set_index(clm["shift_tokens"].astype(int), drop=False)
    llm_start = float(llm.loc[shared[0], "story_ppl"])
    clm_start = float(clm.loc[shared[0], "story_ppl"])

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.4), sharex=True)
    max_x = max(shared) / 1e6
    max_math = max(
        float(llm["math_exact_answer_accuracy"].max()),
        float(clm["math_exact_answer_accuracy"].max()),
        0.01,
    ) * 100
    min_ret = min(
        100 * llm_start / float(llm["story_ppl"].max()),
        100 * clm_start / float(clm["story_ppl"].max()),
        100,
    )
    max_cells = max(float(clm["program_cells"].max()), 12.0)

    def draw(frame_index: int) -> None:
        for axis in axes:
            axis.clear()
            axis.grid(alpha=0.20)
        tokens = shared[: frame_index + 1]
        x = np.asarray(tokens, dtype=float) / 1e6
        llm_math = np.asarray([float(llm.loc[token, "math_exact_answer_accuracy"]) for token in tokens]) * 100
        clm_math = np.asarray([float(clm.loc[token, "math_exact_answer_accuracy"]) for token in tokens]) * 100
        llm_ret = np.asarray([100 * llm_start / float(llm.loc[token, "story_ppl"]) for token in tokens])
        clm_ret = np.asarray([100 * clm_start / float(clm.loc[token, "story_ppl"]) for token in tokens])
        cells = np.asarray([float(clm.loc[token, "program_cells"]) for token in tokens])

        axes[0].plot(x, llm_math, marker="o", label="LLM")
        axes[0].plot(x, clm_math, marker="o", label="CLM")
        axes[0].set_ylabel("Math (%)")
        axes[0].set_ylim(0, max_math * 1.12 + 1)
        axes[0].legend(loc="upper left")
        axes[1].plot(x, llm_ret, marker="o", label="LLM")
        axes[1].plot(x, clm_ret, marker="o", label="CLM")
        axes[1].axhline(100, linestyle="--", linewidth=1)
        axes[1].set_ylabel("Story retention")
        axes[1].set_ylim(max(0, min_ret - 8), max(108, float(max(llm_ret.max(), clm_ret.max())) + 5))
        axes[2].step(x, cells, where="post", linewidth=2)
        axes[2].scatter(x, cells, s=25)
        axes[2].set_ylabel("CLM cells")
        axes[2].set_xlabel("Math-heavy tokens (M)")
        axes[2].set_ylim(11.5, max_cells + 1.5)
        axes[2].set_xlim(0, max_x)

        current = tokens[-1]
        for event in events:
            if event.get("type") != "growth_decision":
                continue
            decision = int(event["decision_shift_tokens"])
            if decision <= current:
                axes[2].axvline(decision / 1e6, alpha=0.25, linewidth=1)
            if event.get("outcome") == "PROMOTE":
                promoted = int(event.get("probation_end_shift_tokens", decision))
                if promoted <= current:
                    axes[2].axvline(promoted / 1e6, linestyle="--", linewidth=1.2)
        if crossover is not None and int(crossover["shift_tokens"]) <= current:
            for axis in axes:
                axis.axvline(float(crossover["shift_tokens"]) / 1e6, linestyle=":", linewidth=1.2)
        fig.suptitle(f"MiniCells grows with experience — {current/1e6:.1f}M shift tokens")
        fig.tight_layout()

    animation = FuncAnimation(fig, draw, frames=len(shared), interval=700, repeat=True)
    destination = results / "growth-animation.gif"
    try:
        animation.save(destination, writer=PillowWriter(fps=2))
    except Exception:
        plt.close(fig)
        return None
    plt.close(fig)
    return str(destination)


def main() -> int:
    args = parser().parse_args()
    results = args.results_dir
    llm_summary = json.loads((results / "llm" / "worker-summary.json").read_text(encoding="utf-8"))
    clm_summary = json.loads((results / "clm" / "worker-summary.json").read_text(encoding="utf-8"))
    if not bool(llm_summary.get("complete")) or not bool(clm_summary.get("complete")):
        raise RuntimeError("cannot report Experiment 025 until both arms are complete")

    llm = _metric_rows(pd.read_csv(results / "llm" / "metrics.csv"))
    clm = _metric_rows(pd.read_csv(results / "clm" / "metrics.csv"))
    llm_rows = llm.to_dict(orient="records")
    clm_rows = clm.to_dict(orient="records")
    crossover = pareto_crossover(llm_rows, clm_rows)
    panel_a = _panel_a(results, llm, clm)

    shared = sorted(set(llm["shift_tokens"].astype(int)) & set(clm["shift_tokens"].astype(int)))
    comparison_rows = []
    for tokens in shared:
        left = llm.loc[llm["shift_tokens"].astype(int) == tokens].iloc[-1]
        right = clm.loc[clm["shift_tokens"].astype(int) == tokens].iloc[-1]
        comparison_rows.append(
            {
                "shift_tokens": tokens,
                "total_experience_tokens": EXPECTED_SOURCE_TOKENS + tokens,
                "llm_story_ppl": float(left["story_ppl"]),
                "clm_story_ppl": float(right["story_ppl"]),
                "story_ppl_clm_over_llm": float(right["story_ppl"] / left["story_ppl"]),
                "llm_math_exact_answer_accuracy": float(left["math_exact_answer_accuracy"]),
                "clm_math_exact_answer_accuracy": float(right["math_exact_answer_accuracy"]),
                "math_accuracy_delta_clm_minus_llm": float(
                    right["math_exact_answer_accuracy"] - left["math_exact_answer_accuracy"]
                ),
                "clm_program_cells": int(right["program_cells"]),
                "clm_total_parameters": int(right["total_parameters"]),
                "clm_active_parameter_proxy": int(right["active_parameter_proxy"]),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(results / "llm-vs-clm-trajectory.csv", index=False)

    clm_events = json.loads((results / "clm" / "events.json").read_text(encoding="utf-8"))
    _public_performance(results, llm, clm, crossover)
    _growth_timeline(results, llm, clm, clm_events, crossover)
    animation = _growth_animation(results, llm, clm, clm_events, crossover)

    promotions = [event for event in clm_events if event.get("outcome") == "PROMOTE"]
    rejections = [event for event in clm_events if event.get("outcome") == "REJECT"]
    final_llm = llm.iloc[-1]
    final_clm = clm.iloc[-1]
    if crossover is not None:
        status = "CLM_30M_ADAPTATION_RETENTION_PARETO_CROSSOVER"
    elif promotions:
        status = "CLM_30M_GROWTH_WITHOUT_PARETO_CROSSOVER"
    else:
        status = "CLM_30M_NO_PERSISTENT_GROWTH_TRIGGERED"

    decision = {
        "format": FORMAT,
        "experiment": "Experiment 025 — 30M Story→Math Developmental Shift",
        "status": status,
        "public_comparison": ["30M fixed Transformer LLM", "30M-source growing CLM"],
        "panel_a": panel_a,
        "pareto_crossover": crossover,
        "growth": {
            "initial_program_cells": int(clm.iloc[0]["program_cells"]),
            "final_program_cells": int(final_clm["program_cells"]),
            "promotions": len(promotions),
            "rejections": len(rejections),
            "events": [
                {
                    "decision_shift_tokens": event.get("decision_shift_tokens"),
                    "probation_end_shift_tokens": event.get("probation_end_shift_tokens"),
                    "outcome": event.get("outcome"),
                    "parent": event.get("birth", {}).get("parent"),
                    "child": event.get("birth", {}).get("child"),
                }
                for event in clm_events
                if event.get("type") == "growth_decision"
            ],
        },
        "final": {
            "llm_story_ppl": float(final_llm["story_ppl"]),
            "clm_story_ppl": float(final_clm["story_ppl"]),
            "llm_math_exact_answer_accuracy": float(final_llm["math_exact_answer_accuracy"]),
            "clm_math_exact_answer_accuracy": float(final_clm["math_exact_answer_accuracy"]),
            "clm_story_ppl_ratio_vs_llm": float(final_clm["story_ppl"] / final_llm["story_ppl"]),
            "clm_math_accuracy_delta_vs_llm": float(
                final_clm["math_exact_answer_accuracy"] - final_llm["math_exact_answer_accuracy"]
            ),
        },
        "cost": {
            "llm_elapsed_seconds": float(llm_summary["elapsed_seconds"]),
            "clm_elapsed_seconds": float(clm_summary["elapsed_seconds"]),
            "llm_physical_training_tokens": int(llm_summary["physical_training_tokens"]),
            "clm_physical_training_tokens": int(clm_summary["physical_training_tokens"]),
            "llm_peak_vram_bytes": int(llm_summary["peak_vram_bytes"]),
            "clm_peak_vram_bytes": int(clm_summary["peak_vram_bytes"]),
        },
        "interpretation_rules": {
            "crossover": (
                "first shared checkpoint where CLM arithmetic exact-answer accuracy >= LLM "
                "and CLM story PPL <= LLM, with at least one strict inequality"
            ),
            "math_scope": "synthetic integer arithmetic, not general mathematical reasoning",
            "growth_claim": (
                "persistent growth is claimed only for a promoted shadow birth; "
                "proposal alone is not birth"
            ),
        },
        "figures": {
            "panel_a": "panel-a-starting-comparability.png",
            "performance": "story-math-performance.png",
            "growth_timeline": "growth-timeline.png",
            "growth_animation": animation,
        },
    }
    _json_write(results / "decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
