from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class AblationSpec:
    name: str
    rms_norm: bool
    carry_bias: bool
    auxiliary_loss: bool

    @property
    def carry_bias_value(self) -> float:
        return 2.0 if self.carry_bias else 0.0

    @property
    def auxiliary_weights(self) -> tuple[float, float] | None:
        return (0.1, 0.2) if self.auxiliary_loss else None

    def factor_codes(self) -> dict[str, int]:
        return {
            "R": 1 if self.rms_norm else -1,
            "C": 1 if self.carry_bias else -1,
            "A": 1 if self.auxiliary_loss else -1,
        }


FACTORIAL_SPECS: tuple[AblationSpec, ...] = (
    AblationSpec("ln-c0-a0", False, False, False),
    AblationSpec("rms-c0-a0", True, False, False),
    AblationSpec("ln-c2-a0", False, True, False),
    AblationSpec("ln-c0-aux", False, False, True),
    AblationSpec("rms-c2-a0", True, True, False),
    AblationSpec("rms-c0-aux", True, False, True),
    AblationSpec("ln-c2-aux", False, True, True),
    AblationSpec("rms-c2-aux", True, True, True),
)

FACTOR_LABELS = {
    "R": "RMSNorm",
    "C": "Carry bias +2",
    "A": "Aux stage loss",
    "RC": "RMSNorm × Carry",
    "RA": "RMSNorm × Aux",
    "CA": "Carry × Aux",
    "RCA": "RMSNorm × Carry × Aux",
}


def validate_factorial_specs(specs: tuple[AblationSpec, ...] = FACTORIAL_SPECS) -> None:
    combinations = {(item.rms_norm, item.carry_bias, item.auxiliary_loss) for item in specs}
    expected = set(itertools.product((False, True), repeat=3))
    if len(specs) != 8 or combinations != expected:
        raise ValueError("Experiment 005B requires the complete 2^3 factorial design")
    if len({item.name for item in specs}) != 8:
        raise ValueError("Experiment 005B variant names must be unique")


def add_factor_codes(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"rms_norm", "carry_bias", "auxiliary_loss"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing factorial columns: {sorted(missing)}")
    coded = frame.copy()
    coded["R"] = coded["rms_norm"].map(lambda value: 1 if bool(value) else -1)
    coded["C"] = coded["carry_bias"].map(lambda value: 1 if bool(value) else -1)
    coded["A"] = coded["auxiliary_loss"].map(lambda value: 1 if bool(value) else -1)
    return coded


def factorial_effects(
    frame: pd.DataFrame,
    *,
    response: str = "validation_nll",
) -> pd.DataFrame:
    """Return standard 2^3 factorial effects using +/-1 coding.

    Effects are computed on validation NLL because NLL is additive. A negative
    effect is beneficial. ``ppl_multiplier`` translates the NLL effect back to
    the perplexity scale: values below 1 improve perplexity.
    """

    if len(frame) != 8:
        raise ValueError(f"expected 8 factorial rows, got {len(frame)}")
    coded = add_factor_codes(frame)
    if response not in coded:
        raise ValueError(f"missing response column {response!r}")
    if coded[["R", "C", "A"]].drop_duplicates().shape[0] != 8:
        raise ValueError("factorial response does not contain all 8 factor combinations")

    rows: list[dict[str, object]] = []
    for term in ("R", "C", "A", "RC", "RA", "CA", "RCA"):
        signs = coded[term[0]].astype(float)
        for factor in term[1:]:
            signs = signs * coded[factor].astype(float)
        # With balanced +/-1 coding, beta = mean(y*x); factorial effect = 2 beta.
        effect = 2.0 * float((coded[response].astype(float) * signs).mean())
        multiplier = math.exp(effect)
        rows.append(
            {
                "term": term,
                "label": FACTOR_LABELS[term],
                "order": len(term),
                "effect_nll": effect,
                "abs_effect_nll": abs(effect),
                "ppl_multiplier": multiplier,
                "ppl_percent_effect": 100.0 * (multiplier - 1.0),
                "direction": "improves" if effect < 0 else "worsens",
            }
        )
    return pd.DataFrame(rows).sort_values(["order", "term"]).reset_index(drop=True)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_factorial_ppl(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values("validation_ppl")
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(ordered["name"], ordered["validation_ppl"])
    axis.set_ylabel("Validation perplexity @ 500K")
    axis.set_title("Experiment 005B — 2³ optimization ablation")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    _save(fig, path)


def save_factorial_learning_curves(checkpoints: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 6))
    for name, group in checkpoints.groupby("model"):
        group = group.sort_values("consumed_tokens")
        axis.plot(group["consumed_tokens"], group["validation_ppl"], marker="o", label=name)
    axis.set_xlabel("Consumed training tokens")
    axis.set_ylabel("Validation perplexity")
    axis.set_title("Experiment 005B — learning trajectories")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    _save(fig, path)


def save_effects(effects: pd.DataFrame, path: Path, *, order: int, title: str) -> None:
    subset = effects[effects["order"] == order].copy().sort_values("effect_nll")
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(subset["label"], subset["effect_nll"])
    axis.axhline(0.0, linewidth=1)
    axis.set_ylabel("Factorial effect on validation NLL\n(negative = better)")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    _save(fig, path)


def save_replication_comparison(
    baseline_005: dict[str, float],
    frame: pd.DataFrame,
    path: Path,
) -> None:
    by_name = frame.set_index("name")
    rows = pd.DataFrame(
        [
            {
                "condition": "TextNCA baseline",
                "005": baseline_005["textnca_ppl"],
                "005B": float(by_name.loc["ln-c0-a0", "validation_ppl"]),
            },
            {
                "condition": "MiniTextNCA S+",
                "005": baseline_005["minitextnca_plus_ppl"],
                "005B": float(by_name.loc["rms-c2-aux", "validation_ppl"]),
            },
        ]
    )
    x = range(len(rows))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar([value - width / 2 for value in x], rows["005"], width=width, label="005")
    axis.bar([value + width / 2 for value in x], rows["005B"], width=width, label="005B")
    axis.set_xticks(list(x), rows["condition"])
    axis.set_ylabel("Validation perplexity @ 500K")
    axis.set_title("005 replication check inside factorial design")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save(fig, path)
