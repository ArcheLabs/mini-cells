from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class DepthAblationVariant:
    code: str
    random_depth: bool
    step_embedding_init_scale: float
    stability_weight: float

    @property
    def low_step_init(self) -> bool:
        return self.step_embedding_init_scale < 1.0

    @property
    def uses_stability_loss(self) -> bool:
        return self.stability_weight > 0.0


# Full 2 x 2 x 2 factorial. A and D reproduce the Experiment 011 recipes.
VARIANTS: tuple[DepthAblationVariant, ...] = (
    DepthAblationVariant("A", False, 1.00, 0.00),
    DepthAblationVariant("B", True, 1.00, 0.00),
    DepthAblationVariant("C", True, 0.25, 0.00),
    DepthAblationVariant("D", True, 0.25, 0.10),
    DepthAblationVariant("E", False, 0.25, 0.00),
    DepthAblationVariant("F", False, 1.00, 0.10),
    DepthAblationVariant("G", False, 0.25, 0.10),
    DepthAblationVariant("H", True, 1.00, 0.10),
)
VARIANT_BY_CODE = {variant.code: variant for variant in VARIANTS}


def variant_by_code(code: str) -> DepthAblationVariant:
    try:
        return VARIANT_BY_CODE[code.upper()]
    except KeyError as exc:
        raise ValueError(f"unknown depth ablation variant: {code}") from exc


def resolve_stage_depths(
    variant: DepthAblationVariant,
    scheduled_depths: tuple[int, int, int],
) -> tuple[int, int, int]:
    if len(scheduled_depths) != 3:
        raise ValueError("scheduled_depths must contain three stages")
    if variant.random_depth:
        if any(depth < 2 or depth > 4 for depth in scheduled_depths):
            raise ValueError("random-depth schedule must stay within [2, 4]")
        return tuple(int(depth) for depth in scheduled_depths)
    return (4, 4, 4)


def step_embedding_rms(model: torch.nn.Module) -> float:
    stages = getattr(model, "stages", None)
    if stages is None:
        raise TypeError("model has no recurrent stages")
    values = []
    for stage in stages:
        embedding = getattr(stage, "step_embedding", None)
        if embedding is None:
            raise TypeError("stage has no step_embedding")
        values.append(embedding.detach().float().reshape(-1))
    flat = torch.cat(values)
    return float(flat.square().mean().sqrt().cpu())


def factorial_contrast(
    values: dict[str, float],
    factors: tuple[str, ...],
) -> float:
    """Return a balanced high-minus-low contrast for the 2^3 design.

    Valid factors are ``random_depth``, ``low_step_init``, and
    ``stability_loss``. For interactions, pass multiple factor names. Callers
    normally use log metrics, making ``exp(contrast)`` the geometric ratio.
    """

    allowed = {"random_depth", "low_step_init", "stability_loss"}
    if not factors or any(factor not in allowed for factor in factors):
        raise ValueError("invalid factorial factors")
    if set(values) != set(VARIANT_BY_CODE):
        raise ValueError("values must contain all eight variants")

    signed = 0.0
    for variant in VARIANTS:
        sign = 1.0
        for factor in factors:
            level = {
                "random_depth": variant.random_depth,
                "low_step_init": variant.low_step_init,
                "stability_loss": variant.uses_stability_loss,
            }[factor]
            sign *= 1.0 if level else -1.0
        signed += sign * float(values[variant.code])
    return signed / 4.0


def geometric_ratio_from_log_contrast(contrast: float) -> float:
    return math.exp(float(contrast))
