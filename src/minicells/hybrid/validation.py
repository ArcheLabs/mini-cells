"""Engineering-only numerical and rollback reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _logits(output: Any) -> torch.Tensor:
    try:
        value = output.logits
    except AttributeError:
        value = output
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError("model output does not contain a tensor")
    return value


@dataclass(frozen=True)
class ZeroStateReport:
    native_g0_max_abs: float
    matched_graph_max_abs: float
    matched_graph_mean_abs: float
    matched_graph_pass: bool
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.matched_graph_pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "native_g0_max_abs": self.native_g0_max_abs,
            "matched_graph_max_abs": self.matched_graph_max_abs,
            "matched_graph_mean_abs": self.matched_graph_mean_abs,
            "matched_graph_pass": self.matched_graph_pass,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class RestorationReport:
    max_abs: float
    mean_abs: float
    passed: bool
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "passed": self.passed,
            "tolerance": self.tolerance,
        }


@torch.no_grad()
def compare_model_outputs(native: Any, cellular: Any, inputs: dict[str, Any], tolerance: float) -> ZeroStateReport:
    native_output = _logits(native(**inputs))
    cellular_output = _logits(cellular(**inputs))
    difference = cellular_output.float() - native_output.float()
    max_abs = float(difference.abs().max().item()) if difference.numel() else 0.0
    mean_abs = float(difference.abs().mean().item()) if difference.numel() else 0.0
    return ZeroStateReport(
        native_g0_max_abs=max_abs,
        matched_graph_max_abs=max_abs,
        matched_graph_mean_abs=mean_abs,
        matched_graph_pass=max_abs <= tolerance,
        tolerance=tolerance,
    )
