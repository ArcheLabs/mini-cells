"""Lightweight expert overlays for resource-bounded Granite evaluation.

Evaluation states differ only in the final MoE expert runtime.  Cloning the
entire 1.3B foundation for MA/MB/MAB/LA/LB/LAB is unnecessary and can exceed a
single-GPU memory budget.  This proxy swaps the expert module for one forward,
then restores the exact parent module in ``finally``.
"""

from __future__ import annotations

from typing import Any

from .model import target_module


class ExpertsOverlayModel:
    """Callable model view that overlays one expert runtime without copying F."""

    def __init__(self, model: Any, target_path: str, experts: Any) -> None:
        self.model = model
        self.target_path = str(target_path)
        self.experts = experts

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        target = target_module(self.model, self.target_path)
        parent_experts = target.experts
        target.experts = self.experts
        try:
            return self.model(*args, **kwargs)
        finally:
            target.experts = parent_experts

    def eval(self) -> "ExpertsOverlayModel":
        self.model.eval()
        if hasattr(self.experts, "eval"):
            self.experts.eval()
        return self


def model_with_experts_overlay(model: Any, inspector: Any, experts: Any) -> ExpertsOverlayModel:
    """Drop-in replacement for the historical full-model deepcopy helper."""
    return ExpertsOverlayModel(model, inspector.target_path, experts).eval()
