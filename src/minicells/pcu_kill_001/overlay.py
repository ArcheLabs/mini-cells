"""Lightweight expert overlays for resource-bounded Granite evaluation.

Evaluation states differ only in the final MoE expert runtime. Cloning the
entire 1.3B foundation for MA/MB/MAB/LA/LB/LAB is unnecessary and can exceed a
single-GPU memory budget. This proxy swaps the expert module for one complete
forward or generation call, then restores the exact resident module in
``finally``.
"""

from __future__ import annotations

from typing import Any

from .model import target_module


class ExpertsOverlayModel:
    """Model view that overlays one expert runtime without copying F."""

    def __init__(self, model: Any, target_path: str, experts: Any) -> None:
        self.model = model
        self.target_path = str(target_path)
        self.experts = experts

    def _swap(self) -> tuple[Any, Any]:
        target = target_module(self.model, self.target_path)
        parent_experts = target.experts
        target.experts = self.experts
        return target, parent_experts

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        target, parent_experts = self._swap()
        try:
            return self.model(*args, **kwargs)
        finally:
            target.experts = parent_experts

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Run HF generation under the same temporary expert overlay.

        Keeping the overlay installed for the whole ``generate`` call is
        essential: Transformers may execute many cached decoding forwards
        internally. Restoring after each individual forward would make a
        branch generation silently fall back to the resident foundation.
        """
        generate = getattr(self.model, "generate", None)
        if not callable(generate):
            raise AttributeError("underlying model has no generate method")
        target, parent_experts = self._swap()
        try:
            return generate(*args, **kwargs)
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
