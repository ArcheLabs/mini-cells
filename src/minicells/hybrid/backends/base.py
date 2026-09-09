"""Backend contract for model-family-specific HybridCLM operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from ..inspector import CellPlacement, ModelInspection


class HybridBackend(ABC):
    """A tested adapter between the generic API and one model family."""

    name = "abstract"

    @abstractmethod
    def inspect(self, model: Any) -> ModelInspection:
        raise NotImplementedError

    @abstractmethod
    def resolve_placement(self, model: Any, placement: CellPlacement) -> CellPlacement:
        raise NotImplementedError

    @abstractmethod
    def cellularize(self, model: Any, placements: Sequence[CellPlacement]) -> Any:
        raise NotImplementedError

    def attach_mutation(self, model: Any, mutation: Any, alpha: float = 1.0) -> None:
        """Optional hook for backends with non-parameter mutation storage."""
        del model, mutation, alpha

    def detach_mutation(self, model: Any, mutation: Any, alpha: float = 1.0) -> None:
        del model, mutation, alpha

    def set_scale(self, model: Any, mutation: Any, alpha: float) -> None:
        del model, mutation, alpha

    def verify_zero_state(self, model: Any, placement: CellPlacement | None = None) -> Any:
        del model, placement
        return None
