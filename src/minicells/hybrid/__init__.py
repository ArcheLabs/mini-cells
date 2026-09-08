"""Public HybridCLM toolkit (v0.1).

Only the Granite MoE backend is supported in this release.  Research runners
remain under ``pcu_kill_001`` and are not imported by the public API.
"""

from .backends.base import HybridBackend
from .backends.granite_moe import GraniteMoEBackend
from .backends.registry import BackendRegistry
from .errors import (
    ArtifactValidationError,
    HybridCLMError,
    MutationLifecycleError,
    PlacementError,
    UnsupportedArchitectureError,
)
from .inspector import CellPlacement, ModelInspection
from .model import HybridCLM
from .mutation import CellMutation
from .validation import RestorationReport, ZeroStateReport

BackendRegistry.register(GraniteMoEBackend)

__all__ = [
    "ArtifactValidationError",
    "BackendRegistry",
    "CellMutation",
    "CellPlacement",
    "GraniteMoEBackend",
    "HybridBackend",
    "HybridCLM",
    "HybridCLMError",
    "ModelInspection",
    "MutationLifecycleError",
    "PlacementError",
    "RestorationReport",
    "UnsupportedArchitectureError",
    "ZeroStateReport",
]

