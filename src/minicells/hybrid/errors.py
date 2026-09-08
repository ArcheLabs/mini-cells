"""Errors raised by the public HybridCLM API.

The public layer is deliberately fail-closed.  A model or mutation that does
not match a verified backend is rejected instead of being modified by a
best-effort heuristic.
"""

from __future__ import annotations


class HybridCLMError(RuntimeError):
    """Base class for HybridCLM lifecycle and validation failures."""


class UnsupportedArchitectureError(HybridCLMError):
    """The model architecture has no tested HybridCLM backend."""


class PlacementError(HybridCLMError, ValueError):
    """A placement is malformed or does not exist in the inspected model."""


class ArtifactValidationError(HybridCLMError, ValueError):
    """A mutation artifact failed schema, hash, or compatibility validation."""


class MutationLifecycleError(HybridCLMError):
    """An attach/detach/scale operation violates the mutation lifecycle."""

