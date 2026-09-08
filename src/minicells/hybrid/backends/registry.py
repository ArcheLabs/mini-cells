"""Fail-closed registry for HybridCLM model backends."""

from __future__ import annotations

from typing import Any, Type

from ..errors import UnsupportedArchitectureError
from .base import HybridBackend


class BackendRegistry:
    _backend_types: list[type[HybridBackend]] = []

    @classmethod
    def register(cls, backend: type[HybridBackend] | HybridBackend) -> type[HybridBackend] | HybridBackend:
        value = backend if isinstance(backend, type) else type(backend)
        if not issubclass(value, HybridBackend):
            raise TypeError("backend must implement HybridBackend")
        if value not in cls._backend_types:
            cls._backend_types.append(value)
        return backend

    @classmethod
    def resolve(cls, model: Any) -> HybridBackend:
        for backend_type in cls._backend_types:
            backend = backend_type()
            try:
                backend.inspect(model)
            except UnsupportedArchitectureError:
                continue
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue
            return backend
        architecture = getattr(getattr(model, "config", None), "model_type", type(model).__name__)
        raise UnsupportedArchitectureError(f"no tested HybridCLM backend for architecture {architecture!r}")

    @classmethod
    def registered(cls) -> tuple[type[HybridBackend], ...]:
        return tuple(cls._backend_types)

