"""Public identity helpers for HybridCLM model and mutation manifests."""

from __future__ import annotations

from typing import Any

from .inspector import ModelInspection, architecture_signature


def model_manifest(model: Any, inspection: ModelInspection | None = None, *, model_id: str | None = None, revision: str | None = None) -> dict[str, Any]:
    inspection = inspection or __import__("minicells.hybrid", fromlist=["HybridCLM"]).HybridCLM.inspect(model)
    return {
        "schema_version": 1,
        "base_model": model_id or getattr(model, "name_or_path", None) or getattr(getattr(model, "config", None), "_name_or_path", None),
        "base_revision": revision or getattr(getattr(model, "config", None), "_commit_hash", None),
        "architecture": inspection.architecture,
        "architecture_hash": inspection.architecture_signature or architecture_signature(model),
        "inspection": inspection.to_dict(),
    }


__all__ = ["model_manifest"]

