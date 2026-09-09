"""Public identity helpers for HybridCLM model and mutation manifests."""

from __future__ import annotations

from typing import Any

from .inspector import ModelInspection, architecture_signature


def model_manifest(model: Any, inspection: ModelInspection | None = None, *, model_id: str | None = None, revision: str | None = None) -> dict[str, Any]:
    inspection = inspection or __import__("minicells.hybrid", fromlist=["HybridCLM"]).HybridCLM.inspect(model)
    try:
        model_name = model.name_or_path
    except AttributeError:
        model_name = None
    try:
        config = model.config
    except AttributeError:
        config = None
    if model_name is None and config is not None:
        try:
            model_name = config._name_or_path
        except AttributeError:
            model_name = None
    try:
        model_revision = config._commit_hash if config is not None else None
    except AttributeError:
        model_revision = None
    return {
        "schema_version": 1,
        "base_model": model_id or model_name,
        "base_revision": revision or model_revision,
        "architecture": inspection.architecture,
        "architecture_hash": inspection.architecture_signature or architecture_signature(model),
        "inspection": inspection.to_dict(),
    }


__all__ = ["model_manifest"]
