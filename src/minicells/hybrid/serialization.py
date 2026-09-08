"""Versioned, hash-checked HybridCLM mutation artifact IO."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .errors import ArtifactValidationError

SCHEMA_VERSION = 1
TENSOR_FILE = "mutation.safetensors"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_artifact(
    directory: str | Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    cell_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> Path:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ArtifactValidationError("safetensors is required for public mutation artifacts") from exc
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not tensors:
        raise ArtifactValidationError("mutation artifact must contain at least one tensor")
    serialized = {str(key): value.detach().cpu().contiguous() for key, value in tensors.items()}
    tensor_path = root / TENSOR_FILE
    save_file(serialized, str(tensor_path))
    records = [
        {
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", "").upper(),
        }
        for key, value in sorted(serialized.items())
    ]
    complete_manifest = {
        "schema_version": SCHEMA_VERSION,
        "base_model": manifest.get("base_model"),
        "base_revision": manifest.get("base_revision"),
        "architecture": manifest.get("architecture", "granitemoe"),
        "architecture_hash": manifest.get("architecture_hash", ""),
        "tensor_manifest": records,
        "dtype": manifest.get("dtype", "F32"),
        "mutation_sha256": sha256_file(tensor_path),
    }
    if not complete_manifest["base_model"] or not complete_manifest["base_revision"]:
        raise ArtifactValidationError("base_model and immutable base_revision are required")
    complete_config = {
        "schema_version": SCHEMA_VERSION,
        "mutation_type": cell_config.get("mutation_type", "expert_delta"),
        "backend": cell_config.get("backend", "granite_moe"),
        "placements": list(cell_config.get("placements", [])),
        "default_alpha": float(cell_config.get("default_alpha", 1.0)),
        "targets": list(cell_config.get("targets", [])),
    }
    _write_json(root / "cell_config.json", complete_config)
    _write_json(root / "manifest.json", complete_manifest)
    _write_json(root / "provenance.json", dict(provenance))
    _write_json(root / "evaluation.json", dict(evaluation))
    return root


def load_artifact(directory: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ArtifactValidationError("safetensors is required for public mutation artifacts") from exc
    root = Path(directory).expanduser().resolve()
    required = ("cell_config.json", "manifest.json", "provenance.json", "evaluation.json", TENSOR_FILE)
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ArtifactValidationError(f"mutation artifact is missing: {', '.join(missing)}")
    try:
        config = json.loads((root / "cell_config.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("mutation artifact contains malformed JSON") from exc
    if config.get("schema_version") != SCHEMA_VERSION or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactValidationError("unsupported mutation artifact schema version")
    for key in ("base_model", "base_revision", "architecture", "architecture_hash", "tensor_manifest", "dtype", "mutation_sha256"):
        if key not in manifest:
            raise ArtifactValidationError(f"manifest is missing {key}")
    tensor_path = root / TENSOR_FILE
    if sha256_file(tensor_path) != manifest["mutation_sha256"]:
        raise ArtifactValidationError("mutation.safetensors SHA256 mismatch")
    tensors = dict(load_file(str(tensor_path), device="cpu"))
    expected = {record.get("key") for record in manifest["tensor_manifest"]}
    if set(tensors) != expected:
        raise ArtifactValidationError("tensor manifest does not match mutation.safetensors")
    for record in manifest["tensor_manifest"]:
        tensor = tensors[record["key"]]
        if list(tensor.shape) != list(record.get("shape", [])):
            raise ArtifactValidationError(f"tensor shape mismatch for {record['key']}")
    return config, manifest, tensors, provenance, evaluation

