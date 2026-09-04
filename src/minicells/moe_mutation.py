from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

MUTATION_SCHEMA_VERSION = "clm.moe-mutation.v1"
MUTATION_MANIFEST_NAME = "mutation.json"
MUTATION_TENSORS_NAME = "mutation.safetensors"


class MoeMutationError(RuntimeError):
    """Raised when a CLM MoE mutation artifact violates its frozen contract."""


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def expert_address(tensor_name: str, expert_index: int) -> str:
    return f"{tensor_name}::expert[{int(expert_index)}]"


def capture_expert_slices(
    parameters: Mapping[str, torch.nn.Parameter],
    tensor_names: list[str] | tuple[str, ...],
    expert_index: int,
) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    for name in tensor_names:
        if name not in parameters:
            raise MoeMutationError(f"model is missing target parameter: {name}")
        parameter = parameters[name]
        if parameter.ndim < 1 or not 0 <= expert_index < parameter.shape[0]:
            raise MoeMutationError(f"invalid expert index {expert_index} for {name}")
        captured[name] = parameter.detach()[expert_index].float().cpu().clone()
    return captured


def restore_expert_slices_(
    parameters: Mapping[str, torch.nn.Parameter],
    slices: Mapping[str, torch.Tensor],
    expert_index: int,
) -> None:
    with torch.no_grad():
        for name, value in slices.items():
            if name not in parameters:
                raise MoeMutationError(f"model is missing target parameter: {name}")
            parameter = parameters[name]
            target = parameter[expert_index]
            if list(target.shape) != list(value.shape):
                raise MoeMutationError(
                    f"slice shape mismatch for {name}: model={list(target.shape)} artifact={list(value.shape)}"
                )
            target.copy_(value.to(device=target.device, dtype=target.dtype))


def save_expert_slice_mutation(
    output_dir: str | Path,
    *,
    base_manifest_identity: str,
    source_model_id: str,
    source_revision: str,
    layer_index: int,
    expert_index: int,
    deltas: Mapping[str, torch.Tensor],
    canonical_tensor_names: Mapping[str, str],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize sparse expert-slice deltas without copying the canonical MoE substrate."""
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise MoeMutationError("safetensors is required for MoE mutation artifacts") from exc

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not deltas:
        raise MoeMutationError("mutation must contain at least one delta tensor")
    if set(deltas) != set(canonical_tensor_names):
        raise MoeMutationError("runtime/canonical tensor mapping must cover every delta exactly")

    serialized: dict[str, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    for index, runtime_name in enumerate(sorted(deltas)):
        delta = deltas[runtime_name].detach().float().cpu().contiguous()
        if delta.ndim < 1:
            raise MoeMutationError(f"delta must be an expert slice tensor: {runtime_name}")
        key = f"delta_{index:02d}"
        serialized[key] = delta
        canonical_name = canonical_tensor_names[runtime_name]
        records.append(
            {
                "key": key,
                "runtime_tensor": runtime_name,
                "canonical_tensor": canonical_name,
                "address": expert_address(canonical_name, expert_index),
                "shape": list(delta.shape),
                "dtype": "F32",
                "l2_norm": float(torch.linalg.vector_norm(delta).item()),
                "max_abs": float(delta.abs().max().item()),
            }
        )

    tensor_path = destination / MUTATION_TENSORS_NAME
    save_file(serialized, str(tensor_path))
    unsigned = {
        "schema_version": MUTATION_SCHEMA_VERSION,
        "kind": "expert_slice_delta",
        "base": {
            "manifest_identity_sha256": base_manifest_identity,
            "model_id": source_model_id,
            "revision": source_revision,
        },
        "target": {
            "layer_index": int(layer_index),
            "expert_index": int(expert_index),
            "expert_is_cell": False,
        },
        "deltas": records,
        "tensor_file": {
            "path": MUTATION_TENSORS_NAME,
            "sha256": _sha256_file(tensor_path),
            "bytes": tensor_path.stat().st_size,
        },
        "metadata": dict(metadata or {}),
    }
    manifest = {**unsigned, "identity_sha256": _stable_hash(unsigned)}
    (destination / MUTATION_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_expert_slice_mutation(
    mutation_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise MoeMutationError("safetensors is required for MoE mutation artifacts") from exc

    root = Path(mutation_dir).resolve()
    manifest_path = root / MUTATION_MANIFEST_NAME
    if not manifest_path.is_file():
        raise MoeMutationError(f"missing mutation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MUTATION_SCHEMA_VERSION:
        raise MoeMutationError(f"unsupported mutation schema: {manifest.get('schema_version')}")
    identity = manifest.get("identity_sha256")
    unsigned = dict(manifest)
    unsigned.pop("identity_sha256", None)
    if identity != _stable_hash(unsigned):
        raise MoeMutationError("mutation manifest identity mismatch")

    tensor_record = manifest.get("tensor_file") or {}
    tensor_path = root / str(tensor_record.get("path", MUTATION_TENSORS_NAME))
    if not tensor_path.is_file():
        raise MoeMutationError(f"missing mutation tensor file: {tensor_path}")
    if tensor_path.stat().st_size != int(tensor_record.get("bytes", -1)):
        raise MoeMutationError("mutation tensor file size mismatch")
    if _sha256_file(tensor_path) != tensor_record.get("sha256"):
        raise MoeMutationError("mutation tensor file hash mismatch")

    tensors = load_file(str(tensor_path), device="cpu")
    expected = {record["key"] for record in manifest["deltas"]}
    if set(tensors) != expected:
        raise MoeMutationError("mutation tensor key set mismatch")
    for record in manifest["deltas"]:
        tensor = tensors[record["key"]]
        if list(tensor.shape) != list(record["shape"]):
            raise MoeMutationError(f"mutation delta shape mismatch: {record['key']}")
    return manifest, tensors


def apply_expert_slice_mutation_(
    model: torch.nn.Module,
    mutation_dir: str | Path,
    *,
    scale: float = 1.0,
) -> dict[str, Any]:
    manifest, tensors = load_expert_slice_mutation(mutation_dir)
    parameters = dict(model.named_parameters())
    expert_index = int(manifest["target"]["expert_index"])
    with torch.no_grad():
        for record in manifest["deltas"]:
            runtime_name = record["runtime_tensor"]
            if runtime_name not in parameters:
                raise MoeMutationError(f"model is missing mutation target: {runtime_name}")
            parameter = parameters[runtime_name]
            if not 0 <= expert_index < parameter.shape[0]:
                raise MoeMutationError(f"invalid expert index for runtime tensor: {runtime_name}")
            target = parameter[expert_index]
            delta = tensors[record["key"]]
            if list(target.shape) != list(delta.shape):
                raise MoeMutationError(f"mutation shape mismatch for runtime tensor: {runtime_name}")
            target.add_(delta.to(device=target.device, dtype=target.dtype), alpha=float(scale))
    return manifest
