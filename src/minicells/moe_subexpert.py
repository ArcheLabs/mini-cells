from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = "clm.moe-subexpert-mutation.v1"
MANIFEST_NAME = "mutation.json"
TENSOR_NAME = "mutation.safetensors"


class MoeSubexpertError(RuntimeError):
    """Raised when an aligned sub-expert mutation violates its contract."""


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_group_shapes(
    gate_up: torch.Tensor,
    down: torch.Tensor,
) -> int:
    if gate_up.ndim != 3 or down.ndim != 3:
        raise MoeSubexpertError("Granite packed expert tensors must both be rank-3")
    if gate_up.shape[0] != down.shape[0]:
        raise MoeSubexpertError("packed expert count mismatch")
    intermediate = int(down.shape[2])
    if gate_up.shape[1] != 2 * intermediate:
        raise MoeSubexpertError("gate/up tensor does not encode two aligned intermediate blocks")
    if gate_up.shape[2] != down.shape[1]:
        raise MoeSubexpertError("hidden dimension mismatch between gate/up and down tensors")
    return intermediate


def group_bounds(intermediate: int, group_size: int, group_index: int) -> tuple[int, int]:
    if group_size <= 0 or intermediate % group_size != 0:
        raise MoeSubexpertError("group_size must evenly divide the expert intermediate width")
    groups = intermediate // group_size
    if not 0 <= group_index < groups:
        raise MoeSubexpertError(f"group_index {group_index} outside [0, {groups})")
    start = group_index * group_size
    return start, start + group_size


def capture_group(
    parameters: Mapping[str, torch.nn.Parameter],
    *,
    gate_up_name: str,
    down_name: str,
    expert_index: int,
    group_index: int,
    group_size: int,
) -> dict[str, torch.Tensor]:
    if gate_up_name not in parameters or down_name not in parameters:
        raise MoeSubexpertError("model is missing one or both packed expert tensors")
    gate_up = parameters[gate_up_name]
    down = parameters[down_name]
    intermediate = validate_group_shapes(gate_up, down)
    if not 0 <= expert_index < gate_up.shape[0]:
        raise MoeSubexpertError("invalid expert index")
    start, end = group_bounds(intermediate, group_size, group_index)
    gate = gate_up.detach()[expert_index, start:end].float().cpu().clone()
    up = gate_up.detach()[expert_index, intermediate + start : intermediate + end].float().cpu().clone()
    down_group = down.detach()[expert_index, :, start:end].float().cpu().clone()
    return {"gate": gate, "up": up, "down": down_group}


def restore_group_(
    parameters: Mapping[str, torch.nn.Parameter],
    original: Mapping[str, torch.Tensor],
    *,
    gate_up_name: str,
    down_name: str,
    expert_index: int,
    group_index: int,
    group_size: int,
) -> None:
    gate_up = parameters[gate_up_name]
    down = parameters[down_name]
    intermediate = validate_group_shapes(gate_up, down)
    start, end = group_bounds(intermediate, group_size, group_index)
    with torch.no_grad():
        gate_up[expert_index, start:end].copy_(
            original["gate"].to(gate_up.device, gate_up.dtype)
        )
        gate_up[expert_index, intermediate + start : intermediate + end].copy_(
            original["up"].to(gate_up.device, gate_up.dtype)
        )
        down[expert_index, :, start:end].copy_(
            original["down"].to(down.device, down.dtype)
        )


def group_delta(
    current: Mapping[str, torch.Tensor], original: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {key: current[key] - original[key] for key in ("gate", "up", "down")}


def save_group_mutation(
    output_dir: str | Path,
    *,
    base_manifest_identity: str,
    source_model_id: str,
    source_revision: str,
    layer_index: int,
    expert_index: int,
    group_index: int,
    group_size: int,
    intermediate_size: int,
    gate_up_runtime_name: str,
    down_runtime_name: str,
    gate_up_canonical_name: str,
    down_canonical_name: str,
    deltas: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise MoeSubexpertError("safetensors is required") from exc

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if set(deltas) != {"gate", "up", "down"}:
        raise MoeSubexpertError("aligned mutation requires gate, up, and down deltas")
    start, end = group_bounds(intermediate_size, group_size, group_index)
    tensors = {key: deltas[key].detach().float().cpu().contiguous() for key in deltas}
    tensor_path = root / TENSOR_NAME
    save_file(tensors, str(tensor_path))
    records = {
        key: {
            "shape": list(value.shape),
            "l2_norm": float(torch.linalg.vector_norm(value).item()),
            "max_abs": float(value.abs().max().item()),
        }
        for key, value in tensors.items()
    }
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "kind": "aligned_intermediate_group_delta",
        "base": {
            "manifest_identity_sha256": base_manifest_identity,
            "model_id": source_model_id,
            "revision": source_revision,
        },
        "target": {
            "layer_index": int(layer_index),
            "expert_index": int(expert_index),
            "group_index": int(group_index),
            "group_size": int(group_size),
            "intermediate_size": int(intermediate_size),
            "channel_start": int(start),
            "channel_end": int(end),
            "expert_fraction": float(group_size / intermediate_size),
            "expert_is_cell": False,
            "group_is_cell": False,
        },
        "runtime_tensors": {
            "gate_up": gate_up_runtime_name,
            "down": down_runtime_name,
        },
        "canonical_tensors": {
            "gate_up": gate_up_canonical_name,
            "down": down_canonical_name,
        },
        "deltas": records,
        "tensor_file": {
            "path": TENSOR_NAME,
            "bytes": tensor_path.stat().st_size,
            "sha256": _sha256_file(tensor_path),
        },
        "metadata": dict(metadata or {}),
    }
    manifest = {**unsigned, "identity_sha256": _stable_hash(unsigned)}
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_group_mutation(
    mutation_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise MoeSubexpertError("safetensors is required") from exc
    root = Path(mutation_dir).resolve()
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise MoeSubexpertError("unsupported sub-expert mutation schema")
    identity = manifest.get("identity_sha256")
    unsigned = dict(manifest)
    unsigned.pop("identity_sha256", None)
    if identity != _stable_hash(unsigned):
        raise MoeSubexpertError("mutation manifest identity mismatch")
    tensor_record = manifest["tensor_file"]
    tensor_path = root / tensor_record["path"]
    if tensor_path.stat().st_size != int(tensor_record["bytes"]):
        raise MoeSubexpertError("mutation tensor size mismatch")
    if _sha256_file(tensor_path) != tensor_record["sha256"]:
        raise MoeSubexpertError("mutation tensor hash mismatch")
    tensors = load_file(str(tensor_path), device="cpu")
    if set(tensors) != {"gate", "up", "down"}:
        raise MoeSubexpertError("mutation tensor key set mismatch")
    return manifest, tensors


def apply_group_mutation_(
    parameters: Mapping[str, torch.nn.Parameter],
    mutation_dir: str | Path,
    *,
    scale: float = 1.0,
) -> dict[str, Any]:
    manifest, tensors = load_group_mutation(mutation_dir)
    target = manifest["target"]
    names = manifest["runtime_tensors"]
    gate_up = parameters[names["gate_up"]]
    down = parameters[names["down"]]
    intermediate = validate_group_shapes(gate_up, down)
    if intermediate != int(target["intermediate_size"]):
        raise MoeSubexpertError("intermediate width mismatch")
    start, end = group_bounds(intermediate, int(target["group_size"]), int(target["group_index"]))
    expert = int(target["expert_index"])
    with torch.no_grad():
        gate_up[expert, start:end].add_(
            tensors["gate"].to(gate_up.device, gate_up.dtype), alpha=float(scale)
        )
        gate_up[expert, intermediate + start : intermediate + end].add_(
            tensors["up"].to(gate_up.device, gate_up.dtype), alpha=float(scale)
        )
        down[expert, :, start:end].add_(
            tensors["down"].to(down.device, down.dtype), alpha=float(scale)
        )
    return manifest
