from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from minicells.moe_subexpert import (
    MoeSubexpertError,
    apply_group_mutation_,
    capture_group,
    load_group_mutation,
    restore_group_,
    save_group_mutation,
    validate_group_shapes,
)

SCHEMA_VERSION = "clm.moe-multicoordinate-mutation.v1"
MANIFEST_NAME = "mutation-set.json"


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _coordinate_key(target: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(target["layer_index"]),
        int(target["expert_index"]),
        int(target["group_index"]),
    )


def _runtime_tensor_names(
    parameters: Mapping[str, torch.nn.Parameter],
    target: Mapping[str, Any],
) -> tuple[str, str]:
    has_gate_up = "gate_up_name" in target
    has_down = "down_name" in target
    if has_gate_up or has_down:
        if not (has_gate_up and has_down):
            raise MoeSubexpertError(
                "coordinate target must provide both gate_up_name and down_name"
            )
        return str(target["gate_up_name"]), str(target["down_name"])

    layer_index = int(target["layer_index"])
    expert_index = int(target["expert_index"])
    marker = f"layers.{layer_index}.block_sparse_moe."
    matches = [
        (name, parameter)
        for name, parameter in parameters.items()
        if marker in name
        and parameter.ndim == 3
        and int(parameter.shape[0]) > expert_index
    ]

    valid: list[tuple[str, str]] = []
    for gate_up_name, gate_up in matches:
        for down_name, down in matches:
            if gate_up_name == down_name:
                continue
            try:
                validate_group_shapes(gate_up, down)
            except MoeSubexpertError:
                continue
            valid.append((gate_up_name, down_name))

    if len(valid) != 1:
        raise MoeSubexpertError(
            "coordinate target omits runtime tensor names and Granite packed tensor "
            f"geometry did not resolve uniquely at layer {layer_index}: "
            f"matches={[(name, list(parameter.shape)) for name, parameter in matches]}, "
            f"valid_orientations={len(valid)}"
        )
    return valid[0]


def validate_coordinate_targets(
    targets: Sequence[Mapping[str, Any]], *, require_unique_experts: bool = False
) -> None:
    keys = [_coordinate_key(target) for target in targets]
    if len(keys) != len(set(keys)):
        raise MoeSubexpertError("multi-coordinate mutation contains duplicate coordinates")
    if require_unique_experts:
        expert_keys = [(layer, expert) for layer, expert, _group in keys]
        if len(expert_keys) != len(set(expert_keys)):
            raise MoeSubexpertError(
                "multi-coordinate mutation requires at most one writable group per expert"
            )


def capture_coordinate_set(
    parameters: Mapping[str, torch.nn.Parameter],
    targets: Sequence[Mapping[str, Any]],
) -> list[dict[str, torch.Tensor]]:
    validate_coordinate_targets(targets)
    captured: list[dict[str, torch.Tensor]] = []
    for target in targets:
        gate_up_name, down_name = _runtime_tensor_names(parameters, target)
        captured.append(
            capture_group(
                parameters,
                gate_up_name=gate_up_name,
                down_name=down_name,
                expert_index=int(target["expert_index"]),
                group_index=int(target["group_index"]),
                group_size=int(target["group_size"]),
            )
        )
    return captured


def restore_coordinate_set_(
    parameters: Mapping[str, torch.nn.Parameter],
    targets: Sequence[Mapping[str, Any]],
    originals: Sequence[Mapping[str, torch.Tensor]],
) -> None:
    if len(targets) != len(originals):
        raise MoeSubexpertError("target/original coordinate count mismatch")
    validate_coordinate_targets(targets)
    for target, original in zip(targets, originals, strict=True):
        gate_up_name, down_name = _runtime_tensor_names(parameters, target)
        restore_group_(
            parameters,
            original,
            gate_up_name=gate_up_name,
            down_name=down_name,
            expert_index=int(target["expert_index"]),
            group_index=int(target["group_index"]),
            group_size=int(target["group_size"]),
        )


def save_mutation_set(
    output_dir: str | Path,
    *,
    base_manifest_identity: str,
    source_model_id: str,
    source_revision: str,
    coordinates: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
    require_unique_experts: bool = True,
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    targets = [dict(row["target"]) for row in coordinates]
    validate_coordinate_targets(targets, require_unique_experts=require_unique_experts)

    children: list[dict[str, Any]] = []
    for index, row in enumerate(coordinates):
        target = row["target"]
        child_dir = root / f"coordinate-{index:03d}"
        child = save_group_mutation(
            child_dir,
            base_manifest_identity=base_manifest_identity,
            source_model_id=source_model_id,
            source_revision=source_revision,
            layer_index=int(target["layer_index"]),
            expert_index=int(target["expert_index"]),
            group_index=int(target["group_index"]),
            group_size=int(target["group_size"]),
            intermediate_size=int(target["intermediate_size"]),
            gate_up_runtime_name=str(target["gate_up_name"]),
            down_runtime_name=str(target["down_name"]),
            gate_up_canonical_name=str(target["gate_up_canonical_name"]),
            down_canonical_name=str(target["down_canonical_name"]),
            deltas=row["deltas"],
            metadata={
                **dict(metadata or {}),
                "coordinate_index": index,
            },
        )
        children.append(
            {
                "path": child_dir.name,
                "identity_sha256": child["identity_sha256"],
                "target": child["target"],
            }
        )

    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "kind": "aligned_intermediate_group_delta_set",
        "base": {
            "manifest_identity_sha256": base_manifest_identity,
            "model_id": source_model_id,
            "revision": source_revision,
        },
        "coordinate_count": len(children),
        "require_unique_experts": bool(require_unique_experts),
        "coordinates": children,
        "metadata": dict(metadata or {}),
    }
    manifest = {**unsigned, "identity_sha256": _stable_hash(unsigned)}
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_mutation_set(mutation_dir: str | Path) -> dict[str, Any]:
    root = Path(mutation_dir).resolve()
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise MoeSubexpertError("unsupported multi-coordinate mutation schema")
    identity = manifest.get("identity_sha256")
    unsigned = dict(manifest)
    unsigned.pop("identity_sha256", None)
    if identity != _stable_hash(unsigned):
        raise MoeSubexpertError("multi-coordinate mutation manifest identity mismatch")
    children = manifest.get("coordinates", [])
    if int(manifest.get("coordinate_count", -1)) != len(children) or not children:
        raise MoeSubexpertError("multi-coordinate mutation coordinate count mismatch")
    targets = [row["target"] for row in children]
    validate_coordinate_targets(
        targets, require_unique_experts=bool(manifest.get("require_unique_experts", False))
    )
    for child in children:
        child_manifest, _ = load_group_mutation(root / child["path"])
        if child_manifest["identity_sha256"] != child["identity_sha256"]:
            raise MoeSubexpertError("child mutation identity mismatch")
        if child_manifest["target"] != child["target"]:
            raise MoeSubexpertError("child mutation target mismatch")
        if child_manifest["base"] != manifest["base"]:
            raise MoeSubexpertError("child mutation base identity mismatch")
    return manifest


def apply_mutation_set_(
    parameters: Mapping[str, torch.nn.Parameter],
    mutation_dir: str | Path,
    *,
    scale: float = 1.0,
) -> dict[str, Any]:
    root = Path(mutation_dir).resolve()
    manifest = load_mutation_set(root)
    for child in manifest["coordinates"]:
        apply_group_mutation_(parameters, root / child["path"], scale=scale)
    return manifest
