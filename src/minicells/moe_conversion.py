from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "clm.moe-substrate.v1"
MANIFEST_NAME = "clm_moe_manifest.json"
SUBSTRATE_DIR = "substrate"


class MoeConversionError(RuntimeError):
    """Raised when a checkpoint cannot satisfy the frozen conversion contract."""


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_range(path: Path, offset: int, length: int, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                raise MoeConversionError(f"unexpected EOF while hashing {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _copy_file(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode not in {"copy", "hardlink"}:
        raise ValueError("copy mode must be 'copy' or 'hardlink'")
    if mode == "hardlink":
        try:
            os.link(source.resolve(), target)
            return
        except OSError:
            pass
    shutil.copy2(source, target, follow_symlinks=True)


def _copy_tree(source: Path, target: Path, mode: str) -> None:
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    for path in _iter_files(source):
        _copy_file(path, target / path.relative_to(source), mode)


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise MoeConversionError(f"invalid safetensors header in {path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        raw_header = handle.read(header_len)
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MoeConversionError(f"invalid safetensors JSON header in {path}") from exc
    if not isinstance(header, dict):
        raise MoeConversionError(f"invalid safetensors header object in {path}")
    return 8 + header_len, header


def _classify_tensor(name: str, shape: list[int], num_experts: int) -> tuple[str, int | None]:
    moe_prefix = ".block_sparse_moe."
    if moe_prefix not in name:
        return "shared_backbone", None
    if ".block_sparse_moe.router." in name:
        return "moe_router", None
    packed = (
        ".block_sparse_moe.input_linear." in name
        or ".block_sparse_moe.output_linear." in name
    )
    if packed and shape and shape[0] == num_experts:
        return "moe_packed_experts", 0
    return "moe_other", None


def _inspect_safetensors(
    path: Path, relative_path: str, num_experts: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_start, header = _read_safetensors_header(path)
    tensors: list[dict[str, Any]] = []
    expert_addresses: list[dict[str, Any]] = []
    for name in sorted(k for k in header if k != "__metadata__"):
        entry = header[name]
        if not isinstance(entry, dict):
            raise MoeConversionError(f"invalid tensor metadata for {name}")
        shape = [int(x) for x in entry.get("shape", [])]
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise MoeConversionError(f"missing data offsets for {name}")
        start, end = (int(offsets[0]), int(offsets[1]))
        if end < start:
            raise MoeConversionError(f"invalid data offsets for {name}")
        role, expert_axis = _classify_tensor(name, shape, num_experts)
        tensor_record = {
            "name": name,
            "file": relative_path,
            "dtype": entry.get("dtype"),
            "shape": shape,
            "byte_length": end - start,
            "sha256": _sha256_range(path, data_start + start, end - start),
            "role": role,
        }
        if expert_axis is not None:
            tensor_record["expert_axis"] = expert_axis
        tensors.append(tensor_record)
        if expert_axis == 0:
            byte_length = end - start
            if byte_length % num_experts != 0:
                raise MoeConversionError(f"packed expert tensor is not evenly sliceable: {name}")
            slice_bytes = byte_length // num_experts
            for expert in range(num_experts):
                expert_addresses.append(
                    {
                        "address": f"{name}::expert[{expert}]",
                        "tensor": name,
                        "file": relative_path,
                        "axis": 0,
                        "index": expert,
                        "shape": shape[1:],
                        "byte_length": slice_bytes,
                        "sha256": _sha256_range(
                            path,
                            data_start + start + expert * slice_bytes,
                            slice_bytes,
                        ),
                    }
                )
    return tensors, expert_addresses


def inspect_hf_moe_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any]:
    root = Path(checkpoint_dir).resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise MoeConversionError(f"missing config.json in {root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "granitemoe":
        raise MoeConversionError(
            "MOE Conversion 001 is frozen to model_type=granitemoe, "
            f"got {config.get('model_type')!r}"
        )
    num_experts = int(config.get("num_local_experts", 0))
    top_k = int(config.get("num_experts_per_tok", 0))
    if num_experts <= 0 or top_k <= 0 or top_k > num_experts:
        raise MoeConversionError("invalid Granite MoE expert/router configuration")

    safetensor_files = sorted(root.glob("*.safetensors"))
    if not safetensor_files:
        raise MoeConversionError("safe_serialization is required; no *.safetensors files found")

    files = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in _iter_files(root)
    ]
    tensors: list[dict[str, Any]] = []
    expert_addresses: list[dict[str, Any]] = []
    for path in safetensor_files:
        tensor_records, address_records = _inspect_safetensors(
            path, str(path.relative_to(root)), num_experts
        )
        tensors.extend(tensor_records)
        expert_addresses.extend(address_records)

    if not any(tensor["role"] == "moe_router" for tensor in tensors):
        raise MoeConversionError("no Granite MoE router tensors discovered")
    if not any(tensor["role"] == "moe_packed_experts" for tensor in tensors):
        raise MoeConversionError("no Granite MoE packed expert tensors discovered")

    return {
        "config": {
            "model_type": config["model_type"],
            "architectures": config.get("architectures", []),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "num_local_experts": num_experts,
            "num_experts_per_tok": top_k,
        },
        "files": files,
        "tensors": tensors,
        "expert_addresses": expert_addresses,
    }


def create_clm_moe_bundle(
    checkpoint_dir: str | Path,
    bundle_dir: str | Path,
    *,
    source_model_id: str,
    source_revision: str | None = None,
    copy_mode: str = "hardlink",
) -> dict[str, Any]:
    source = Path(checkpoint_dir).resolve()
    destination = Path(bundle_dir).resolve()
    inspection = inspect_hf_moe_checkpoint(source)
    _copy_tree(source, destination / SUBSTRATE_DIR, copy_mode)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "conversion": {
            "kind": "substrate_wrap",
            "execution_semantics": "preserve_hf_moe",
            "expert_is_cell": False,
            "mutation_state": "none",
        },
        "source": {
            "model_id": source_model_id,
            "revision": source_revision,
        },
        **inspection,
    }
    manifest["identity_sha256"] = _stable_hash(manifest)
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    path = Path(bundle_dir) / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise MoeConversionError(f"unsupported manifest schema: {manifest.get('schema_version')}")
    identity = manifest.get("identity_sha256")
    unsigned = dict(manifest)
    unsigned.pop("identity_sha256", None)
    if identity != _stable_hash(unsigned):
        raise MoeConversionError("manifest identity hash mismatch")
    return manifest


def verify_clm_moe_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir).resolve()
    manifest = load_manifest(root)
    substrate = root / SUBSTRATE_DIR
    checked = 0
    for record in manifest["files"]:
        path = substrate / record["path"]
        if not path.is_file():
            raise MoeConversionError(f"missing substrate file: {record['path']}")
        if path.stat().st_size != record["bytes"] or _sha256_file(path) != record["sha256"]:
            raise MoeConversionError(f"substrate file identity mismatch: {record['path']}")
        checked += 1
    return {
        "status": "PASS",
        "identity_sha256": manifest["identity_sha256"],
        "files_checked": checked,
        "tensor_count": len(manifest["tensors"]),
        "expert_address_count": len(manifest["expert_addresses"]),
    }


def materialize_hf_checkpoint(
    bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    copy_mode: str = "hardlink",
) -> Path:
    root = Path(bundle_dir).resolve()
    verify_clm_moe_bundle(root)
    output = Path(output_dir).resolve()
    _copy_tree(root / SUBSTRATE_DIR, output, copy_mode)
    return output
