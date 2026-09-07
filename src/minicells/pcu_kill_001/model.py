"""Pinned Granite loader and model identity manifest."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .cellular import GraniteArchitectureInspector, UnsupportedFoundationArchitecture, patch_moe_block
from .registry import module_tensor_hash


MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-base"


class DependencyUnavailable(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(config: Any) -> str:
    payload = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_granite(model_id: str = MODEL_ID, revision: str | None = None, device: str = "cpu", local_files_only: bool = False) -> tuple[Any, nn.Module, dict[str, Any]]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise DependencyUnavailable("transformers is required for the Granite backend") from exc
    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    resolved_revision = revision
    source = model_id
    try:
        from huggingface_hub import HfApi, snapshot_download

        if resolved_revision is None:
            resolved_revision = HfApi().model_info(model_id).sha
        source = snapshot_download(model_id, revision=resolved_revision, local_files_only=local_files_only)
        kwargs = {"local_files_only": True}
    except Exception as exc:
        # Never silently fall back to a moving main branch.  A formal freeze
        # requires a concrete Hub commit and local files to hash.
        if not resolved_revision:
            raise DependencyUnavailable(
                f"could not resolve an immutable Granite revision for {model_id}"
            ) from exc
        kwargs["revision"] = resolved_revision
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32, **kwargs)
    model.to(torch.device(device)).eval()
    manifest = model_identity_manifest(model_id, resolved_revision, model, tokenizer, source_root=source if Path(str(source)).is_dir() else None)
    return tokenizer, model, manifest


def model_identity_manifest(model_id: str, revision: str | None, model: nn.Module, tokenizer: Any | None = None, source_root: str | Path | None = None) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        raise UnsupportedFoundationArchitecture("loaded model has no config")
    manifest: dict[str, Any] = {
        "schema": "minicells.pcu-kill-001.model-manifest.v1",
        "model_repo": model_id,
        "model_revision": revision or getattr(config, "_commit_hash", None),
        "config_sha256": _config_hash(config),
        "weight_file_sha256": [],
        "tokenizer_sha256": [],
        "foundation_tensor_sha256": module_tensor_hash(model),
    }
    source = source_root or getattr(config, "_name_or_path", None) or getattr(model, "name_or_path", None)
    if source and Path(str(source)).is_dir():
        source_path = Path(str(source))
        weight_names = ("*.safetensors", "*.bin", "*.pt")
        files = sorted({path for pattern in weight_names for path in source_path.glob(pattern) if path.is_file()})
        manifest["weight_file_sha256"] = [{"file": path.name, "sha256": _sha256(path)} for path in files]
        tokenizer_names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
        manifest["tokenizer_sha256"] = [{"file": name, "sha256": _sha256(source_path / name)} for name in tokenizer_names if (source_path / name).is_file()]
    return manifest


def target_module(model: nn.Module, target_path: str) -> nn.Module:
    value: Any = model
    for part in target_path.split("."):
        value = getattr(value, part)
    if not isinstance(value, nn.Module):
        raise UnsupportedFoundationArchitecture(f"target path is not a module: {target_path}")
    return value


def cellularize_model(model: nn.Module, inspector: GraniteArchitectureInspector | None = None) -> tuple[nn.Module, GraniteArchitectureInspector]:
    """Clone and patch only the final MoE expert compute module."""
    inspector = inspector or GraniteArchitectureInspector.inspect(model)
    cellular = deepcopy(model)
    block = target_module(cellular, inspector.target_path)
    patch_moe_block(block, inspector.partition)
    return cellular.eval(), inspector


def model_state_hashes(model: nn.Module) -> dict[str, str]:
    result = {}
    for name, value in sorted(model.state_dict().items()):
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        result[name] = digest.hexdigest()
    return result
