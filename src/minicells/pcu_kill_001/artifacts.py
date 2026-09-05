"""Content-addressed tensor artifact IO with honest file formats."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_tensor_artifact(state: Mapping[str, Tensor], directory: Path, stem: str = "DELTA_CELLS") -> tuple[Path, str]:
    """Use safetensors when installed; otherwise use an explicit ``.pt`` file."""
    directory.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
    except ImportError:
        path = directory / f"{stem}.pt"
        torch.save({key: value.detach().cpu() for key, value in state.items()}, path)
        return path, "torch.save"
    path = directory / f"{stem}.safetensors"
    save_file({key: value.detach().cpu().contiguous() for key, value in state.items()}, str(path))
    return path, "safetensors"


def load_tensor_artifact(path: Path, expected_sha256: str | None = None) -> dict[str, Tensor]:
    """Load only an explicitly supported format and verify bytes first."""
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(f"artifact SHA-256 mismatch for {path}")
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("safetensors is required to load this artifact") from exc
        return dict(load_file(str(path), device="cpu"))
    if path.suffix == ".pt":
        value = torch.load(path, map_location="cpu")
        if not isinstance(value, dict) or not all(isinstance(key, str) and isinstance(tensor, Tensor) for key, tensor in value.items()):
            raise ValueError("tensor artifact has an invalid state schema")
        return value
    raise ValueError(f"unsupported tensor artifact format: {path.suffix}")
