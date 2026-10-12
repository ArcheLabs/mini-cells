"""Atomic resumable checkpoint helpers with strict run-identity matching."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_identity(*, source_commit: str, seed: int, arm: str,
                  config_sha256: str, protocol_sha256: str,
                  assets_manifest_sha256: str) -> dict[str, Any]:
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    for label, value in (("config", config_sha256), ("protocol", protocol_sha256),
                         ("assets", assets_manifest_sha256)):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{label} identity must be a full lowercase SHA256")
    if seed < 0 or not arm:
        raise ValueError("seed must be non-negative and arm must be non-empty")
    return {"experiment_id": "005-compositional-cell-graph", "source_commit": source_commit,
            "seed": int(seed), "arm": arm, "config_sha256": config_sha256,
            "protocol_sha256": protocol_sha256,
            "assets_manifest_sha256": assets_manifest_sha256}


def save_checkpoint(path: str | Path, *, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, step: int,
                    scaler: torch.amp.GradScaler | None,
                    identity: dict[str, Any], generator: np.random.Generator,
                    extra: dict[str, Any] | None = None) -> None:
    """Persist every stochastic state and graph buffer before atomically replacing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "identity": dict(identity),
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "python_rng": random.getstate(),
        "numpy_rng": generator.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra": extra or {},
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path: str | Path, *, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler | None,
                    expected_identity: dict[str, Any],
                    generator: np.random.Generator,
                    map_location: str | torch.device = "cpu") -> tuple[int, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("identity") != expected_identity:
        raise ValueError("checkpoint source/config/protocol/data/seed/arm identity mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    random.setstate(payload["python_rng"])
    generator.bit_generator.state = payload["numpy_rng"]
    torch.set_rng_state(payload["torch_rng"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return int(payload["step"]), dict(payload.get("extra", {}))
