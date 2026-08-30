from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .language_localized_learning import LocalizedLearningState


CHECKPOINT_FORMAT = "minicells.proposal-utility-checkpoint.v1"
TRAINING_PROTOCOL_ID = "minicells.proposal-utility-training.v1"
CHECKPOINT_ENV = "MINICELLS_019_CHECKPOINT_DIR"
FORCE_RETRAIN_ENV = "MINICELLS_019_FORCE_RETRAIN"


def checkpoint_root() -> Path | None:
    raw = os.environ.get(CHECKPOINT_ENV, "").strip()
    if not raw:
        return None
    root = Path(raw).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def force_retrain() -> bool:
    return os.environ.get(FORCE_RETRAIN_ENV, "0").strip().lower() in {"1", "true", "yes"}


def phase1_path(root: Path, replicate: int) -> Path:
    return root / f"r{replicate}-phase1.pt"


def donor_path(root: Path, replicate: int, family: str) -> Path:
    safe = family.replace("/", "_")
    return root / f"r{replicate}-donor-{safe}.pt"


def localized_state_payload(state: LocalizedLearningState) -> dict[str, torch.Tensor]:
    return {
        "base_alive": state.base_alive.detach().cpu().clone(),
        "base_adjacency": state.base_adjacency.detach().cpu().clone(),
        "base_memory": state.base_memory.detach().cpu().clone(),
    }


def localized_state_from_payload(payload: dict[str, torch.Tensor]) -> LocalizedLearningState:
    return LocalizedLearningState(
        base_alive=payload["base_alive"].detach().clone(),
        base_adjacency=payload["base_adjacency"].detach().clone(),
        base_memory=payload["base_memory"].detach().clone(),
    )


def cpu_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    *,
    kind: str,
    replicate: int,
    family: str | None = None,
) -> dict[str, Any] | None:
    if force_retrain() or not path.is_file() or path.stat().st_size == 0:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"unsupported Experiment 019 checkpoint format: {path}")
    if payload.get("training_protocol") != TRAINING_PROTOCOL_ID:
        raise RuntimeError(
            f"Experiment 019 checkpoint training protocol mismatch: {path}; "
            "use --force-retrain instead of mixing model generations"
        )
    if payload.get("kind") != kind or int(payload.get("replicate", -1)) != int(replicate):
        raise RuntimeError(f"Experiment 019 checkpoint identity mismatch: {path}")
    if family is not None and payload.get("family") != family:
        raise RuntimeError(f"Experiment 019 checkpoint family mismatch: {path}")
    return payload
