from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .runtime import COWCLMError, COWRuntime, ExpertSite


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode("utf-8"))
    h.update(json.dumps(list(value.shape)).encode("utf-8"))
    h.update(raw)
    return h.hexdigest()


@dataclass(frozen=True)
class COWCellArtifact:
    cell_id: str
    parent_id: str
    parent_digest: str
    foundation_model_id: str
    foundation_revision: str
    expert_sites: tuple[ExpertSite, ...]
    state: dict[str, torch.Tensor]

    def metadata(self) -> dict[str, Any]:
        return {
            "format": "COW_CLM_CELL_V1",
            "cell_id": self.cell_id,
            "parent_id": self.parent_id,
            "parent_digest": self.parent_digest,
            "foundation_model_id": self.foundation_model_id,
            "foundation_revision": self.foundation_revision,
            "expert_sites": [site.as_dict() for site in self.expert_sites],
            "state_digests": {
                name: _tensor_digest(tensor) for name, tensor in sorted(self.state.items())
            },
        }

    def digest(self) -> str:
        payload = json.dumps(self.metadata(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def export_cell(runtime: COWRuntime, cell_id: str) -> COWCellArtifact:
    cell = runtime.cell(cell_id)
    return COWCellArtifact(
        cell_id=cell.cell_id,
        parent_id=cell.parent_id,
        parent_digest=cell.parent_digest,
        foundation_model_id=runtime.foundation_model_id,
        foundation_revision=runtime.foundation_revision,
        expert_sites=cell.expert_sites,
        state=runtime.cell_state(cell_id),
    )


def save_cell_artifact(path: Path, artifact: COWCellArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": artifact.metadata(),
            "digest": artifact.digest(),
            "state": artifact.state,
        },
        path,
    )


def load_cell_artifact(path: Path) -> COWCellArtifact:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    artifact = COWCellArtifact(
        cell_id=str(metadata["cell_id"]),
        parent_id=str(metadata["parent_id"]),
        parent_digest=str(metadata["parent_digest"]),
        foundation_model_id=str(metadata["foundation_model_id"]),
        foundation_revision=str(metadata["foundation_revision"]),
        expert_sites=tuple(ExpertSite.from_dict(value) for value in metadata["expert_sites"]),
        state={name: tensor for name, tensor in payload["state"].items()},
    )
    if artifact.metadata() != metadata:
        raise COWCLMError(f"artifact metadata does not match tensor state: {path}")
    if artifact.digest() != str(payload["digest"]):
        raise COWCLMError(f"artifact digest mismatch: {path}")
    return artifact


def apply_cell_artifact(runtime: COWRuntime, artifact: COWCellArtifact) -> str:
    if artifact.parent_id != runtime.ROOT_CELL_ID:
        raise COWCLMError("COW-CLM v0.1 artifact must descend directly from root")
    if artifact.parent_digest != runtime.root_digest:
        raise COWCLMError("artifact parent digest differs from runtime root")
    if artifact.foundation_model_id != runtime.foundation_model_id:
        raise COWCLMError("artifact foundation model differs from runtime")
    if artifact.foundation_revision != runtime.foundation_revision:
        raise COWCLMError("artifact foundation revision differs from runtime")
    runtime.fork_experts(artifact.cell_id, artifact.expert_sites)
    runtime.load_cell_state_(artifact.cell_id, artifact.state)
    return artifact.cell_id
