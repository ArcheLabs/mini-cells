"""Independent Cell mutation objects and safe artifact loading."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .errors import ArtifactValidationError
from .inspector import CellPlacement, ModelInspection
from .serialization import load_artifact, save_artifact


class CellMutation:
    """A content-addressed, alpha-scalable collection of Cell parameter deltas.

    Tensors are always kept detached on CPU.  The optional ``targets`` records
    identify a full parameter or a first-axis expert slice; no pickle-backed
    executable state is accepted.
    """

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        placements: tuple[CellPlacement, ...] | list[CellPlacement] = (),
        base_model: str | None = None,
        base_revision: str | None = None,
        architecture: str = "granitemoe",
        architecture_hash: str = "",
        default_alpha: float = 1.0,
        provenance: Mapping[str, Any] | None = None,
        evaluation: Mapping[str, Any] | None = None,
        targets: list[Mapping[str, Any]] | None = None,
        source_path: str | Path | None = None,
    ) -> None:
        if not tensors:
            raise ArtifactValidationError("a CellMutation requires at least one tensor")
        if not math.isfinite(float(default_alpha)):
            raise ValueError("default_alpha must be finite")
        self.tensors = {
            str(key): value.detach().cpu().clone() for key, value in tensors.items()
        }
        self.placements = tuple(placements)
        self.base_model = base_model
        self.base_revision = base_revision
        self.architecture = architecture
        self.architecture_hash = architecture_hash
        self.default_alpha = float(default_alpha)
        self.provenance = dict(provenance or {})
        self.evaluation = dict(evaluation or {})
        self.targets = [dict(value) for value in (targets or [])]
        self.source_path = Path(source_path).resolve() if source_path else None

    @classmethod
    def from_pretrained(cls, path_or_repo: str | Path, *, revision: str | None = None) -> CellMutation:
        path = Path(path_or_repo).expanduser()
        if not path.is_dir():
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ArtifactValidationError("a local artifact directory or huggingface_hub is required") from exc
            try:
                path = Path(snapshot_download(str(path_or_repo), revision=revision, allow_patterns=[
                    "mutation.safetensors", "cell_config.json", "manifest.json", "provenance.json", "evaluation.json"
                ]))
            except Exception as exc:
                raise ArtifactValidationError(f"could not download mutation artifact {path_or_repo!r}") from exc
        config, manifest, tensors, provenance, evaluation = load_artifact(path)
        placements = tuple(CellPlacement.from_dict(value) for value in config.get("placements", []))
        return cls(
            tensors,
            placements=placements,
            base_model=manifest.get("base_model"),
            base_revision=manifest.get("base_revision"),
            architecture=manifest.get("architecture", "granitemoe"),
            architecture_hash=manifest.get("architecture_hash", ""),
            default_alpha=float(config.get("default_alpha", 1.0)),
            provenance=provenance,
            evaluation=evaluation,
            targets=config.get("targets", []),
            source_path=path,
        )

    def save_pretrained(self, path: str | Path) -> Path:
        if not self.base_model or not self.base_revision:
            raise ArtifactValidationError("base_model and immutable base_revision are required")
        config = {
            "mutation_type": "expert_delta",
            "backend": "granite_moe",
            "placements": [placement.to_dict() for placement in self.placements],
            "default_alpha": self.default_alpha,
            "targets": self.targets,
        }
        manifest = {
            "base_model": self.base_model,
            "base_revision": self.base_revision,
            "architecture": self.architecture,
            "architecture_hash": self.architecture_hash,
            "dtype": "F32",
        }
        provenance = {
            "training_source_commit": None,
            "training_protocol": None,
            "dataset_identity": None,
            "dataset_hash": None,
            "seed": None,
            "optimizer_configuration": None,
            "training_steps": None,
            "trainable_parameter_count": int(sum(value.numel() for value in self.tensors.values())),
            "created_at": None,
            "status": "engineering",
            "research_status": "Engineering Evidence",
            "engineering_status": "ready_for_review",
            "formal_status": "NOT_STARTED",
            **self.provenance,
        }
        evaluation = {"status": "Engineering Evidence", **self.evaluation}
        root = save_artifact(path, self.tensors, cell_config=config, manifest=manifest, provenance=provenance, evaluation=evaluation)
        self.source_path = root
        return root

    def set_alpha(self, alpha: float) -> CellMutation:
        value = float(alpha)
        if not math.isfinite(value):
            raise ValueError("alpha must be finite")
        self.default_alpha = value
        return self

    def tensor_targets(self) -> tuple[dict[str, Any], ...]:
        if self.targets:
            return tuple(self.targets)
        return tuple({"key": key, "name": key, "index": None} for key in self.tensors)

    def validate_against(
        self,
        model: Any,
        *,
        inspection: ModelInspection | None = None,
        base_model: str | None = None,
        base_revision: str | None = None,
        allow_override: bool = False,
    ) -> None:
        """Validate identity, architecture, placement, shapes, and target names."""
        if inspection is None:
            from .backends.registry import BackendRegistry

            inspection = BackendRegistry.resolve(model).inspect(model)
        failures: list[str] = []
        try:
            model_name = model.name_or_path
        except AttributeError:
            model_name = None
        try:
            config = model.config
        except AttributeError:
            config = None
        if model_name is None and config is not None:
            try:
                model_name = config._name_or_path
            except AttributeError:
                model_name = None
        expected_model = base_model or model_name
        if self.base_model and expected_model and self.base_model != expected_model:
            failures.append(f"base model mismatch ({expected_model!r} != {self.base_model!r})")
        try:
            model_revision = config._commit_hash if config is not None else None
        except AttributeError:
            model_revision = None
        expected_revision = base_revision or model_revision
        if self.base_revision and expected_revision and self.base_revision != expected_revision:
            failures.append(f"base revision mismatch ({expected_revision!r} != {self.base_revision!r})")
        if self.architecture and self.architecture != inspection.architecture:
            failures.append(f"architecture mismatch ({inspection.architecture!r} != {self.architecture!r})")
        if self.architecture_hash and self.architecture_hash != inspection.architecture_signature:
            failures.append("architecture signature mismatch")
        parameters = dict(model.named_parameters())
        for target in self.tensor_targets():
            key = str(target.get("key"))
            name = str(target.get("name", key))
            if key not in self.tensors:
                failures.append(f"tensor manifest references missing tensor {key!r}")
                continue
            if name not in parameters:
                failures.append(f"model is missing mutation target {name!r}")
                continue
            parameter = parameters[name]
            index = target.get("index")
            try:
                shape = tuple(parameter.shape) if index is None else tuple(parameter[int(index)].shape)
            except (IndexError, TypeError, ValueError):
                failures.append(f"invalid target index for {name!r}")
                continue
            if shape != tuple(self.tensors[key].shape):
                failures.append(f"shape mismatch for {name!r}")
        if failures and not allow_override:
            raise ArtifactValidationError("; ".join(failures))

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.base_model).encode())
        digest.update(str(self.base_revision).encode())
        digest.update(self.architecture.encode())
        for key in sorted(self.tensors):
            tensor = self.tensors[key].contiguous()
            digest.update(key.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()
