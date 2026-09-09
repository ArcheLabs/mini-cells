"""The small, stable public HybridCLM lifecycle API."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Sequence
from copy import deepcopy
from typing import Any

import torch

from .backends.base import HybridBackend
from .backends.granite_moe import GraniteMoEBackend
from .backends.registry import BackendRegistry
from .errors import ArtifactValidationError, MutationLifecycleError
from .inspector import CellPlacement, ModelInspection, normalize_placements
from .mutation import CellMutation
from .validation import RestorationReport, ZeroStateReport, compare_model_outputs


class HybridCLM:
    """A frozen foundation model with explicitly managed Cell mutations.

    v0.1 supports explicit placements and the verified Granite MoE backend.
    Scientific placement search, router training, and dependency-withdrawal
    (the future ``beta`` mechanism) intentionally remain outside this class.
    """

    def __init__(
        self,
        model: Any,
        *,
        backend: HybridBackend | None = None,
        base_model: str | None = None,
        base_revision: str | None = None,
        clone: bool = True,
    ) -> None:
        self.backend = backend or BackendRegistry.resolve(model)
        # Keep the caller's frozen foundation as the native reference and make
        # one cellularized working copy.  A 1B checkpoint should not be copied
        # twice merely to support the zero-state diagnostic.
        self._foundation_model = model
        self.model = deepcopy(model) if clone else model
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
        self.base_model = base_model or model_name
        try:
            model_revision = config._commit_hash if config is not None else None
        except AttributeError:
            model_revision = None
        self.base_revision = base_revision or model_revision
        self._inspection = self.backend.inspect(self.model)
        self._placements: tuple[CellPlacement, ...] = ()
        self._baseline_state: dict[str, torch.Tensor] | None = None
        self._attached: dict[int, dict[str, Any]] = {}

    @classmethod
    def register_backend(cls, backend: type[HybridBackend] | HybridBackend) -> Any:
        return BackendRegistry.register(backend)

    @classmethod
    def inspect_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> ModelInspection:
        _tokenizer, model, _manifest = cls._load_pretrained(model_id, revision, device, local_files_only)
        return cls.inspect(model)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> HybridCLM:
        _tokenizer, model, manifest = cls._load_pretrained(model_id, revision, device, local_files_only)
        return cls(
            model,
            backend=GraniteMoEBackend(),
            base_model=model_id,
            base_revision=manifest.get("model_revision") or revision,
        )

    @staticmethod
    def _load_pretrained(model_id: str, revision: str | None, device: str, local_files_only: bool) -> tuple[Any, Any, dict[str, Any]]:
        from ..pcu_kill_001.model import load_granite

        return load_granite(model_id, revision=revision, device=device, local_files_only=local_files_only)

    @property
    def placements(self) -> tuple[CellPlacement, ...]:
        return self._placements

    def inspect(self_or_model: Any = None, model: Any = None) -> ModelInspection:
        """Inspect either a wrapped instance or a bare model.

        The intentionally dual-shaped method supports both documented forms:
        ``hybrid.inspect()`` and ``HybridCLM.inspect(model)``.
        """
        if isinstance(self_or_model, HybridCLM) and model is None:
            return self_or_model._inspection
        candidate = model if model is not None else self_or_model
        if candidate is None:
            raise TypeError("inspect requires a model")
        return BackendRegistry.resolve(candidate).inspect(candidate)

    def cellularize(self, placements: Sequence[CellPlacement] | CellPlacement) -> HybridCLM:
        requested = normalize_placements(placements)
        resolved = tuple(self.backend.resolve_placement(self.model, placement) for placement in requested)
        self.model = self.backend.cellularize(self.model, resolved)
        self._placements = resolved
        self._baseline_state = {
            name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()
        }
        return self

    def attach(self, mutation: CellMutation) -> HybridCLM:
        if not isinstance(mutation, CellMutation):
            raise TypeError("attach expects a CellMutation")
        if not self._placements:
            raise MutationLifecycleError("cellularize the model before attaching a mutation")
        key = id(mutation)
        if key in self._attached:
            raise MutationLifecycleError("mutation is already attached")
        mutation.validate_against(
            self.model,
            inspection=self._inspection,
            base_model=self.base_model,
            base_revision=self.base_revision,
        )
        if mutation.placements:
            for declared in mutation.placements:
                matching = next((item for item in self._placements if item.layer == declared.layer), None)
                if matching is None:
                    raise ArtifactValidationError(f"mutation placement layer {declared.layer} is not cellularized")
                if declared.module_signature and declared.module_signature != matching.module_signature:
                    raise ArtifactValidationError(f"mutation module signature mismatch at layer {declared.layer}")
                if declared.experts != "all" and matching.experts != "all" and not set(declared.experts).issubset(set(matching.experts)):
                    raise ArtifactValidationError(f"mutation expert placement is outside layer {declared.layer}")
        originals = self._capture_targets(mutation)
        self._apply_delta(mutation, mutation.default_alpha)
        self._attached[key] = {"mutation": mutation, "alpha": mutation.default_alpha, "originals": originals}
        return self

    def detach(self, mutation: CellMutation) -> HybridCLM:
        record = self._attached.pop(id(mutation), None)
        if record is None:
            raise MutationLifecycleError("mutation is not attached")
        self._restore_targets(record["originals"])
        return self

    def set_alpha(self, mutation: CellMutation, alpha: float) -> HybridCLM:
        value = float(alpha)
        if not math.isfinite(value):
            raise ValueError("alpha must be finite")
        record = self._attached.get(id(mutation))
        if record is None:
            raise MutationLifecycleError("mutation is not attached")
        self._restore_targets(record["originals"])
        self._apply_delta(mutation, value)
        record["alpha"] = value
        return self

    def _capture_targets(self, mutation: CellMutation) -> dict[str, torch.Tensor]:
        parameters = dict(self.model.named_parameters())
        originals: dict[str, torch.Tensor] = {}
        for target in mutation.tensor_targets():
            name = str(target.get("name", target.get("key")))
            if name not in parameters:
                raise MutationLifecycleError(f"model is missing mutation target {name!r}")
            parameter = parameters[name]
            index = target.get("index")
            destination = parameter if index is None else parameter[int(index)]
            originals[f"{name}::{index}"] = destination.detach().clone()
        return originals

    def _restore_targets(self, originals: dict[str, torch.Tensor]) -> None:
        parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for identity, value in originals.items():
                name, _, index = identity.rpartition("::")
                if name not in parameters:
                    raise MutationLifecycleError(f"model target disappeared: {name!r}")
                destination = parameters[name] if index == "None" else parameters[name][int(index)]
                destination.copy_(value.to(device=destination.device, dtype=destination.dtype))

    def _apply_delta(self, mutation: CellMutation, scale: float) -> None:
        parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for target in mutation.tensor_targets():
                key = str(target.get("key"))
                name = str(target.get("name", key))
                if name not in parameters:
                    raise MutationLifecycleError(f"model is missing mutation target {name!r}")
                parameter = parameters[name]
                index = target.get("index")
                destination = parameter if index is None else parameter[int(index)]
                delta = mutation.tensors[key].to(device=destination.device, dtype=destination.dtype)
                if tuple(destination.shape) != tuple(delta.shape):
                    raise MutationLifecycleError(f"mutation shape mismatch for {name!r}")
                destination.add_(delta, alpha=float(scale))

    def verify_zero_state(
        self,
        inputs: dict[str, Any] | None = None,
        *,
        tolerance: float = 2e-5,
        **model_inputs: Any,
    ) -> ZeroStateReport:
        if not self._placements:
            raise MutationLifecycleError("cellularize the model before zero-state verification")
        payload = dict(inputs or {})
        payload.update(model_inputs)
        if not payload:
            raise ValueError("zero-state verification requires model inputs")
        return compare_model_outputs(self._foundation_model, self.model, payload, float(tolerance))

    def verify_restoration(self, *, tolerance: float = 0.0) -> RestorationReport:
        if self._baseline_state is None:
            raise MutationLifecycleError("cellularize the model before restoration verification")
        current = self.model.state_dict()
        differences = [
            (current[name].detach().cpu().float() - expected.float()).abs()
            for name, expected in self._baseline_state.items()
            if name in current
        ]
        max_abs = max((float(value.max().item()) for value in differences if value.numel()), default=0.0)
        mean_abs = sum((float(value.mean().item()) for value in differences if value.numel()), 0.0) / max(len(differences), 1)
        return RestorationReport(max_abs, mean_abs, max_abs <= float(tolerance), float(tolerance))

    @contextlib.contextmanager
    def mutation_enabled(self, mutation: CellMutation, *, alpha: float | None = None) -> Iterator[HybridCLM]:
        was_attached = id(mutation) in self._attached
        previous_alpha = float(self._attached[id(mutation)]["alpha"]) if was_attached else None
        if not was_attached:
            self.attach(mutation)
        if alpha is not None:
            self.set_alpha(mutation, alpha)
        try:
            yield self
        finally:
            if not was_attached:
                self.detach(mutation)
            elif previous_alpha is not None:
                self.set_alpha(mutation, previous_alpha)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self.model, "generate"):
            raise AttributeError("underlying model does not provide generate()")
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)
