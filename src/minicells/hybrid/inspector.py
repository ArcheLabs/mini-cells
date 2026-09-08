"""Typed structural model inspection and explicit Cell placement."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable

from .errors import PlacementError


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CellPlacement:
    """An explicit, serializable location at which Cells may be managed.

    ``experts="all"`` is accepted as a convenience only when a backend can
    prove that all experts are addressable.  No automatic scientific placement
    heuristic is implied by this class.
    """

    layer: int
    experts: tuple[int, ...] | str = "all"
    module_path: str | None = None
    placement_type: str = "expert"
    architecture_signature: str | None = None
    module_signature: str | None = None

    def __post_init__(self) -> None:
        layer = int(self.layer)
        if layer < 0:
            raise PlacementError("layer must be non-negative")
        object.__setattr__(self, "layer", layer)
        if self.placement_type != "expert":
            raise PlacementError("v0.1 supports only expert placements")
        if isinstance(self.experts, str):
            if self.experts != "all":
                raise PlacementError("experts must be a sequence of ids or 'all'")
        else:
            try:
                values = tuple(sorted({int(index) for index in self.experts}))
            except (TypeError, ValueError) as exc:
                raise PlacementError("expert ids must be integers") from exc
            if any(index < 0 for index in values):
                raise PlacementError("expert ids must be non-negative")
            if not values:
                raise PlacementError("at least one expert is required")
            object.__setattr__(self, "experts", values)
        if self.module_path is not None and not self.module_path.strip():
            raise PlacementError("module_path cannot be empty")

    @property
    def layer_index(self) -> int:
        return self.layer

    @property
    def expert_ids(self) -> tuple[int, ...] | str:
        return self.experts

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer,
            "module_path": self.module_path,
            "expert_ids": list(self.experts) if self.experts != "all" else "all",
            "placement_type": self.placement_type,
            "architecture_signature": self.architecture_signature,
            "module_signature": self.module_signature,
        }

    as_dict = to_dict

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CellPlacement":
        if not isinstance(value, dict):
            raise PlacementError("placement must be an object")
        experts = value.get("expert_ids", value.get("experts", "all"))
        return cls(
            layer=int(value.get("layer_index", value.get("layer", -1))),
            experts=experts if experts == "all" else tuple(experts),
            module_path=value.get("module_path"),
            placement_type=str(value.get("placement_type", "expert")),
            architecture_signature=value.get("architecture_signature"),
            module_signature=value.get("module_signature"),
        )


@dataclass(frozen=True)
class ModelInspection:
    """Structural facts discovered by a backend, never an optimality claim."""

    architecture: str
    backend: str
    num_layers: int
    moe_layers: tuple[int, ...]
    experts_per_layer: dict[int, int]
    router_type: str
    activation: str
    supports_cellularization: bool
    supported_placement_types: tuple[str, ...] = ("expert",)
    support_level: str = "SUPPORTED"
    architecture_signature: str = ""
    module_signatures: dict[int, str] = field(default_factory=dict)
    target_paths: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        levels = {"SUPPORTED", "EXPERIMENTAL", "INSPECT_ONLY", "UNSUPPORTED"}
        if self.support_level not in levels:
            raise ValueError(f"unknown support level: {self.support_level}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "backend": self.backend,
            "num_layers": self.num_layers,
            "moe_layers": list(self.moe_layers),
            "experts_per_layer": {str(key): value for key, value in self.experts_per_layer.items()},
            "router_type": self.router_type,
            "activation": self.activation,
            "supports_cellularization": self.supports_cellularization,
            "supported_placement_types": list(self.supported_placement_types),
            "support_level": self.support_level,
            "architecture_signature": self.architecture_signature,
            "module_signatures": {str(key): value for key, value in self.module_signatures.items()},
            "target_paths": {str(key): value for key, value in self.target_paths.items()},
        }

    as_dict = to_dict

    @property
    def module_signature(self) -> str:
        """Single-target convenience for callers inspecting one-layer models."""
        if not self.module_signatures:
            return ""
        return self.module_signatures[sorted(self.module_signatures)[-1]]


def normalize_placements(placements: Iterable[CellPlacement] | CellPlacement) -> tuple[CellPlacement, ...]:
    if isinstance(placements, CellPlacement):
        result = (placements,)
    else:
        try:
            result = tuple(placements)
        except TypeError as exc:
            raise PlacementError("placements must be CellPlacement values") from exc
    if not result:
        raise PlacementError("at least one placement is required")
    if any(not isinstance(value, CellPlacement) for value in result):
        raise PlacementError("placements must be CellPlacement values")
    layers = [value.layer for value in result]
    if layers != sorted(set(layers)):
        raise PlacementError("placements must contain each layer at most once")
    return result


def architecture_signature(model: Any) -> str:
    """Hash stable config attributes without depending on repository strings."""
    config = getattr(model, "config", None)
    if config is None:
        payload: Any = {"class": type(model).__qualname__}
    elif hasattr(config, "to_dict"):
        payload = config.to_dict()
    else:
        payload = {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_") and isinstance(value, (str, int, float, bool, list, tuple, type(None)))
        }
    return _json_hash(payload)
