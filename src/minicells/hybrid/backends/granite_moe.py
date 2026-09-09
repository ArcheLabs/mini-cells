"""Verified Granite-style fused SwiGLU MoE backend.

The adapter deliberately reuses the already validated router-preserving Cell
partitioner in :mod:`minicells.pcu_kill_001`.  It recognizes an architecture
from config/module structure, not from a mutable Hub repository name, and
rejects anything outside the tested fused gate/up + down layout.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

from torch import nn

from ...pcu_kill_001.cellular import CellularExperts, GraniteArchitectureInspector, patch_moe_block
from ...pcu_kill_001.model import target_module
from ..errors import PlacementError, UnsupportedArchitectureError
from ..inspector import CellPlacement, ModelInspection, architecture_signature
from .base import HybridBackend


def _module_signature(module: nn.Module) -> str:
    payload = [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(module.named_parameters())
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _is_granite_like(model: Any) -> bool:
    try:
        config = model.config
    except AttributeError:
        config = None
    try:
        model_type = str(config.model_type if config is not None else "").lower()
    except AttributeError:
        model_type = ""
    try:
        backbone = model.model
    except AttributeError:
        backbone = None
    names = f"{type(model).__name__} {type(backbone).__name__} {model_type}".lower()
    return "granite" in names


class GraniteMoEBackend(HybridBackend):
    name = "granite_moe"

    def inspect(self, model: Any) -> ModelInspection:
        if not _is_granite_like(model):
            raise UnsupportedArchitectureError("model is not a recognized Granite MoE architecture")
        already_cellular = any(
            isinstance(module.experts, CellularExperts)
            for module in model.modules()
            if hasattr(module, "experts")
        )
        if not already_cellular:
            try:
                GraniteArchitectureInspector.inspect(model, require_granite=False)
            except Exception as exc:
                raise UnsupportedArchitectureError(f"unsupported Granite MoE layout: {exc}") from exc

        paths: dict[int, str] = {}
        signatures: dict[int, str] = {}
        experts_per_layer: dict[int, int] = {}
        for name, module in model.named_modules():
            if not hasattr(module, "experts") or not hasattr(module, "router"):
                continue
            experts = module.experts
            router = module.router
            if experts is None or router is None:
                continue
            if isinstance(experts, CellularExperts):
                count = int(experts.num_experts)
            else:
                try:
                    count = int(experts.num_experts)
                except (AttributeError, TypeError, ValueError):
                    continue
            numbers = re.findall(r"layers\.(\d+)", name)
            layer = int(numbers[-1]) if numbers else len(paths)
            paths[layer] = name
            signatures[layer] = _module_signature(module)
            experts_per_layer[layer] = count
        if not paths:
            raise UnsupportedArchitectureError("Granite model has no router-preserving MoE block")
        try:
            config = model.config
        except AttributeError:
            config = None
        try:
            configured_layers = config.num_hidden_layers if config is not None else None
        except AttributeError:
            configured_layers = None
        num_layers = int(configured_layers if configured_layers is not None else max(paths) + 1)
        try:
            experts_per_token = config.num_experts_per_tok if config is not None else None
        except AttributeError:
            experts_per_token = None
        return ModelInspection(
            architecture="granitemoe",
            backend=self.name,
            num_layers=num_layers,
            moe_layers=tuple(sorted(paths)),
            experts_per_layer=experts_per_layer,
            router_type="top_k" if experts_per_token else "inherited",
            activation="swiglu",
            supports_cellularization=True,
            supported_placement_types=("expert",),
            support_level="SUPPORTED",
            architecture_signature=architecture_signature(model),
            module_signatures=signatures,
            target_paths=paths,
        )

    def resolve_placement(self, model: Any, placement: CellPlacement) -> CellPlacement:
        inspection = self.inspect(model)
        if placement.layer not in inspection.moe_layers:
            raise PlacementError(f"layer {placement.layer} is not a supported Granite MoE layer")
        count = inspection.experts_per_layer[placement.layer]
        experts = tuple(range(count)) if placement.experts == "all" else tuple(placement.experts)
        if any(index >= count for index in experts):
            raise PlacementError(f"expert id outside layer {placement.layer} range 0..{count - 1}")
        if placement.module_path is not None and placement.module_path != inspection.target_paths[placement.layer]:
            raise PlacementError("placement module_path does not match the inspected model")
        return CellPlacement(
            layer=placement.layer,
            experts=experts,
            module_path=inspection.target_paths[placement.layer],
            placement_type=placement.placement_type,
            architecture_signature=placement.architecture_signature or inspection.architecture_signature,
            module_signature=placement.module_signature or inspection.module_signatures[placement.layer],
        )

    def cellularize(self, model: Any, placements: Sequence[CellPlacement]) -> Any:
        inspection = self.inspect(model)
        for original in placements:
            if original.layer not in inspection.moe_layers:
                raise PlacementError(f"layer {original.layer} is not a supported Granite MoE layer")
            count = inspection.experts_per_layer[original.layer]
            experts = tuple(range(count)) if original.experts == "all" else tuple(original.experts)
            if any(index >= count for index in experts):
                raise PlacementError(f"expert id outside layer {original.layer} range 0..{count - 1}")
            placement = CellPlacement(
                layer=original.layer,
                experts=experts,
                module_path=original.module_path or inspection.target_paths[original.layer],
                placement_type=original.placement_type,
                architecture_signature=original.architecture_signature or inspection.architecture_signature,
                module_signature=original.module_signature or inspection.module_signatures[original.layer],
            )
            path = placement.module_path or inspection.target_paths[placement.layer]
            block = target_module(model, path)
            if not hasattr(block, "experts") or not isinstance(block.experts, CellularExperts):
                patch_moe_block(block)
        model._minicells_hybrid_placements = tuple(placements)
        return model
