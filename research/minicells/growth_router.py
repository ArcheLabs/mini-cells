"""Explicit, reconstructable hierarchical routing for CLM-0.3.

The root router is deliberately kept separate from the growth tree.  A birth
only replaces one root leaf with a binary split, so a child can never see
traffic which was outside its parent's root region.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Union

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RouteLeaf:
    expert_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"leaf": self.expert_id}


@dataclass(frozen=True)
class RouteSplit:
    split_id: str
    left: "RouteNode"
    right: "RouteNode"

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split_id,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


RouteNode = Union[RouteLeaf, RouteSplit]


def route_node_from_dict(value: dict[str, Any]) -> RouteNode:
    if "leaf" in value:
        return RouteLeaf(str(value["leaf"]))
    if "split" in value:
        return RouteSplit(
            str(value["split"]),
            route_node_from_dict(value["left"]),
            route_node_from_dict(value["right"]),
        )
    raise ValueError(f"invalid route node: {value!r}")


def _replace_leaf(node: RouteNode, parent_id: str, replacement: RouteNode) -> tuple[RouteNode, bool]:
    if isinstance(node, RouteLeaf):
        return (replacement, True) if node.expert_id == parent_id else (node, False)
    left, changed = _replace_leaf(node.left, parent_id, replacement)
    if changed:
        return RouteSplit(node.split_id, left, node.right), True
    right, changed = _replace_leaf(node.right, parent_id, replacement)
    return (RouteSplit(node.split_id, node.left, right), True) if changed else (node, False)


def iter_leaves(node: RouteNode) -> Iterable[str]:
    if isinstance(node, RouteLeaf):
        yield node.expert_id
    else:
        yield from iter_leaves(node.left)
        yield from iter_leaves(node.right)


def iter_splits(node: RouteNode) -> Iterable[str]:
    if isinstance(node, RouteSplit):
        yield node.split_id
        yield from iter_splits(node.left)
        yield from iter_splits(node.right)


class BinaryLineageRouter(nn.Module):
    """A pointwise cosine-prototype router with exactly two outputs."""

    def __init__(self, dim: int, *, scale: float = 4.0) -> None:
        super().__init__()
        if dim < 1 or scale <= 0:
            raise ValueError("dim must be positive and scale must be positive")
        self.dim = dim
        self.scale = float(scale)
        self.prototypes = nn.Parameter(F.normalize(torch.randn(2, dim), dim=-1))

    @torch.no_grad()
    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        if tuple(prototypes.shape) != (2, self.dim):
            raise ValueError(f"expected prototypes {(2, self.dim)}, got {tuple(prototypes.shape)}")
        self.prototypes.copy_(F.normalize(prototypes.to(self.prototypes), dim=-1))

    def forward(self, perception: torch.Tensor) -> torch.Tensor:
        return self.scale * F.normalize(perception, dim=-1) @ F.normalize(self.prototypes, dim=-1).T


def straight_through_top1(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = logits.softmax(dim=-1)
    indices = probabilities.argmax(dim=-1)
    hard = F.one_hot(indices, num_classes=logits.shape[-1]).to(logits.dtype)
    return hard + probabilities - probabilities.detach(), probabilities, indices


class HierarchicalGrowthRouter(nn.Module):
    """Root 4-way routing plus an explicit recursive lineage tree."""

    def __init__(
        self,
        stage: int,
        root_router: nn.Module,
        dim: int,
        *,
        root_expert_count: int = 4,
        router_scale: float = 4.0,
    ) -> None:
        super().__init__()
        if root_expert_count < 1:
            raise ValueError("root_expert_count must be positive")
        self.stage = int(stage)
        self.root_router = root_router
        self.dim = dim
        self.root_expert_count = root_expert_count
        self.router_scale = float(router_scale)
        self.split_routers = nn.ModuleDict()
        self.roots: tuple[RouteNode, ...] = tuple(
            RouteLeaf(f"s{stage}-e{index}") for index in range(root_expert_count)
        )

    @property
    def expert_ids(self) -> tuple[str, ...]:
        return tuple(expert_id for root in self.roots for expert_id in iter_leaves(root))

    @property
    def split_ids(self) -> tuple[str, ...]:
        return tuple(split_id for root in self.roots for split_id in iter_splits(root))

    def structure(self) -> dict[str, Any]:
        return {"stage": self.stage, "roots": [root.to_dict() for root in self.roots]}

    def restore_structure(self, structure: dict[str, Any]) -> None:
        if int(structure.get("stage", -1)) != self.stage:
            raise ValueError("growth router stage does not match checkpoint")
        roots = tuple(route_node_from_dict(item) for item in structure["roots"])
        if len(roots) != self.root_expert_count:
            raise ValueError("checkpoint has the wrong root count")
        self.roots = roots
        wanted = set(self.split_ids)
        for split_id in wanted - set(self.split_routers):
            self.split_routers[split_id] = BinaryLineageRouter(self.dim, scale=self.router_scale)
        for split_id in tuple(self.split_routers):
            if split_id not in wanted:
                del self.split_routers[split_id]

    @torch.no_grad()
    def add_split(
        self,
        parent_id: str,
        child_id: str,
        split_id: str,
        prototypes: torch.Tensor,
    ) -> BinaryLineageRouter:
        if split_id in self.split_routers:
            raise ValueError(f"split already exists: {split_id}")
        if parent_id not in self.expert_ids:
            raise KeyError(f"parent is not a current leaf: {parent_id}")
        router = BinaryLineageRouter(self.dim, scale=self.router_scale)
        router.set_prototypes(prototypes)
        replacement = RouteSplit(split_id, RouteLeaf(parent_id), RouteLeaf(child_id))
        new_roots: list[RouteNode] = []
        changed = False
        for root in self.roots:
            updated, did_change = _replace_leaf(root, parent_id, replacement)
            new_roots.append(updated)
            changed = changed or did_change
        if not changed:
            raise KeyError(f"parent is not reachable from stage roots: {parent_id}")
        self.roots = tuple(new_roots)
        self.split_routers[split_id] = router
        return router

    def _route_node(
        self,
        node: RouteNode,
        perception: torch.Tensor,
        incoming: torch.Tensor,
        gates: dict[str, torch.Tensor],
        split_choices: dict[str, torch.Tensor],
    ) -> None:
        if isinstance(node, RouteLeaf):
            gates[node.expert_id] = gates.get(node.expert_id, torch.zeros_like(incoming)) + incoming
            return
        logits = self.split_routers[node.split_id](perception)
        branch, _probabilities, indices = straight_through_top1(logits)
        split_choices[node.split_id] = indices
        self._route_node(node.left, perception, incoming * branch[..., 0], gates, split_choices)
        self._route_node(node.right, perception, incoming * branch[..., 1], gates, split_choices)

    def route(
        self, perception: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        gates, root_indices, choices, _probabilities = self.route_with_details(perception)
        return gates, root_indices, choices

    def route_with_details(
        self, perception: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        root_logits = self.root_router(perception)
        if root_logits.shape[-1] != self.root_expert_count:
            raise ValueError("root router output count does not match growth tree")
        root_gates, probabilities, root_indices = straight_through_top1(root_logits)
        gates: dict[str, torch.Tensor] = {}
        choices: dict[str, torch.Tensor] = {}
        for index, root in enumerate(self.roots):
            self._route_node(root, perception, root_gates[..., index], gates, choices)
        return gates, root_indices, choices, probabilities

    def get_extra_state(self) -> dict[str, Any]:
        return {"stage": self.stage, "structure": self.structure()}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.restore_structure(state["structure"])


def clone_module(module: nn.Module) -> nn.Module:
    """Small named helper used by growth and checkpoint tests."""

    return copy.deepcopy(module)


GrowthRouter = HierarchicalGrowthRouter
