from __future__ import annotations

import contextlib
import copy
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class FunctionalCellizationError(RuntimeError):
    """Raised when the functional Cell overlay cannot be applied safely."""


@dataclass(frozen=True)
class RouteSummary:
    primary_cell: torch.Tensor
    probabilities: torch.Tensor
    active_cells: tuple[int, ...]


@dataclass(frozen=True)
class CellMutation:
    cell_index: int
    state: dict[str, torch.Tensor]


def _clone_tensor_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


class FunctionalCellOverlay(nn.Module):
    """Cross-layer sparse residual Cells over a frozen transformer.

    Routing is computed once at the first write site and reused by all later sites
    in the same forward pass. ``up`` starts at exactly zero, making installation
    functionally identical to the frozen foundation before Cell training.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        layer_indices: Sequence[int],
        max_cells: int = 16,
        initial_active_cells: int = 8,
        rank: int = 8,
        temperature: float = 0.35,
        top_k: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or rank <= 0 or max_cells <= 0:
            raise ValueError("hidden_size, rank and max_cells must be positive")
        if not layer_indices:
            raise ValueError("at least one layer index is required")
        if tuple(sorted(set(layer_indices))) != tuple(layer_indices):
            raise ValueError("layer_indices must be strictly increasing and unique")
        if not 1 <= initial_active_cells <= max_cells:
            raise ValueError("initial_active_cells must be in [1, max_cells]")
        if not 1 <= top_k <= initial_active_cells:
            raise ValueError("top_k must be in [1, initial_active_cells]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.hidden_size = int(hidden_size)
        self.layer_indices = tuple(int(value) for value in layer_indices)
        self.max_cells = int(max_cells)
        self.rank = int(rank)
        self.temperature = float(temperature)
        self.top_k = int(top_k)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        keys = torch.randn(self.max_cells, self.hidden_size, generator=generator)
        down = torch.randn(
            len(self.layer_indices),
            self.max_cells,
            self.hidden_size,
            self.rank,
            generator=generator,
        ) / (self.hidden_size**0.5)
        self.keys = nn.Parameter(F.normalize(keys, dim=-1))
        self.down = nn.Parameter(down)
        self.up = nn.Parameter(
            torch.zeros(
                len(self.layer_indices),
                self.max_cells,
                self.rank,
                self.hidden_size,
            )
        )
        active = torch.zeros(self.max_cells, dtype=torch.bool)
        active[:initial_active_cells] = True
        self.register_buffer("active_mask", active)

        self._cached_route: torch.Tensor | None = None
        self._cached_soft_route: torch.Tensor | None = None
        self._installed_handles: list[Any] = []

    @property
    def active_cells(self) -> tuple[int, ...]:
        return tuple(
            int(index)
            for index in torch.nonzero(self.active_mask, as_tuple=False).flatten().tolist()
        )

    def clear_forward_cache(self) -> None:
        self._cached_route = None
        self._cached_soft_route = None

    def _routing_probabilities(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise FunctionalCellizationError(
                f"expected [batch,sequence,{self.hidden_size}] hidden states, "
                f"found {tuple(hidden.shape)}"
            )
        query = F.normalize(hidden.float(), dim=-1)
        keys = F.normalize(self.keys.float(), dim=-1)
        logits = torch.einsum("btd,kd->btk", query, keys) / self.temperature
        logits = logits.masked_fill(~self.active_mask.view(1, 1, -1), -torch.inf)
        return F.softmax(logits, dim=-1).to(hidden.dtype)

    def compute_route(self, hidden: torch.Tensor, *, hard: bool = True) -> RouteSummary:
        soft = self._routing_probabilities(hidden)
        active = self.active_cells
        if not hard:
            route = soft
        else:
            count = min(self.top_k, len(active))
            values, indices = torch.topk(soft, k=count, dim=-1)
            hard_route = torch.zeros_like(soft).scatter_(-1, indices, values)
            hard_route = hard_route / hard_route.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            route = hard_route + soft - soft.detach()
        return RouteSummary(
            primary_cell=soft.argmax(dim=-1),
            probabilities=route,
            active_cells=active,
        )

    def _transform(self, hidden: torch.Tensor, route: torch.Tensor, site: int) -> torch.Tensor:
        work = hidden.float()
        down = self.down[site].float()
        up = self.up[site].float()
        low = F.silu(torch.einsum("btd,kdr->btkr", work, down))
        delta = torch.einsum("btkr,krd->btkd", low, up)
        transformed = work + torch.einsum("btk,btkd->btd", route.float(), delta)
        return transformed.to(hidden.dtype)

    def forward_site(self, hidden: torch.Tensor, site: int) -> torch.Tensor:
        if site == 0:
            summary = self.compute_route(hidden, hard=True)
            self._cached_route = summary.probabilities
            self._cached_soft_route = self._routing_probabilities(hidden)
        elif self._cached_route is None:
            raise FunctionalCellizationError(
                "write site executed before the read site; model layer ordering changed"
            )
        route = self._cached_route
        assert route is not None
        if route.shape[:2] != hidden.shape[:2]:
            raise FunctionalCellizationError(
                "cached route shape differs from downstream hidden-state shape"
            )
        return self._transform(hidden, route, site)

    def prompt_routes(self, positions: torch.Tensor) -> RouteSummary:
        if self._cached_soft_route is None:
            raise FunctionalCellizationError("no routing trace is available")
        if positions.ndim != 1 or positions.numel() != self._cached_soft_route.shape[0]:
            raise FunctionalCellizationError("prompt positions do not match route batch")
        rows = torch.arange(positions.numel(), device=positions.device)
        soft = self._cached_soft_route[rows, positions]
        return RouteSummary(
            primary_cell=soft.argmax(dim=-1),
            probabilities=soft,
            active_cells=self.active_cells,
        )

    def next_inactive_cell(self) -> int:
        inactive = torch.nonzero(~self.active_mask, as_tuple=False).flatten()
        if not inactive.numel():
            raise FunctionalCellizationError("no inactive Cell capacity remains")
        return int(inactive[0].item())

    @torch.no_grad()
    def spawn_child(self, parent_index: int, *, child_index: int | None = None) -> int:
        if parent_index not in self.active_cells:
            raise FunctionalCellizationError("parent Cell must be active")
        child = self.next_inactive_cell() if child_index is None else int(child_index)
        if self.active_mask[child]:
            raise FunctionalCellizationError("child Cell is already active")
        self.keys[child].copy_(self.keys[parent_index])
        self.down[:, child].copy_(self.down[:, parent_index])
        self.up[:, child].copy_(self.up[:, parent_index])
        self.active_mask[child] = True
        return child

    def cell_state(self, cell_index: int) -> dict[str, torch.Tensor]:
        if not 0 <= cell_index < self.max_cells:
            raise IndexError(cell_index)
        return _clone_tensor_dict(
            {
                "key": self.keys[cell_index],
                "down": self.down[:, cell_index],
                "up": self.up[:, cell_index],
                "active": self.active_mask[cell_index],
            }
        )

    @torch.no_grad()
    def load_cell_state_(self, cell_index: int, state: dict[str, torch.Tensor]) -> None:
        expected = {"key", "down", "up", "active"}
        if set(state) != expected:
            raise FunctionalCellizationError(
                f"unexpected Cell state fields: {sorted(state)}"
            )
        device = self.keys.device
        self.keys[cell_index].copy_(state["key"].to(device=device, dtype=self.keys.dtype))
        self.down[:, cell_index].copy_(
            state["down"].to(device=device, dtype=self.down.dtype)
        )
        self.up[:, cell_index].copy_(state["up"].to(device=device, dtype=self.up.dtype))
        self.active_mask[cell_index] = bool(state["active"].item())

    def export_mutation(self, cell_index: int) -> CellMutation:
        return CellMutation(cell_index=cell_index, state=self.cell_state(cell_index))

    @torch.no_grad()
    def apply_mutation_(self, mutation: CellMutation) -> None:
        self.load_cell_state_(mutation.cell_index, mutation.state)

    def snapshot(self) -> dict[str, torch.Tensor]:
        return _clone_tensor_dict(self.state_dict())

    def restore_(self, snapshot: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(copy.deepcopy(snapshot), strict=True)

    def zero_output_is_exact(self) -> bool:
        return bool(torch.count_nonzero(self.up.detach()).item() == 0)

    @contextlib.contextmanager
    def installed(self, model: nn.Module) -> Iterator[FunctionalCellOverlay]:
        if self._installed_handles:
            raise FunctionalCellizationError("overlay hooks are already installed")
        try:
            for site, layer_index in enumerate(self.layer_indices):
                try:
                    module = model.get_submodule(f"model.layers.{layer_index}")
                except AttributeError as exc:
                    raise FunctionalCellizationError(
                        f"foundation has no model.layers.{layer_index}"
                    ) from exc

                def hook(
                    _module: nn.Module,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    *,
                    site_index: int = site,
                ) -> Any:
                    hidden = output[0] if isinstance(output, tuple) else output
                    if not torch.is_tensor(hidden):
                        raise FunctionalCellizationError(
                            "decoder layer did not return a tensor hidden state"
                        )
                    transformed = self.forward_site(hidden, site_index)
                    if isinstance(output, tuple):
                        return (transformed, *output[1:])
                    return transformed

                self._installed_handles.append(module.register_forward_hook(hook))
            yield self
        finally:
            for handle in self._installed_handles:
                handle.remove()
            self._installed_handles.clear()
            self.clear_forward_cache()


def freeze_foundation_(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def mask_cell_gradients_(
    overlay: FunctionalCellOverlay,
    cell_indices: Sequence[int],
    *,
    include_keys: bool,
) -> None:
    selected = torch.zeros(
        overlay.max_cells, dtype=torch.bool, device=overlay.keys.device
    )
    selected[[int(value) for value in cell_indices]] = True
    for parameter in (overlay.down, overlay.up):
        if parameter.grad is not None:
            parameter.grad[:, ~selected] = 0
    if overlay.keys.grad is not None:
        if include_keys:
            overlay.keys.grad[~selected] = 0
        else:
            overlay.keys.grad.zero_()


def disjoint_mutations(*mutations: CellMutation) -> tuple[CellMutation, ...]:
    seen: set[int] = set()
    for mutation in mutations:
        if mutation.cell_index in seen:
            raise FunctionalCellizationError(
                f"overlapping mutation for Cell {mutation.cell_index}"
            )
        seen.add(mutation.cell_index)
    return mutations
