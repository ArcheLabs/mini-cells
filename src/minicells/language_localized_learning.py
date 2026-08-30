from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .language_growing_organism import FORK_PERTURB, GrowingCellularLM, StructuralEvent, StructuralProbe


MAX_LOCAL_NEWBORNS = 3
LOCAL_EMA_BETA = 0.70
LOCAL_CONNECT_SCORE = 0.75
LOCAL_PRUNE_SCORE = -0.50
LOCAL_REWRITE_PRESSURE_RATIO = 0.50
LOCAL_PRESSURE_Z = 0.50
LOCAL_PERSISTENCE = 1
LOCAL_FORK_COOLDOWN = 2


@dataclass(frozen=True)
class LocalizedLearningState:
    base_alive: torch.Tensor
    base_adjacency: torch.Tensor
    base_memory: torch.Tensor

    @classmethod
    def capture(cls, model: GrowingCellularLM) -> "LocalizedLearningState":
        return cls(
            base_alive=model.alive_mask.detach().clone(),
            base_adjacency=model.adjacency.detach().clone(),
            base_memory=model.cell_memory.detach().clone(),
        )

    def newborn_mask(self, model: GrowingCellularLM) -> torch.Tensor:
        return model.alive_mask & ~self.base_alive.to(model.alive_mask.device)

    def newborn_cells(self, model: GrowingCellularLM) -> list[int]:
        return [int(v) for v in torch.nonzero(self.newborn_mask(model), as_tuple=False).flatten().tolist()]

    def base_memory_drift(self, model: GrowingCellularLM) -> torch.Tensor:
        mask = self.base_alive.to(model.cell_memory.device)
        delta = (model.cell_memory.detach() - self.base_memory.to(model.cell_memory.device)).float().norm(dim=-1)
        return delta[mask].max() if bool(mask.any()) else delta.new_zeros(())


@torch.no_grad()
def conservative_fork(
    model: GrowingCellularLM,
    parent: int,
    *,
    step: int,
    direction: torch.Tensor,
) -> int | None:
    """Fork new capacity without changing the protected parent's phenotype.

    Experiment 016 used a symmetric parent/child split. For localized continual
    learning the old tissue is the retained capability, so the child inherits a
    perturbed phenotype while the parent remains bit-identical.
    """
    if not model.variant.structural_plasticity or parent == 0 or not bool(model.alive_mask[parent]):
        return None
    inactive = torch.nonzero(~model.alive_mask, as_tuple=False).flatten()
    if inactive.numel() == 0:
        return None
    child = int(inactive[0].item())
    vector = direction.to(device=model.cell_memory.device, dtype=model.cell_memory.dtype)
    vector = vector / vector.norm().clamp_min(1e-8)
    center = model.cell_memory[parent].clone()
    model.cell_memory[child].copy_(center + FORK_PERTURB * vector)
    model.alive_mask[child] = True
    model.parent[child] = parent
    model.birth_step[child] = int(step)
    model.adjacency[parent, child] = True
    model.adjacency[child, parent] = True
    model.protected_edges[parent, child] = True
    model.protected_edges[child, parent] = True
    return child


def mask_to_newborn_gradients(model: GrowingCellularLM, state: LocalizedLearningState) -> None:
    """Divert phenotype learning into newborn tissue only."""
    if model.cell_memory.grad is None:
        return
    newborn = state.newborn_mask(model).to(model.cell_memory.grad.device)
    model.cell_memory.grad.mul_(newborn[:, None].to(model.cell_memory.grad.dtype))


def _zscore(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() < 2:
        return torch.zeros_like(values)
    mean = selected.mean()
    std = selected.std(unbiased=False).clamp_min(1e-8)
    return (values - mean) / std


class LocalizedGrowthController:
    """Structural policy that protects old-old structure and grows a skill branch.

    Old cells remain valid information sources and receivers, but only edges
    touching newborn tissue may be added or pruned. Persistent counterfactual
    rewrite pressure on frozen old phenotype triggers a conservative fork, but
    useful communication is always attempted first.
    """

    def __init__(
        self,
        state: LocalizedLearningState,
        *,
        max_newborns: int = MAX_LOCAL_NEWBORNS,
        ema_beta: float = LOCAL_EMA_BETA,
        connect_score: float = LOCAL_CONNECT_SCORE,
        prune_score: float = LOCAL_PRUNE_SCORE,
        rewrite_pressure_ratio: float = LOCAL_REWRITE_PRESSURE_RATIO,
        pressure_z: float = LOCAL_PRESSURE_Z,
        persistence: int = LOCAL_PERSISTENCE,
        fork_cooldown: int = LOCAL_FORK_COOLDOWN,
    ) -> None:
        self.state = state
        self.max_newborns = int(max_newborns)
        self.ema_beta = float(ema_beta)
        self.connect_score = float(connect_score)
        self.prune_score = float(prune_score)
        self.rewrite_pressure_ratio = float(rewrite_pressure_ratio)
        self.pressure_z = float(pressure_z)
        self.persistence = int(persistence)
        self.fork_cooldown = int(fork_cooldown)
        size = len(state.base_alive)
        self.utility_ema = torch.zeros(size, size)
        self.pressure_ema = torch.zeros(size)
        self.connect_streak = torch.zeros(size, size, dtype=torch.long)
        self.prune_streak = torch.zeros(size, size, dtype=torch.long)
        self.rewrite_streak = torch.zeros(size, dtype=torch.long)
        self.initial_rewrite_pressure: float | None = None
        self.probes_since_fork = fork_cooldown
        self.events: list[StructuralEvent] = []

    def _base_candidates(self, model: GrowingCellularLM) -> torch.Tensor:
        mask = self.state.base_alive.detach().cpu().clone()
        mask &= model.alive_mask.detach().cpu()
        mask[0] = False
        return mask

    def _update_ema(self, probe: StructuralProbe) -> None:
        beta = self.ema_beta
        self.utility_ema.mul_(beta).add_(probe.edge_utility.detach().float().cpu(), alpha=1.0 - beta)
        self.pressure_ema.mul_(beta).add_(probe.pressure.detach().float().cpu(), alpha=1.0 - beta)

    def allocate_initial(self, model: GrowingCellularLM, probe: StructuralProbe, *, step: int = 0) -> StructuralEvent:
        """Create one seed cell where the frozen organism most wants to rewrite itself."""
        self._update_ema(probe)
        base = self._base_candidates(model)
        if not bool(base.any()):
            raise RuntimeError("localized learning requires a non-interface base cell")
        cells = torch.nonzero(base, as_tuple=False).flatten()
        parent = int(cells[probe.pressure[cells].argmax()].item())
        initial_pressure = float(probe.pressure[parent])
        self.initial_rewrite_pressure = max(initial_pressure, 1e-12)
        child = conservative_fork(model, parent, step=step, direction=probe.split_direction[parent])
        if child is None:
            raise RuntimeError("failed to allocate initial localized-learning cell")
        event = StructuralEvent(
            step=step,
            event="localized_fork",
            parent=parent,
            child=child,
            pressure_score=initial_pressure,
            conflict=float(probe.conflict[parent]),
        )
        self.events.append(event)
        self.probes_since_fork = 0
        return event

    def apply(self, model: GrowingCellularLM, probe: StructuralProbe, *, step: int) -> list[StructuralEvent]:
        if self.initial_rewrite_pressure is None:
            raise RuntimeError("allocate_initial must be called before apply")
        self._update_ema(probe)
        alive = model.alive_mask.detach().cpu()
        newborn = self.state.newborn_mask(model).detach().cpu()
        base = self._base_candidates(model)
        candidate = alive[:, None] & alive[None, :]
        candidate.fill_diagonal_(False)
        touches_newborn = newborn[:, None] | newborn[None, :]
        structural_candidate = candidate & touches_newborn
        utility_z = _zscore(self.utility_ema, structural_candidate)
        new_events: list[StructuralEvent] = []

        absent = structural_candidate & ~model.adjacency.detach().cpu()
        self.connect_streak = torch.where(
            absent & (utility_z > self.connect_score) & (self.utility_ema > 0),
            self.connect_streak + 1,
            torch.zeros_like(self.connect_streak),
        )
        if bool((self.connect_streak >= self.persistence).any()):
            scores = utility_z.masked_fill(self.connect_streak < self.persistence, float("-inf"))
            flat = int(scores.argmax().item())
            receiver, source = divmod(flat, model.max_cells)
            if model.connect(receiver, source):
                event = StructuralEvent(
                    step=step,
                    event="localized_connect",
                    receiver=receiver,
                    source=source,
                    utility_score=float(utility_z[receiver, source]),
                )
                self.events.append(event)
                new_events.append(event)
                self.connect_streak[receiver, source] = 0

        learned = model.adjacency.detach().cpu() & ~model.protected_edges.detach().cpu() & structural_candidate
        self.prune_streak = torch.where(
            learned & (utility_z < self.prune_score) & (self.utility_ema < 0),
            self.prune_streak + 1,
            torch.zeros_like(self.prune_streak),
        )
        if bool((self.prune_streak >= self.persistence).any()):
            scores = utility_z.masked_fill(self.prune_streak < self.persistence, float("inf"))
            flat = int(scores.argmin().item())
            receiver, source = divmod(flat, model.max_cells)
            if model.prune(receiver, source):
                event = StructuralEvent(
                    step=step,
                    event="localized_prune",
                    receiver=receiver,
                    source=source,
                    utility_score=float(utility_z[receiver, source]),
                )
                self.events.append(event)
                new_events.append(event)
                self.prune_streak[receiver, source] = 0

        self.probes_since_fork += 1
        connect_happened = any(event.event == "localized_connect" for event in new_events)
        newborn_count = int(newborn.sum().item())
        can_fork = newborn_count < self.max_newborns and self.probes_since_fork >= self.fork_cooldown
        if can_fork and not connect_happened and bool(base.any()):
            pressure_z = _zscore(self.pressure_ema, base)
            ratio = self.pressure_ema / self.initial_rewrite_pressure
            rewrite = base & (pressure_z > self.pressure_z) & (ratio > self.rewrite_pressure_ratio)
            self.rewrite_streak = torch.where(rewrite, self.rewrite_streak + 1, torch.zeros_like(self.rewrite_streak))
            eligible = rewrite & (self.rewrite_streak >= self.persistence)
            if bool(eligible.any()):
                cells = torch.nonzero(eligible, as_tuple=False).flatten()
                parent = int(cells[self.pressure_ema[cells].argmax()].item())
                child = conservative_fork(model, parent, step=step, direction=probe.split_direction[parent])
                if child is not None:
                    event = StructuralEvent(
                        step=step,
                        event="localized_fork",
                        parent=parent,
                        child=child,
                        pressure_score=float(self.pressure_ema[parent]),
                        conflict=float(probe.conflict[parent]),
                    )
                    self.events.append(event)
                    new_events.append(event)
                    self.rewrite_streak[parent] = 0
                    self.probes_since_fork = 0
        return new_events


@torch.no_grad()
def graft_localized_tissue(
    recipient: GrowingCellularLM,
    donor: GrowingCellularLM,
    cells: Iterable[int],
) -> None:
    """Graft newborn phenotype plus its boundary edges without copying old phenotype."""
    selected = sorted(set(int(cell) for cell in cells))
    selected_set = set(selected)
    for cell in selected:
        if cell <= 0 or cell >= donor.max_cells or not bool(donor.alive_mask[cell]):
            raise ValueError("graft cells must be live non-interface donor cells")
        recipient.cell_memory[cell].copy_(donor.cell_memory[cell])
        recipient.alive_mask[cell] = True
        recipient.parent[cell] = donor.parent[cell]
        recipient.birth_step[cell] = donor.birth_step[cell]
    donor_alive = [int(v) for v in torch.nonzero(donor.alive_mask, as_tuple=False).flatten().tolist()]
    for receiver in donor_alive:
        for source in donor_alive:
            if receiver not in selected_set and source not in selected_set:
                continue
            if receiver not in selected_set and not bool(recipient.alive_mask[receiver]):
                continue
            if source not in selected_set and not bool(recipient.alive_mask[source]):
                continue
            recipient.adjacency[receiver, source] = donor.adjacency[receiver, source]
            recipient.protected_edges[receiver, source] = donor.protected_edges[receiver, source]


@torch.no_grad()
def set_newborn_tissue_active(model: GrowingCellularLM, state: LocalizedLearningState, active: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Temporarily ablate/restore newborn tissue; returns snapshots for restoration."""
    saved_alive = model.alive_mask.clone()
    saved_adjacency = model.adjacency.clone()
    newborn = state.newborn_mask(model)
    if not active:
        model.alive_mask[newborn] = False
        model.adjacency[newborn, :] = False
        model.adjacency[:, newborn] = False
    return saved_alive, saved_adjacency


@torch.no_grad()
def restore_structure(model: GrowingCellularLM, saved_alive: torch.Tensor, saved_adjacency: torch.Tensor) -> None:
    model.alive_mask.copy_(saved_alive)
    model.adjacency.copy_(saved_adjacency)
