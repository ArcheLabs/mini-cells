from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .language_growing_organism import (
    ACTIVITY_BUDGET,
    ACTIVITY_RATE,
    EDGE_COUPLING,
    STEP_SIZE,
    GrowingCellularLM,
    StructuralProbe,
    _relative_residual,
)
from .language_localized_learning import LocalizedLearningState
from .language_models import LanguageModelOutput


RECRUITMENT_FLOOR = 0.02
RECRUITMENT_TEMPERATURE = 10.0
HOMEOSTATIC_QUANTILE = 0.95


@dataclass(frozen=True)
class HomeostaticProfile:
    """Normal Phase-1 state statistics indexed by recurrent step and cell."""

    mean: torch.Tensor  # [steps, max_cells, dim]
    scale: torch.Tensor  # [steps, max_cells, dim]
    threshold: torch.Tensor  # [steps, max_cells]
    quantile: float = HOMEOSTATIC_QUANTILE

    def to(self, device: torch.device) -> "HomeostaticProfile":
        return HomeostaticProfile(
            mean=self.mean.to(device),
            scale=self.scale.to(device),
            threshold=self.threshold.to(device),
            quantile=self.quantile,
        )


@dataclass
class RecruitmentForward:
    output: LanguageModelOutput
    stability_loss: torch.Tensor
    recruitment_trace: torch.Tensor  # [steps, batch, length, alive_cells]
    final_state: torch.Tensor


@torch.no_grad()
def _base_state_trace(model: GrowingCellularLM, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    alive_indices = torch.nonzero(model.alive_mask, as_tuple=False).flatten()
    memory = model.cell_memory.index_select(0, alive_indices)
    batch, length = input_ids.shape
    positions = torch.arange(length, device=input_ids.device)
    token_state = model.token_embedding(input_ids) + model.position_embedding(positions)[None, :, :]
    state = memory[None, None, :, :].expand(batch, length, -1, -1).clone()
    state[:, :, 0, :] = state[:, :, 0, :] + token_state
    activity = torch.full(
        (batch, length, len(alive_indices)),
        ACTIVITY_BUDGET / len(alive_indices),
        device=input_ids.device,
        dtype=torch.float32,
    )
    weights = model._graph_weights(alive_indices, None).to(device=input_ids.device)
    traces: list[torch.Tensor] = []
    for _ in range(model.iterations):
        traces.append(state.detach().float())
        reaction = model.rule(state, memory)
        activity = model._replicator_activity(activity, reaction)
        diffusion = model._diffusion(state, weights)
        relative = activity / (ACTIVITY_BUDGET / len(alive_indices))
        gain = relative.clamp(0.05, 3.0).unsqueeze(-1).to(state.dtype)
        state = state + STEP_SIZE * gain * (reaction + diffusion)
    return torch.stack(traces), alive_indices


@torch.no_grad()
def calibrate_homeostasis(
    model: GrowingCellularLM,
    input_batches: Iterable[torch.Tensor],
    *,
    quantile: float = HOMEOSTATIC_QUANTILE,
) -> HomeostaticProfile:
    """Estimate per-step local homeostasis from retained language before adaptation."""
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must be in (0.5, 1.0)")
    model.eval()
    device = model.cell_memory.device
    alive = [int(v) for v in torch.nonzero(model.alive_mask, as_tuple=False).flatten().tolist()]
    samples: list[dict[int, list[torch.Tensor]]] = [{cell: [] for cell in alive} for _ in range(model.iterations)]
    for inputs in input_batches:
        trace, alive_indices = _base_state_trace(model, inputs.to(device))
        local_alive = alive_indices.cpu().tolist()
        trace = trace.cpu()
        for step in range(model.iterations):
            for local_index, cell in enumerate(local_alive):
                samples[step][int(cell)].append(trace[step, :, :, local_index, :].reshape(-1, model.dim))
    if not all(samples[step][cell] for step in range(model.iterations) for cell in alive):
        raise RuntimeError("homeostatic calibration received incomplete live-cell samples")

    mean = torch.zeros(model.iterations, model.max_cells, model.dim, dtype=torch.float32)
    scale = torch.ones(model.iterations, model.max_cells, model.dim, dtype=torch.float32)
    threshold = torch.full((model.iterations, model.max_cells), float("inf"), dtype=torch.float32)
    for step in range(model.iterations):
        for cell in alive:
            values = torch.cat(samples[step][cell], dim=0)
            cell_mean = values.mean(dim=0)
            cell_scale = values.std(dim=0, unbiased=False).clamp_min(1e-3)
            novelty = ((values - cell_mean) / cell_scale).square().mean(dim=-1).sqrt()
            mean[step, cell] = cell_mean
            scale[step, cell] = cell_scale
            threshold[step, cell] = torch.quantile(novelty, quantile)
    return HomeostaticProfile(mean=mean, scale=scale, threshold=threshold, quantile=quantile)


def _base_graph_weights(
    model: GrowingCellularLM,
    alive_indices: torch.Tensor,
    localized_state: LocalizedLearningState,
) -> torch.Tensor:
    adjacency = localized_state.base_adjacency.to(model.adjacency.device)
    adjacency = adjacency.index_select(0, alive_indices).index_select(1, alive_indices)
    scores = adjacency.to(dtype=model.cell_memory.dtype)
    return EDGE_COUPLING * scores / scores.sum(dim=-1, keepdim=True).clamp_min(1.0)


def _cell_recruitment(
    model: GrowingCellularLM,
    state: torch.Tensor,
    alive_indices: torch.Tensor,
    localized_state: LocalizedLearningState,
    profile: HomeostaticProfile,
    *,
    step_index: int,
    force_recruitment: float | None,
) -> torch.Tensor:
    batch, length, cells, _ = state.shape
    gates = torch.ones(batch, length, cells, device=state.device, dtype=torch.float32)
    newborn_global = localized_state.newborn_cells(model)
    if not newborn_global:
        return gates
    global_to_local = {int(cell): index for index, cell in enumerate(alive_indices.tolist())}
    for child in newborn_global:
        local_child = global_to_local.get(int(child))
        if local_child is None:
            continue
        if force_recruitment is not None:
            gate = torch.full((batch, length), float(force_recruitment), device=state.device, dtype=torch.float32).clamp(0.0, 1.0)
        else:
            parent = int(model.parent[child].item())
            local_parent = global_to_local.get(parent)
            if local_parent is None:
                raise RuntimeError("newborn parent must remain alive for local recruitment")
            parent_state = state[:, :, local_parent, :].float()
            mean = profile.mean[step_index, parent].to(state.device)
            scale = profile.scale[step_index, parent].to(state.device)
            threshold = profile.threshold[step_index, parent].to(state.device)
            novelty = ((parent_state - mean) / scale).square().mean(dim=-1).sqrt()
            gate = RECRUITMENT_FLOOR + (1.0 - RECRUITMENT_FLOOR) * torch.sigmoid(
                RECRUITMENT_TEMPERATURE * (novelty - threshold)
            )
        gates[:, :, local_child] = gate
    return gates


def _gated_replicator_activity(
    previous: torch.Tensor | None,
    reaction: torch.Tensor,
    availability: torch.Tensor,
) -> torch.Tensor:
    drive = reaction.float().square().mean(dim=-1).add(1e-8).sqrt()
    mass = availability.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    mean = (availability * drive).sum(dim=-1, keepdim=True) / mass
    variance = (availability * (drive - mean).square()).sum(dim=-1, keepdim=True) / mass
    fitness = (drive - mean) / variance.sqrt().clamp_min(1e-4)
    growth = torch.exp(ACTIVITY_RATE * fitness).clamp_max(20.0)
    prior = availability if previous is None else previous.float() * availability
    updated = prior * growth
    return ACTIVITY_BUDGET * updated / updated.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _conditional_diffusion(
    state: torch.Tensor,
    base_weights: torch.Tensor,
    full_weights: torch.Tensor,
    gates: torch.Tensor,
    newborn_positions: list[int],
) -> torch.Tensor:
    if not newborn_positions:
        dynamic = base_weights[None, None, :, :].expand(state.shape[0], state.shape[1], -1, -1)
    else:
        # The structural delta (birth edges + learned newborn edges) is itself a
        # conductance. At gate=0 the exact Phase-1 graph is restored; at gate=1
        # the current adapted graph is recovered.
        alpha = gates[..., newborn_positions].amax(dim=-1, keepdim=True)
        dynamic = base_weights[None, None, :, :] + alpha.unsqueeze(-1) * (full_weights - base_weights)[None, None, :, :]
    source = torch.einsum("blrs,blsd->blrd", dynamic, state)
    row_mass = dynamic.sum(dim=-1, keepdim=True)
    return source - row_mass * state


def forward_with_recruitment(
    model: GrowingCellularLM,
    input_ids: torch.Tensor,
    localized_state: LocalizedLearningState,
    profile: HomeostaticProfile,
    *,
    iterations: int | None = None,
    edge_probe: torch.Tensor | None = None,
    force_recruitment: float | None = None,
) -> RecruitmentForward:
    """Run local excitability over both communication and metabolic participation."""
    if input_ids.ndim != 2 or input_ids.shape[1] > model.max_context:
        raise ValueError(f"input_ids must be [batch, <= {model.max_context}]")
    steps = model.iterations if iterations is None else int(iterations)
    if steps < 1 or steps > profile.mean.shape[0]:
        raise ValueError("iterations must be positive and covered by the homeostatic profile")
    alive_indices = torch.nonzero(model.alive_mask, as_tuple=False).flatten()
    if alive_indices.numel() < 1 or int(alive_indices[0]) != 0:
        raise RuntimeError("interface cell 0 must remain alive")
    memory = model.cell_memory.index_select(0, alive_indices)
    batch, length = input_ids.shape
    positions = torch.arange(length, device=input_ids.device)
    token_state = model.token_embedding(input_ids) + model.position_embedding(positions)[None, :, :]
    state = memory[None, None, :, :].expand(batch, length, -1, -1).clone()
    state[:, :, 0, :] = state[:, :, 0, :] + token_state
    full_weights = model._graph_weights(alive_indices, edge_probe).to(device=input_ids.device)
    base_weights = _base_graph_weights(model, alive_indices, localized_state).to(device=input_ids.device)
    profile = profile.to(input_ids.device)
    global_alive = alive_indices.tolist()
    newborn_positions = [global_alive.index(cell) for cell in localized_state.newborn_cells(model) if cell in global_alive]
    activity: torch.Tensor | None = None
    traces: list[torch.Tensor] = []
    last_before = state
    for step_index in range(steps):
        last_before = state
        gates = _cell_recruitment(
            model,
            state,
            alive_indices,
            localized_state,
            profile,
            step_index=step_index,
            force_recruitment=force_recruitment,
        )
        reaction = model.rule(state, memory)
        activity = _gated_replicator_activity(activity, reaction, gates)
        diffusion = _conditional_diffusion(state, base_weights, full_weights, gates, newborn_positions)
        effective_cells = gates.sum(dim=-1).clamp_min(1.0)
        baseline_activity = ACTIVITY_BUDGET / effective_cells
        relative = activity / baseline_activity.unsqueeze(-1)
        gain = relative.clamp(0.0, 3.0).unsqueeze(-1).to(state.dtype)
        state = state + STEP_SIZE * gain * (reaction + diffusion)
        traces.append(gates)
    logits = model.lm_head(model.final_norm(state[:, :, 0, :]))
    stability = _relative_residual(last_before, state)
    return RecruitmentForward(
        output=LanguageModelOutput(logits),
        stability_loss=stability,
        recruitment_trace=torch.stack(traces),
        final_state=state,
    )


def newborn_recruitment_mean(result: RecruitmentForward, model: GrowingCellularLM, localized_state: LocalizedLearningState) -> torch.Tensor:
    newborn = localized_state.newborn_cells(model)
    if not newborn:
        return result.recruitment_trace.new_zeros(())
    alive = torch.nonzero(model.alive_mask, as_tuple=False).flatten().tolist()
    positions = [alive.index(cell) for cell in newborn if cell in alive]
    if not positions:
        return result.recruitment_trace.new_zeros(())
    return result.recruitment_trace[..., positions].float().mean()


def make_recruitment_probe(
    model: GrowingCellularLM,
    localized_state: LocalizedLearningState,
    profile: HomeostaticProfile,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    loss_fn,
) -> StructuralProbe:
    if not microbatches:
        raise ValueError("microbatches must not be empty")
    device = model.cell_memory.device
    edge_probe = torch.zeros(model.max_cells, model.max_cells, device=device, requires_grad=True)
    inputs = torch.cat([item[0] for item in microbatches], dim=0)
    targets = torch.cat([item[1] for item in microbatches], dim=0)
    result = forward_with_recruitment(model, inputs, localized_state, profile, edge_probe=edge_probe)
    loss = loss_fn(result.output.logits, targets)
    edge_grad = torch.autograd.grad(loss, edge_probe, retain_graph=False)[0]
    edge_utility = -edge_grad.detach()

    memory_gradients = []
    for inputs_mb, targets_mb in microbatches:
        result_mb = forward_with_recruitment(model, inputs_mb, localized_state, profile)
        loss_mb = loss_fn(result_mb.output.logits, targets_mb)
        grad = torch.autograd.grad(loss_mb, model.cell_memory, retain_graph=False)[0]
        memory_gradients.append(grad.detach())
    gradients = torch.stack(memory_gradients, dim=0).float()
    pressure = gradients.square().mean(dim=(0, 2)).sqrt()
    summed = gradients.sum(dim=0)
    numerator = summed.square().sum(dim=-1)
    denominator = gradients.square().sum(dim=-1).sum(dim=0) * gradients.shape[0]
    conflict = (1.0 - numerator / denominator.clamp_min(1e-12)).clamp(0.0, 1.0)
    directions = torch.zeros_like(model.cell_memory.detach().float())
    for cell in range(model.max_cells):
        centered = gradients[:, cell, :] - gradients[:, cell, :].mean(dim=0, keepdim=True)
        if float(centered.square().sum()) > 1e-12:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            directions[cell] = vh[0]
        else:
            directions[cell, cell % model.dim] = 1.0
    return StructuralProbe(edge_utility.cpu(), pressure.cpu(), conflict.cpu(), directions.cpu(), float(loss.detach()))
