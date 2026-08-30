from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .language_conditional_recruitment import (
    RecruitmentForward,
    _base_graph_weights,
    _conditional_diffusion,
    _gated_replicator_activity,
)
from .language_growing_organism import (
    ACTIVITY_BUDGET,
    STEP_SIZE,
    GrowingCellularLM,
    StructuralProbe,
    _relative_residual,
)
from .language_localized_learning import LocalizedLearningState
from .language_models import LanguageModelOutput


PRESSURE_RECRUITMENT_FLOOR = 0.02
PRESSURE_RECRUITMENT_TEMPERATURE = 8.0
PRESSURE_HOMEOSTATIC_QUANTILE = 0.95


@dataclass(frozen=True)
class PressureProfile:
    """Phase-1 old-tissue computational-pressure statistics.

    Statistics are indexed by recurrent step and global cell id. The sensor is
    calibrated and evaluated on a shadow organism containing only the retained
    Phase-1 tissue, so newborn activity cannot feed back into recruitment.
    """

    mean: torch.Tensor  # [steps, max_cells]
    scale: torch.Tensor  # [steps, max_cells]
    threshold: torch.Tensor  # [steps, max_cells]
    quantile: float = PRESSURE_HOMEOSTATIC_QUANTILE

    def to(self, device: torch.device) -> "PressureProfile":
        return PressureProfile(
            mean=self.mean.to(device),
            scale=self.scale.to(device),
            threshold=self.threshold.to(device),
            quantile=self.quantile,
        )


@dataclass
class PressureRecruitmentForward(RecruitmentForward):
    shadow_pressure_trace: torch.Tensor  # [steps, batch, length, max_cells]


def _shadow_layout(
    model: GrowingCellularLM,
    localized_state: LocalizedLearningState,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize a feedback-isolated copy of the exact Phase-1 organism."""
    base_alive = localized_state.base_alive.to(model.alive_mask.device)
    base_indices = torch.nonzero(base_alive, as_tuple=False).flatten()
    if base_indices.numel() < 1 or int(base_indices[0]) != 0:
        raise RuntimeError("Phase-1 interface cell 0 must remain present")
    memory = localized_state.base_memory.to(model.cell_memory.device).index_select(0, base_indices)
    batch, length = input_ids.shape
    positions = torch.arange(length, device=input_ids.device)
    token_state = model.token_embedding(input_ids) + model.position_embedding(positions)[None, :, :]
    state = memory[None, None, :, :].expand(batch, length, -1, -1).clone()
    state[:, :, 0, :] = state[:, :, 0, :] + token_state
    activity = torch.full(
        (batch, length, len(base_indices)),
        ACTIVITY_BUDGET / len(base_indices),
        device=input_ids.device,
        dtype=torch.float32,
    )
    weights = _base_graph_weights(model, base_indices, localized_state).to(input_ids.device)
    return state, memory, activity, weights


def _shadow_step(
    model: GrowingCellularLM,
    state: torch.Tensor,
    memory: torch.Tensor,
    activity: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reaction = model.rule(state, memory)
    diffusion = model._diffusion(state, weights)
    # Pressure is the local unresolved update demand of the old-only organism.
    pressure = (reaction + diffusion).float().square().mean(dim=-1).add(1e-12).sqrt()
    activity = model._replicator_activity(activity, reaction)
    relative = activity / (ACTIVITY_BUDGET / state.shape[2])
    gain = relative.clamp(0.05, 3.0).unsqueeze(-1).to(state.dtype)
    next_state = state + STEP_SIZE * gain * (reaction + diffusion)
    return next_state, activity, pressure


@torch.no_grad()
def shadow_pressure_trace(
    model: GrowingCellularLM,
    input_ids: torch.Tensor,
    localized_state: LocalizedLearningState,
    *,
    iterations: int | None = None,
) -> torch.Tensor:
    """Return old-only pressure; newborn phenotype/edges cannot affect it."""
    steps = model.iterations if iterations is None else int(iterations)
    if steps < 1:
        raise ValueError("iterations must be positive")
    shadow, memory, activity, weights = _shadow_layout(model, localized_state, input_ids)
    base_indices = torch.nonzero(localized_state.base_alive.to(model.alive_mask.device), as_tuple=False).flatten()
    traces: list[torch.Tensor] = []
    for _ in range(steps):
        shadow, activity, pressure = _shadow_step(model, shadow, memory, activity, weights)
        global_pressure = pressure.new_zeros(pressure.shape[0], pressure.shape[1], model.max_cells)
        global_pressure[..., base_indices] = pressure
        traces.append(global_pressure)
    return torch.stack(traces)


@torch.no_grad()
def calibrate_pressure_homeostasis(
    model: GrowingCellularLM,
    localized_state: LocalizedLearningState,
    input_batches: Iterable[torch.Tensor],
    *,
    quantile: float = PRESSURE_HOMEOSTATIC_QUANTILE,
) -> PressureProfile:
    """Calibrate old-only local update demand on retained Phase-1 language."""
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must be in (0.5, 1.0)")
    model.eval()
    alive = [int(v) for v in torch.nonzero(localized_state.base_alive, as_tuple=False).flatten().tolist()]
    samples: list[dict[int, list[torch.Tensor]]] = [
        {cell: [] for cell in alive} for _ in range(model.iterations)
    ]
    seen = 0
    for inputs in input_batches:
        trace = shadow_pressure_trace(model, inputs.to(model.cell_memory.device), localized_state)
        trace = trace.detach().float().cpu()
        seen += 1
        for step in range(model.iterations):
            for cell in alive:
                samples[step][cell].append(trace[step, :, :, cell].reshape(-1))
    if seen == 0 or not all(samples[step][cell] for step in range(model.iterations) for cell in alive):
        raise RuntimeError("pressure calibration received incomplete Phase-1 samples")

    mean = torch.zeros(model.iterations, model.max_cells, dtype=torch.float32)
    scale = torch.ones(model.iterations, model.max_cells, dtype=torch.float32)
    threshold = torch.full((model.iterations, model.max_cells), float("inf"), dtype=torch.float32)
    for step in range(model.iterations):
        for cell in alive:
            values = torch.cat(samples[step][cell])
            cell_mean = values.mean()
            cell_scale = values.std(unbiased=False).clamp_min(1e-4)
            mean[step, cell] = cell_mean
            scale[step, cell] = cell_scale
            threshold[step, cell] = torch.quantile(values, quantile)
    return PressureProfile(mean=mean, scale=scale, threshold=threshold, quantile=quantile)


def _pressure_gates(
    model: GrowingCellularLM,
    alive_indices: torch.Tensor,
    localized_state: LocalizedLearningState,
    profile: PressureProfile,
    pressure: torch.Tensor,
    *,
    step_index: int,
    force_recruitment: float | None,
) -> torch.Tensor:
    batch, length, _ = pressure.shape
    gates = torch.ones(batch, length, len(alive_indices), device=pressure.device, dtype=torch.float32)
    newborn = localized_state.newborn_cells(model)
    if not newborn:
        return gates
    global_to_local = {int(cell): index for index, cell in enumerate(alive_indices.tolist())}
    for child in newborn:
        local_child = global_to_local.get(int(child))
        if local_child is None:
            continue
        if force_recruitment is not None:
            gate = torch.full(
                (batch, length),
                float(force_recruitment),
                device=pressure.device,
                dtype=torch.float32,
            ).clamp(0.0, 1.0)
        else:
            parent = int(model.parent[child].item())
            threshold = profile.threshold[step_index, parent].to(pressure.device)
            scale = profile.scale[step_index, parent].to(pressure.device)
            standardized_excess = (pressure[..., parent] - threshold) / scale.clamp_min(1e-4)
            gate = PRESSURE_RECRUITMENT_FLOOR + (1.0 - PRESSURE_RECRUITMENT_FLOOR) * torch.sigmoid(
                PRESSURE_RECRUITMENT_TEMPERATURE * standardized_excess
            )
        gates[..., local_child] = gate
    return gates


def forward_with_pressure_recruitment(
    model: GrowingCellularLM,
    input_ids: torch.Tensor,
    localized_state: LocalizedLearningState,
    profile: PressureProfile,
    *,
    iterations: int | None = None,
    edge_probe: torch.Tensor | None = None,
    force_recruitment: float | None = None,
) -> PressureRecruitmentForward:
    """Recruit newborn tissue from feedback-isolated old-tissue pressure."""
    if input_ids.ndim != 2 or input_ids.shape[1] > model.max_context:
        raise ValueError(f"input_ids must be [batch, <= {model.max_context}]")
    steps = model.iterations if iterations is None else int(iterations)
    if steps < 1 or steps > profile.mean.shape[0]:
        raise ValueError("iterations must be positive and covered by the pressure profile")

    alive_indices = torch.nonzero(model.alive_mask, as_tuple=False).flatten()
    if alive_indices.numel() < 1 or int(alive_indices[0]) != 0:
        raise RuntimeError("interface cell 0 must remain alive")
    memory = model.cell_memory.index_select(0, alive_indices)
    batch, length = input_ids.shape
    positions = torch.arange(length, device=input_ids.device)
    token_state = model.token_embedding(input_ids) + model.position_embedding(positions)[None, :, :]
    state = memory[None, None, :, :].expand(batch, length, -1, -1).clone()
    state[:, :, 0, :] = state[:, :, 0, :] + token_state

    full_weights = model._graph_weights(alive_indices, edge_probe).to(input_ids.device)
    base_weights = _base_graph_weights(model, alive_indices, localized_state).to(input_ids.device)
    global_alive = alive_indices.tolist()
    newborn_positions = [global_alive.index(cell) for cell in localized_state.newborn_cells(model) if cell in global_alive]

    shadow, shadow_memory, shadow_activity, shadow_weights = _shadow_layout(model, localized_state, input_ids)
    base_indices = torch.nonzero(localized_state.base_alive.to(model.alive_mask.device), as_tuple=False).flatten()
    profile = profile.to(input_ids.device)
    activity: torch.Tensor | None = None
    recruitment_traces: list[torch.Tensor] = []
    pressure_traces: list[torch.Tensor] = []
    last_before = state

    for step_index in range(steps):
        last_before = state
        shadow, shadow_activity, local_pressure = _shadow_step(
            model, shadow, shadow_memory, shadow_activity, shadow_weights
        )
        global_pressure = local_pressure.new_zeros(batch, length, model.max_cells)
        global_pressure[..., base_indices] = local_pressure
        gates = _pressure_gates(
            model,
            alive_indices,
            localized_state,
            profile,
            global_pressure,
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
        recruitment_traces.append(gates)
        pressure_traces.append(global_pressure)

    logits = model.lm_head(model.final_norm(state[:, :, 0, :]))
    stability = _relative_residual(last_before, state)
    return PressureRecruitmentForward(
        output=LanguageModelOutput(logits),
        stability_loss=stability,
        recruitment_trace=torch.stack(recruitment_traces),
        final_state=state,
        shadow_pressure_trace=torch.stack(pressure_traces),
    )


def newborn_recruitment_mean(
    result: PressureRecruitmentForward,
    model: GrowingCellularLM,
    localized_state: LocalizedLearningState,
) -> torch.Tensor:
    newborn = localized_state.newborn_cells(model)
    if not newborn:
        return result.recruitment_trace.new_zeros(())
    alive = torch.nonzero(model.alive_mask, as_tuple=False).flatten().tolist()
    positions = [alive.index(cell) for cell in newborn if cell in alive]
    if not positions:
        return result.recruitment_trace.new_zeros(())
    return result.recruitment_trace[..., positions].float().mean()


def newborn_parent_pressure_mean(
    result: PressureRecruitmentForward,
    model: GrowingCellularLM,
    localized_state: LocalizedLearningState,
) -> torch.Tensor:
    newborn = localized_state.newborn_cells(model)
    if not newborn:
        return result.shadow_pressure_trace.new_zeros(())
    parents = sorted({int(model.parent[cell].item()) for cell in newborn})
    return result.shadow_pressure_trace[..., parents].float().mean()


def make_pressure_recruitment_probe(
    model: GrowingCellularLM,
    localized_state: LocalizedLearningState,
    profile: PressureProfile,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    loss_fn,
) -> StructuralProbe:
    """Edge utility under pressure-gated dynamics; extra forks are disabled in 018b."""
    if not microbatches:
        raise ValueError("microbatches must not be empty")
    device = model.cell_memory.device
    edge_probe = torch.zeros(model.max_cells, model.max_cells, device=device, requires_grad=True)
    inputs = torch.cat([item[0] for item in microbatches], dim=0)
    targets = torch.cat([item[1] for item in microbatches], dim=0)
    result = forward_with_pressure_recruitment(
        model, inputs, localized_state, profile, edge_probe=edge_probe
    )
    loss = loss_fn(result.output.logits, targets)
    edge_grad = torch.autograd.grad(loss, edge_probe, retain_graph=False)[0]
    edge_utility = -edge_grad.detach()

    memory_gradients = []
    for inputs_mb, targets_mb in microbatches:
        result_mb = forward_with_pressure_recruitment(model, inputs_mb, localized_state, profile)
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
