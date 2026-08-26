from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .language_growing_organism import (
    ACTIVITY_BUDGET,
    STABILITY_WEIGHT,
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
    """Per-cell Phase-1 activity statistics used as a local novelty reference."""

    mean: torch.Tensor
    scale: torch.Tensor
    threshold: torch.Tensor
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
    recruitment_trace: torch.Tensor
    final_state: torch.Tensor

    @property
    def mean_recruitment(self) -> torch.Tensor:
        return self.recruitment_trace.float().mean()


@torch.no_grad()
def calibrate_homeostasis(
    model: GrowingCellularLM,
    input_batches: Iterable[torch.Tensor],
    *,
    quantile: float = HOMEOSTATIC_QUANTILE,
) -> HomeostaticProfile:
    """Estimate the normal Phase-1 state manifold for every currently live cell.

    The profile is calibrated before any adaptation cell exists. It is not a
    task classifier: it only records what each old cell normally looks like on
    the retained language distribution.
    """
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must be in (0.5, 1.0)")
    model.eval()
    device = model.cell_memory.device
    alive = torch.nonzero(model.alive_mask, as_tuple=False).flatten().tolist()
    samples: dict[int, list[torch.Tensor]] = {int(cell): [] for cell in alive}
    for inputs in input_batches:
        result = model.forward_variable(inputs.to(device), collect_observability=True)
        diagnostics = result.diagnostics
        assert diagnostics is not None
        states = diagnostics.cell_states.detach().float().cpu()
        local_alive = diagnostics.alive_indices.cpu().tolist()
        for local_index, cell in enumerate(local_alive):
            samples[int(cell)].append(states[:, :, local_index, :].reshape(-1, states.shape[-1]))
    if not all(samples[cell] for cell in alive):
        raise RuntimeError("homeostatic calibration received no samples for a live cell")

    mean = torch.zeros(model.max_cells, model.dim, dtype=torch.float32)
    scale = torch.ones(model.max_cells, model.dim, dtype=torch.float32)
    threshold = torch.full((model.max_cells,), float("inf"), dtype=torch.float32)
    for cell in alive:
        values = torch.cat(samples[cell], dim=0)
        cell_mean = values.mean(dim=0)
        cell_scale = values.std(dim=0, unbiased=False).clamp_min(1e-3)
        novelty = ((values - cell_mean) / cell_scale).square().mean(dim=-1).sqrt()
        mean[cell] = cell_mean
        scale[cell] = cell_scale
        threshold[cell] = torch.quantile(novelty, quantile)
    return HomeostaticProfile(mean=mean, scale=scale, threshold=threshold, quantile=quantile)


def _dynamic_diffusion(state: torch.Tensor, weights: torch.Tensor, cell_gate: torch.Tensor) -> torch.Tensor:
    # Old cells have gate 1. A newborn-old edge therefore receives exactly the
    # newborn gate; newborn-newborn edges require both tissues to be excitable.
    edge_gate = cell_gate.unsqueeze(-1) * cell_gate.unsqueeze(-2)
    dynamic = weights[None, None, :, :] * edge_gate.to(weights.dtype)
    source = torch.einsum("blrs,blsd->blrd", dynamic, state)
    row_mass = dynamic.sum(dim=-1, keepdim=True)
    return source - row_mass * state


def _cell_recruitment(
    model: GrowingCellularLM,
    state: torch.Tensor,
    alive_indices: torch.Tensor,
    localized_state: LocalizedLearningState,
    profile: HomeostaticProfile,
    *,
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
            gate = torch.full(
                (batch, length),
                float(force_recruitment),
                device=state.device,
                dtype=torch.float32,
            ).clamp(0.0, 1.0)
        else:
            parent = int(model.parent[child].item())
            local_parent = global_to_local.get(parent)
            if local_parent is None:
                raise RuntimeError("newborn parent must remain alive for local recruitment")
            parent_state = state[:, :, local_parent, :].float()
            mean = profile.mean[parent].to(state.device)
            scale = profile.scale[parent].to(state.device)
            threshold = profile.threshold[parent].to(state.device)
            novelty = ((parent_state - mean) / scale).square().mean(dim=-1).sqrt()
            gate = RECRUITMENT_FLOOR + (1.0 - RECRUITMENT_FLOOR) * torch.sigmoid(
                RECRUITMENT_TEMPERATURE * (novelty - threshold)
            )
        gates[:, :, local_child] = gate
    return gates


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
    """Run the shared cellular genome with local state-dependent tissue conductance."""
    if input_ids.ndim != 2 or input_ids.shape[1] > model.max_context:
        raise ValueError(f"input_ids must be [batch, <= {model.max_context}]")
    steps = model.iterations if iterations is None else int(iterations)
    if steps < 1:
        raise ValueError("iterations must be positive")
    alive_indices = torch.nonzero(model.alive_mask, as_tuple=False).flatten()
    if alive_indices.numel() < 1 or int(alive_indices[0]) != 0:
        raise RuntimeError("interface cell 0 must remain alive")
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
    weights = model._graph_weights(alive_indices, edge_probe).to(device=input_ids.device)
    profile = profile.to(input_ids.device)
    traces: list[torch.Tensor] = []
    last_before = state
    for _ in range(steps):
        last_before = state
        reaction = model.rule(state, memory)
        activity = model._replicator_activity(activity, reaction)
        gates = _cell_recruitment(
            model,
            state,
            alive_indices,
            localized_state,
            profile,
            force_recruitment=force_recruitment,
        )
        diffusion = _dynamic_diffusion(state, weights, gates)
        relative = activity / (ACTIVITY_BUDGET / len(alive_indices))
        gain = relative.clamp(0.05, 3.0).unsqueeze(-1).to(state.dtype)
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


def make_recruitment_probe(
    model: GrowingCellularLM,
    localized_state: LocalizedLearningState,
    profile: HomeostaticProfile,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    loss_fn,
) -> StructuralProbe:
    """Structural utility/pressure probe under the same conditional dynamics."""
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
