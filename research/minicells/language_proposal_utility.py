from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .language_conditional_recruitment import (
    _base_graph_weights,
    _conditional_diffusion,
    _gated_replicator_activity,
)
from .language_growing_organism import ACTIVITY_BUDGET, STEP_SIZE, GrowingCellularLM, _relative_residual
from .language_localized_learning import LocalizedLearningState
from .language_models import LanguageModelOutput


UTILITY_EPSILON = 0.02

LOCAL_FEATURES = (
    "parent_state_rms",
    "parent_settling_rms",
    "proposal_parent_rms",
    "proposal_parent_alignment",
    "child_memory_norm",
    "child_parent_memory_cosine",
)

BOUNDARY_FEATURES = LOCAL_FEATURES + (
    "interface_settling_rms",
    "proposal_interface_rms",
    "proposal_interface_alignment",
    "base_entropy",
    "base_margin",
    "probe_logit_shift_rms",
    "probe_kl",
    "probe_entropy_delta",
    "probe_margin_delta",
)


@dataclass
class UtilityForward:
    output: LanguageModelOutput
    stability_loss: torch.Tensor
    final_state: torch.Tensor
    last_delta: torch.Tensor
    alive_indices: torch.Tensor


def _recruitment_matrix(
    recruitment: float | torch.Tensor,
    *,
    batch: int,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(recruitment, torch.Tensor):
        value = recruitment.to(device=device, dtype=torch.float32)
        if value.ndim == 0:
            value = value.expand(batch)
        if value.ndim != 1 or value.shape[0] != batch:
            raise ValueError("tensor recruitment must be scalar or [batch]")
    else:
        value = torch.full((batch,), float(recruitment), device=device, dtype=torch.float32)
    return value[:, None].expand(batch, length)


def forward_with_fixed_recruitment(
    model: GrowingCellularLM,
    input_ids: torch.Tensor,
    localized_state: LocalizedLearningState,
    recruitment: float | torch.Tensor,
    *,
    iterations: int | None = None,
) -> UtilityForward:
    """Continuously interpolate between Phase-1 and the adapted one-cell organism.

    recruitment=0 exactly restores the Phase-1 graph/metabolic competition while
    recruitment=1 exactly restores the ordinary fully active adapted organism.
    The same scalar can be differentiated per example to define marginal tissue
    utility without a learned router or task label.
    """
    if input_ids.ndim != 2 or input_ids.shape[1] > model.max_context:
        raise ValueError(f"input_ids must be [batch, <= {model.max_context}]")
    steps = model.iterations if iterations is None else int(iterations)
    if steps < 1:
        raise ValueError("iterations must be positive")

    alive_indices = torch.nonzero(model.alive_mask, as_tuple=False).flatten()
    if alive_indices.numel() < 1 or int(alive_indices[0]) != 0:
        raise RuntimeError("interface cell 0 must remain alive")
    newborn_global = localized_state.newborn_cells(model)
    if not newborn_global:
        raise RuntimeError("proposal utility requires at least one newborn tissue cell")

    memory = model.cell_memory.index_select(0, alive_indices)
    batch, length = input_ids.shape
    positions = torch.arange(length, device=input_ids.device)
    token_state = model.token_embedding(input_ids) + model.position_embedding(positions)[None, :, :]
    state = memory[None, None, :, :].expand(batch, length, -1, -1).clone()
    state[:, :, 0, :] = state[:, :, 0, :] + token_state

    full_weights = model._graph_weights(alive_indices, None).to(input_ids.device)
    base_weights = _base_graph_weights(model, alive_indices, localized_state).to(input_ids.device)
    global_alive = alive_indices.tolist()
    newborn_positions = [global_alive.index(cell) for cell in newborn_global if cell in global_alive]
    if len(newborn_positions) != len(newborn_global):
        raise RuntimeError("all newborn cells must remain alive")

    recruited = _recruitment_matrix(recruitment, batch=batch, length=length, device=input_ids.device)
    activity: torch.Tensor | None = None
    last_before = state
    for _ in range(steps):
        last_before = state
        gates = torch.ones(batch, length, len(alive_indices), device=input_ids.device, dtype=torch.float32)
        gates[..., newborn_positions] = recruited.unsqueeze(-1)
        reaction = model.rule(state, memory)
        activity = _gated_replicator_activity(activity, reaction, gates)
        diffusion = _conditional_diffusion(state, base_weights, full_weights, gates, newborn_positions)
        effective_cells = gates.sum(dim=-1).clamp_min(1.0)
        baseline_activity = ACTIVITY_BUDGET / effective_cells
        relative = activity / baseline_activity.unsqueeze(-1)
        gain = relative.clamp(0.0, 3.0).unsqueeze(-1).to(state.dtype)
        state = state + STEP_SIZE * gain * (reaction + diffusion)

    logits = model.lm_head(model.final_norm(state[:, :, 0, :]))
    return UtilityForward(
        output=LanguageModelOutput(logits),
        stability_loss=_relative_residual(last_before, state),
        final_state=state,
        last_delta=state - last_before,
        alive_indices=alive_indices,
    )


def per_example_masked_nll(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=logits.device, dtype=torch.bool)
    selected_logits = logits[:, mask, :]
    selected_targets = targets[:, mask]
    losses = F.cross_entropy(
        selected_logits.reshape(-1, logits.shape[-1]),
        selected_targets.reshape(-1),
        reduction="none",
    ).view(logits.shape[0], -1)
    return losses.mean(dim=-1)


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean(dim=tuple(range(1, value.ndim))).add(1e-12).sqrt()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.float().reshape(left.shape[0], -1)
    right = right.float().reshape(right.shape[0], -1)
    numerator = (left * right).sum(dim=-1)
    denominator = left.norm(dim=-1) * right.norm(dim=-1)
    return numerator / denominator.clamp_min(1e-12)


def _entropy_and_margin(logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    selected = logits[:, mask.to(logits.device, dtype=torch.bool), :].float()
    log_probs = selected.log_softmax(dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean(dim=-1)
    top2 = probs.topk(k=2, dim=-1).values
    margin = (top2[..., 0] - top2[..., 1]).mean(dim=-1)
    return entropy, margin


def extract_label_free_features(
    model: GrowingCellularLM,
    localized_state: LocalizedLearningState,
    base: UtilityForward,
    probe: UtilityForward,
    mask: torch.Tensor,
    *,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    """Features may inspect the input-conditioned organism/proposal, never targets."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    newborn = localized_state.newborn_cells(model)
    if len(newborn) != 1:
        raise RuntimeError(f"Experiment 019 requires exactly one newborn per candidate, got {newborn}")
    child = newborn[0]
    parent = int(model.parent[child].item())
    global_to_local = {int(cell): index for index, cell in enumerate(base.alive_indices.tolist())}
    if parent not in global_to_local:
        raise RuntimeError("newborn parent must be part of the Phase-1 organism")
    parent_local = global_to_local[parent]
    interface_local = global_to_local[0]
    selected = mask.to(base.final_state.device, dtype=torch.bool)

    parent_state = base.final_state[:, selected, parent_local, :]
    parent_delta = base.last_delta[:, selected, parent_local, :]
    interface_state = base.final_state[:, selected, interface_local, :]
    interface_delta = base.last_delta[:, selected, interface_local, :]
    proposal_parent = (probe.final_state[:, selected, parent_local, :] - parent_state) / epsilon
    proposal_interface = (probe.final_state[:, selected, interface_local, :] - interface_state) / epsilon

    base_entropy, base_margin = _entropy_and_margin(base.output.logits, mask)
    probe_entropy, probe_margin = _entropy_and_margin(probe.output.logits, mask)
    base_log_probs = base.output.logits[:, selected, :].float().log_softmax(dim=-1)
    probe_log_probs = probe.output.logits[:, selected, :].float().log_softmax(dim=-1)
    base_probs = base_log_probs.exp()
    kl = (base_probs * (base_log_probs - probe_log_probs)).sum(dim=-1).mean(dim=-1)
    logit_shift = (probe.output.logits[:, selected, :].float() - base.output.logits[:, selected, :].float()) / epsilon

    batch = base.output.logits.shape[0]
    child_memory = model.cell_memory[child].detach().float()
    parent_memory = localized_state.base_memory[parent].to(child_memory.device).float()
    memory_cos = F.cosine_similarity(child_memory, parent_memory, dim=0)

    return {
        "parent_state_rms": _rms(parent_state),
        "parent_settling_rms": _rms(parent_delta) / _rms(parent_state).clamp_min(1e-6),
        "proposal_parent_rms": _rms(proposal_parent),
        "proposal_parent_alignment": _cosine(proposal_parent, parent_delta),
        "child_memory_norm": child_memory.norm().expand(batch),
        "child_parent_memory_cosine": memory_cos.expand(batch),
        "interface_settling_rms": _rms(interface_delta) / _rms(interface_state).clamp_min(1e-6),
        "proposal_interface_rms": _rms(proposal_interface),
        "proposal_interface_alignment": _cosine(proposal_interface, interface_delta),
        "base_entropy": base_entropy,
        "base_margin": base_margin,
        "probe_logit_shift_rms": _rms(logit_shift),
        "probe_kl": kl / (epsilon * epsilon),
        "probe_entropy_delta": (probe_entropy - base_entropy) / epsilon,
        "probe_margin_delta": (probe_margin - base_margin) / epsilon,
    }


def measure_proposal_batch(
    model: GrowingCellularLM,
    localized_state: LocalizedLearningState,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float = UTILITY_EPSILON,
) -> dict[str, torch.Tensor]:
    """Measure oracle marginal utility plus target-free proposal observables."""
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("epsilon must be in (0, 1)")
    batch = inputs.shape[0]
    recruitment = torch.zeros(batch, device=inputs.device, dtype=torch.float32, requires_grad=True)
    base = forward_with_fixed_recruitment(model, inputs, localized_state, recruitment)
    loss0 = per_example_masked_nll(base.output.logits.float(), targets, mask)
    gradient = torch.autograd.grad(loss0.sum(), recruitment, retain_graph=False)[0]
    oracle_gradient = -gradient

    with torch.no_grad():
        probe = forward_with_fixed_recruitment(model, inputs, localized_state, float(epsilon))
        loss_probe = per_example_masked_nll(probe.output.logits.float(), targets, mask)
        oracle_fd = (loss0.detach() - loss_probe) / epsilon
        features = extract_label_free_features(model, localized_state, base, probe, mask, epsilon=epsilon)

    return {
        "loss_closed": loss0.detach(),
        "loss_probe": loss_probe.detach(),
        "oracle_gradient": oracle_gradient.detach(),
        "oracle_fd": oracle_fd.detach(),
        **{name: value.detach() for name, value in features.items()},
    }
