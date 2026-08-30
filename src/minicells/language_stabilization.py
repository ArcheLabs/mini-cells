from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from .language_2d import LatentTissueNCALM, NCAStage2D
from .language_models import LanguageModelOutput, NCAStage, TextNCALM


@dataclass(frozen=True)
class StabilizingForward:
    output: LanguageModelOutput
    stage_depths: tuple[int, ...]
    stage_residual_rms: tuple[torch.Tensor, ...]
    stability_loss: torch.Tensor


def make_depth_schedule(
    steps: int,
    *,
    seed: int,
    min_depth: int = 2,
    max_depth: int = 4,
    stages: int = 3,
) -> tuple[tuple[int, ...], ...]:
    """Deterministic randomized recurrent-depth schedule shared by 1D and 2D.

    The schedule is generated on CPU so both architectures see exactly the same
    recurrent-depth sequence. This isolates topology from a lucky depth draw.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    if min_depth < 1 or max_depth < min_depth:
        raise ValueError("invalid depth range")
    if stages < 1:
        raise ValueError("stages must be positive")
    generator = random.Random(seed)
    return tuple(
        tuple(generator.randint(min_depth, max_depth) for _ in range(stages))
        for _ in range(steps)
    )


def scale_step_embeddings(model: torch.nn.Module, scale: float) -> None:
    """Attenuate absolute iteration identity without changing parameter count."""

    if scale < 0:
        raise ValueError("step embedding scale must be non-negative")
    stages = getattr(model, "stages", None)
    if stages is None:
        raise TypeError("model has no recurrent stages")
    with torch.no_grad():
        for stage in stages:
            if not hasattr(stage, "step_embedding"):
                raise TypeError("stage has no step embedding")
            stage.step_embedding.mul_(scale)


def _relative_residual(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    delta_rms = (after.float() - before.float()).square().mean().add(1e-12).sqrt()
    # Detaching the denominator prevents the regularizer from reducing the score
    # merely by inflating the hidden-state norm.
    state_rms = before.float().square().mean().add(1e-6).sqrt().detach()
    return delta_rms / state_rms


def run_1d_stage_variable(
    stage: NCAStage,
    state: torch.Tensor,
    *,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if iterations < 1 or iterations > stage.iterations:
        raise ValueError("iterations must be within the trained stage range")
    batch, length, dim = state.shape
    last_before = state
    for index in range(iterations):
        last_before = state
        conditioned = state + stage.step_embedding[index].view(1, 1, dim)
        attention_delta = stage.attention(stage.norm_attention(conditioned))
        candidate_state = state + attention_delta
        ffn_delta = stage.ffn(stage.norm_ffn(candidate_state))
        proposal = attention_delta + ffn_delta
        state = stage.gru(
            proposal.reshape(batch * length, dim),
            state.reshape(batch * length, dim),
        ).view(batch, length, dim)
    return state, _relative_residual(last_before, state)


def run_2d_stage_variable(
    stage: NCAStage2D,
    state: torch.Tensor,
    *,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if iterations < 1 or iterations > stage.iterations:
        raise ValueError("iterations must be within the trained stage range")
    if state.ndim != 4:
        raise ValueError("2D stage expects [batch, length, tissue, dim]")
    batch, length, tissue, dim = state.shape
    last_before = state
    for index in range(iterations):
        last_before = state
        conditioned = state + stage.step_embedding[index].view(1, 1, 1, dim)
        rows = conditioned.permute(0, 2, 1, 3).reshape(batch * tissue, length, dim)
        horizontal = stage.attention(stage.norm_attention(rows))
        horizontal = (
            horizontal.reshape(batch, tissue, length, dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        vertical = stage.vertical(stage.norm_vertical(conditioned))
        candidate_state = state + horizontal + vertical
        ffn_delta = stage.ffn(stage.norm_ffn(candidate_state))
        proposal = horizontal + vertical + ffn_delta
        state = stage.gru(
            proposal.reshape(batch * length * tissue, dim),
            state.reshape(batch * length * tissue, dim),
        ).view(batch, length, tissue, dim)
    return state, _relative_residual(last_before, state)


def run_textnca_variable(
    model: TextNCALM,
    input_ids: torch.Tensor,
    *,
    stage_depths: tuple[int, ...],
) -> StabilizingForward:
    if input_ids.ndim != 2 or input_ids.shape[1] > model.max_context:
        raise ValueError(f"input_ids must be [batch, <= {model.max_context}]")
    if len(stage_depths) != len(model.stages):
        raise ValueError("stage_depths must match recurrent stage count")
    length = input_ids.shape[1]
    positions = torch.arange(length, device=input_ids.device)
    state = model.token_embedding(input_ids) + model.position_embedding(positions)[None, :, :]
    residuals: list[torch.Tensor] = []
    for stage, depth in zip(model.stages, stage_depths):
        state, residual = run_1d_stage_variable(stage, state, iterations=depth)
        residuals.append(residual)
    logits = model.lm_head(model.final_norm(state))
    stability = torch.stack(residuals).mean()
    return StabilizingForward(
        output=LanguageModelOutput(logits),
        stage_depths=stage_depths,
        stage_residual_rms=tuple(residuals),
        stability_loss=stability,
    )


def run_2d_variable(
    model: LatentTissueNCALM,
    input_ids: torch.Tensor,
    *,
    stage_depths: tuple[int, ...],
) -> StabilizingForward:
    if len(stage_depths) != len(model.stages):
        raise ValueError("stage_depths must match recurrent stage count")
    state = model._initial_state(input_ids)
    residuals: list[torch.Tensor] = []
    for stage, depth in zip(model.stages, stage_depths):
        state, residual = run_2d_stage_variable(stage, state, iterations=depth)
        residuals.append(residual)
    token_state = state[:, :, 0, :]
    logits = model.lm_head(model.final_norm(token_state))
    stability = torch.stack(residuals).mean()
    return StabilizingForward(
        output=LanguageModelOutput(logits),
        stage_depths=stage_depths,
        stage_residual_rms=tuple(residuals),
        stability_loss=stability,
    )


def stabilizing_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    stage_depths: tuple[int, ...],
) -> StabilizingForward:
    if isinstance(model, LatentTissueNCALM):
        return run_2d_variable(model, input_ids, stage_depths=stage_depths)
    if isinstance(model, TextNCALM):
        return run_textnca_variable(model, input_ids, stage_depths=stage_depths)
    raise TypeError(f"unsupported stabilizing model type: {type(model).__name__}")
