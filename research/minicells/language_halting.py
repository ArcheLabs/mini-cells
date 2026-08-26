from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .language_2d import LatentTissueNCALM, NCAStage2D
from .language_models import LanguageModelOutput, NCAStage, TextNCALM


@dataclass(frozen=True)
class AdaptiveForward:
    output: LanguageModelOutput
    stage_steps: tuple[int, ...]
    stage_residuals: tuple[tuple[float, ...], ...]

    @property
    def total_steps(self) -> int:
        return sum(self.stage_steps)


def state_delta_rms(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """Absolute RMS state update used as the first adaptive-halting signal."""

    return (after.float() - before.float()).square().mean().sqrt()


def _should_halt(
    residual: torch.Tensor,
    *,
    step: int,
    threshold: float | None,
    min_iterations: int,
) -> bool:
    if threshold is None or step < min_iterations:
        return False
    # A real early exit requires a control-flow decision. This scalar sync is
    # intentionally part of Experiment 010's wall-clock measurement.
    return bool((residual <= threshold).detach().item())


def run_1d_stage_adaptive(
    stage: NCAStage,
    state: torch.Tensor,
    *,
    threshold: float | None,
    min_iterations: int,
) -> tuple[torch.Tensor, int, tuple[float, ...]]:
    """Run one trained 1D stage with a readout-position stopping criterion.

    Experiment 010 evaluates one autoregressive prefix at a time. Halting is based
    only on the final input position, whose next-token prediction is being scored;
    later sequence positions therefore cannot influence the stopping decision.
    """

    if min_iterations < 1 or min_iterations > stage.iterations:
        raise ValueError("min_iterations must be within the trained stage iteration range")
    batch, length, dim = state.shape
    residuals: list[float] = []
    steps = 0

    for index in range(stage.iterations):
        before = state
        conditioned = state + stage.step_embedding[index].view(1, 1, dim)
        attention_delta = stage.attention(stage.norm_attention(conditioned))
        candidate_state = state + attention_delta
        ffn_delta = stage.ffn(stage.norm_ffn(candidate_state))
        proposal = attention_delta + ffn_delta
        state = stage.gru(
            proposal.reshape(batch * length, dim),
            state.reshape(batch * length, dim),
        ).view(batch, length, dim)
        residual = state_delta_rms(before[:, -1, :], state[:, -1, :])
        residuals.append(float(residual.detach().cpu()))
        steps = index + 1
        if _should_halt(
            residual,
            step=steps,
            threshold=threshold,
            min_iterations=min_iterations,
        ):
            break
    return state, steps, tuple(residuals)


def run_textnca_adaptive(
    model: TextNCALM,
    input_ids: torch.Tensor,
    *,
    threshold: float | None,
    min_iterations: int = 1,
) -> AdaptiveForward:
    if input_ids.ndim != 2 or input_ids.shape[1] > model.max_context:
        raise ValueError(f"input_ids must be [batch, <= {model.max_context}]")
    length = input_ids.shape[1]
    positions = torch.arange(length, device=input_ids.device)
    state = model.token_embedding(input_ids) + model.position_embedding(positions)[None, :, :]
    stage_steps: list[int] = []
    stage_residuals: list[tuple[float, ...]] = []

    for stage in model.stages:
        state, steps, residuals = run_1d_stage_adaptive(
            stage,
            state,
            threshold=threshold,
            min_iterations=min_iterations,
        )
        stage_steps.append(steps)
        stage_residuals.append(residuals)

    logits = model.lm_head(model.final_norm(state))
    return AdaptiveForward(
        output=LanguageModelOutput(logits),
        stage_steps=tuple(stage_steps),
        stage_residuals=tuple(stage_residuals),
    )


def run_2d_stage_adaptive(
    stage: NCAStage2D,
    state: torch.Tensor,
    *,
    threshold: float | None,
    min_iterations: int,
) -> tuple[torch.Tensor, int, tuple[float, ...]]:
    """Run one trained 2D stage using the final-position tissue column to halt."""

    if min_iterations < 1 or min_iterations > stage.iterations:
        raise ValueError("min_iterations must be within the trained stage iteration range")
    if state.ndim != 4:
        raise ValueError("2D adaptive stage expects [batch, length, tissue, dim]")
    batch, length, tissue, dim = state.shape
    residuals: list[float] = []
    steps = 0

    for index in range(stage.iterations):
        before = state
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
        residual = state_delta_rms(before[:, -1, :, :], state[:, -1, :, :])
        residuals.append(float(residual.detach().cpu()))
        steps = index + 1
        if _should_halt(
            residual,
            step=steps,
            threshold=threshold,
            min_iterations=min_iterations,
        ):
            break
    return state, steps, tuple(residuals)


def run_2d_adaptive(
    model: LatentTissueNCALM,
    input_ids: torch.Tensor,
    *,
    threshold: float | None,
    min_iterations: int = 1,
) -> AdaptiveForward:
    state = model._initial_state(input_ids)
    stage_steps: list[int] = []
    stage_residuals: list[tuple[float, ...]] = []

    for stage in model.stages:
        state, steps, residuals = run_2d_stage_adaptive(
            stage,
            state,
            threshold=threshold,
            min_iterations=min_iterations,
        )
        stage_steps.append(steps)
        stage_residuals.append(residuals)

    token_state = state[:, :, 0, :]
    logits = model.lm_head(model.final_norm(token_state))
    return AdaptiveForward(
        output=LanguageModelOutput(logits),
        stage_steps=tuple(stage_steps),
        stage_residuals=tuple(stage_residuals),
    )


def adaptive_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    threshold: float | None,
    min_iterations: int = 1,
) -> AdaptiveForward:
    """Adaptive prefix inference for a TextNCA or 2D latent-tissue NCA.

    The current prototype uses one stopping decision for the whole input example,
    so Experiment 010 evaluates batch size 1 and scores only the next token after
    the complete prefix. A future training-time implementation should move to
    per-position/per-cell active masks or an equivalent causal sparse scheduler.
    """

    if isinstance(model, LatentTissueNCALM):
        return run_2d_adaptive(
            model,
            input_ids,
            threshold=threshold,
            min_iterations=min_iterations,
        )
    if isinstance(model, TextNCALM):
        return run_textnca_adaptive(
            model,
            input_ids,
            threshold=threshold,
            min_iterations=min_iterations,
        )
    raise TypeError(f"unsupported adaptive-halting model type: {type(model).__name__}")


def fixed_forward(model: torch.nn.Module, input_ids: torch.Tensor) -> LanguageModelOutput:
    output = model(input_ids)
    if not isinstance(output, LanguageModelOutput):
        raise TypeError("language model did not return LanguageModelOutput")
    return output


def logits_kl(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference_log = F.log_softmax(reference.float(), dim=-1)
    candidate_log = F.log_softmax(candidate.float(), dim=-1)
    value = F.kl_div(candidate_log, reference_log.exp(), reduction="batchmean")
    return float(value.detach().cpu())
