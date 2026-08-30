from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .language_2d import LatentTissueNCALM, NCAStage2D
from .language_models import LanguageModelOutput, NCAStage, TextNCALM


@dataclass(frozen=True)
class SettlingForward:
    output: LanguageModelOutput
    probe_output: LanguageModelOutput
    stage_depths: tuple[int, ...]
    stage_probe_residuals: tuple[torch.Tensor, ...]
    state_stability_loss: torch.Tensor
    logit_consistency_loss: torch.Tensor


@dataclass(frozen=True)
class RelaxationForward:
    output: LanguageModelOutput
    stage_depths: tuple[int, ...]
    stage_last_residuals: tuple[torch.Tensor, ...]


def relative_residual(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """Scale-free RMS update magnitude used as the fixed-point residual."""

    delta_rms = (after.float() - before.float()).square().mean().add(1e-12).sqrt()
    state_rms = before.float().square().mean().add(1e-6).sqrt().detach()
    return delta_rms / state_rms


def _step_1d(stage: NCAStage, state: torch.Tensor) -> torch.Tensor:
    """Apply the stage's shared cellular rule once, without iteration identity."""

    batch, length, dim = state.shape
    attention_delta = stage.attention(stage.norm_attention(state))
    candidate_state = state + attention_delta
    ffn_delta = stage.ffn(stage.norm_ffn(candidate_state))
    proposal = attention_delta + ffn_delta
    return stage.gru(
        proposal.reshape(batch * length, dim),
        state.reshape(batch * length, dim),
    ).view(batch, length, dim)


def _step_2d(stage: NCAStage2D, state: torch.Tensor) -> torch.Tensor:
    """Apply one factorized 2D cellular update with no absolute step embedding."""

    if state.ndim != 4:
        raise ValueError("2D settling stage expects [batch, length, tissue, dim]")
    batch, length, tissue, dim = state.shape
    rows = state.permute(0, 2, 1, 3).reshape(batch * tissue, length, dim)
    horizontal = stage.attention(stage.norm_attention(rows))
    horizontal = (
        horizontal.reshape(batch, tissue, length, dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    vertical = stage.vertical(stage.norm_vertical(state))
    candidate_state = state + horizontal + vertical
    ffn_delta = stage.ffn(stage.norm_ffn(candidate_state))
    proposal = horizontal + vertical + ffn_delta
    return stage.gru(
        proposal.reshape(batch * length * tissue, dim),
        state.reshape(batch * length * tissue, dim),
    ).view(batch, length, tissue, dim)


def run_1d_shared(
    stage: NCAStage,
    state: torch.Tensor,
    *,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    last_before = state
    for _ in range(iterations):
        last_before = state
        state = _step_1d(stage, state)
    return state, relative_residual(last_before, state)


def run_2d_shared(
    stage: NCAStage2D,
    state: torch.Tensor,
    *,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    last_before = state
    for _ in range(iterations):
        last_before = state
        state = _step_2d(stage, state)
    return state, relative_residual(last_before, state)


def _kl_teacher_student(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    teacher = F.softmax(teacher_logits.float().detach(), dim=-1)
    student_log = F.log_softmax(student_logits.float(), dim=-1)
    return F.kl_div(student_log, teacher, reduction="none").sum(dim=-1).mean()


def run_textnca_settling(
    model: TextNCALM,
    input_ids: torch.Tensor,
    *,
    stage_depths: tuple[int, ...],
) -> SettlingForward:
    if input_ids.ndim != 2 or input_ids.shape[1] > model.max_context:
        raise ValueError(f"input_ids must be [batch, <= {model.max_context}]")
    if len(stage_depths) != len(model.stages):
        raise ValueError("stage_depths must match recurrent stage count")

    length = input_ids.shape[1]
    positions = torch.arange(length, device=input_ids.device)
    state = model.token_embedding(input_ids) + model.position_embedding(positions)[None, :, :]
    probe_residuals: list[torch.Tensor] = []
    final_probe_state: torch.Tensor | None = None

    for index, (stage, depth) in enumerate(zip(model.stages, stage_depths)):
        state, _ = run_1d_shared(stage, state, iterations=depth)
        probe = _step_1d(stage, state)
        probe_residuals.append(relative_residual(state, probe))
        if index == len(model.stages) - 1:
            final_probe_state = probe

    if final_probe_state is None:
        raise RuntimeError("settling model has no stages")
    logits = model.lm_head(model.final_norm(state))
    probe_logits = model.lm_head(model.final_norm(final_probe_state))
    return SettlingForward(
        output=LanguageModelOutput(logits),
        probe_output=LanguageModelOutput(probe_logits),
        stage_depths=stage_depths,
        stage_probe_residuals=tuple(probe_residuals),
        state_stability_loss=torch.stack(probe_residuals).mean(),
        logit_consistency_loss=_kl_teacher_student(logits, probe_logits),
    )


def run_2d_settling(
    model: LatentTissueNCALM,
    input_ids: torch.Tensor,
    *,
    stage_depths: tuple[int, ...],
) -> SettlingForward:
    if len(stage_depths) != len(model.stages):
        raise ValueError("stage_depths must match recurrent stage count")

    state = model._initial_state(input_ids)
    probe_residuals: list[torch.Tensor] = []
    final_probe_state: torch.Tensor | None = None

    for index, (stage, depth) in enumerate(zip(model.stages, stage_depths)):
        state, _ = run_2d_shared(stage, state, iterations=depth)
        probe = _step_2d(stage, state)
        probe_residuals.append(relative_residual(state, probe))
        if index == len(model.stages) - 1:
            final_probe_state = probe

    if final_probe_state is None:
        raise RuntimeError("settling model has no stages")
    token_state = state[:, :, 0, :]
    probe_token_state = final_probe_state[:, :, 0, :]
    logits = model.lm_head(model.final_norm(token_state))
    probe_logits = model.lm_head(model.final_norm(probe_token_state))
    return SettlingForward(
        output=LanguageModelOutput(logits),
        probe_output=LanguageModelOutput(probe_logits),
        stage_depths=stage_depths,
        stage_probe_residuals=tuple(probe_residuals),
        state_stability_loss=torch.stack(probe_residuals).mean(),
        logit_consistency_loss=_kl_teacher_student(logits, probe_logits),
    )


def settling_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    stage_depths: tuple[int, ...],
) -> SettlingForward:
    if isinstance(model, LatentTissueNCALM):
        return run_2d_settling(model, input_ids, stage_depths=stage_depths)
    if isinstance(model, TextNCALM):
        return run_textnca_settling(model, input_ids, stage_depths=stage_depths)
    raise TypeError(f"unsupported settling model type: {type(model).__name__}")


def run_textnca_relaxation(
    model: TextNCALM,
    input_ids: torch.Tensor,
    *,
    stage_depths: tuple[int, ...],
) -> RelaxationForward:
    if input_ids.ndim != 2 or input_ids.shape[1] > model.max_context:
        raise ValueError(f"input_ids must be [batch, <= {model.max_context}]")
    if len(stage_depths) != len(model.stages):
        raise ValueError("stage_depths must match recurrent stage count")
    length = input_ids.shape[1]
    positions = torch.arange(length, device=input_ids.device)
    state = model.token_embedding(input_ids) + model.position_embedding(positions)[None, :, :]
    residuals: list[torch.Tensor] = []
    for stage, depth in zip(model.stages, stage_depths):
        state, residual = run_1d_shared(stage, state, iterations=depth)
        residuals.append(residual)
    logits = model.lm_head(model.final_norm(state))
    return RelaxationForward(LanguageModelOutput(logits), stage_depths, tuple(residuals))


def run_2d_relaxation(
    model: LatentTissueNCALM,
    input_ids: torch.Tensor,
    *,
    stage_depths: tuple[int, ...],
) -> RelaxationForward:
    if len(stage_depths) != len(model.stages):
        raise ValueError("stage_depths must match recurrent stage count")
    state = model._initial_state(input_ids)
    residuals: list[torch.Tensor] = []
    for stage, depth in zip(model.stages, stage_depths):
        state, residual = run_2d_shared(stage, state, iterations=depth)
        residuals.append(residual)
    token_state = state[:, :, 0, :]
    logits = model.lm_head(model.final_norm(token_state))
    return RelaxationForward(LanguageModelOutput(logits), stage_depths, tuple(residuals))


def relaxation_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    stage_depths: tuple[int, ...],
) -> RelaxationForward:
    """Free-run the same shared update rule for arbitrary recurrent depth.

    Unlike Experiments 009-011, this path has no learned absolute iteration
    identity and is intentionally valid beyond the four iterations used by the
    original TextNCA stage implementation.
    """

    if isinstance(model, LatentTissueNCALM):
        return run_2d_relaxation(model, input_ids, stage_depths=stage_depths)
    if isinstance(model, TextNCALM):
        return run_textnca_relaxation(model, input_ids, stage_depths=stage_depths)
    raise TypeError(f"unsupported relaxation model type: {type(model).__name__}")
