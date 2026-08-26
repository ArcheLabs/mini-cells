from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .language_2d import VerticalDepthwiseMixer
from .language_models import LanguageModelOutput, LocalCausalSelfAttention, TextNCALM
from .language_sparse_topology import fixed_row_encoding


TISSUE_HEIGHT = 8
STABILITY_WEIGHT = 0.1
ACTIVITY_BUDGET = 3.0
ACTIVITY_MOMENTUM = 0.25
SYNAPTIC_BUDGET = 0.25
PLASTICITY_RATE = 0.25
NONLOCAL_PRIOR = 0.25


@dataclass
class PlasticTopologyDiagnostics:
    activity: torch.Tensor
    connectome: torch.Tensor
    activity_trace: torch.Tensor
    connectome_trace: torch.Tensor
    reaction_rms_trace: torch.Tensor
    diffusion_rms_trace: torch.Tensor
    residual_trace: torch.Tensor
    activity_entropy_trace: torch.Tensor
    connectome_entropy_trace: torch.Tensor
    nonlocal_mass_trace: torch.Tensor

    @property
    def effective_active_fraction(self) -> torch.Tensor:
        weights = self.activity.float()
        participation = weights.sum(dim=-1).square() / weights.square().sum(dim=-1).clamp_min(1e-12)
        return (participation / weights.shape[-1]).mean()

    @property
    def mean_nonlocal_mass(self) -> torch.Tensor:
        return self.nonlocal_mass_trace.float().mean()

    @property
    def mean_diffusion_to_reaction(self) -> torch.Tensor:
        ratio = self.diffusion_rms_trace.float() / self.reaction_rms_trace.float().clamp_min(1e-8)
        return ratio.mean()


@dataclass
class PlasticTissueForward:
    output: LanguageModelOutput
    stability_loss: torch.Tensor
    diagnostics: PlasticTopologyDiagnostics | None = None


def _relative_residual(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    delta_rms = (after.float() - before.float()).square().mean().add(1e-12).sqrt()
    state_rms = before.float().square().mean().add(1e-6).sqrt().detach()
    return delta_rms / state_rms


def _normalized_entropy(weights: torch.Tensor, dim: int) -> torch.Tensor:
    probabilities = weights.float() / weights.float().sum(dim=dim, keepdim=True).clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=dim)
    support = weights.shape[dim]
    return entropy / math.log(max(2, support))


def _initial_connectome(
    batch: int,
    length: int,
    tissue: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    row = torch.arange(tissue, device=device)
    receiver = row[:, None]
    source = row[None, :]
    distance = (receiver - source).abs()
    prior = torch.full((tissue, tissue), NONLOCAL_PRIOR, device=device, dtype=torch.float32)
    prior = torch.where(distance == 1, torch.ones_like(prior), prior)
    prior = torch.where(distance == 0, torch.zeros_like(prior), prior)
    prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    prior = prior * SYNAPTIC_BUDGET
    return prior.to(dtype=dtype).view(1, 1, tissue, tissue).expand(batch, length, -1, -1).clone()


def _activity_from_state(activity_state: torch.Tensor) -> torch.Tensor:
    # Finite metabolic resource: the tissue distributes a fixed activity budget
    # continuously, rather than selecting a hard top-k expert set.
    return ACTIVITY_BUDGET * torch.softmax(activity_state.float(), dim=-1).to(activity_state.dtype)


class PlasticReactionDiffusionStage(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        ffn_dim: int,
        window: int,
        iterations: int,
        carry_bias: float,
    ) -> None:
        super().__init__()
        self.iterations = iterations
        self.norm_attention = nn.LayerNorm(dim)
        self.norm_vertical = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.attention = LocalCausalSelfAttention(dim, heads, window)
        self.vertical = VerticalDepthwiseMixer(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.step_embedding = nn.Parameter(torch.zeros(iterations, dim))
        nn.init.normal_(self.step_embedding, mean=0.0, std=0.02)
        self.gru = nn.GRUCell(dim, dim)
        self._initialize_gru(carry_bias)

    def _initialize_gru(self, carry_bias: float) -> None:
        if carry_bias == 0.0:
            return
        hidden = self.gru.hidden_size
        with torch.no_grad():
            self.gru.bias_ih[hidden : 2 * hidden].fill_(carry_bias / 2.0)
            self.gru.bias_hh[hidden : 2 * hidden].fill_(carry_bias / 2.0)

    def _local_reaction(self, state: torch.Tensor, step: int) -> torch.Tensor:
        batch, length, tissue, dim = state.shape
        conditioned = state + self.step_embedding[step].view(1, 1, 1, dim)
        rows = conditioned.permute(0, 2, 1, 3).reshape(batch * tissue, length, dim)
        horizontal = self.attention(self.norm_attention(rows))
        horizontal = (
            horizontal.reshape(batch, tissue, length, dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        vertical = self.vertical(self.norm_vertical(conditioned))
        candidate_state = state + horizontal + vertical
        ffn_delta = self.ffn(self.norm_ffn(candidate_state))
        proposal = horizontal + vertical + ffn_delta
        proposed = self.gru(
            proposal.reshape(batch * length * tissue, dim),
            state.reshape(batch * length * tissue, dim),
        ).view(batch, length, tissue, dim)
        return proposed - state

    @staticmethod
    def _update_activity(activity_state: torch.Tensor, reaction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        drive = reaction.float().square().mean(dim=-1).add(1e-8).sqrt()
        mean = drive.mean(dim=-1, keepdim=True)
        std = drive.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        normalized_drive = (drive - mean) / std
        next_state = (1.0 - ACTIVITY_MOMENTUM) * activity_state + ACTIVITY_MOMENTUM * normalized_drive
        activity = _activity_from_state(next_state)
        return next_state, activity

    @staticmethod
    def _update_connectome(
        connectome: torch.Tensor,
        reaction: torch.Tensor,
        activity: torch.Tensor,
    ) -> torch.Tensor:
        normalized = F.normalize(reaction.float(), dim=-1)
        similarity = torch.einsum("blrd,blsd->blrs", normalized, normalized).clamp_min(0.0)
        tissue = reaction.shape[2]
        eye = torch.eye(tissue, device=reaction.device, dtype=torch.bool).view(1, 1, tissue, tissue)
        similarity = similarity.masked_fill(eye, 0.0)
        coactivity = torch.sqrt(activity.unsqueeze(-1) * activity.unsqueeze(-2)).to(similarity.dtype)
        similarity = similarity * coactivity
        # Replicator-style Hebbian plasticity: correlated, co-active links gain a larger share
        # of a fixed per-cell synaptic budget. No edge is a trainable parameter.
        growth = torch.exp(PLASTICITY_RATE * similarity).to(dtype=connectome.dtype)
        next_connectome = connectome * growth
        next_connectome = next_connectome.masked_fill(eye, 0.0)
        next_connectome = (
            SYNAPTIC_BUDGET
            * next_connectome
            / next_connectome.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        )
        return next_connectome

    @staticmethod
    def _diffusion(
        state: torch.Tensor,
        connectome: torch.Tensor,
        *,
        intervention: str,
    ) -> torch.Tensor:
        if intervention == "diffusion_off":
            return torch.zeros_like(state)
        weights = connectome
        if intervention == "connectome_shuffled":
            weights = torch.roll(weights, shifts=1, dims=-1)
            tissue = state.shape[2]
            eye = torch.eye(tissue, device=state.device, dtype=torch.bool).view(1, 1, tissue, tissue)
            weights = weights.masked_fill(eye, 0.0)
            weights = (
                SYNAPTIC_BUDGET
                * weights
                / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            )
        elif intervention not in ("normal", "plasticity_off"):
            raise ValueError(f"unknown intervention: {intervention}")
        source_mean = torch.einsum("blrs,blsd->blrd", weights, state)
        row_mass = weights.sum(dim=-1, keepdim=True)
        return source_mean - row_mass * state

    def forward_variable(
        self,
        state: torch.Tensor,
        activity_state: torch.Tensor,
        connectome: torch.Tensor,
        *,
        iterations: int,
        intervention: str = "normal",
        collect_observability: bool = False,
        ablate_row: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, list[torch.Tensor]] | None,
    ]:
        if iterations < 1 or iterations > self.iterations:
            raise ValueError("iterations must be within the trained stage range")
        if ablate_row is not None and (ablate_row < 0 or ablate_row >= state.shape[2]):
            raise ValueError("ablate_row is outside tissue height")
        traces = None
        if collect_observability:
            traces = {
                "activity": [],
                "connectome": [],
                "reaction_rms": [],
                "diffusion_rms": [],
                "residual": [],
                "activity_entropy": [],
                "connectome_entropy": [],
                "nonlocal_mass": [],
            }
        last_before = state
        for step in range(iterations):
            if ablate_row is not None:
                state = state.clone()
                state[..., ablate_row, :] = 0.0
            last_before = state
            reaction = self._local_reaction(state, step)
            activity_state, activity = self._update_activity(activity_state, reaction)
            if intervention != "plasticity_off":
                connectome = self._update_connectome(connectome, reaction, activity)
            diffusion = self._diffusion(state, connectome, intervention=intervention)
            gain = activity / (1.0 + activity)
            delta = gain.unsqueeze(-1) * (reaction + diffusion)
            state = state + delta
            if ablate_row is not None:
                state = state.clone()
                state[..., ablate_row, :] = 0.0
            if traces is not None:
                tissue = state.shape[2]
                row = torch.arange(tissue, device=state.device)
                receiver = row[:, None]
                source = row[None, :]
                nonlocal_mask = ((receiver - source).abs() > 1).view(1, 1, tissue, tissue)
                total_mass = connectome.float().sum(dim=(-1, -2)).clamp_min(1e-12)
                nonlocal_mass = (
                    connectome.float().masked_fill(~nonlocal_mask, 0.0).sum(dim=(-1, -2))
                    / total_mass
                )
                traces["activity"].append(activity.detach())
                traces["connectome"].append(connectome.detach())
                traces["reaction_rms"].append(reaction.float().square().mean().sqrt().detach())
                traces["diffusion_rms"].append(diffusion.float().square().mean().sqrt().detach())
                traces["residual"].append(_relative_residual(last_before, state).detach())
                traces["activity_entropy"].append(_normalized_entropy(activity, dim=-1).mean().detach())
                traces["connectome_entropy"].append(_normalized_entropy(connectome, dim=-1).mean().detach())
                traces["nonlocal_mass"].append(nonlocal_mass.mean().detach())
        residual = _relative_residual(last_before, state)
        return state, activity_state, connectome, residual, traces


class PlasticReactionDiffusionCLM(nn.Module):
    """CLM whose sparse activity and tissue routing are dynamical state, not modules."""

    def __init__(
        self,
        *,
        vocab_size: int,
        tissue_height: int = TISSUE_HEIGHT,
        max_context: int = 32,
        dim: int = 128,
        heads: int = 4,
        ffn_dim: int = 512,
        windows: tuple[int, int, int] = (8, 32, 32),
        iterations: tuple[int, int, int] = (4, 4, 4),
        carry_bias: float = 2.0,
    ) -> None:
        super().__init__()
        if len(windows) != 3 or len(iterations) != 3:
            raise ValueError("Experiment 015b uses exactly three recurrent stages")
        self.tissue_height = tissue_height
        self.max_context = max_context
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_context, dim)
        self.stages = nn.ModuleList(
            [
                PlasticReactionDiffusionStage(
                    dim=dim,
                    heads=heads,
                    ffn_dim=ffn_dim,
                    window=window,
                    iterations=steps,
                    carry_bias=carry_bias,
                )
                for window, steps in zip(windows, iterations)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(TextNCALM._init_weights)
        self.register_buffer(
            "row_encoding",
            fixed_row_encoding(tissue_height, dim) * 0.10,
            persistent=True,
        )

    def _initial_dynamics(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.max_context:
            raise ValueError(f"input_ids must be [batch, <= {self.max_context}]")
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        base = self.position_embedding(positions)[None, :, None, :] + self.row_encoding[None, None, :, :]
        state = base.expand(batch, -1, -1, -1).clone()
        state[:, :, 0, :] += self.token_embedding(input_ids)
        activity_state = torch.zeros(batch, length, self.tissue_height, device=state.device, dtype=torch.float32)
        activity_state[..., 0] = 1.0
        connectome = _initial_connectome(
            batch,
            length,
            self.tissue_height,
            device=state.device,
            dtype=state.dtype,
        )
        return state, activity_state, connectome

    def forward_variable(
        self,
        input_ids: torch.Tensor,
        *,
        stage_depths: tuple[int, int, int],
        intervention: str = "normal",
        collect_observability: bool = False,
        ablate_row: int | None = None,
    ) -> PlasticTissueForward:
        if len(stage_depths) != len(self.stages):
            raise ValueError("stage_depths must match recurrent stage count")
        state, activity_state, connectome = self._initial_dynamics(input_ids)
        residuals: list[torch.Tensor] = []
        merged: dict[str, list[torch.Tensor]] | None = None
        if collect_observability:
            merged = {
                "activity": [],
                "connectome": [],
                "reaction_rms": [],
                "diffusion_rms": [],
                "residual": [],
                "activity_entropy": [],
                "connectome_entropy": [],
                "nonlocal_mass": [],
            }
        for stage, depth in zip(self.stages, stage_depths):
            state, activity_state, connectome, residual, traces = stage.forward_variable(
                state,
                activity_state,
                connectome,
                iterations=depth,
                intervention=intervention,
                collect_observability=collect_observability,
                ablate_row=ablate_row,
            )
            residuals.append(residual)
            if merged is not None and traces is not None:
                for key, values in traces.items():
                    merged[key].extend(values)
        logits = self.lm_head(self.final_norm(state[:, :, 0, :]))
        diagnostics = None
        if merged is not None:
            diagnostics = PlasticTopologyDiagnostics(
                activity=merged["activity"][-1],
                connectome=merged["connectome"][-1],
                activity_trace=torch.stack(merged["activity"], dim=0),
                connectome_trace=torch.stack(merged["connectome"], dim=0),
                reaction_rms_trace=torch.stack(merged["reaction_rms"]),
                diffusion_rms_trace=torch.stack(merged["diffusion_rms"]),
                residual_trace=torch.stack(merged["residual"]),
                activity_entropy_trace=torch.stack(merged["activity_entropy"]),
                connectome_entropy_trace=torch.stack(merged["connectome_entropy"]),
                nonlocal_mass_trace=torch.stack(merged["nonlocal_mass"]),
            )
        return PlasticTissueForward(
            output=LanguageModelOutput(logits),
            stability_loss=torch.stack(residuals).mean(),
            diagnostics=diagnostics,
        )

    def forward(self, input_ids: torch.Tensor) -> LanguageModelOutput:
        return self.forward_variable(input_ids, stage_depths=(4, 4, 4)).output


def build_plastic_reaction_diffusion_model(
    vocab_size: int,
    **kwargs,
) -> PlasticReactionDiffusionCLM:
    return PlasticReactionDiffusionCLM(vocab_size=vocab_size, **kwargs)
