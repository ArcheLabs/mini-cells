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
ACTIVITY_RATE = 0.50
LOCAL_COUPLING = 0.08
PLASTICITY_RATE = 0.75
LONG_RANGE_MAX_COUPLING = LOCAL_COUPLING


@dataclass(frozen=True)
class EmergentTopologyVariant:
    code: str
    name: str
    plastic_long_range: bool


VARIANTS = (
    EmergentTopologyVariant("L", "local-substrate", False),
    EmergentTopologyVariant("E", "emergent-plastic-topology", True),
)
VARIANT_BY_CODE = {variant.code: variant for variant in VARIANTS}


@dataclass
class EmergentTopologyDiagnostics:
    activity: torch.Tensor
    plastic_distribution: torch.Tensor
    plastic_weights: torch.Tensor
    activity_trace: torch.Tensor
    plastic_distribution_trace: torch.Tensor
    plastic_strength_trace: torch.Tensor
    reaction_rms_trace: torch.Tensor
    local_diffusion_rms_trace: torch.Tensor
    plastic_diffusion_rms_trace: torch.Tensor
    residual_trace: torch.Tensor
    activity_entropy_trace: torch.Tensor
    plastic_entropy_trace: torch.Tensor
    plastic_tv_trace: torch.Tensor

    @property
    def effective_active_fraction(self) -> torch.Tensor:
        weights = self.activity.float()
        participation = weights.sum(dim=-1).square() / weights.square().sum(dim=-1).clamp_min(1e-12)
        return (participation / weights.shape[-1]).mean()

    @property
    def mean_plastic_strength(self) -> torch.Tensor:
        return self.plastic_strength_trace.float().mean()

    @property
    def mean_plastic_to_reaction(self) -> torch.Tensor:
        ratio = self.plastic_diffusion_rms_trace.float() / self.reaction_rms_trace.float().clamp_min(1e-8)
        return ratio.mean()


@dataclass
class EmergentTopologyForward:
    output: LanguageModelOutput
    stability_loss: torch.Tensor
    diagnostics: EmergentTopologyDiagnostics | None = None


def variant_by_code(code: str) -> EmergentTopologyVariant:
    try:
        return VARIANT_BY_CODE[code.upper()]
    except KeyError as exc:
        raise ValueError(f"unknown emergent topology variant: {code}") from exc


def _relative_residual(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    delta_rms = (after.float() - before.float()).square().mean().add(1e-12).sqrt()
    state_rms = before.float().square().mean().add(1e-6).sqrt().detach()
    return delta_rms / state_rms


def _normalized_entropy(weights: torch.Tensor, dim: int) -> torch.Tensor:
    probabilities = weights.float() / weights.float().sum(dim=dim, keepdim=True).clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=dim)
    support = weights.shape[dim]
    return entropy / math.log(max(2, support))


def _masked_row_entropy(distribution: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    probabilities = distribution.float().masked_fill(~allowed, 0.0)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    support = allowed.sum(dim=-1).float().clamp_min(2.0)
    return entropy / support.log()


def _local_weights(tissue: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    row = torch.arange(tissue, device=device)
    receiver = row[:, None]
    source = row[None, :]
    adjacency = ((receiver - source).abs() == 1).to(torch.float32)
    adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return (LOCAL_COUPLING * adjacency).to(dtype=dtype)


def _long_range_mask(tissue: int, *, device: torch.device) -> torch.Tensor:
    row = torch.arange(tissue, device=device)
    receiver = row[:, None]
    source = row[None, :]
    return (receiver - source).abs() > 1


def _uniform_plastic_distribution(
    batch: int,
    length: int,
    tissue: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    allowed = _long_range_mask(tissue, device=device)
    distribution = allowed.to(torch.float32)
    distribution = distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return distribution.to(dtype=dtype).view(1, 1, tissue, tissue).expand(batch, length, -1, -1).clone()


def _permute_plastic_distribution(distribution: torch.Tensor) -> torch.Tensor:
    """Permute only legal long-range source identities while preserving row mass/entropy exactly."""
    tissue = distribution.shape[-1]
    allowed = _long_range_mask(tissue, device=distribution.device)
    shuffled = torch.zeros_like(distribution)
    for receiver in range(tissue):
        sources = torch.nonzero(allowed[receiver], as_tuple=False).flatten()
        values = distribution[..., receiver, sources]
        shuffled[..., receiver, sources] = torch.roll(values, shifts=1, dims=-1)
    return shuffled


def _initial_activity(batch: int, length: int, tissue: int, *, device: torch.device) -> torch.Tensor:
    return torch.full(
        (batch, length, tissue),
        ACTIVITY_BUDGET / tissue,
        device=device,
        dtype=torch.float32,
    )


class EmergentTopologyStage(nn.Module):
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
        horizontal = horizontal.reshape(batch, tissue, length, dim).permute(0, 2, 1, 3).contiguous()
        vertical = self.vertical(self.norm_vertical(conditioned))
        candidate = state + horizontal + vertical
        ffn_delta = self.ffn(self.norm_ffn(candidate))
        proposal = horizontal + vertical + ffn_delta
        proposed = self.gru(
            proposal.reshape(batch * length * tissue, dim),
            state.reshape(batch * length * tissue, dim),
        ).view(batch, length, tissue, dim)
        return proposed - state

    @staticmethod
    def _replicator_activity(activity: torch.Tensor, reaction: torch.Tensor) -> torch.Tensor:
        drive = reaction.float().square().mean(dim=-1).add(1e-8).sqrt()
        mean = drive.mean(dim=-1, keepdim=True)
        std = drive.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        fitness = (drive - mean) / std
        growth = torch.exp(ACTIVITY_RATE * fitness).clamp_max(20.0)
        next_activity = activity.float() * growth
        next_activity = ACTIVITY_BUDGET * next_activity / next_activity.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return next_activity

    @staticmethod
    def _update_plastic_distribution(
        distribution: torch.Tensor,
        reaction: torch.Tensor,
        activity: torch.Tensor,
    ) -> torch.Tensor:
        tissue = reaction.shape[2]
        allowed = _long_range_mask(tissue, device=reaction.device).view(1, 1, tissue, tissue)
        normalized = F.normalize(reaction.float(), dim=-1)
        similarity = torch.einsum("blrd,blsd->blrs", normalized, normalized).clamp_min(0.0)
        coactivity = torch.sqrt(activity.unsqueeze(-1) * activity.unsqueeze(-2)).float()
        fitness = similarity * coactivity
        growth = torch.exp(PLASTICITY_RATE * fitness).clamp_max(20.0).to(distribution.dtype)
        next_distribution = distribution * growth
        next_distribution = next_distribution.masked_fill(~allowed, 0.0)
        next_distribution = next_distribution / next_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return next_distribution

    @staticmethod
    def _plastic_weights(distribution: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tissue = distribution.shape[-1]
        allowed = _long_range_mask(tissue, device=distribution.device).view(1, 1, tissue, tissue)
        entropy = _masked_row_entropy(distribution, allowed)
        strength = LONG_RANGE_MAX_COUPLING * (1.0 - entropy).clamp(0.0, 1.0)
        weights = distribution * strength.unsqueeze(-1).to(distribution.dtype)
        uniform = allowed.to(distribution.dtype)
        uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1.0)
        tv = 0.5 * (distribution - uniform).abs().sum(dim=-1)
        return weights, strength, tv

    @staticmethod
    def _diffusion(state: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        source_mean = torch.einsum("blrs,blsd->blrd", weights, state)
        row_mass = weights.sum(dim=-1, keepdim=True)
        return source_mean - row_mass * state

    def forward_variable(
        self,
        state: torch.Tensor,
        activity: torch.Tensor,
        plastic_distribution: torch.Tensor,
        *,
        iterations: int,
        variant: EmergentTopologyVariant,
        intervention: str = "normal",
        collect_observability: bool = False,
        ablate_row: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, list[torch.Tensor]] | None]:
        if iterations < 1 or iterations > self.iterations:
            raise ValueError("iterations must be within the trained stage range")
        if ablate_row is not None and (ablate_row < 0 or ablate_row >= state.shape[2]):
            raise ValueError("ablate_row is outside tissue height")
        valid = {"normal", "plastic_diffusion_off", "plasticity_off", "topology_shuffled", "local_diffusion_off"}
        if intervention not in valid:
            raise ValueError(f"unknown intervention: {intervention}")

        traces = None
        if collect_observability:
            traces = {
                "activity": [],
                "plastic_distribution": [],
                "plastic_strength": [],
                "reaction_rms": [],
                "local_diffusion_rms": [],
                "plastic_diffusion_rms": [],
                "residual": [],
                "activity_entropy": [],
                "plastic_entropy": [],
                "plastic_tv": [],
            }

        batch, length, tissue, _ = state.shape
        local = _local_weights(tissue, device=state.device, dtype=state.dtype).view(1, 1, tissue, tissue)
        last_before = state

        for step in range(iterations):
            if ablate_row is not None:
                state = state.clone()
                state[..., ablate_row, :] = 0.0
            last_before = state
            reaction = self._local_reaction(state, step)
            activity = self._replicator_activity(activity, reaction)

            if variant.plastic_long_range and intervention != "plasticity_off":
                plastic_distribution = self._update_plastic_distribution(plastic_distribution, reaction, activity)

            plastic_weights, strength, plastic_tv = self._plastic_weights(plastic_distribution)
            if not variant.plastic_long_range or intervention in {"plastic_diffusion_off", "plasticity_off"}:
                plastic_weights = torch.zeros_like(plastic_weights)
                strength = torch.zeros_like(strength)
            elif intervention == "topology_shuffled":
                shuffled = _permute_plastic_distribution(plastic_distribution)
                plastic_weights, strength, _ = self._plastic_weights(shuffled)

            local_diffusion = self._diffusion(state, local.expand(batch, length, -1, -1))
            if intervention == "local_diffusion_off":
                local_diffusion = torch.zeros_like(local_diffusion)
            plastic_diffusion = self._diffusion(state, plastic_weights)

            gain = activity / (1.0 + activity)
            delta = gain.unsqueeze(-1).to(state.dtype) * (reaction + local_diffusion + plastic_diffusion)
            state = state + delta
            if ablate_row is not None:
                state = state.clone()
                state[..., ablate_row, :] = 0.0

            if traces is not None:
                allowed = _long_range_mask(tissue, device=state.device).view(1, 1, tissue, tissue)
                traces["activity"].append(activity.detach())
                traces["plastic_distribution"].append(plastic_distribution.detach())
                traces["plastic_strength"].append(strength.detach())
                traces["reaction_rms"].append(reaction.float().square().mean().sqrt().detach())
                traces["local_diffusion_rms"].append(local_diffusion.float().square().mean().sqrt().detach())
                traces["plastic_diffusion_rms"].append(plastic_diffusion.float().square().mean().sqrt().detach())
                traces["residual"].append(_relative_residual(last_before, state).detach())
                traces["activity_entropy"].append(_normalized_entropy(activity, dim=-1).mean().detach())
                traces["plastic_entropy"].append(_masked_row_entropy(plastic_distribution, allowed).mean().detach())
                traces["plastic_tv"].append(plastic_tv.mean().detach())

        residual = _relative_residual(last_before, state)
        return state, activity, plastic_distribution, residual, traces


class EmergentPlasticTopologyCLM(nn.Module):
    """CLM with a fixed local NCA substrate and a separately emergent long-range topology."""

    def __init__(
        self,
        *,
        vocab_size: int,
        variant: EmergentTopologyVariant,
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
            raise ValueError("Experiment 015c uses exactly three recurrent stages")
        self.variant = variant
        self.tissue_height = tissue_height
        self.max_context = max_context
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_context, dim)
        self.stages = nn.ModuleList(
            [
                EmergentTopologyStage(
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
        self.register_buffer("row_encoding", fixed_row_encoding(tissue_height, dim) * 0.10, persistent=True)

    def _initial_dynamics(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.max_context:
            raise ValueError(f"input_ids must be [batch, <= {self.max_context}]")
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        base = self.position_embedding(positions)[None, :, None, :] + self.row_encoding[None, None, :, :]
        state = base.expand(batch, -1, -1, -1).clone()
        state[:, :, 0, :] += self.token_embedding(input_ids)
        activity = _initial_activity(batch, length, self.tissue_height, device=state.device)
        plastic_distribution = _uniform_plastic_distribution(
            batch,
            length,
            self.tissue_height,
            device=state.device,
            dtype=state.dtype,
        )
        return state, activity, plastic_distribution

    def forward_variable(
        self,
        input_ids: torch.Tensor,
        *,
        stage_depths: tuple[int, int, int],
        intervention: str = "normal",
        collect_observability: bool = False,
        ablate_row: int | None = None,
    ) -> EmergentTopologyForward:
        if len(stage_depths) != len(self.stages):
            raise ValueError("stage_depths must match recurrent stage count")
        state, activity, plastic_distribution = self._initial_dynamics(input_ids)
        residuals: list[torch.Tensor] = []
        merged: dict[str, list[torch.Tensor]] | None = None
        if collect_observability:
            merged = {
                "activity": [],
                "plastic_distribution": [],
                "plastic_strength": [],
                "reaction_rms": [],
                "local_diffusion_rms": [],
                "plastic_diffusion_rms": [],
                "residual": [],
                "activity_entropy": [],
                "plastic_entropy": [],
                "plastic_tv": [],
            }

        for stage, depth in zip(self.stages, stage_depths):
            state, activity, plastic_distribution, residual, traces = stage.forward_variable(
                state,
                activity,
                plastic_distribution,
                iterations=depth,
                variant=self.variant,
                intervention=intervention,
                collect_observability=collect_observability,
                ablate_row=ablate_row,
            )
            residuals.append(residual)
            if merged is not None and traces is not None:
                for key in merged:
                    merged[key].extend(traces[key])

        token_state = state[:, :, 0, :]
        logits = self.lm_head(self.final_norm(token_state))
        diagnostics = None
        if merged is not None:
            activity_trace = torch.stack(merged["activity"], dim=0)
            distribution_trace = torch.stack(merged["plastic_distribution"], dim=0)
            strength_trace = torch.stack(merged["plastic_strength"], dim=0)
            final_weights, _, _ = EmergentTopologyStage._plastic_weights(plastic_distribution)
            if not self.variant.plastic_long_range:
                final_weights = torch.zeros_like(final_weights)
            diagnostics = EmergentTopologyDiagnostics(
                activity=activity,
                plastic_distribution=plastic_distribution,
                plastic_weights=final_weights,
                activity_trace=activity_trace,
                plastic_distribution_trace=distribution_trace,
                plastic_strength_trace=strength_trace,
                reaction_rms_trace=torch.stack(merged["reaction_rms"]),
                local_diffusion_rms_trace=torch.stack(merged["local_diffusion_rms"]),
                plastic_diffusion_rms_trace=torch.stack(merged["plastic_diffusion_rms"]),
                residual_trace=torch.stack(merged["residual"]),
                activity_entropy_trace=torch.stack(merged["activity_entropy"]),
                plastic_entropy_trace=torch.stack(merged["plastic_entropy"]),
                plastic_tv_trace=torch.stack(merged["plastic_tv"]),
            )
        stability = torch.stack(residuals).mean()
        return EmergentTopologyForward(LanguageModelOutput(logits), stability, diagnostics)

    def forward(self, input_ids: torch.Tensor) -> LanguageModelOutput:
        return self.forward_variable(input_ids, stage_depths=(4, 4, 4)).output


def build_emergent_topology_model(*, vocab_size: int, variant_code: str, **kwargs: object) -> EmergentPlasticTopologyCLM:
    return EmergentPlasticTopologyCLM(
        vocab_size=vocab_size,
        variant=variant_by_code(variant_code),
        **kwargs,
    )
