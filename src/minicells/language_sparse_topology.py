from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from .language_2d import VerticalDepthwiseMixer
from .language_models import LanguageModelOutput, LocalCausalSelfAttention, TextNCALM


VARIANT_CODES = ("A", "B", "C")
TISSUE_HEIGHT = 8
ACTIVE_LATENT = 2
STABILITY_WEIGHT = 0.1
BALANCE_WEIGHT = 0.01


@dataclass(frozen=True)
class SparseTopologyVariant:
    code: str
    name: str
    sparse_activity: bool
    dynamic_edges: bool


VARIANTS = (
    SparseTopologyVariant("A", "dense-local", False, False),
    SparseTopologyVariant("B", "sparse-local", True, False),
    SparseTopologyVariant("C", "sparse-dynamic", True, True),
)
VARIANT_BY_CODE = {variant.code: variant for variant in VARIANTS}


@dataclass
class TopologyDiagnostics:
    activity: torch.Tensor
    edges: torch.Tensor
    logical_active_fraction: torch.Tensor


@dataclass
class SparseTopologyForward:
    output: LanguageModelOutput
    stability_loss: torch.Tensor
    balance_loss: torch.Tensor
    diagnostics: TopologyDiagnostics | None = None


def variant_by_code(code: str) -> SparseTopologyVariant:
    try:
        return VARIANT_BY_CODE[code.upper()]
    except KeyError as exc:
        raise ValueError(f"unknown sparse topology variant: {code}") from exc


def fixed_row_encoding(tissue_height: int, dim: int) -> torch.Tensor:
    """Deterministic coordinate features; these are buffers, never expert parameters."""
    if tissue_height < 2 or dim < 2:
        raise ValueError("tissue_height and dim must both be at least two")
    y = torch.linspace(-1.0, 1.0, tissue_height, dtype=torch.float32)[:, None]
    half = dim // 2
    frequencies = torch.arange(1, half + 1, dtype=torch.float32)[None, :]
    phase = math.pi * y * frequencies
    encoded = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
    if encoded.shape[1] < dim:
        encoded = F.pad(encoded, (0, dim - encoded.shape[1]))
    return encoded[:, :dim]


def _relative_residual(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    delta_rms = (after.float() - before.float()).square().mean().add(1e-12).sqrt()
    state_rms = before.float().square().mean().add(1e-6).sqrt().detach()
    return delta_rms / state_rms


class SparseTopologyStage(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        ffn_dim: int,
        window: int,
        iterations: int,
        tissue_height: int,
        active_latent: int,
        carry_bias: float,
    ) -> None:
        super().__init__()
        if tissue_height < 3:
            raise ValueError("tissue_height must be at least three")
        if active_latent < 1 or active_latent >= tissue_height:
            raise ValueError("active_latent must be in [1, tissue_height-1]")
        self.iterations = iterations
        self.tissue_height = tissue_height
        self.active_latent = active_latent
        self.norm_attention = nn.LayerNorm(dim)
        self.norm_vertical = nn.LayerNorm(dim)
        self.norm_gate = nn.LayerNorm(dim)
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
        self.gate_head = nn.Linear(dim, 1, bias=False)
        self._initialize_gru(carry_bias)
        row = torch.arange(tissue_height)
        receiver = row[:, None]
        source = row[None, :]
        allowed = (receiver - source).abs() > 1
        self.register_buffer("dynamic_allowed", allowed, persistent=False)

    def _initialize_gru(self, carry_bias: float) -> None:
        if carry_bias == 0.0:
            return
        hidden = self.gru.hidden_size
        with torch.no_grad():
            self.gru.bias_ih[hidden : 2 * hidden].fill_(carry_bias / 2.0)
            self.gru.bias_hh[hidden : 2 * hidden].fill_(carry_bias / 2.0)

    def _activity_gate(
        self,
        perception: torch.Tensor,
        *,
        variant: SparseTopologyVariant,
        ablate_row: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, tissue, _ = perception.shape
        if not variant.sparse_activity:
            hard = torch.ones(batch, length, tissue, device=perception.device, dtype=perception.dtype)
            if ablate_row is not None:
                hard[..., ablate_row] = 0.0
            zero = perception.new_zeros(())
            return hard, hard.detach(), zero
        latent_logits = self.gate_head(self.norm_gate(perception[:, :, 1:, :])).squeeze(-1)
        if ablate_row is not None and ablate_row > 0:
            latent_logits = latent_logits.clone()
            latent_logits[..., ablate_row - 1] = torch.finfo(latent_logits.dtype).min
        soft = torch.softmax(latent_logits.float(), dim=-1).to(dtype=perception.dtype)
        topk = torch.topk(latent_logits, k=self.active_latent, dim=-1).indices
        hard_latent = torch.zeros_like(soft).scatter_(-1, topk, 1.0)
        latent_gate = hard_latent + soft - soft.detach()
        row0 = torch.ones(batch, length, 1, device=perception.device, dtype=perception.dtype)
        gate = torch.cat((row0, latent_gate), dim=-1)
        hard = torch.cat((row0, hard_latent), dim=-1)
        if ablate_row == 0:
            gate = gate.clone()
            hard = hard.clone()
            gate[..., 0] = 0.0
            hard[..., 0] = 0.0
        mean_soft = soft.float().mean(dim=(0, 1))
        target = torch.full_like(mean_soft, 1.0 / mean_soft.numel())
        balance = (mean_soft - target).square().mean()
        return gate, hard.detach(), balance

    def _dynamic_message(
        self,
        perception: torch.Tensor,
        *,
        hard_gate: torch.Tensor,
        variant: SparseTopologyVariant,
        ablate_row: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, tissue, _ = perception.shape
        if not variant.dynamic_edges:
            return torch.zeros_like(perception), perception.new_zeros(batch, length, tissue, tissue)
        # No expert/router-specific parameters. Connectivity emerges from shared cell state.
        normalized = F.normalize(perception.float(), dim=-1).to(dtype=perception.dtype)
        # score[..., receiver, source]
        score = torch.einsum("blrd,blsd->blrs", normalized, normalized) * 5.0
        allowed = self.dynamic_allowed.view(1, 1, tissue, tissue)
        if ablate_row is not None:
            allowed = allowed.clone()
            allowed[..., :, ablate_row] = False
        score = score.masked_fill(~allowed, torch.finfo(score.dtype).min)
        soft = torch.softmax(score.float(), dim=-1).to(dtype=perception.dtype)
        source = score.argmax(dim=-1, keepdim=True)
        hard = torch.zeros_like(soft).scatter_(-1, source, 1.0)
        weights = hard + soft - soft.detach()
        message = torch.einsum("blrs,blsd->blrd", weights, perception)
        message = message * hard_gate.unsqueeze(-1)
        # source->receiver orientation for aggregation.
        edge_usage = hard.transpose(-1, -2) * hard_gate.unsqueeze(-2)
        return message, edge_usage.detach()

    def forward_variable(
        self,
        state: torch.Tensor,
        *,
        iterations: int,
        variant: SparseTopologyVariant,
        ablate_row: int | None = None,
        collect_topology: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TopologyDiagnostics | None]:
        if state.ndim != 4:
            raise ValueError("sparse tissue state must be [batch, length, tissue, dim]")
        if iterations < 1 or iterations > self.iterations:
            raise ValueError("iterations must be within the trained stage range")
        if ablate_row is not None and (ablate_row < 0 or ablate_row >= state.shape[2]):
            raise ValueError("ablate_row is outside tissue height")
        batch, length, tissue, dim = state.shape
        activity_sum = state.new_zeros(batch, length, tissue) if collect_topology else None
        edge_sum = state.new_zeros(batch, length, tissue, tissue) if collect_topology else None
        balance_terms: list[torch.Tensor] = []
        last_before = state
        for index in range(iterations):
            if ablate_row is not None:
                state = state.clone()
                state[..., ablate_row, :] = 0.0
            last_before = state
            conditioned = state + self.step_embedding[index].view(1, 1, 1, dim)
            rows = conditioned.permute(0, 2, 1, 3).reshape(batch * tissue, length, dim)
            horizontal = self.attention(self.norm_attention(rows))
            horizontal = horizontal.reshape(batch, tissue, length, dim).permute(0, 2, 1, 3).contiguous()
            vertical = self.vertical(self.norm_vertical(conditioned))
            perception = conditioned + horizontal + vertical
            gate, hard_gate, balance = self._activity_gate(
                perception,
                variant=variant,
                ablate_row=ablate_row,
            )
            dynamic, hard_edges = self._dynamic_message(
                perception,
                hard_gate=hard_gate,
                variant=variant,
                ablate_row=ablate_row,
            )
            candidate_state = state + horizontal + vertical + dynamic
            ffn_delta = self.ffn(self.norm_ffn(candidate_state))
            proposal = horizontal + vertical + dynamic + ffn_delta
            proposed = self.gru(
                proposal.reshape(batch * length * tissue, dim),
                state.reshape(batch * length * tissue, dim),
            ).view(batch, length, tissue, dim)
            state = state + gate.unsqueeze(-1) * (proposed - state)
            if ablate_row is not None:
                state = state.clone()
                state[..., ablate_row, :] = 0.0
            balance_terms.append(balance)
            if collect_topology:
                assert activity_sum is not None and edge_sum is not None
                activity_sum.add_(hard_gate)
                edge_sum.add_(hard_edges)
        residual = _relative_residual(last_before, state)
        balance_loss = torch.stack(balance_terms).mean() if balance_terms else state.new_zeros(())
        diagnostics = None
        if collect_topology:
            assert activity_sum is not None and edge_sum is not None
            activity = activity_sum / iterations
            edges = edge_sum / iterations
            diagnostics = TopologyDiagnostics(
                activity=activity,
                edges=edges,
                logical_active_fraction=activity.float().mean(),
            )
        return state, residual, balance_loss, diagnostics


class SparseTopologyCLM(nn.Module):
    """Eight-row CLM with shared cell rules and optional sparse/dynamic tissue topology."""

    def __init__(
        self,
        *,
        vocab_size: int,
        variant: SparseTopologyVariant,
        tissue_height: int = TISSUE_HEIGHT,
        active_latent: int = ACTIVE_LATENT,
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
            raise ValueError("Experiment 015 uses exactly three recurrent stages")
        self.variant = variant
        self.tissue_height = tissue_height
        self.active_latent = active_latent
        self.max_context = max_context
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_context, dim)
        self.stages = nn.ModuleList(
            [
                SparseTopologyStage(
                    dim=dim,
                    heads=heads,
                    ffn_dim=ffn_dim,
                    window=window,
                    iterations=steps,
                    tissue_height=tissue_height,
                    active_latent=active_latent,
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

    def _initial_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.max_context:
            raise ValueError(f"input_ids must be [batch, <= {self.max_context}]")
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        base = self.position_embedding(positions)[None, :, None, :] + self.row_encoding[None, None, :, :]
        state = base.expand(batch, -1, -1, -1).clone()
        state[:, :, 0, :] += self.token_embedding(input_ids)
        return state

    def forward_variable(
        self,
        input_ids: torch.Tensor,
        *,
        stage_depths: tuple[int, int, int],
        collect_topology: bool = False,
        ablate_row: int | None = None,
    ) -> SparseTopologyForward:
        if len(stage_depths) != len(self.stages):
            raise ValueError("stage_depths must match recurrent stage count")
        state = self._initial_state(input_ids)
        residuals: list[torch.Tensor] = []
        balances: list[torch.Tensor] = []
        activity_accum = None
        edge_accum = None
        diagnostic_stages = 0
        for stage, depth in zip(self.stages, stage_depths):
            state, residual, balance, diagnostics = stage.forward_variable(
                state,
                iterations=depth,
                variant=self.variant,
                ablate_row=ablate_row,
                collect_topology=collect_topology,
            )
            residuals.append(residual)
            balances.append(balance)
            if diagnostics is not None:
                diagnostic_stages += 1
                activity_accum = diagnostics.activity if activity_accum is None else activity_accum + diagnostics.activity
                edge_accum = diagnostics.edges if edge_accum is None else edge_accum + diagnostics.edges
        token_state = state[:, :, 0, :]
        logits = self.lm_head(self.final_norm(token_state))
        stability = torch.stack(residuals).mean()
        balance = torch.stack(balances).mean()
        diagnostics = None
        if collect_topology:
            assert activity_accum is not None and edge_accum is not None and diagnostic_stages > 0
            activity = activity_accum / diagnostic_stages
            edges = edge_accum / diagnostic_stages
            diagnostics = TopologyDiagnostics(
                activity=activity,
                edges=edges,
                logical_active_fraction=activity.float().mean(),
            )
        return SparseTopologyForward(
            output=LanguageModelOutput(logits),
            stability_loss=stability,
            balance_loss=balance,
            diagnostics=diagnostics,
        )

    def forward(self, input_ids: torch.Tensor) -> LanguageModelOutput:
        return self.forward_variable(input_ids, stage_depths=(4, 4, 4)).output


def build_sparse_topology_model(
    vocab_size: int,
    *,
    variant: str | SparseTopologyVariant,
    **kwargs,
) -> SparseTopologyCLM:
    resolved = variant_by_code(variant) if isinstance(variant, str) else variant
    return SparseTopologyCLM(vocab_size=vocab_size, variant=resolved, **kwargs)
