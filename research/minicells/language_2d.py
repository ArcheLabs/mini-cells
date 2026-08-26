from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .language_models import LanguageModelOutput, LocalCausalSelfAttention, RMSNorm, TextNCALM


@dataclass(frozen=True)
class TissueDiagnostics:
    row_update_rms: tuple[tuple[float, ...], ...]
    row_cosine_to_token: tuple[float, ...]


class VerticalDepthwiseMixer(nn.Module):
    """Local same-position communication along the latent-tissue axis.

    The mixer is deliberately depthwise so that adding a tissue dimension changes
    the topology and compute much more than it changes parameter capacity.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=True,
        )
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.conv.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 4:
            raise ValueError("vertical mixer expects [batch, length, tissue, dim]")
        batch, length, tissue, dim = state.shape
        columns = state.permute(0, 1, 3, 2).reshape(batch * length, dim, tissue)
        mixed = self.conv(columns)
        return mixed.reshape(batch, length, dim, tissue).permute(0, 1, 3, 2).contiguous()


class NCAStage2D(nn.Module):
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

    @staticmethod
    def _zero_row(state: torch.Tensor, row: int | None) -> torch.Tensor:
        if row is None:
            return state
        if row < 0 or row >= state.shape[2]:
            raise ValueError(f"ablation row {row} is outside tissue height {state.shape[2]}")
        mask = torch.ones(state.shape[2], device=state.device, dtype=state.dtype)
        mask[row] = 0
        return state * mask.view(1, 1, -1, 1)

    def forward(
        self,
        state: torch.Tensor,
        *,
        ablate_row: int | None = None,
        collect_updates: bool = False,
    ) -> tuple[torch.Tensor, tuple[tuple[float, ...], ...]]:
        if state.ndim != 4:
            raise ValueError("2D NCA stage expects [batch, length, tissue, dim]")
        batch, length, tissue, dim = state.shape
        state = self._zero_row(state, ablate_row)
        updates: list[tuple[float, ...]] = []

        for step in range(self.iterations):
            before = state
            conditioned = state + self.step_embedding[step].view(1, 1, 1, dim)

            # Horizontal communication is causal along the text/time axis and is
            # independently shared across every tissue row.
            rows = conditioned.permute(0, 2, 1, 3).reshape(batch * tissue, length, dim)
            horizontal = self.attention(self.norm_attention(rows))
            horizontal = (
                horizontal.reshape(batch, tissue, length, dim)
                .permute(0, 2, 1, 3)
                .contiguous()
            )

            # Vertical communication never changes text position, so it cannot
            # leak future tokens into a causal language prediction.
            vertical = self.vertical(self.norm_vertical(conditioned))
            candidate_state = state + horizontal + vertical
            ffn_delta = self.ffn(self.norm_ffn(candidate_state))
            proposal = horizontal + vertical + ffn_delta
            state = self.gru(
                proposal.reshape(batch * length * tissue, dim),
                state.reshape(batch * length * tissue, dim),
            ).view(batch, length, tissue, dim)
            state = self._zero_row(state, ablate_row)

            if collect_updates:
                delta = (state - before).float().pow(2).mean(dim=(0, 1, 3)).sqrt()
                updates.append(tuple(float(value) for value in delta.detach().cpu()))

        return state, tuple(updates)


class LatentTissueNCALM(nn.Module):
    """Causal language NCA with an explicit latent-tissue dimension.

    Axis 1 is sequence/time. Axis 2 is latent tissue. Only tissue row 0 receives
    token embeddings and only row 0 is decoded by the LM head; the remaining rows
    must earn their usefulness through local cellular communication.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        tissue_height: int = 4,
        max_context: int = 128,
        dim: int = 128,
        heads: int = 4,
        ffn_dim: int = 512,
        windows: tuple[int, int, int] = (8, 32, 128),
        iterations: tuple[int, int, int] = (4, 4, 4),
        carry_bias: float = 2.0,
        tie_embeddings: bool = True,
        stage_supervision: bool = False,
    ) -> None:
        super().__init__()
        if tissue_height < 1:
            raise ValueError("tissue_height must be positive")
        if len(windows) != 3 or len(iterations) != 3:
            raise ValueError("Experiment 009 requires exactly three NCA stages")
        self.tissue_height = tissue_height
        self.max_context = max_context
        self.stage_supervision = stage_supervision
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_context, dim)
        self.row_embedding = nn.Embedding(tissue_height, dim)
        self.stages = nn.ModuleList(
            [
                NCAStage2D(
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
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(TextNCALM._init_weights)
        nn.init.normal_(self.row_embedding.weight, mean=0.0, std=0.02)

    def _initial_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.max_context:
            raise ValueError(f"input_ids must be [batch, <= {self.max_context}]")
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        rows = torch.arange(self.tissue_height, device=input_ids.device)
        base = self.position_embedding(positions)[None, :, None, :] + self.row_embedding(rows)[
            None, None, :, :
        ]
        state = base.expand(batch, -1, -1, -1).clone()
        state[:, :, 0, :] = state[:, :, 0, :] + self.token_embedding(input_ids)
        return state

    def _run(
        self,
        input_ids: torch.Tensor,
        *,
        ablate_row: int | None = None,
        collect_updates: bool = False,
    ) -> tuple[LanguageModelOutput, torch.Tensor, tuple[tuple[tuple[float, ...], ...], ...]]:
        state = self._initial_state(input_ids)
        intermediate_logits: list[torch.Tensor] = []
        diagnostics: list[tuple[tuple[float, ...], ...]] = []
        for index, stage in enumerate(self.stages):
            state, stage_updates = stage(
                state,
                ablate_row=ablate_row,
                collect_updates=collect_updates,
            )
            diagnostics.append(stage_updates)
            if self.stage_supervision and index < len(self.stages) - 1:
                token_state = state[:, :, 0, :]
                intermediate_logits.append(self.lm_head(self.final_norm(token_state)))
        token_state = state[:, :, 0, :]
        final_logits = self.lm_head(self.final_norm(token_state))
        if self.stage_supervision:
            output = LanguageModelOutput(final_logits, tuple([*intermediate_logits, final_logits]))
        else:
            output = LanguageModelOutput(final_logits)
        return output, state, tuple(diagnostics)

    def forward(self, input_ids: torch.Tensor) -> LanguageModelOutput:
        output, _, _ = self._run(input_ids)
        return output

    def forward_with_ablation(self, input_ids: torch.Tensor, row: int) -> LanguageModelOutput:
        if row == 0:
            raise ValueError("row 0 is the token row; Experiment 009 only ablates latent rows")
        output, _, _ = self._run(input_ids, ablate_row=row)
        return output

    @torch.no_grad()
    def diagnose(self, input_ids: torch.Tensor) -> TissueDiagnostics:
        was_training = self.training
        self.eval()
        _, state, stage_updates = self._run(input_ids, collect_updates=True)
        token = state[:, :, 0, :].float()
        cosines = [1.0]
        for row in range(1, self.tissue_height):
            latent = state[:, :, row, :].float()
            cosine = F.cosine_similarity(token, latent, dim=-1).mean()
            cosines.append(float(cosine.detach().cpu()))
        if was_training:
            self.train()
        return TissueDiagnostics(
            row_update_rms=tuple(
                tuple(value for update in stage for value in update) for stage in stage_updates
            ),
            row_cosine_to_token=tuple(cosines),
        )


def build_minicells_2d(vocab_size: int, *, tissue_height: int = 4) -> LatentTissueNCALM:
    return LatentTissueNCALM(
        vocab_size=vocab_size,
        tissue_height=tissue_height,
        carry_bias=2.0,
        tie_embeddings=True,
        stage_supervision=False,
    )
