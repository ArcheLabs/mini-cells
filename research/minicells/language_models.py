from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LanguageModelOutput:
    logits: torch.Tensor
    stage_logits: tuple[torch.Tensor, ...] = ()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * scale.to(dtype=x.dtype)) * self.weight


class LocalCausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, window: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        if window < 1:
            raise ValueError("window must be positive")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.window = window
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def _mask(self, length: int, device: torch.device) -> torch.Tensor:
        query = torch.arange(length, device=device)[:, None]
        key = torch.arange(length, device=device)[None, :]
        allowed = (key <= query) & (key >= query - self.window + 1)
        return allowed[None, None, :, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self._mask(length, x.device),
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, self.dim)
        return self.out(attended)


class NCAStage(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        ffn_dim: int,
        window: int,
        iterations: int,
        rms_norm: bool,
        carry_bias: float,
    ) -> None:
        super().__init__()
        norm = RMSNorm if rms_norm else nn.LayerNorm
        self.iterations = iterations
        self.norm_attention = norm(dim)
        self.norm_ffn = norm(dim)
        self.attention = LocalCausalSelfAttention(dim, heads, window)
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
        # PyTorch GRU gate order is reset, update, new. A positive update-gate
        # bias makes the initial recurrent dynamics prefer carrying old state.
        with torch.no_grad():
            self.gru.bias_ih[hidden : 2 * hidden].fill_(carry_bias / 2.0)
            self.gru.bias_hh[hidden : 2 * hidden].fill_(carry_bias / 2.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch, length, dim = state.shape
        for step in range(self.iterations):
            conditioned = state + self.step_embedding[step].view(1, 1, dim)
            attention_delta = self.attention(self.norm_attention(conditioned))
            candidate_state = state + attention_delta
            ffn_delta = self.ffn(self.norm_ffn(candidate_state))
            proposal = attention_delta + ffn_delta
            state = self.gru(
                proposal.reshape(batch * length, dim),
                state.reshape(batch * length, dim),
            ).view(batch, length, dim)
        return state


class TextNCALM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_context: int = 128,
        dim: int = 128,
        heads: int = 4,
        ffn_dim: int = 512,
        windows: tuple[int, int, int] = (8, 32, 128),
        iterations: tuple[int, int, int] = (4, 4, 4),
        rms_norm: bool = False,
        carry_bias: float = 0.0,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if len(windows) != 3 or len(iterations) != 3:
            raise ValueError("Experiment 005 requires exactly three NCA stages")
        self.max_context = max_context
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_context, dim)
        self.stages = nn.ModuleList(
            [
                NCAStage(
                    dim=dim,
                    heads=heads,
                    ffn_dim=ffn_dim,
                    window=window,
                    iterations=steps,
                    rms_norm=rms_norm,
                    carry_bias=carry_bias,
                )
                for window, steps in zip(windows, iterations)
            ]
        )
        self.final_norm = RMSNorm(dim) if rms_norm else nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> LanguageModelOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.max_context:
            raise ValueError(f"input_ids must be [batch, <= {self.max_context}]")
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        state = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        stage_logits: list[torch.Tensor] = []
        for stage in self.stages:
            state = stage(state)
            stage_logits.append(self.lm_head(self.final_norm(state)))
        return LanguageModelOutput(stage_logits[-1], tuple(stage_logits))


class TransformerBlock(nn.Module):
    def __init__(self, *, dim: int, heads: int, ffn_dim: int, max_context: int) -> None:
        super().__init__()
        self.norm_attention = RMSNorm(dim)
        self.attention = LocalCausalSelfAttention(dim, heads, max_context)
        self.norm_ffn = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_attention(x))
        return x + self.ffn(self.norm_ffn(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_context: int = 128,
        dim: int = 128,
        heads: int = 4,
        ffn_dim: int = 512,
        layers: int = 5,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.max_context = max_context
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_context, dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    heads=heads,
                    ffn_dim=ffn_dim,
                    max_context=max_context,
                )
                for _ in range(layers)
            ]
        )
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(TextNCALM._init_weights)

    def forward(self, input_ids: torch.Tensor) -> LanguageModelOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.max_context:
            raise ValueError(f"input_ids must be [batch, <= {self.max_context}]")
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        state = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            state = block(state)
        return LanguageModelOutput(self.lm_head(self.final_norm(state)))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_textnca_control(vocab_size: int) -> TextNCALM:
    return TextNCALM(
        vocab_size=vocab_size,
        rms_norm=False,
        carry_bias=0.0,
        tie_embeddings=True,
    )


def build_minitextnca_plus(vocab_size: int) -> TextNCALM:
    return TextNCALM(
        vocab_size=vocab_size,
        rms_norm=True,
        carry_bias=2.0,
        tie_embeddings=True,
    )


def build_parameter_matched_transformer(
    vocab_size: int,
    target_parameters: int,
) -> tuple[TransformerLM, dict[str, int | float]]:
    candidates: list[tuple[float, TransformerLM, int, int, int]] = []
    for layers in range(3, 9):
        for ffn_dim in (384, 448, 512, 576, 640):
            model = TransformerLM(
                vocab_size=vocab_size,
                layers=layers,
                ffn_dim=ffn_dim,
            )
            parameters = count_parameters(model)
            relative_error = abs(parameters - target_parameters) / target_parameters
            candidates.append((relative_error, model, layers, ffn_dim, parameters))
    error, model, layers, ffn_dim, parameters = min(candidates, key=lambda item: item[0])
    return model, {
        "layers": layers,
        "ffn_dim": ffn_dim,
        "parameters": parameters,
        "target_parameters": target_parameters,
        "relative_parameter_error": error,
    }
