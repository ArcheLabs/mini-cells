"""A tiny Granite-shaped backend used only for engineering/unit verification."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ToyExperts(nn.Module):
    def __init__(self, hidden_size: int = 16, intermediate_size: int = 16, experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = experts
        self.hidden_dim = hidden_size
        self.intermediate_dim = intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(experts, 2 * intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(experts, hidden_size, intermediate_size))
        self.act_fn = F.silu
        self.top_k = top_k
        nn.init.normal_(self.gate_up_proj, std=0.08)
        nn.init.normal_(self.down_proj, std=0.08)

    def forward(self, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor) -> Tensor:
        output = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx_tensor in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
            expert_idx = int(expert_idx_tensor[0])
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = hidden_states[token_idx]
            gate, up = F.linear(current, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current = F.linear(F.silu(gate) * up, self.down_proj[expert_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            output.index_add_(0, token_idx, current.to(output.dtype))
        return output


class ToyRouter(nn.Module):
    def __init__(self, hidden_size: int, experts: int, top_k: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(experts, hidden_size))
        self.top_k = top_k
        nn.init.normal_(self.weight, std=0.08)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        probabilities = F.softmax(F.linear(hidden_states, self.weight), dim=-1)
        weights, indices = torch.topk(probabilities, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return indices, weights


class ToyMoE(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, experts: int, top_k: int) -> None:
        super().__init__()
        self.input_size = hidden_size
        self.router = ToyRouter(hidden_size, experts, top_k)
        self.experts = ToyExperts(hidden_size, intermediate_size, experts, top_k)

    def forward(self, layer_input: Tensor) -> Tensor:
        shape = layer_input.shape
        hidden = layer_input.reshape(-1, shape[-1])
        indices, weights = self.router(hidden)
        return self.experts(hidden, indices, weights).reshape(shape)


class ToyBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, experts: int, top_k: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.block_sparse_moe = ToyMoE(hidden_size, intermediate_size, experts, top_k)
        self.residual_multiplier = 1.0

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)
        return residual + hidden_states * self.residual_multiplier


class ToyBackbone(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, intermediate_size: int, experts: int, top_k: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([ToyBlock(hidden_size, intermediate_size, experts, top_k)])
        self.norm = nn.LayerNorm(hidden_size)


class ToyGraniteLikeModel(nn.Module):
    def __init__(self, seed: int = 26090501, vocab_size: int = 96, hidden_size: int = 16, intermediate_size: int = 16, experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.config = SimpleNamespace(
            model_type="toy-granitemoe",
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_local_experts=experts,
            num_experts_per_tok=top_k,
            num_hidden_layers=1,
            _name_or_path="toy://pcu-kill-001",
        )
        self.model = ToyBackbone(vocab_size, hidden_size, intermediate_size, experts, top_k)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, **_: object) -> SimpleNamespace:
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(self.model.norm(hidden)))


def make_toy_model(seed: int = 26090501) -> ToyGraniteLikeModel:
    return ToyGraniteLikeModel(seed=seed)
