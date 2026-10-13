"""Experiment 001: capacity matched dense and sparse causal language models."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class RoutedFeedForward(nn.Module):
    """Common block contract for routed FFNs; implementations own their routing policy."""
    collect_stats = False


class GlobalRouter(RoutedFeedForward):
    def __init__(self, width, hidden, experts, top_k):
        super().__init__()
        self.router = nn.Linear(width, experts, bias=False)
        self.cells = nn.ModuleList(FeedForward(width, hidden) for _ in range(experts))
        self.top_k = top_k
        self.collect_stats = False
        self.counts = None
        self.entropy_sum = None
        self.token_count = 0

    def reset_stats(self):
        self.counts = None
        self.entropy_sum = None
        self.token_count = 0

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        probabilities = self.router(flat).float().softmax(-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        output = torch.zeros_like(flat)
        for expert, cell in enumerate(self.cells):
            rows, slots = torch.where(indices == expert)
            if rows.numel():
                values = cell(flat[rows]) * weights[rows, slots, None].to(flat.dtype)
                output.index_add_(0, rows, values)
        assigned = F.one_hot(indices, len(self.cells)).float().mean((0, 1))
        balance = len(self.cells) * (assigned * probabilities.mean(0)).sum()
        if self.collect_stats:
            counts = torch.bincount(indices.flatten(), minlength=len(self.cells)).detach()
            entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum().detach()
            self.counts = counts if self.counts is None else self.counts + counts
            self.entropy_sum = entropy if self.entropy_sum is None else self.entropy_sum + entropy
            self.token_count += flat.shape[0]
        return output.reshape_as(x), balance


class Block(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        width = config['width']
        self.heads = config['heads']
        self.norm_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn = (GlobalRouter(width, config['expert_hidden'], config['experts'], config['top_k'])
                    if kind == 'global-router' else
                    FeedForward(width, config['expert_hidden'] * config['experts']))

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm_attention(x)).reshape(
            batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.projection(attended.transpose(1, 2).reshape(batch, length, width))
        normalized = self.norm_ffn(x)
        if isinstance(self.ffn, RoutedFeedForward):
            result, balance = self.ffn(normalized)
        else:
            result, balance = self.ffn(normalized), x.new_zeros(())
        return x + result, balance


class LanguageModel(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        if kind not in {'fixed', 'global-router'}:
            raise ValueError('Unknown model kind')
        self.kind = kind
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['width'])
        self.position = nn.Embedding(config['sequence_length'], config['width'])
        self.blocks = nn.ModuleList(Block(config, kind) for _ in range(config['layers']))
        self.norm = nn.LayerNorm(config['width'])
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embedding(tokens) + self.position(positions)
        auxiliary = x.new_zeros(())
        for block in self.blocks:
            x, balance = block(x)
            auxiliary = auxiliary + balance
        logits = F.linear(self.norm(x), self.embedding.weight)
        return logits, auxiliary / len(self.blocks)

    def routing_modules(self):
        return [block.ffn for block in self.blocks if isinstance(block.ffn, RoutedFeedForward)]

    def parameter_metrics(self):
        total = sum(p.numel() for p in self.parameters())
        config = self.config
        inactive = (config['experts'] - config['top_k']) * 2 * config['width'] * config['expert_hidden'] * config['layers']
        active = total - inactive if self.kind == 'global-router' else total
        # Matmul-only forward estimate: attention projections, score/value products,
        # FFN, router, and tied output projection. Excludes norms, softmax and dispatch.
        width, length = config['width'], config['sequence_length']
        hidden = config['expert_hidden'] * (config['top_k'] if self.kind == 'global-router' else config['experts'])
        per_layer = 8 * width * width + 4 * length * width + 4 * width * hidden
        if self.kind == 'global-router':
            per_layer += 2 * width * config['experts']
        return {'params': total, 'activated_params_per_token_estimate': active,
                'forward_flops_per_token_estimate': config['layers'] * per_layer + 2 * width * config['vocab_size'],
                'flops_estimate_scope': 'matmuls only; excludes dispatch, norms, activations and softmax'}
