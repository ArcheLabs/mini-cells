"""Single attention insertion for the paired Qwen layer-location experiment."""
import math

import torch
from torch import nn
from torch.nn import functional as F

MODEL = "Qwen/Qwen3-0.6B-Base"
REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
UPSTREAM = "c60b4efaa3a3c6df4deebd57cbccf75947b87cad"
LAYERS = (3, 22)


class AttentionMemory(nn.Module):
    def __init__(self, attention, memory):
        super().__init__()
        self.attention = attention
        self.memory = memory
        self.query = nn.Linear(1024, 512, bias=False)
        self.key = nn.Linear(1024, 512, bias=False)
        self.value = nn.Linear(1024, 512, bias=False)
        self.output = nn.Linear(512, 1024, bias=False)
        self.gate = nn.Linear(1024, 1024)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(0.05 / 0.95))
        self.reset_episode()

    def reset_episode(self):
        self.cell = None
        self.position = 0
        self.read_enabled = True
        self.write_enabled = True

    def forward(self, hidden_states, *args, **kwargs):
        native, *auxiliary = self.attention(hidden_states, *args, **kwargs)
        x = hidden_states.float()
        q = F.normalize(self.query(x), dim=-1).unsqueeze(1)
        k = F.normalize(self.key(x), dim=-1).unsqueeze(1)
        v = self.value(x).unsqueeze(1)
        if not self.write_enabled:
            if self.cell is None:
                return (native, *auxiliary)
            read, _ = self.memory.read_only(
                (q.squeeze(1) * q.shape[-1] ** -0.25, k.squeeze(1), v.squeeze(1)),
                self.cell,
            )
        else:
            read, _, self.cell = self.memory(
                q, k, v, state=self.cell, detach_mem_state=False, seq_offset=self.position
            )
        self.position += x.shape[1]
        if not self.read_enabled:
            return (native, *auxiliary)
        gate = self.gate(x).sigmoid()
        mixed = (1 - gate) * native.float() + gate * self.output(read)
        return (mixed.to(native.dtype), *auxiliary)


def install(model, layer, memory):
    if layer not in LAYERS:
        raise ValueError(f"Expected layer index in {LAYERS}")
    config = model.config
    expected = dict(model_type="qwen3", num_hidden_layers=28, hidden_size=1024,
                    num_attention_heads=16, num_key_value_heads=8,
                    head_dim=128, intermediate_size=3072)
    if any(getattr(config, name, None) != value for name, value in expected.items()):
        raise ValueError("Unexpected Qwen architecture")
    blocks = model.model.layers
    if any(isinstance(block.self_attn, AttentionMemory) for block in blocks):
        raise ValueError("The experiment permits exactly one Memory insertion")
    model.requires_grad_(False)
    attention = blocks[layer].self_attn
    wrapper = AttentionMemory(attention, memory).to(device=next(attention.parameters()).device)
    blocks[layer].self_attn = wrapper
    return wrapper


def build_memory(upstream_checkout):
    """Load the pinned upstream implementation and keep its adaptive updates."""
    from pathlib import Path
    import subprocess
    import sys

    root = Path(upstream_checkout).resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if revision != UPSTREAM:
        raise ValueError(f"Expected Titans revision {UPSTREAM}, got {revision}")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "titans_pytorch"],
        cwd=root, text=True,
    ).strip():
        raise ValueError("Pinned Titans package has local modifications")
    sys.path.insert(0, str(root))
    try:
        from titans_pytorch import neural_memory
    finally:
        sys.path.pop(0)
    if root not in Path(neural_memory.__file__).resolve().parents:
        raise RuntimeError("Titans was imported from a different checkout")
    memory = neural_memory.NeuralMemory(
        dim=512, heads=1, chunk_size=1, bypass_projections=True,
        heterogeneous_qkv=True,
        detach_inner_grads=False, mem_model_norm_add_residual=False,
        default_model_kwargs={"depth": 2, "expansion_factor": 4.0},
    )
    # Upstream retains the template module alongside its active parameter bank.
    active = {id(parameter) for parameter in memory.memory_model_parameters}
    for parameter in memory.memory_model.parameters():
        if id(parameter) not in active:
            parameter.requires_grad_(False)
    return memory
