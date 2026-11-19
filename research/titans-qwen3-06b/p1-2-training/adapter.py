"""Independent memory address path for P1.2-TR."""
from copy import deepcopy
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p1-meta"))
from p1 import NativeMemory, install as install_native

ARMS = {"A0": (512, False), "A1": (512, False), "A2": (16, False),
        "A3": (512, True), "A4": (16, True)}


class ProjectedMemory(NativeMemory):
    def __init__(self, attention, memory, trainable):
        super().__init__(attention, memory)
        for name in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm"):
            setattr(self, "memory_" + name,
                    deepcopy(getattr(attention, name)).float().requires_grad_(trainable))

    def project(self, hidden):
        x = hidden.float()
        q = self.memory_q_norm(self.memory_q_proj(x).unflatten(-1, (16, 128)))
        k = self.memory_k_norm(self.memory_k_proj(x).unflatten(-1, (8, 128)))
        scale = 128 ** -.25
        return q.flatten(2)*scale, k.flatten(2)*scale, self.memory_v_proj(x)

    def forward(self, hidden_states, position_embeddings, **kwargs):
        self.pending = self.project(hidden_states)

        def fuse(_module, inputs):
            if self.state is None:
                return None
            read, _ = self.memory.read_only(self.pending, self.state)
            if self.read_override == "zero":
                read = torch.zeros_like(read)
            gate = (self.memory.history_mix_proj(hidden_states.float())
                    + self.memory.history_mix.reshape(1, 1, -1)).sigmoid()
            return (((1-gate)*inputs[0].float()+gate*read).to(inputs[0].dtype),)

        handle = self.attention.o_proj.register_forward_pre_hook(fuse)
        try:
            return self.attention(hidden_states, position_embeddings=position_embeddings, **kwargs)
        finally:
            handle.remove()


def install(model, arm):
    chunk, trainable = ARMS[arm]
    if arm == "A0":
        return install_native(model)
    from titans_pytorch.neural_memory import NeuralMemory

    config = model.config
    if (config.num_attention_heads, config.num_key_value_heads, config.head_dim) != (16, 8, 128):
        raise ValueError("Unexpected Qwen geometry")
    model.requires_grad_(False)
    block = model.model.layers[3]
    memory = NeuralMemory(
        dim=1024, dim_head=128, heads=8, retrieve_heads=16,
        chunk_size=chunk, batch_size=512, bypass_projections=True,
        heterogeneous_qkv=True, detach_inner_grads=False,
        mem_model_norm_add_residual=False,
        default_model_kwargs={"depth": 2, "expansion_factor": 1.0},
        max_grad_norm=1.0, default_step_transform_max_lr=.0025,
        history_context_dim=1024, history_mix_init=.1,
    )
    memory.retrieve_chunk_size = 512
    if (memory.chunk_size, memory.store_chunk_size, memory.retrieve_chunk_size,
        memory.batch_size) != (chunk, chunk, 512, 512):
        raise RuntimeError("Unexpected pinned Titans chunk fields")
    active = {id(p) for p in memory.memory_model_parameters}
    for p in memory.memory_model.parameters():
        if id(p) not in active:
            p.requires_grad_(False)
    wrapper = ProjectedMemory(block.self_attn, memory.to(next(model.parameters()).device), trainable)
    block.self_attn = wrapper
    return wrapper
