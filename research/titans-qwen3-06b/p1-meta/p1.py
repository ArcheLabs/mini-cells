"""Native Qwen3 attention with one deferred Titans store per chunk."""
import torch
from torch import nn
from torch.nn import functional as F

LAYER = 3
CHUNK = 512


class NativeMemory(nn.Module):
    def __init__(self, attention, memory):
        super().__init__()
        self.attention = attention
        self.memory = memory
        self.state = None
        self.pending = None

    def forward(self, hidden_states, position_embeddings, **kwargs):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        captured = {}
        modules = {"q": self.attention.q_norm, "k": self.attention.k_norm,
                   "v": self.attention.v_proj}

        def capture(name):
            def hook(_module, _inputs, output):
                captured[name] = output
            return hook

        handles = [module.register_forward_hook(capture(name))
                   for name, module in modules.items()]

        def fuse(_module, inputs):
            q, k = apply_rotary_pos_emb(
                captured["q"].transpose(1, 2), captured["k"].transpose(1, 2),
                *position_embeddings,
            )
            scale = 128 ** -0.25
            self.pending = (
                q.transpose(1, 2).flatten(2).float() * scale,
                k.transpose(1, 2).flatten(2).float() * scale,
                captured["v"].float(),
            )
            if self.state is None:
                return None
            read, _ = self.memory.read_only(self.pending, self.state)
            gate = (self.memory.history_mix_proj(hidden_states.float()).reshape(
                *hidden_states.shape[:2], 16, 128
            ) + self.memory.history_mix.reshape(1, 1, 16, 128)).sigmoid().flatten(2)
            return (((1 - gate) * inputs[0].float() + gate * read).to(inputs[0].dtype),)

        handles.append(self.attention.o_proj.register_forward_pre_hook(fuse))
        try:
            return self.attention(hidden_states, position_embeddings=position_embeddings, **kwargs)
        finally:
            for handle in handles:
                handle.remove()

    def commit(self):
        self.state = self.memory.store_only(
            self.pending, state=self.state, detach_mem_state=False,
        )
        self.pending = None


def install(model):
    from titans_pytorch.neural_memory import NeuralMemory

    config = model.config
    if (config.num_attention_heads, config.num_key_value_heads, config.head_dim) != (16, 8, 128):
        raise ValueError("Unexpected Qwen GQA geometry")
    model.requires_grad_(False)
    block = model.model.layers[LAYER]
    memory = NeuralMemory(
        dim=1024, dim_head=128, heads=8, retrieve_heads=16, chunk_size=CHUNK,
        batch_size=CHUNK, bypass_projections=True, heterogeneous_qkv=True,
        detach_inner_grads=False, mem_model_norm_add_residual=False,
        default_model_kwargs={"depth": 2, "expansion_factor": 1.0},
        max_grad_norm=1.0, default_step_transform_max_lr=.0025,
        history_context_dim=1024, history_mix_init=.1,
    )
    active = {id(p) for p in memory.memory_model_parameters}
    for p in memory.memory_model.parameters():
        if id(p) not in active:
            p.requires_grad_(False)
    wrapper = NativeMemory(block.self_attn, memory.to(next(model.parameters()).device))
    block.self_attn = wrapper
    return wrapper


def episode_loss(model, wrapper, tokens, *, audit=None):
    """One graph across all chunks, with an independent Qwen forward per chunk."""
    if tokens.shape != (1, 4 * CHUNK):
        raise ValueError("Expected one 2048-token episode")
    wrapper.state = None
    losses = []
    try:
        for offset in range(0, tokens.shape[1], CHUNK):
            ids = tokens[:, offset:offset + CHUNK]
            positions = torch.arange(offset, offset + CHUNK, device=ids.device)[None]
            logits = model(input_ids=ids, position_ids=positions, use_cache=False).logits
            losses.append(F.cross_entropy(logits[:, :-1].float().flatten(0, 1),
                                          ids[:, 1:].flatten()))
            if audit is not None and offset == CHUNK:
                fast = wrapper.state.updates
                slow = tuple(wrapper.memory.memory_model_parameters)
                gradients = torch.autograd.grad(losses[-1], (*fast, *slow), retain_graph=True)
                for name, group in (("early_fast", gradients[:len(fast)]),
                                    ("slow_initialization", gradients[len(fast):])):
                    norm = torch.stack([grad.float().norm() for grad in group]).norm()
                    if not torch.isfinite(norm) or norm.item() == 0:
                        raise RuntimeError(f"Invalid meta-gradient: {name}")
                    audit[name] = norm.item()
            wrapper.commit()
        return torch.stack(losses)
    finally:
        wrapper.state = None
        wrapper.pending = None
