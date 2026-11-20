"""Gate 1 native Qwen addresses and upstream one-token Titans recurrence."""
import torch
from torch import nn

MODEL = "Qwen/Qwen3-0.6B-Base"
REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
TITANS = "c60b4efaa3a3c6df4deebd57cbccf75947b87cad"
LAYER = 22
SEGMENT = 128
ARMS = {"G1-A": (False, .0025), "G1-B": (False, 1.),
        "G1-C": (True, .0025), "G1-D": (True, 1.)}


class OnlineMemory(nn.Module):
    def __init__(self, attention, memory):
        super().__init__()
        self.attention, self.memory = attention, memory
        self.reset_episode()

    def reset_episode(self):
        self.state = None
        self.position = 0
        self.read_only = False
        self.zero_read = False

    def forward(self, hidden_states, position_embeddings, **kwargs):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        if hidden_states.shape[1] != 1:
            raise ValueError("Gate 1 requires one-token forwards")
        captured = {}

        def capture(name):
            def hook(_module, _args, output):
                captured[name] = output
            return hook

        def fuse(_module, inputs):
            q, k = apply_rotary_pos_emb(captured["q"].transpose(1, 2),
                                        captured["k"].transpose(1, 2), *position_embeddings)
            v = captured["v"].unflatten(-1, (8, 128)).transpose(1, 2)
            if self.read_only:
                if self.state is None:
                    return None
                addr = (q.float() * 128**-.25).transpose(1,2).flatten(2)
                read, _ = self.memory.read_only((addr,None,None),self.state)
                if self.zero_read:
                    read = torch.zeros_like(read)
                gate = (self.memory.history_mix_proj(hidden_states.float())
                        + self.memory.history_mix.flatten()).sigmoid()
                return (((1-gate)*inputs[0].float()+gate*read).to(inputs[0].dtype),)
            read, gate, self.state = self.memory(q.float(), k.float(), v.float(),
                state=self.state, context=hidden_states.float(),
                detach_mem_state=False, seq_offset=self.position)
            self.position += 1
            if self.state.seq_index != self.position:
                raise RuntimeError("Expected exactly one stored token")
            gate = gate.flatten(2)
            return (((1-gate)*inputs[0].float()+gate*read).to(inputs[0].dtype),)

        handles = [module.register_forward_hook(capture(name)) for name, module in
                   (("q", self.attention.q_norm), ("k", self.attention.k_norm),
                    ("v", self.attention.v_proj))]
        handles.append(self.attention.o_proj.register_forward_pre_hook(fuse))
        try:
            return self.attention(hidden_states, position_embeddings=position_embeddings, **kwargs)
        finally:
            for handle in handles:
                handle.remove()


def install(model, arm, *, identity_validated=False):
    from titans_pytorch.neural_memory import NeuralMemory

    cfg = model.config
    if (len(model.model.layers), cfg.hidden_size, cfg.num_attention_heads,
        cfg.num_key_value_heads, cfg.head_dim) != (28, 1024, 16, 8, 128):
        raise ValueError("Unexpected backbone geometry")
    identity, lr = ARMS[arm]
    if identity and not identity_validated:
        raise ValueError("Identity initialization requires Gate 0 compatibility evidence")
    model.requires_grad_(False)
    block = model.model.layers[LAYER]
    attention = block.self_attn
    memory = NeuralMemory(dim=1024, dim_head=128, heads=8, retrieve_heads=16,
        chunk_size=1, batch_size=1, bypass_projections=True, heterogeneous_qkv=True,
        detach_inner_grads=False, mem_model_norm_add_residual=False,
        default_model_kwargs={"depth": 2, "expansion_factor": 1.0}, max_grad_norm=1.,
        default_step_transform_max_lr=lr, history_context_dim=1024, history_mix_init=.05
    ).to(next(model.parameters()).device)
    if identity:
        memory.initialize_kv_memory(attention.k_proj.weight, attention.v_proj.weight, identity=True)
    active = {id(p) for p in memory.memory_model_parameters}
    for p in memory.memory_model.parameters():
        if id(p) not in active:
            p.requires_grad_(False)
    block.self_attn = OnlineMemory(attention, memory)
    return block.self_attn
