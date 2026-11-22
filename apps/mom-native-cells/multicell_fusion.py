"""Explicit read-only fusion for the pinned MoM runtime."""
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
import math
import time

import torch
from torch import nn
from torch.nn import functional as F


MODES = ("single_cell", "legacy_equal", "hierarchical_equal", "hierarchical_gate",
         "global_factorized", "global_topk_gate", "global_all4_gate")
LEARNED = ("hierarchical_gate", "global_topk_gate", "global_all4_gate")


@dataclass(frozen=True)
class FusionConfig:
    mode: str = "legacy_equal"
    router_prior: float = 0.0
    null_candidate: bool = False

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Unknown fusion mode {self.mode}")
        if not math.isfinite(self.router_prior) or self.router_prior < 0:
            raise ValueError("Router prior must be finite and nonnegative")
        if self.router_prior and self.mode != "global_all4_gate":
            raise ValueError("Router prior is supported in All-4")
        if self.null_candidate and self.mode not in LEARNED:
            raise ValueError("Null requires a learned fusion mode")


@dataclass
class CellReadResult:
    routed_outputs: torch.Tensor  # batch, token, selected lane, head, width
    router_probabilities: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    shared_output: torch.Tensor  # batch, token, head, width


class FusionGate(nn.Module):
    def __init__(self, width=1024, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width * 3, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, 1, bias=False))

    def forward(self, hidden, candidates):
        q = F.layer_norm(hidden.float(), (hidden.shape[-1],)).unsqueeze(-2)
        v = F.layer_norm(candidates.float(), (candidates.shape[-1],))
        q = q.expand_as(v)
        return self.net(torch.cat((q, v, q * v), dim=-1)).squeeze(-1)


def weights(gate, hidden, candidates, config, prior=None):
    if config.null_candidate:
        candidates = torch.cat((candidates, torch.zeros_like(candidates[..., :1, :])), dim=-2)
    logits = gate(hidden, candidates)
    if prior is not None:
        prior = prior.clamp_min(torch.finfo(torch.float32).tiny).log() * config.router_prior
        if config.null_candidate:
            prior = F.pad(prior, (0, 1))
        logits = logits + prior
    return logits.softmax(dim=-1)


def fuse(hidden, reads, config, gate=None, removed_unit=None):
    routed = torch.stack([r.routed_outputs for r in reads], dim=2).float()
    shared = torch.stack([r.shared_output for r in reads], dim=2).float()
    native = reads[0].topk_weights.float()
    if config.mode == "global_all4_gate":
        ix = reads[0].topk_indices[:, :, None, :, None, None]
        ix = ix.expand(*routed.shape[:3], ix.shape[3], *routed.shape[-2:])
        selected = routed.gather(3, ix)
    else:
        selected = routed
    within = (selected * native[:, :, None, :, None, None]).sum(dim=3)
    cell_values = within + shared
    if config.mode in LEARNED:
        alpha = weights(gate, hidden, cell_values.flatten(-2), config)
    else:
        alpha = torch.full_like(cell_values[..., 0, 0], 1 / len(reads))
    cell = alpha[..., :len(reads)]
    beta = None
    if config.mode.startswith("hierarchical"):
        mixed = (cell_values * cell[..., None, None]).sum(dim=2)
    else:
        if config.mode == "global_factorized":
            beta = (cell.unsqueeze(-1) * native.unsqueeze(2)).flatten(2)
        else:
            prior = None
            if config.router_prior:
                prior = reads[0].router_probabilities.unsqueeze(2)
                prior = prior.expand(*routed.shape[:4]).flatten(2)
            beta = weights(gate, hidden, routed.flatten(2, 3).flatten(-2), config, prior)
        if removed_unit is not None:
            if config.mode == "global_all4_gate":
                allowed = torch.arange(routed.shape[3], device=hidden.device) != removed_unit
                allowed = allowed.expand(*routed.shape[:4]).flatten(2)
            else:
                allowed = (reads[0].topk_indices != removed_unit).unsqueeze(2)
                allowed = allowed.expand(*routed.shape[:4]).flatten(2)
            if config.null_candidate:
                allowed = F.pad(allowed, (0, 1), value=True)
            beta = beta * allowed
            beta = beta / beta.sum(-1, keepdim=True).clamp_min(torch.finfo(beta.dtype).tiny)
        candidates = routed.flatten(2, 3)
        routed_weights = beta[..., :candidates.shape[2]]
        mixed = (candidates * routed_weights[..., None, None]).sum(dim=2)
        mixed = mixed + (shared * cell[..., None, None]).sum(dim=2)
    if len(reads) == 1 and config.mode in ("hierarchical_equal", "global_factorized"):
        # Match native FP16 rounding at the two intermediate reductions.
        read = reads[0]
        mixed = (read.routed_outputs * read.topk_weights[..., None, None]).sum(dim=2)
        mixed = mixed + read.shared_output
    return mixed.to(reads[0].shared_output.dtype), alpha, beta


class Diagnostics:
    def __init__(self, timing=False, removed_unit=None):
        self.layers = []
        self.timing = timing
        self.removed_unit = removed_unit
        self.read_seconds = 0.0
        self.fusion_seconds = 0.0

    @contextmanager
    def measure(self, field, device):
        if not self.timing:
            yield
            return
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        yield
        torch.cuda.synchronize(device)
        setattr(self, field, getattr(self, field) + time.perf_counter() - started)

    def record(self, layer, alpha, beta, selected, all4, null):
        def stats(p):
            p = p.detach().float()
            return dict(mean=p.mean(dim=(0, 1)).cpu().tolist(),
                        entropy=float(-(p * p.clamp_min(1e-30).log()).sum(-1).mean()),
                        top1=p.argmax(-1).cpu().tolist(),
                        null_weight=float(p[..., -1].mean()) if null else None)
        self.layers.append(dict(layer=layer, cell=stats(alpha), unit=stats(beta) if beta is not None else None,
                                unit_layout="all4" if all4 else "native_topk",
                                topk=selected.detach().cpu().tolist()))


def pack_projections(attn, packed):
    q = attn.q_proj(packed)
    if attn.single_kv_proj:
        k, v = attn.shared_k(packed), attn.shared_v(packed)
        beta, a = attn.shared_b(packed), attn.shared_a(packed)
    else:
        k, v, beta, a = (
            torch.stack([projection(x) for projection, x in zip(experts, packed, strict=True)])
            for experts in (attn.k_proj, attn.v_proj, attn.b_proj, attn.a_proj)
        )
    return q, k, v, beta.sigmoid(), -attn.A_log.float().exp() * F.softplus(a.float() + attn.dt_bias)


def recurrence(attn, ops, projected, conv, states, use_cache, active):
    q, k, v, beta, g = projected
    banks, batch, length, _ = q.shape
    convolved = []
    for layer, value, history in zip((attn.q_conv1d, attn.k_conv1d, attn.v_conv1d),
                                     (q, k, v), conv, strict=True):
        value, updated = layer(x=value.flatten(0, 1), cache=history, output_final_state=use_cache)
        convolved.append((value.reshape(banks, batch, length, attn.num_heads, -1), updated))
    (q, q_conv), (k, k_conv), (v, v_conv) = convolved
    q, k = ops.l2_norm(q), ops.l2_norm(k)
    outputs = []
    for bank, state in enumerate(states):
        if not active[bank]:
            # Native decode preserves recurrent state for unselected expert banks.
            outputs.append(torch.zeros_like(v[bank]))
            continue
        result, final = ops.fused_recurrent_gated_delta_rule(
            q=q[bank], k=k[bank], v=v[bank], g=g[bank], beta=beta[bank], initial_state=state,
            output_final_state=use_cache, cu_seqlens=None, head_first=False,
        )
        outputs.append(result)
        states[bank] = final
    return torch.stack(outputs), (q_conv, k_conv, v_conv)


def read_layer(attn, ops, hidden, caches, config, gate, diagnostics):
    batch, length, _ = hidden.shape
    if batch != 1 or not 1 <= length <= 64:
        raise ValueError("Experimental fusion expects one unpadded segment of up to 64 tokens")
    router_logits = attn.gate(hidden)
    probs = router_logits.softmax(dim=-1, dtype=torch.float32)
    top, selected = probs.topk(attn.topk, dim=-1)
    top = (top / top.sum(dim=-1, keepdim=True)).to(hidden)
    all4 = config.mode == "global_all4_gate"
    if all4:
        packed = hidden.unsqueeze(0).expand(attn.num_memories, -1, -1, -1)
        active = [True] * attn.num_memories
    else:
        mask = torch.zeros_like(probs, dtype=torch.int).scatter(-1, selected, 1)
        active = mask.bool().any(dim=(0, 1)).tolist()
        packed, indices, order, max_len, valid = ops.transform(
            hidden, mask, attn.num_memories, selected, attn.capacity)
        selectors = torch.eye(attn.topk, dtype=hidden.dtype, device=hidden.device)
    projected = pack_projections(attn, packed)
    shared = (attn.q_proj(hidden).unsqueeze(0), attn.shared_k(hidden).unsqueeze(0),
              attn.shared_v(hidden).unsqueeze(0), attn.shared_b(hidden).sigmoid().unsqueeze(0),
              (-attn.A_log.float().exp() *
               F.softplus(attn.shared_a(hidden).float() + attn.dt_bias)).unsqueeze(0))
    reads = []
    timing = diagnostics.measure if diagnostics else lambda *_: nullcontext()
    with timing("read_seconds", hidden.device):
        for cache in caches:
            prior = cache[attn.layer_idx] if len(cache) > attn.layer_idx else None
            banks = list(prior["recurrent_state"]) if prior else [None] * (attn.num_memories + 1)
            conv = prior["conv_state"] if prior else ([None, None],) * 3
            routed_states = banks[:-1]
            routed, routed_conv = recurrence(attn, ops, projected, [c[0] for c in conv],
                                             routed_states, use_cache=True, active=active)
            shared_banks = banks[-1:]
            shared_read, shared_conv = recurrence(attn, ops, shared, [c[1] for c in conv],
                                                  shared_banks, use_cache=True, active=(True,))
            cache.update(recurrent_state=[*routed_states, *shared_banks],
                         conv_state=tuple([a, b] for a, b in zip(routed_conv, shared_conv, strict=True)),
                         layer_idx=attn.layer_idx, offset=packed.shape[2])
            if all4:
                outputs = routed.permute(1, 2, 0, 3, 4)
            else:
                outputs = torch.stack([
                    ops.reconstruct(routed[:, :, -max_len:], indices, order, batch, length, attn.topk,
                                    selector.expand(batch, length, attn.topk), valid)
                    for selector in selectors
                ], dim=2)
            reads.append(CellReadResult(outputs, probs, selected, top, shared_read[0]))
    with timing("fusion_seconds", hidden.device):
        result, alpha, beta = fuse(hidden, reads, config, gate,
                                   diagnostics.removed_unit if diagnostics else None)
    if diagnostics is not None and not diagnostics.timing:
        diagnostics.record(attn.layer_idx, alpha, beta, selected, all4, config.null_candidate)
    if attn.use_gate:
        result = attn.o_norm(result, attn.g_proj(hidden).unflatten(-1, (attn.num_heads, -1)))
    else:
        result = attn.o_norm(result)
    return attn.o_proj(result.flatten(-2)), router_logits


@contextmanager
def temporary_forward(module, forward):
    previous = module.__dict__.get("forward")
    module.forward = forward.__get__(module)
    try:
        yield
    finally:
        if previous is None:
            del module.forward
        else:
            module.forward = previous


class FusionReader:
    def __init__(self, backbone):
        self.backbone = backbone

    @torch.inference_mode()
    def step(self, tokens, caches, config, gate=None, diagnostics=None):
        if not caches or config.mode in ("single_cell", "legacy_equal"):
            if config.mode == "single_cell" and len(caches) > 1:
                raise ValueError("single_cell requires at most one Cell")
            caches = caches or [self.backbone.stream({})]
            return self.backbone.step(tokens, caches, [1 / len(caches)] * len(caches))
        if config.mode in LEARNED and gate is None:
            raise ValueError("Load the trained Fusion Gate for this experimental mode")
        from mom.layers import mom_gated_deltanet as ops

        model = self.backbone.model.model

        def forward(attn, hidden_states, attention_mask=None, past_key_values=None,
                    use_cache=True, output_attentions=False, **kwargs):
            if attention_mask is not None or not use_cache or output_attentions:
                raise ValueError("Experimental fusion requires unpadded cached inference")
            val, router = read_layer(attn, ops, hidden_states, caches, config, gate, diagnostics)
            return val, None, past_key_values, router

        with ExitStack() as stack:
            for block in model.layers:
                attn = block.attn
                stack.enter_context(temporary_forward(attn, forward))
            ids = torch.tensor([tokens], device=self.backbone.device)
            result = model(input_ids=ids, past_key_values=caches[0], use_cache=True, return_dict=True)
        return result.last_hidden_state

    def start(self, states):
        return [self.backbone.stream(state) for state in states]


@torch.inference_mode()
def read_query(store, question, max_new_tokens, config=FusionConfig(), gate=None):
    """Read every registered Cell through independent working streams."""
    tokenizer = store.backbone.tokenizer
    prompt = tokenizer.encode(f"Question: {question}\nAnswer:", add_special_tokens=False)
    if not 1 <= max_new_tokens <= 512 or not 1 <= len(prompt) <= 64:
        raise ValueError("Use a nonempty query of up to 64 tokens and 1–512 generation tokens")
    reader = FusionReader(store.backbone)
    ids = [path.stem for path in sorted(store.root.glob("*.zip"))]
    caches = reader.start(store.read(cell_id)[1] for cell_id in ids)
    if not caches:
        caches = [store.backbone.stream({})]
        config = FusionConfig()
    answer = []
    for _ in range(max_new_tokens):
        last = reader.step(prompt, caches, config, gate)[:, -1]
        token = int(store.backbone.model.lm_head(last).argmax(-1))
        if token == tokenizer.eos_token_id:
            break
        answer.append(token)
        prompt = [token]
    return dict(answer=tokenizer.decode(answer, skip_special_tokens=True), cells_used=ids,
                generation_tokens=len(answer), fusion_mode=config.mode)
