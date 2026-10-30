"""Reference adapter; original full-forward modules and weights are never changed.

Supports original C0, residual-only X3/X4 and T1. Dense unpadded batches only;
padding masks are deliberately rejected instead of silently changing semantics.
"""

import copy
import uuid

import torch
from torch.nn import functional as F

from .common import model_hash, native, original_language_models


def rope_at(x, position):
    # Same half-split convention and float32 angle construction as native._rope.
    d = x.shape[-1]
    even = d - d % 2
    if not even:
        return x
    half = even // 2
    inv = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    angle = position * inv
    sin, cos = angle.sin().to(x.dtype), angle.cos().to(x.dtype)
    a, b = x[..., :half], x[..., half:even]
    out = torch.cat((a * cos - b * sin, a * sin + b * cos), -1)
    return torch.cat((out, x[..., even:]), -1) if d % 2 else out


class IncrementalLM:
    def __init__(self, model, *, checkpoint_sha256=None):
        base, phase_b, _ = native()
        if isinstance(model, (base.HistoricalScaledCLM, original_language_models().TextNCALM)):
            self.kind = "C0"
        elif (
            isinstance(model, phase_b.PhaseBLM)
            and model.use_rope
            and all(s.update == "residual-only" and s.comm_gate is None for s in model.stages)
        ):
            self.kind = "X3_X4"
        elif isinstance(model, base.ModernTransformerLM):
            self.kind = "T1"
        else:
            raise ValueError("UNSUPPORTED_OR_UNAUDITED_MODEL")
        if model.training:
            raise ValueError("EVAL_MODE_REQUIRED")
        self.model = model
        self.weight_identity = model_hash(model)
        self.identity = checkpoint_sha256 or self.weight_identity
        self.max_context = getattr(model, "max_context", base.CONTEXT_LENGTH)
        self.iteration_updates = 0
        self.kv_history_slots_read = 0

    def init_cache(self, batch_size, device, dtype, *, max_context):
        param = next(self.model.parameters())
        if batch_size < 1 or not 1 <= max_context <= self.max_context:
            raise ValueError("INVALID_CONTEXT_OR_BATCH")
        if torch.device(device) != param.device or dtype != param.dtype:
            raise ValueError("MODEL_CACHE_DTYPE_DEVICE_MISMATCH")
        return {
            "schema": "epclm.cache.v1",
            "identity": self.identity,
            "kind": self.kind,
            "session_id": uuid.uuid4().hex,
            "implementation": type(self).__name__,
            "position": 0,
            "batch": batch_size,
            "device": str(param.device),
            "dtype": str(dtype),
            "max_context": max_context,
            "layers": {},
            "states": {},
            "reset": True,
        }

    def _check(self, cache, ids, position):
        if self.model.training:
            raise ValueError("EVAL_MODE_REQUIRED")
        if (cache.get("schema"), cache.get("identity"), cache.get("kind")) != (
            "epclm.cache.v1",
            self.identity,
            self.kind,
        ):
            raise ValueError("CACHE_IDENTITY_MISMATCH")
        if cache.get("implementation") != type(self).__name__:
            raise ValueError("CACHE_IMPLEMENTATION_MISMATCH")
        if position != cache["position"] or position >= cache["max_context"]:
            raise ValueError("CACHE_POSITION_OR_CONTEXT_MISMATCH")
        if ids.shape != (cache["batch"], 1) or ids.dtype != torch.long:
            raise ValueError("DENSE_SINGLE_TOKEN_LONG_REQUIRED")
        p = next(self.model.parameters())
        if (
            str(ids.device) != cache["device"]
            or p.device != ids.device
            or str(p.dtype) != cache["dtype"]
        ):
            raise ValueError("CACHE_DEVICE_DTYPE_MISMATCH")
        expected = self._layout()
        if set(cache["layers"]) != (set(expected) if position else set()):
            raise ValueError("CACHE_STAGE_ITERATION_LAYOUT_MISMATCH")
        for key, (heads, dim, window) in expected.items():
            if not position:
                break
            entry = cache["layers"][key]
            length = min(position, window)
            if entry["tag"] != key or entry["frontier"] != position:
                raise ValueError("CACHE_LAYER_TAG_OR_FRONTIER_MISMATCH")
            for value in (entry["k"], entry["v"]):
                if (
                    value.shape != (cache["batch"], heads, length, dim)
                    or value.device != p.device
                    or value.dtype != p.dtype
                ):
                    raise ValueError("CACHE_TENSOR_LAYOUT_MISMATCH")

    def _layout(self):
        if self.kind == "T1":
            return {
                f"block:{i}": (b.attn.kv_heads, b.attn.head_dim, self.max_context)
                for i, b in enumerate(self.model.blocks)
            }
        result = {}
        for i, stage in enumerate(self.model.stages):
            a = stage.attention if self.kind == "C0" else stage.attn
            for j in range(stage.iterations):
                result[f"stage:{i}/iteration:{j}"] = (a.heads, a.head_dim, a.window)
        return result

    def _attend(self, a, x, cache, key, position, rope, gqa=False):
        b, _, dim = x.shape
        if gqa:
            q = a.q(x).view(b, 1, a.heads, a.head_dim).transpose(1, 2)
            k = a.k(x).view(b, 1, a.kv_heads, a.head_dim).transpose(1, 2)
            v = a.v(x).view(b, 1, a.kv_heads, a.head_dim).transpose(1, 2)
        else:
            q, k, v = a.qkv(x).view(b, 1, 3, a.heads, a.head_dim).unbind(2)
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if rope:
            q, k = rope_at(q, position), rope_at(k, position)
        window = self.max_context if gqa else a.window
        k, v = self._append(cache, key, position, k, v, window)
        self.kv_history_slots_read += b * k.shape[1] * k.shape[2]
        if gqa:
            rep = a.heads // a.kv_heads
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        # Every cached key is past/current: is_causal=True with q length=1 is WRONG.
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        projection = a.o if gqa else a.out
        return projection(out.transpose(1, 2).contiguous().view(b, 1, dim))

    def _append(self, cache, key, position, k, v, window):
        previous = cache["layers"].get(key)
        if previous:
            k = torch.cat((previous["k"], k), 2)
            v = torch.cat((previous["v"], v), 2)
        k, v = k[:, :, -window:], v[:, :, -window:]
        cache["layers"][key] = {"tag": key, "frontier": position + 1, "k": k, "v": v}
        return k, v

    @torch.inference_mode()
    def decode_one(self, token_id, cache, *, absolute_position):
        self._check(cache, token_id, absolute_position)
        m, pos = self.model, absolute_position
        cache["states"] = {}
        if self.kind == "T1":
            x = m.token_embedding(token_id)
            for i, block in enumerate(m.blocks):
                x = x + self._attend(block.attn, block.n1(x), cache, f"block:{i}", pos, True, True)
                x = x + block.ffn(block.n2(x))
                cache["states"][f"block:{i}"] = x.clone()
                self.iteration_updates += token_id.shape[0]
            logits = m.lm_head(m.final_norm(x))
        else:
            historical = self.kind == "C0"
            x = m.token_embedding(token_id) if historical else m.token(token_id)
            embedding = m.position_embedding if historical else m.pos
            if embedding is not None:
                x = x + embedding(torch.tensor([pos], device=x.device))[None]
            for i, stage in enumerate(m.stages):
                for j in range(stage.iterations):
                    key = f"stage:{i}/iteration:{j}"
                    conditioned = (
                        x + (stage.step_embedding[j] if historical else stage.step[j])[None, None]
                    )
                    norm = stage.norm_attention if historical else stage.n1
                    attn = stage.attention if historical else stage.attn
                    delta = self._attend(attn, norm(conditioned), cache, key, pos, not historical)
                    candidate = x + delta
                    ff = stage.ffn if historical else stage.ff
                    n2 = stage.norm_ffn if historical else stage.n2
                    proposal = delta + ff(n2(candidate))
                    if historical:
                        x = stage.gru(proposal[:, 0], x[:, 0])[:, None]
                    else:
                        _, pb, _ = native()
                        x = x + pb.N1_ALPHA * proposal
                    cache["states"][key] = x.clone()
                    self.iteration_updates += token_id.shape[0]
            logits = m.lm_head(m.final_norm(x)) if historical else m.head(m.norm(x))
        cache["position"] += 1
        cache["reset"] = False
        return logits[:, -1], cache

    def prefill(self, input_ids, cache, *, start_position=0, attention_mask=None):
        if attention_mask is not None:
            raise ValueError("PADDING_MASK_UNSUPPORTED_USE_UNPADDED_BATCHES")
        if input_ids.ndim != 2 or not input_ids.shape[1]:
            raise ValueError("NONEMPTY_DENSE_PREFIX_REQUIRED")
        if (
            start_position != cache["position"]
            or start_position + input_ids.shape[1] > cache["max_context"]
        ):
            raise ValueError("PREFIX_CONTEXT_OR_POSITION_MISMATCH")
        for i in range(input_ids.shape[1]):
            logits, cache = self.decode_one(
                input_ids[:, i : i + 1], cache, absolute_position=start_position + i
            )
        return logits, cache

    def reset(self, cache):
        fresh = self.init_cache(
            cache["batch"],
            cache["device"],
            next(self.model.parameters()).dtype,
            max_context=cache["max_context"],
        )
        cache.clear()
        cache.update(fresh)
        return cache

    def snapshot(self, cache):
        return copy.deepcopy(cache)

    def restore(self, snapshot):
        cache = copy.deepcopy(snapshot)
        if cache.get("identity") != self.identity or cache.get("kind") != self.kind:
            raise ValueError("CACHE_IDENTITY_MISMATCH")
        if model_hash(self.model) != self.weight_identity:
            raise ValueError("MODEL_WEIGHTS_CHANGED")
        return cache

    @staticmethod
    def cache_bytes(cache):
        return sum(
            v.numel() * v.element_size()
            for entry in cache["layers"].values()
            for k, v in entry.items()
            if k in ("k", "v")
        ) + sum(x.numel() * x.element_size() for x in cache["states"].values())


class PreallocatedIncrementalLM(IncrementalLM):
    """A1: preallocate each layer's bounded context buffer, no per-token KV cat.

    Stores up to max_context, rather than only the local window. Reports this
    capacity cost explicitly. No fused kernels or CUDA graph claims.
    """

    def _append(self, cache, key, position, k, v, window):
        entry = cache["layers"].get(key)
        if entry is None:
            shape = (k.shape[0], k.shape[1], cache["max_context"], k.shape[-1])
            entry = {"k_storage": k.new_empty(shape), "v_storage": v.new_empty(shape)}
        entry["k_storage"][:, :, position : position + 1].copy_(k)
        entry["v_storage"][:, :, position : position + 1].copy_(v)
        start = max(0, position - window + 1)
        entry.update(
            tag=key,
            frontier=position + 1,
            k=entry["k_storage"][:, :, start : position + 1],
            v=entry["v_storage"][:, :, start : position + 1],
        )
        cache["layers"][key] = entry
        return entry["k"], entry["v"]

    @staticmethod
    def cache_bytes(cache):
        return sum(
            v.numel() * v.element_size()
            for entry in cache["layers"].values()
            for k, v in entry.items()
            if k in ("k_storage", "v_storage")
        ) + sum(x.numel() * x.element_size() for x in cache["states"].values())
