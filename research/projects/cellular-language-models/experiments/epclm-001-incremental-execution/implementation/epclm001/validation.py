import torch

from .common import full_logits, model_hash, native, original_language_models
from .incremental import IncrementalLM


def engineering_model(kind):
    # Original module classes with tiny dimensions only. Not baseline weights.
    base, pb, _ = native()
    torch.manual_seed(1831)
    if kind == "E0A":
        return (
            original_language_models()
            .TextNCALM(vocab_size=32, dim=16, heads=2, ffn_dim=24, carry_bias=2.0)
            .eval()
        )
    if kind in ("X3", "X4"):
        return pb.PhaseBLM(
            dim=16,
            heads=2,
            ffn_dim=24,
            vocab_size=32,
            use_rope=True,
            update="residual-only",
            activation="swiglu",
            norm_kind="rmsnorm" if kind == "X4" else "layernorm",
        ).eval()
    keys = {
        "C0_DIM": 16,
        "C0_HEADS": 2,
        "C0_FFN": 24,
        "T1_DIM": 16,
        "T1_HEADS": 4,
        "T1_KV_HEADS": 2,
        "T1_FFN": 24,
        "T1_LAYERS": 2,
    }
    old = {k: getattr(base, k) for k in keys}
    try:
        for k, v in keys.items():
            setattr(base, k, v)
        return (
            base.HistoricalScaledCLM(32) if kind == "C0" else base.ModernTransformerLM(32)
        ).eval()
    finally:
        for k, v in old.items():
            setattr(base, k, v)


@torch.inference_mode()
def parity(model, tokens, *, atol, rtol, checkpoint_sha256=None, adapter_class=IncrementalLM):
    adapter = adapter_class(model, checkpoint_sha256=checkpoint_sha256)
    cache = adapter.init_cache(
        tokens.shape[0], tokens.device, next(model.parameters()).dtype, max_context=tokens.shape[1]
    )
    rows = []
    before = model_hash(model)
    for position in range(tokens.shape[1]):
        incremental, cache = adapter.decode_one(
            tokens[:, position : position + 1], cache, absolute_position=position
        )
        full = full_logits(model, tokens[:, : position + 1])[:, -1]
        diff = (full - incremental).abs()
        topk = min(5, full.shape[-1])
        overlap = (
            (incremental.topk(topk).indices[..., :, None] == full.topk(topk).indices[..., None, :])
            .any(-1)
            .float()
            .mean()
        )
        rows.append(
            {
                "prefix_length": position + 1,
                "max_abs": diff.max().item(),
                "mean_abs": diff.mean().item(),
                "max_rel": (diff / full.abs().clamp_min(1e-8)).max().item(),
                "top1_changed": int((full.argmax(-1) != incremental.argmax(-1)).sum()),
                "topk_overlap": overlap.item(),
                "cache_bytes": adapter.cache_bytes(cache),
                "passed": bool(torch.allclose(full, incremental, atol=atol, rtol=rtol)),
            }
        )
    return {
        "passed": all(r["passed"] for r in rows),
        "atol": atol,
        "rtol": rtol,
        "rows": rows,
        "model_unchanged": before == model_hash(model),
        "physical_iteration_updates": adapter.iteration_updates,
        "kv_history_slots_read": adapter.kv_history_slots_read,
    }
