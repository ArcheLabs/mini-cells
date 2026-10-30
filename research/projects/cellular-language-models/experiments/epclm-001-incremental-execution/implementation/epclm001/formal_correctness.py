"""Real-model cache contracts, using the existing reference adapter and parity."""

import torch

from .common import full_logits, native
from .incremental import IncrementalLM, PreallocatedIncrementalLM
from .validation import parity


@torch.inference_mode()
def contracts(model, ids, adapter_class, tolerance, checkpoint_sha):
    base, _, _ = native()
    adapter = adapter_class(model, checkpoint_sha256=checkpoint_sha)
    cache = adapter.init_cache(
        ids.shape[0], ids.device, next(model.parameters()).dtype, max_context=128
    )
    expected = {}
    counts = {}
    handles = []
    if adapter.kind == "T1":
        modules = [(f"block:{i}", block.attn) for i, block in enumerate(model.blocks)]
    else:
        modules = [(f"stage:{i}", stage.attn) for i, stage in enumerate(model.stages)]
    for stem, attn in modules:

        def hook(module, args, stem=stem):
            value = args[0]
            b, t, _d = value.shape
            index = counts.get(stem, 0)
            counts[stem] = index + 1
            key = stem if adapter.kind == "T1" else stem + f"/iteration:{index}"
            if adapter.kind == "T1":
                k = module.k(value).view(b, t, module.kv_heads, module.head_dim).transpose(1, 2)
                v = module.v(value).view(b, t, module.kv_heads, module.head_dim).transpose(1, 2)
                window = 128
            else:
                _, k, v = module.qkv(value).view(b, t, 3, module.heads, module.head_dim).unbind(2)
                k, v = k.transpose(1, 2), v.transpose(1, 2)
                window = module.window
            expected[key] = {
                "k": base._rope(k)[:, :, -window:].clone(),
                "v": v[:, :, -window:].clone(),
                "input": value.clone(),
            }

        handles.append(attn.register_forward_pre_hook(hook))
    full = full_logits(model, ids)
    for handle in handles:
        handle.remove()
    logits, cache = adapter.prefill(ids, cache)
    torch.testing.assert_close(logits, full[:, -1], **tolerance)
    for key, value in expected.items():
        for field in ("k", "v"):
            torch.testing.assert_close(cache["layers"][key][field], value[field], **tolerance)
    # Prefix causality on observable stage/iteration communication inputs.
    altered = ids.clone()
    altered[:, 64:] = torch.flip(ids[:, 64:], dims=(1,))
    past = {key: value["input"][:, :64].clone() for key, value in expected.items()}
    expected.clear()
    counts.clear()
    handles = []
    for stem, attn in modules:

        def observe(module, args, stem=stem):
            index = counts.get(stem, 0)
            counts[stem] = index + 1
            key = stem if adapter.kind == "T1" else stem + f"/iteration:{index}"
            expected[key] = args[0][:, :64].clone()

        handles.append(attn.register_forward_pre_hook(observe))
    changed = full_logits(model, altered)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(full[:, :64], changed[:, :64], **tolerance)
    for key, value in past.items():
        torch.testing.assert_close(value, expected[key], **tolerance)
    adapter.reset(cache)
    _, cache = adapter.prefill(ids[:, :7], cache)
    snapshot = adapter.snapshot(cache)
    _, cache = adapter.prefill(ids[:, 7:33], cache, start_position=7)
    out, cache = adapter.prefill(ids[:, 33:], cache, start_position=33)
    restored = adapter.restore(snapshot)
    _, restored = adapter.prefill(ids[:, 7:96], restored, start_position=7)
    out2, restored = adapter.prefill(ids[:, 96:], restored, start_position=96)
    torch.testing.assert_close(out, out2, **tolerance)
    torch.testing.assert_close(out, full[:, -1], **tolerance)
    wrong = adapter.snapshot(snapshot)
    wrong["identity"] = "wrong-checkpoint"
    try:
        adapter.restore(wrong)
    except ValueError:
        pass
    else:
        raise RuntimeError("WRONG_CHECKPOINT_CACHE_ACCEPTED")
    independent = adapter.init_cache(
        ids.shape[0], ids.device, next(model.parameters()).dtype, max_context=128
    )
    adapter.prefill(ids[:, :1], independent)
    if cache["position"] != 128 or independent["position"] != 1:
        raise RuntimeError("CACHE_INSTANCE_ISOLATION_FAILED")
    return {
        "passed": True,
        "coverage": [
            "stage_iteration_KV",
            "stage_iteration_prefix_causality",
            "chunk_partition",
            "reset_restore",
            "cache_isolation",
            "wrong_checkpoint",
            "absolute_position",
            "full_context_boundary",
        ],
    }


def check_model(model, inputs, config, checkpoint_sha, budget):
    results = []
    for cls in (IncrementalLM, PreallocatedIncrementalLM):
        for batch in (1, 2):
            for role, offsets in (
                ("diagnostic", config["correctness_offsets"]),
                ("holdout", config["holdout_offsets"][:2]),
            ):
                budget.check()
                sequences = [inputs[str(offsets[i % len(offsets)])] for i in range(batch)]
                ids = torch.tensor(
                    sequences, dtype=torch.long, device=next(model.parameters()).device
                )
                report = parity(
                    model,
                    ids,
                    **config["tolerances"],
                    checkpoint_sha256=checkpoint_sha,
                    adapter_class=cls,
                )
                if not report["passed"] or not report["model_unchanged"]:
                    raise RuntimeError("REAL_MODEL_PREFIX_PARITY_FAILED")
                report["contracts"] = contracts(
                    model, ids, cls, config["tolerances"], checkpoint_sha
                )
                results.append(
                    {"adapter": cls.__name__, "batch": batch, "input_role": role, "report": report}
                )
    return {
        "passed": True,
        "cases": results,
        "dtype": "float32",
        "source": "PINNED_TRAINED_WEIGHTS",
    }
