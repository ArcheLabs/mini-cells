"""Locate the first observed divergence in the pinned MoM 1.3B backbone."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import torch

from run import (ROOT, SPLIT, MODEL_13B, REVISION_13B, MiniCells, compare_caches,
                 differences, encode, environment, prompts, reference_step, write)

FORMAT = "minicells.root-cause.001.phase-a2.v1"


QKV = ("q", "k", "v")
CONV_FUNCTIONS = {
    "causal_conv1d": "fla_triton_prefill",
    "causal_conv1d_update": "fla_triton_decode",
    "causal_conv1d_fn": "causal_conv1d_cuda_prefill",
    "causal_conv1d_update_cuda": "causal_conv1d_cuda_decode",
}
STAGES = (
    "attention_input", "router_logits",
    *(f"expert_{name}_{side}_conv" for side in ("before", "after") for name in QKV),
    *(f"shared_{name}_{side}_conv" for name in QKV for side in ("before", "after")),
    "attention_output", "block_output", "logits",
)


def packed_positions(indices, order, mask, topk, offset, width):
    positions = (order[indices] // topk + offset).masked_fill(~mask, -1)
    return torch.nn.functional.pad(positions, (width - positions.shape[1], 0), value=-1).cpu()


def backend_observer(name, function, calls):
    def observe(*args, **kwargs):
        calls.append(name)
        return function(*args, **kwargs)
    return observe


@torch.inference_mode()
def trace(backbone, chunks):
    from fla.modules import convolution
    from mom.layers import mom_gated_deltanet as ops

    tensors = defaultdict(list)
    expert_tensors = defaultdict(dict)
    routing, backends, kernel_calls = [], [], []
    margins = {}
    step, cache = reference_step(backbone)
    offset = 0
    original = ops.transform
    layers = backbone.model.model.layers
    availability = {name: callable(getattr(convolution, name)) for name in CONV_FUNCTIONS}

    def record(layer, stage, tensor):
        tensors[layer, stage].append(tensor.detach().float().cpu())

    positions = None

    def packing(x, routing_mask, num_memories, selected_memories, capacity):
        nonlocal positions
        result = original(x, routing_mask, num_memories, selected_memories, capacity)
        packed, indices, order, _, mask = result
        positions = packed_positions(indices, order, mask, selected_memories.shape[-1], offset, packed.shape[2])
        valid_order = [row[row >= 0].tolist() for row in positions]
        routing.append(dict(layer=len(routing) % len(layers), offset=offset, length=x.shape[1],
                            selected=selected_memories[0].tolist(), packed_positions=valid_order,
                            chronological=all(row == sorted(row) for row in valid_order)))
        return result

    def add(layer, stage, module, value):
        def hook(_module, args, kwargs, output):
            record(layer, stage, value(args, kwargs, output))
        hooks.callback(module.register_forward_hook(hook, with_kwargs=True).remove)

    def add_router(layer, module):
        def hook(_module, _args, logits):
            record(layer, "router_logits", logits)
            scores = logits.softmax(dim=-1, dtype=torch.float32).topk(3, dim=-1).values
            top_logits = logits[0].float().topk(3, dim=-1).values
            margins[layer, offset] = dict(
                top2_top3_probability_margin=(scores[0, :, 1] - scores[0, :, 2]).tolist(),
                top2_top3_logit_margin=(top_logits[:, 1] - top_logits[:, 2]).tolist())
        hooks.callback(module.register_forward_hook(hook).remove)

    def add_convolution(layer, name, module):
        def hook(conv, _args, kwargs, output):
            x, y = kwargs["x"], output[0]
            path = "shared" if x.shape[0] == 1 else "expert"
            backends.append(dict(layer=layer, offset=offset, projection=name, path=path,
                                 input_shape=list(x.shape), use_fast_conv1d=conv.use_fast_conv1d,
                                 invoked_functions=kernel_calls[:],
                                 backends=[CONV_FUNCTIONS[fn] for fn in kernel_calls]))
            kernel_calls.clear()
            for side, value in (("before", x), ("after", y)):
                stage = f"{path}_{name}_{side}_conv"
                if path == "shared":
                    record(layer, stage, value)
                else:
                    assert positions.shape == value.shape[:2]
                    value = value.detach().float().cpu()
                    for expert, row in enumerate(positions):
                        valid = row >= 0
                        expert_tensors[layer, stage, expert].update(zip(row[valid].tolist(), value[expert, valid]))
        hooks.callback(module.register_forward_hook(hook, with_kwargs=True).remove)

    with ExitStack() as hooks:
        hooks.enter_context(patch.object(ops, "transform", packing))
        for name, available in availability.items():
            if available:
                fn = getattr(convolution, name)
                hooks.enter_context(patch.object(convolution, name, backend_observer(name, fn, kernel_calls)))
        for i, block in enumerate(layers):
            add(i, "attention_input", block.attn, lambda a, k, o: a[0] if a else k["hidden_states"])
            add_router(i, block.attn.gate)
            for name in QKV:
                add_convolution(i, name, getattr(block.attn, f"{name}_conv1d"))
            add(i, "attention_output", block.attn, lambda a, k, o: o[0])
            add(i, "block_output", block, lambda a, k, o: o[0])
        for chunk in chunks:
            record(len(layers), "logits", backbone.model.lm_head(step(chunk)))
            offset += len(chunk)

    steps = len(layers) * len(chunks)
    if (len(routing) != steps or len(backends) != steps * 2 * len(QKV)
            or not all(call["invoked_functions"] for call in backends)
            or len(tensors) != len(layers) * (len(STAGES) - 1 - 2 * len(QKV)) + 1
            or any(len(seq) != len(chunks) for seq in tensors.values())):
        raise RuntimeError("Unexpected MoM trace coverage")
    aligned = {key: torch.cat(seq, dim=1) for key, seq in tensors.items()}
    for entry in routing:
        entry.update(margins[entry["layer"], entry["offset"]])
    evidence = dict(routing=routing, convolution_functions_available=availability, convolution_calls=backends)
    return aligned, expert_tensors, evidence, cache()


def comparison_row(layer, stage, a, b, positions):
    comparison = differences(a, b, SPLIT)
    if "max_abs" not in comparison:
        raise RuntimeError(f"Invalid tensor for layer {layer}, stage {stage}")
    error = (a - b).abs()
    changed = error.any(dim=-1)[0].nonzero().flatten()
    outside = (error > SPLIT["atol"] + SPLIT["rtol"] * a.abs()).any(dim=-1)[0].nonzero().flatten()
    return dict(layer=layer, stage=stage, exact=bool(torch.equal(a, b)),
                first_changed_token=positions[changed[0]] if changed.numel() else None,
                first_outside_tolerance_token=positions[outside[0]] if outside.numel() else None,
                comparison=comparison)


def compare_trace(reference, observed, expert_reference, expert_observed):
    if reference.keys() != observed.keys() or expert_reference.keys() != expert_observed.keys():
        raise RuntimeError("Trace stages differ between schedules")
    rows = [comparison_row(layer, stage, a, observed[layer, stage], range(a.shape[1]))
            for (layer, stage), a in reference.items()]
    for (layer, stage, expert), a in expert_reference.items():
        b = expert_observed[layer, stage, expert]
        overlap = sorted(a.keys() & b.keys())
        row = dict(layer=layer, stage=stage, comparison=None, exact=None,
                   first_changed_token=None, first_outside_tolerance_token=None)
        if overlap:
            row = comparison_row(layer, stage, torch.stack([a[t] for t in overlap])[None],
                                 torch.stack([b[t] for t in overlap])[None], overlap)
        rows.append(dict(row, expert=expert, compared_tokens=overlap,
                         whole_only_tokens=sorted(a.keys() - b.keys()), split_only_tokens=sorted(b.keys() - a.keys())))
    return sorted(rows, key=lambda row: (row["layer"], STAGES.index(row["stage"]), row.get("expert", -1)))


def first_difference(rows, token_field):
    return min((row for row in rows if row[token_field] is not None),
               key=lambda row: (row["layer"], STAGES.index(row["stage"]), row[token_field]),
               default=None)


def expert_measurements(tensors):
    rows = []
    for (layer, stage, expert), positions in sorted(tensors.items()):
        tokens = sorted(positions)
        rms, max_abs = [], []
        if tokens:
            values = torch.stack([positions[t] for t in tokens])
            if not torch.isfinite(values).all():
                raise RuntimeError(f"Nonfinite expert activations at layer {layer}, stage {stage}")
            rms = values.square().mean(dim=-1).sqrt().tolist()
            max_abs = values.abs().amax(dim=-1).tolist()
        rows.append(dict(layer=layer, stage=stage, expert=expert, tokens=tokens, rms=rms, max_abs=max_abs))
    return rows


def compare_routing(reference, observed):
    def align(entries):
        return {(entry["layer"], entry["offset"] + index): (experts, margin, logit_margin)
                for entry in entries for index, (experts, margin, logit_margin) in enumerate(zip(
                    entry["selected"], entry["top2_top3_probability_margin"], entry["top2_top3_logit_margin"]))}

    a, b = align(reference), align(observed)
    if a.keys() != b.keys():
        raise RuntimeError("Router token coverage differs between schedules")
    changed, reordered = [], []
    for layer, token in sorted(a):
        whole, whole_margin, whole_logit = a[layer, token]
        split, split_margin, split_logit = b[layer, token]
        if set(whole) != set(split):
            changed.append(dict(layer=layer, token=token, whole=whole, split=split,
                                whole_probability_margin=whole_margin, split_probability_margin=split_margin,
                                whole_logit_margin=whole_logit, split_logit_margin=split_logit))
        elif whole != split:
            reordered.append(dict(layer=layer, token=token, whole=whole, split=split))
    total = len(a)
    return dict(layer_tokens=total, set_agreement=(total - len(changed)) / total,
                order_agreement=(total - len(changed) - len(reordered)) / total, changed_sets=len(changed),
                same_set_different_order=len(reordered), first_set_change=changed[0] if changed else None,
                first_rank_only_change=reordered[0] if reordered else None,
                set_changes=changed)


def run(backbone):
    fixture = next(row for row in prompts() if row["id"] == "france")
    ids = encode(backbone.tokenizer, fixture["text"], bos=True)
    if not 6 <= len(ids) <= 64:
        raise RuntimeError("Phase A.2 fixture must use fused recurrent mode")
    schedules = {"whole": [ids], "prefill_decode": [ids[:-4], *[[i] for i in ids[-4:]]],
                 "two_chunks": [ids[:-4], ids[-4:]], "tokenwise": [[i] for i in ids]}
    full, experts, evidence, cache = trace(backbone, schedules["whole"])
    result = dict(fixture=fixture, tokens=ids, whole=dict(evidence, expert_qkv=expert_measurements(experts)),
                  comparisons={})
    for name, chunks in schedules.items():
        if name == "whole":
            continue
        observed, split_experts, split, state = trace(backbone, chunks)
        rows = compare_trace(full, observed, experts, split_experts)
        result["comparisons"][name] = dict(
            chunks=list(map(len, chunks)), trace=dict(split, expert_qkv=expert_measurements(split_experts)),
            routing_stability=compare_routing(evidence["routing"], split["routing"]), stages=rows,
            first_exact_difference=first_difference(rows, "first_changed_token"),
            first_outside_tolerance=first_difference(rows, "first_outside_tolerance_token"),
            cache=compare_caches(cache, state), cache_cursor=state.get_seq_length(),
            whole_cache_cursor=cache.get_seq_length())
        del state
        print(f"LAYER_TRACE_COMPLETED {name}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True)
    result = dict(format=FORMAT, status="INVALID", engineering_status="FAIL",
                  model_revision=REVISION_13B, tolerance=SPLIT, environment=environment())
    started = time.monotonic()
    try:
        if (torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "Tesla T4"
                or torch.cuda.get_device_capability(0) != (7, 5)):
            raise RuntimeError("Expected one visible Tesla T4 at cuda:0")
        torch.cuda.set_device(0)
        torch.set_num_threads(2)
        result["environment"].update(cuda_runtime=torch.version.cuda, torch=str(torch.__version__),
                                     fla=version("flash-linear-attention"), triton=version("triton"))
        with MiniCells.open(MODEL_13B, out / "work" / "empty", device="cuda:0", dtype="fp16") as mc:
            result["identity"] = mc.model_info
            if mc.cells.list() or mc.model_info["model_revision"] != REVISION_13B:
                raise RuntimeError("Expected pinned weights and an empty Cell registry")
            result.update(run(mc._backbone))
        result.update(status="INCONCLUSIVE", engineering_status="PASS",
                      diagnosis="Layer differences recorded. Phase A remains closed.")
    except Exception as exc:
        result["diagnosis"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write(out, "phase_a2.json", result)
        write(out, "decision.json", {k: result[k] for k in ("format", "status", "engineering_status", "diagnosis")})
        lines = ["# Phase A.2 layer trace", "", result["diagnosis"]]
        for name, comparison in result.get("comparisons", {}).items():
            first = comparison["first_outside_tolerance"]
            detail = f"layer {first['layer']} / {first['stage']}" if first else "none"
            lines.append(f"- {name} first observed stage outside tolerance — {detail}")
            routing = comparison["routing_stability"]
            lines.append(f"- {name} selected expert set agreement — {routing['set_agreement']:.2%}")
        if "whole" in result:
            backends = sorted({backend for trace_result in
                               [result["whole"], *(c["trace"] for c in result["comparisons"].values())]
                               for call in trace_result["convolution_calls"] for backend in call["backends"]})
            lines.append(f"- Observed convolution backends — {', '.join(backends)}")
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write(out, "manifest.json", dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            artifacts_sha256={f.name: sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}))


if __name__ == "__main__":
    main()
