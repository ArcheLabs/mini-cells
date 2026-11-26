"""Locate the first observed divergence in the pinned MoM 1.3B backbone."""
import argparse
from collections import defaultdict
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


def packed_positions(indices, order, mask, topk, offset):
    positions = order[indices] // topk + offset
    return [row[valid].tolist() for row, valid in zip(positions, mask)]


@torch.inference_mode()
def trace(backbone, chunks):
    from mom.layers import mom_gated_deltanet as ops

    tensors = defaultdict(list)
    routing = []
    handles = []
    step, cache = reference_step(backbone)
    offset = 0

    def record(layer, stage, tensor):
        tensors[layer, stage].append(tensor.detach().float().cpu())

    def packing(x, routing_mask, num_memories, selected_memories, capacity):
        result = original(x, routing_mask, num_memories, selected_memories, capacity)
        _, indices, order, _, mask = result
        positions = packed_positions(indices, order, mask, selected_memories.shape[-1], offset)
        routing.append(dict(offset=offset, length=x.shape[1], selected=selected_memories[0].tolist(),
                            packed_positions=positions,
                            chronological=all(row == sorted(row) for row in positions)))
        return result

    def add(layer, stage, module, value):
        def hook(_module, args, kwargs, output):
            tensor = value(args, kwargs, output)
            if tensor is not None:
                record(layer, stage, tensor)
        handles.append(module.register_forward_hook(hook, with_kwargs=True))

    def shared_in(_args, kwargs, _output):
        x = kwargs["x"]
        if x.shape[0] == 1:
            return x

    def shared_out(_args, _kwargs, output):
        if output[0].shape[0] == 1:
            return output[0]

    layers = backbone.model.model.layers
    for i, block in enumerate(layers):
        add(i, "attention_input", block.attn, lambda a, k, o: a[0] if a else k["hidden_states"])
        add(i, "router_logits", block.attn.gate, lambda a, k, o: o)
        for name in ("q", "k", "v"):
            conv = getattr(block.attn, f"{name}_conv1d")
            add(i, f"shared_{name}_before_conv", conv, shared_in)
        for name in ("q", "k", "v"):
            add(i, f"shared_{name}_after_conv", getattr(block.attn, f"{name}_conv1d"), shared_out)
        add(i, "attention_output", block.attn, lambda a, k, o: o[0])
        add(i, "block_output", block, lambda a, k, o: o[0])

    original = ops.transform
    try:
        with patch.object(ops, "transform", packing):
            for chunk in chunks:
                record(len(layers), "logits", backbone.model.lm_head(step(chunk)))
                offset += len(chunk)
    finally:
        for handle in handles:
            handle.remove()

    # Hook callbacks fire in execution order. Whole and split traces are aligned by token.
    if (len(routing) != len(layers) * len(chunks) or len(tensors) != len(layers) * (len(STAGES) - 1) + 1
            or any(len(seq) != len(chunks) for seq in tensors.values())):
        raise RuntimeError("Unexpected MoM trace coverage")
    for index, entry in enumerate(routing):
        entry["layer"] = index % len(layers)
    return {key: torch.cat(seq, dim=1) for key, seq in tensors.items()}, routing, cache()


STAGES = ("attention_input", "router_logits", "shared_q_before_conv", "shared_k_before_conv",
          "shared_v_before_conv", "shared_q_after_conv", "shared_k_after_conv", "shared_v_after_conv",
          "attention_output", "block_output", "logits")


def compare_trace(reference, observed):
    if reference.keys() != observed.keys():
        raise RuntimeError("Trace stages differ between schedules")
    rows = []
    for layer, stage in sorted(reference, key=lambda key: (key[0], STAGES.index(key[1]))):
        a, b = reference[layer, stage], observed[layer, stage]
        comparison = differences(a, b, SPLIT)
        if "max_abs" not in comparison:
            raise RuntimeError(f"Invalid tensor for layer {layer}, stage {stage}")
        error = (a - b).abs().amax(dim=-1)[0]
        changed = error.ne(0).nonzero()
        outside = ((a - b).abs() > SPLIT["atol"] + SPLIT["rtol"] * a.abs()).any(dim=-1)[0].nonzero()
        rows.append(dict(layer=layer, stage=stage, exact=bool(torch.equal(a, b)),
                         first_changed_token=int(changed[0, 0]) if changed.numel() else None,
                         first_outside_tolerance_token=int(outside[0, 0]) if outside.numel() else None,
                         comparison=comparison))
    return rows


def run(backbone):
    fixture = next(row for row in prompts() if row["id"] == "france")
    ids = encode(backbone.tokenizer, fixture["text"], bos=True)
    if not 6 <= len(ids) <= 64:
        raise RuntimeError("Phase A.2 fixture must use fused recurrent mode")
    schedules = {"whole": [ids], "prefill_decode": [ids[:-4], *[[i] for i in ids[-4:]]],
                 "two_chunks": [ids[:-4], ids[-4:]], "tokenwise": [[i] for i in ids]}
    full, packing, cache = trace(backbone, schedules["whole"])
    result = dict(fixture=fixture, tokens=ids, whole_packing=packing, comparisons={})
    for name, chunks in schedules.items():
        if name == "whole":
            continue
        observed, order, state = trace(backbone, chunks)
        rows = compare_trace(full, observed)
        result["comparisons"][name] = dict(
            chunks=list(map(len, chunks)), packing=order, stages=rows,
            first_exact_difference=next((r for r in rows if not r["exact"]), None),
            first_outside_tolerance=next((r for r in rows if not r["comparison"]["pass_"]), None),
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
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write(out, "manifest.json", dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            artifacts_sha256={f.name: sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}))


if __name__ == "__main__":
    main()
