"""Isolate the effect of expert padding on convolution history after A.3."""
import argparse
from contextlib import contextmanager
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import torch
from torch.nn import functional as F

from run import (MODEL_13B, REVISION_13B, ROOT, SPLIT, MiniCells, encode,
                 environment, prompts, write)
from run_a1 import compare
from run_a2 import packed_positions, trace
from run_a3 import interventions, run_arm

FORMAT = "minicells.root-cause.001.phase-a31.v1"
A3_RESULTS = "cfa5e9c0d0ee4ff8c89cc49dd605be5f968f9f4b"
ANCHOR_EXPERT = 3
ANCHOR_TOKEN = 3


def skip_padding(forward, conv, x, valid, *, cache=None, output_final_state=False):
    """Compute padded outputs on scratch state and persist only valid inputs."""
    width = x.shape[1]
    lengths = valid.sum(dim=1).tolist()
    if all(length == width for length in lengths):
        return forward(conv, x, cache=cache, output_final_state=output_final_state)
    outputs, states = [], []
    for lane, count in enumerate(lengths):
        split = width - count
        inputs = x[lane:lane + 1]
        state = None if cache is None else cache[lane:lane + 1]
        parts = []
        if split:
            scratch = None if state is None else state.clone()
            padded, _ = forward(conv, inputs[:, :split].contiguous(), cache=scratch,
                                output_final_state=output_final_state)
            parts.append(padded)
        if count:
            result, state = forward(conv, inputs[:, split:].contiguous(), cache=state,
                                    output_final_state=output_final_state)
            parts.append(result)
        elif output_final_state and state is None:
            state = x.new_zeros(1, x.shape[-1], conv.kernel_size[0])
        outputs.append(torch.cat(parts, dim=1))
        if state is not None:
            states.append(state)
    if cache is None and states:
        cache = torch.cat(states)
    return torch.cat(outputs), cache


@contextmanager
def expert_padding_policy(ops, convolution):
    original_transform = ops.transform
    original_forward = convolution.ShortConvolution.forward
    valid = None

    def transform(x, routing_mask, num_memories, selected_memories, capacity):
        nonlocal valid
        result = original_transform(x, routing_mask, num_memories, selected_memories, capacity)
        packed, _, _, _, mask = result
        valid = F.pad(mask, (packed.shape[2] - mask.shape[1], 0), value=False)
        return result

    def forward(self, x, residual=None, mask=None, cache=None, output_final_state=False,
                cu_seqlens=None, **kwargs):
        if x.shape[0] == 1:
            return original_forward(self, x, residual, mask, cache, output_final_state, cu_seqlens, **kwargs)
        if (x.shape[:2] != valid.shape or residual is not None or mask is not None or cu_seqlens is not None
                or kwargs.get("seq_idx") is not None):
            raise RuntimeError("Expected dense batch-one MoM expert convolution")
        return skip_padding(original_forward, self, x, valid, cache=cache, output_final_state=output_final_state)

    with patch.object(ops, "transform", transform), patch.object(convolution.ShortConvolution, "forward", forward):
        yield


def capture_anchor(backbone, ids):
    """Hold the actual layer-0 whole-pass expert router and Q inputs fixed."""
    attn = backbone.model.model.layers[0].attn
    conv = attn.q_conv1d
    captures = []

    def grab(_module, _args, kwargs):
        x = kwargs["x"]
        if x.shape[0] > 1:
            captures.append(x.detach().clone())

    handle = conv.register_forward_pre_hook(grab, with_kwargs=True)
    try:
        _, _, evidence, _ = trace(backbone, [ids])
    finally:
        handle.remove()
    q, = captures
    routing = evidence["routing"][0]
    token_maps = [{token: value for token, value in zip(tokens, row[-len(tokens):])}
                  if tokens else {} for row, tokens in zip(q, routing["packed_positions"])]
    return conv, q, token_maps, routing["selected"], attn.capacity


def repack_inputs(transform, q, token_maps, selected, capacity, offset, length):
    """Rebuild the real padded layout without recomputing projections or routing."""
    lanes, _, channels = q.shape
    choices = selected[:, offset:offset + length]
    routing_mask = torch.zeros(1, length, lanes, device=q.device, dtype=torch.int32).scatter_(-1, choices, 1)
    marker = q.new_zeros(1, length, 1)
    packed, indices, sort, _, mask = transform(marker, routing_mask, lanes, choices, capacity)
    positions = packed_positions(indices, sort, mask, choices.shape[-1], offset, packed.shape[2])
    x = q.new_zeros(lanes, packed.shape[2], channels)
    for lane, row in enumerate(positions.tolist()):
        for slot, token in enumerate(row):
            if token >= 0:
                x[lane, slot] = token_maps[lane][token]
    return x, positions


def compact_oracle(conv, q, token_maps):
    """Torch FP32 causal convolution on each expert's real inputs."""
    expected, caches = [], []
    width = conv.kernel_size[0]
    channels = q.shape[-1]
    for mapping in token_maps:
        if not mapping:
            expected.append({})
            caches.append(q.new_zeros(channels, width))
            continue
        tokens = sorted(mapping)
        x = torch.stack([mapping[t] for t in tokens]).transpose(0, 1)
        y = F.conv1d(x.float().unsqueeze(0), conv.weight.float(),
                    bias=conv.bias.float() if conv.bias is not None else None,
                    groups=channels, padding=width - 1)[0, :, :len(tokens)].transpose(0, 1)
        if conv.activation in ("silu", "swish"):
            y = F.silu(y)
        expected.append(dict(zip(tokens, y.to(q.dtype))))
        caches.append(F.pad(x, (max(0, width - x.shape[1]), 0))[:, -width:])
    return expected, torch.stack(caches)


def replay_expert_q(conv, q, transform, token_maps, choices, capacity, sizes, policy):
    output = [{} for _ in token_maps]
    offset, cache = 0, None
    window = [-1] * conv.kernel_size[0]
    anchor_window = None
    for length in sizes:
        x, positions = repack_inputs(transform, q, token_maps, choices, capacity, offset, length)
        if policy:
            y, cache = skip_padding(type(conv).forward, conv, x, positions >= 0,
                                    cache=cache, output_final_state=True)
        else:
            y, cache = conv(x, cache=cache, output_final_state=True)
        for lane, row in enumerate(positions.tolist()):
            for slot, token in enumerate(row):
                if token >= 0:
                    output[lane][token] = y[lane, slot]
                if lane == ANCHOR_EXPERT and (token >= 0 or not policy):
                    window = [*window[1:], token]
                    if token == ANCHOR_TOKEN:
                        anchor_window = window
        offset += length
    return output, cache, anchor_window


def fixed_route_probe(backbone, ops, schedules):
    conv, q, maps, routing, capacity = capture_anchor(backbone, schedules["whole"][0])
    choices = torch.tensor([routing], device=q.device)
    expected, expected_cache = compact_oracle(conv, q, maps)
    rows = {}
    for label, policy in (("a3_both", False), ("skip_expert_padding", True)):
        results = {}
        for name, chunks in schedules.items():
            observed, state, window = replay_expert_q(conv, q, ops.transform, maps, choices, capacity,
                                                       list(map(len, chunks)), policy)
            per_expert = []
            for expert, mapping in enumerate(expected):
                compared = sorted(mapping)
                per_expert.append(dict(
                    expert=expert, tokens=compared,
                    output=compare(torch.stack([mapping[t] for t in compared]),
                                   torch.stack([observed[expert][t] for t in compared])) if compared else None,
                    cache_exact=bool(torch.equal(expected_cache[expert], state[expert]))))
            results[name] = dict(
                experts=per_expert, anchor_window=window,
                anchor_output=compare(expected[ANCHOR_EXPERT][ANCHOR_TOKEN], observed[ANCHOR_EXPERT][ANCHOR_TOKEN]))
        rows[label] = results
    gates = {
        "unmodified_padding_reproduces_anchor": rows["a3_both"]["whole"]["anchor_output"]["pass_"] and all(
            not rows["a3_both"][name]["anchor_output"]["pass_"] for name in schedules if name != "whole"),
        "valid_only_experts_match_torch_oracle": all(
            (row["output"] is None or row["output"]["pass_"]) and row["cache_exact"]
            for case in rows["skip_expert_padding"].values() for row in case["experts"]),
    }
    return dict(layer=0, expert=ANCHOR_EXPERT, token=ANCHOR_TOKEN, fixed_top2=routing, gates=gates, arms=rows)


@torch.inference_mode()
def experiment(backbone, result, out):
    from fla.modules import convolution
    from mom.layers import mom_gated_deltanet as ops

    fixture = next(row for row in prompts() if row["id"] == "france" and row["style"] == "continuation")
    ids = encode(backbone.tokenizer, fixture["text"], bos=True)
    if ids != [1, 415, 5565, 302, 4843, 349]:
        raise RuntimeError("Expected the exact A.2 and A.3 token fixture")
    schedules = {"whole": [ids], "prefill_decode": [ids[:-4], *[[i] for i in ids[-4:]]],
                 "two_chunks": [ids[:-4], ids[-4:]], "tokenwise": [[i] for i in ids]}
    result.update(fixture=fixture, tokens=ids, arms={})
    with interventions(ops, convolution, packing=True, history=True):
        probe = fixed_route_probe(backbone, ops, schedules)
        result["fixed_route_probe"] = probe
        write(out, "fixed_route_probe.json", probe)
        if not all(probe["gates"].values()):
            raise RuntimeError("A.3.1 fixed-route probe failed")
        print("A31_FIXED_ROUTE_PROBE_PASSED", flush=True)

        baseline, baseline_logits = run_arm(backbone, schedules)
        result["arms"]["a3_both"] = baseline
        print("A31_ARM_COMPLETED a3_both", flush=True)
        with expert_padding_policy(ops, convolution):
            candidate, _ = run_arm(backbone, schedules, baseline_logits)
        result["arms"]["skip_expert_padding"] = candidate
        print("A31_ARM_COMPLETED skip_expert_padding", flush=True)

    result["mean_abs_logit_error_reduction"] = {
        name: baseline["splits"][name]["logits"]["mean_abs"] - case["logits"]["mean_abs"]
        for name, case in candidate["splits"].items()
    }
    result["candidate_logit_gate"] = all(case["logits"]["pass_"] for case in candidate["splits"].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True)
    result = dict(format=FORMAT, status="INVALID", engineering_status="FAIL",
                  model_revision=REVISION_13B, tolerance=SPLIT, environment=environment(),
                  a3_results_commit=A3_RESULTS)
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
            experiment(mc._backbone, result, out)
        result.update(status="INCONCLUSIVE", engineering_status="PASS",
                      diagnosis="A.3.1 padding intervention measured. A.4 remains a separate gate.")
    except Exception as exc:
        result["diagnosis"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write(out, "phase_a31.json", result)
        write(out, "decision.json", {key: result[key] for key in ("format", "status", "engineering_status", "diagnosis")})
        lines = ["# Phase A.3.1 expert convolution padding", "", result["diagnosis"], "",
                 "| Arm | Schedule | Logits max abs | Mean abs | In tolerance |",
                 "| --- | --- | ---: | ---: | --- |"]
        for name, arm in result.get("arms", {}).items():
            for schedule, split in arm["splits"].items():
                logits = split["logits"]
                lines.append(f"| {name} | {schedule} | {logits['max_abs']:.5g} | "
                             f"{logits['mean_abs']:.5g} | {logits['pass_']} |")
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write(out, "manifest.json", dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            artifacts_sha256={f.name: sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}))


if __name__ == "__main__":
    main()
