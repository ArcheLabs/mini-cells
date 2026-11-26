"""Isolate padded expert transitions in the pinned MoM Delta recurrence."""
import argparse
from collections import deque
from contextlib import ExitStack, contextmanager
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
from run_a31 import expert_padding_policy

FORMAT = "minicells.root-cause.001.phase-a32.v1"
A31_RESULTS = "96941e148aea43f6791708cfc33fbca1d05d95c1"
INPUT_FIELDS = ("q", "k", "v", "g", "beta")
FOCUS = ((0, "attention_output"), (1, "attention_output"))


def no_padding_transitions(arguments, valid):
    gate = valid[None, :, None]
    return dict(arguments, g=arguments["g"].masked_fill(~gate, 0),
                beta=arguments["beta"].masked_fill(~gate, 0))


@contextmanager
def expert_delta_policy(ops, layers, *, apply, capture=None, allow_chunk=False, observe=None):
    original_transform = ops.transform
    original_delta = ops.fused_recurrent_gated_delta_rule
    pending = deque()
    call_index = 0

    def transform(x, routing_mask, num_memories, selected_memories, capacity):
        nonlocal call_index
        if pending:
            raise RuntimeError("Incomplete expert Delta calls in previous MoM layer")
        result = original_transform(x, routing_mask, num_memories, selected_memories, capacity)
        packed, indices, order, _, mask = result
        layer = call_index % len(layers)
        chunk = call_index // len(layers)
        call_index += 1
        valid = F.pad(mask, (packed.shape[2] - mask.shape[1], 0), value=False)
        chunk_kernel = allow_chunk and x.shape[1] > 64 and layers[layer].attn.mode == "chunk"
        kind = "chunk" if chunk_kernel else "recurrent"
        # Only the pinned short path skips empty lanes.
        active = [True] * num_memories if chunk_kernel else packed.reshape(num_memories, -1).any(dim=1).tolist()
        positions = (packed_positions(indices, order, mask, selected_memories.shape[-1], 0, packed.shape[2])
                     if capture is not None and (layer, chunk) == (1, 1) else None)
        for expert, runs_delta in enumerate(active):
            if runs_delta:
                pending.append((kind, expert, valid[expert], None if positions is None else positions[expert]))
        if layers[layer].attn.shared_mem:
            pending.append((kind, None, None, None))
        return result

    def observed_delta(original, kind):
        def invoke(**kwargs):
            if not pending:
                raise RuntimeError("Unexpected Delta call outside the recorded MoM layer")
            expected_kind, expert, valid, positions = pending.popleft()
            if kind != expected_kind:
                raise RuntimeError(f"Expected {expected_kind} Delta dispatch, observed {kind}")
            if expert is not None:
                if kwargs["q"].shape[:2] != (1, valid.numel()) or kwargs["cu_seqlens"] is not None:
                    raise RuntimeError("Expected the pinned dense expert Delta call")
                if positions is not None:
                    capture(expert, positions, kwargs)
                if apply:
                    kwargs = no_padding_transitions(kwargs, valid)
            if observe is not None:
                observe("attempted", kind, kwargs["q"])
            result = original(**kwargs)
            if observe is not None:
                observe("returned", kind, kwargs["q"])
            return result
        return invoke

    with ExitStack() as stack:
        stack.enter_context(patch.object(ops, "transform", transform))
        stack.enter_context(patch.object(ops, "fused_recurrent_gated_delta_rule", observed_delta(original_delta, "recurrent")))
        if allow_chunk:
            chunk_delta = observed_delta(ops.chunk_gated_delta_rule, "chunk")
            stack.enter_context(patch.object(ops, "chunk_gated_delta_rule", chunk_delta))
        yield
        if pending:
            raise RuntimeError("Incomplete final expert Delta layer")


def replay_delta(original, arguments, positions):
    valid = positions.to(arguments["q"].device) >= 0

    def call(overrides):
        # Keep the captured boundary state identical in every condition.
        return original(**dict(overrides, initial_state=arguments["initial_state"].clone()))

    compact = dict(arguments, **{key: arguments[key][:, valid].contiguous() for key in INPUT_FIELDS})
    ref, ref_state = call(compact)
    native, native_state = call(arguments)
    candidate, candidate_state = call(no_padding_transitions(arguments, valid))

    def measures(output, state):
        return dict(output=compare(ref, output[:, valid]), final_state=compare(ref_state, state))
    return dict(
        incoming_state_max_abs=float(arguments["initial_state"].abs().max()),
        padding_positions=(~valid).nonzero().flatten().tolist(),
        native_padding_log_decay_max_abs=float(arguments["g"][:, ~valid].abs().max()),
        native_padding_update_gate_max_abs=float(arguments["beta"][:, ~valid].abs().max()),
        native=measures(native, native_state), candidate=measures(candidate, candidate_state))


def fixed_layer_probe(backbone, ops, chunks):
    captures = {}

    def capture(expert, positions, arguments):
        state = arguments["initial_state"]
        if (positions < 0).any() and state is not None and bool(state.any()):
            copies = {key: arguments[key].detach().clone() for key in (*INPUT_FIELDS, "initial_state")}
            captures[expert] = positions, dict(arguments, **copies)

    with expert_delta_policy(ops, backbone.model.model.layers, apply=False, capture=capture):
        _, _, evidence, _ = trace(backbone, chunks)
    if not captures:
        raise RuntimeError("No layer-1 expert had both padding and a nonzero incoming state")
    offset = len(chunks[0])
    rows = []
    for expert, (positions, arguments) in sorted(captures.items()):
        row = replay_delta(ops.fused_recurrent_gated_delta_rule, arguments, positions)
        rows.append(dict(row, expert=expert, packed_tokens=[int(p) + offset if p >= 0 else None
                                                           for p in positions.tolist()]))
    route = next(row for row in evidence["routing"] if row["layer"] == 1 and row["offset"] == offset)
    return dict(
        layer=1, offset=offset, fixed_top2=route["selected"], experts=rows,
        candidate_pass=all(all(v["pass_"] for v in row["candidate"].values()) for row in rows),
        native_mismatch=any(not v["pass_"] for row in rows for v in row["native"].values()))


@torch.inference_mode()
def experiment(backbone, result, out):
    from fla.modules import convolution
    from mom.layers import mom_gated_deltanet as ops

    fixture = next(row for row in prompts() if row["id"] == "france" and row["style"] == "continuation")
    ids = encode(backbone.tokenizer, fixture["text"], bos=True)
    if ids != [1, 415, 5565, 302, 4843, 349]:
        raise RuntimeError("Expected the exact A.3.1 token fixture")
    schedules = {"whole": [ids], "prefill_decode": [ids[:-4], *[[i] for i in ids[-4:]]],
                 "two_chunks": [ids[:-4], ids[-4:]], "tokenwise": [[i] for i in ids]}
    result.update(fixture=fixture, tokens=ids, arms={})
    layers = backbone.model.model.layers
    with interventions(ops, convolution, packing=True, history=True), expert_padding_policy(ops, convolution):
        probe = fixed_layer_probe(backbone, ops, schedules["two_chunks"])
        result["fixed_layer_probe"] = probe
        write(out, "fixed_layer_probe.json", probe)
        if not probe["candidate_pass"]:
            raise RuntimeError("A.3.2 masked recurrence did not match compact expert replay")
        print("A32_FIXED_LAYER_PROBE_PASSED", flush=True)

        baseline, baseline_logits = run_arm(backbone, schedules, focus=FOCUS)
        result["arms"]["a31"] = baseline
        print("A32_ARM_COMPLETED a31", flush=True)
        with expert_delta_policy(ops, layers, apply=True):
            candidate, _ = run_arm(backbone, schedules, baseline_logits, focus=FOCUS)
        result["arms"]["skip_padded_delta"] = candidate
        print("A32_ARM_COMPLETED skip_padded_delta", flush=True)

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
                  a31_results_commit=A31_RESULTS)
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
                      diagnosis="A.3.2 expert recurrence intervention measured. A.4 remains a separate gate.")
    except Exception as exc:
        result["diagnosis"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write(out, "phase_a32.json", result)
        write(out, "decision.json", {key: result[key] for key in ("format", "status", "engineering_status", "diagnosis")})
        lines = ["# Phase A.3.2 expert recurrence padding", "", result["diagnosis"], "",
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
