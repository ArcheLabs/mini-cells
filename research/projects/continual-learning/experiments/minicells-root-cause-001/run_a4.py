"""Reaccept generated output and chunk-boundary parity on the corrected MoM runtime."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from itertools import accumulate
from pathlib import Path
import subprocess
import time
from traceback import format_exc

import torch

from run import (MODEL_13B, REVISION_13B, ROOT, SPLIT, PARITY, OUTPUT_TOKENS, MIN_MATCHES,
                 GenerationProbe, MiniCells, compare_caches, differences, encode, environment, generate,
                 native_step, phase_a, prompts, reference_step, write)
from run_a3 import interventions, run_arm
from run_a31 import expert_padding_policy
from run_a32 import expert_delta_policy

FORMAT = "minicells.root-cause.001.phase-a4.v1"
A32_RESULTS = "6db20d54ee066816e2319e71be390fb32a026dbc"
LONG_TEXT = (
    "In the morning the library opens its doors. Visitors read at the long tables, "
    "while the librarian checks returned books and writes a list of new titles. "
    "Outside, cyclists cross the square on their way to the station. "
)
LONG_SCHEDULES = dict(
    recurrent_blocks=[64, 64, 4],
    tokenwise=[1] * 132,
    native_whole=[132],
    two_native_chunks=[66, 66],
    native_prefill_decode=[128, 1, 1, 1, 1],
    boundary_mixed=[65, 64, 3],
)


def greedy_for_schedules(backbone, schedules):
    results = {}
    for name, chunks in schedules.items():
        step, _ = reference_step(backbone)
        for chunk in chunks[:-1]:
            step(chunk)
        probe = GenerationProbe(backbone.tokenizer.eos_token_id)
        text, ids = generate(backbone, chunks[-1], step, OUTPUT_TOKENS, 0, observe=probe.logits)
        results[name] = dict(tokens=ids, stop_reason=probe.reason, text=text)
    return dict(pass_=all(row == results["whole"] for row in results.values()), schedules=results)


def short_checks(mc, ops, result, out):
    backbone = mc._backbone
    ids = encode(backbone.tokenizer, next(row["text"] for row in prompts() if row["id"] == "france"), bos=True)
    schedules = dict(whole=[ids], prefill_decode=[ids[:-4], *[[i] for i in ids[-4:]]],
                     two_chunks=[ids[:-4], ids[-4:]], tokenwise=[[i] for i in ids])
    with expert_delta_policy(ops, backbone.model.model.layers, apply=True):
        result["short"] = phase_a(mc, out)
        result["short_anchor"], _ = run_arm(backbone, schedules)
        result["greedy_schedule_parity"] = greedy_for_schedules(backbone, schedules)
    result["short_gates"] = dict(
        result["short"]["gates"],
        anchor_split_cache="PASS" if all(
            s["logits"]["pass_"] and s["cache"]["pass_"] and s["cache_cursor"] == len(ids)
            for s in result["short_anchor"]["splits"].values()) else "FAIL",
        greedy_schedule="PASS" if result["greedy_schedule_parity"]["pass_"] else "FAIL")
    print("A4_SHORT_GATES", result["short_gates"], flush=True)


def dispatch_matches(counter, sizes):
    expected = {("chunk", size, "torch.bfloat16") if size > 64 else ("recurrent", size, "torch.float16")
                for size in sizes}
    return expected == counter.keys()


def measure_long(backbone, ops, ids, sizes):
    from triton.errors import TritonError

    events = {"attempted": Counter(), "returned": Counter()}

    def observe(event, kind, q):
        events[event][kind, q.shape[1], str(q.dtype)] += 1

    step, cache = reference_step(backbone)
    rows, cursors = [], []
    evidence = dict(sizes=sizes, status="FAIL")
    try:
        with expert_delta_policy(ops, backbone.model.model.layers, apply=True, allow_chunk=True, observe=observe):
            start = 0
            for size in sizes:
                rows.append(backbone.model.lm_head(step(ids[start:start + size])).float())
                cursors.append(cache().get_seq_length())
                start += size
            torch.cuda.synchronize(backbone.device)
    except (TritonError, RuntimeError) as exc:
        evidence.update(error_type=type(exc).__name__, error=str(exc), traceback=format_exc(limit=12))
        print(f"A4_LONG_RUNTIME_ERROR {sizes[:4]} {type(exc).__name__}: {exc}", flush=True)
    else:
        logits = torch.cat(rows, dim=1)
        finite = bool(torch.isfinite(logits).all())
        cursor_ok = cursors == list(accumulate(sizes))
        dispatch_ok = dispatch_matches(events["returned"], sizes) and events["attempted"] == events["returned"]
        evidence.update(finite=finite, cursor_ok=cursor_ok, dispatch_ok=dispatch_ok,
                        status="PASS" if finite and cursor_ok and dispatch_ok else "FAIL")
    evidence.update(cursors=cursors, kernel_events={
        event: [dict(kind=kind, tokens=length, dtype=dtype, calls=count)
                for (kind, length, dtype), count in sorted(counter.items())]
        for event, counter in events.items()})
    if evidence["status"] != "PASS":
        return evidence, None, None
    return evidence, logits, cache()


def probe_sdk_long(backbone, ops, ids):
    from triton.errors import TritonError

    step, cache = native_step(backbone)
    reference, reference_cache = reference_step(backbone)
    try:
        with expert_delta_policy(ops, backbone.model.model.layers, apply=True, allow_chunk=True):
            observed = backbone.model.lm_head(step(ids[:65]))
            expected = backbone.model.lm_head(reference(ids[:65]))
            torch.cuda.synchronize(backbone.device)
    except ValueError as exc:
        if "The upstream Chunk kernel uses BF16" not in str(exc):
            raise
        return dict(status="BLOCKED", error=str(exc))
    except (TritonError, RuntimeError) as exc:
        return dict(status="FAIL", error_type=type(exc).__name__, error=str(exc), traceback=format_exc(limit=12))
    logits = differences(expected, observed, PARITY)
    caches = compare_caches(reference_cache(), cache())
    cursors = [reference_cache().get_seq_length(), cache().get_seq_length()]
    passed = logits["pass_"] and caches["pass_"] and cursors == [65, 65]
    return dict(status="PASS" if passed else "FAIL", logits=logits, cache=caches, cursors=cursors)


def long_checks(backbone, ops, result):
    encoded = encode(backbone.tokenizer, LONG_TEXT * 5, bos=True)
    ids = encoded[:132]
    if len(ids) != 132:
        raise RuntimeError("Expected 132 tokens after truncating the fixed long fixture")
    evidence = dict(tokens=ids, sdk_single_call_65=probe_sdk_long(backbone, ops, ids), schedules={})
    result["long"] = evidence
    reference_logits, reference_state = None, None
    for name, sizes in LONG_SCHEDULES.items():
        status = evidence["schedules"].get("native_whole", {}).get("status")
        if status == "FAIL" and any(size > 64 for size in sizes):
            evidence["schedules"][name] = dict(status="BLOCKED", sizes=sizes, reason="native_whole failed")
            continue
        row, logits, state = measure_long(backbone, ops, ids, sizes)
        evidence["schedules"][name] = row
        if logits is not None:
            if name == "recurrent_blocks":
                reference_logits, reference_state = logits, state
            elif reference_logits is not None:
                row["vs_recurrent_blocks"] = dict(logits=differences(reference_logits, logits, SPLIT),
                                                 cache=compare_caches(reference_state, state))
        del logits, state
        print(f"A4_LONG_SCHEDULE {name} {row['status']}", flush=True)
    schedules = evidence["schedules"]
    comparisons = [row.get("vs_recurrent_blocks", {}) for name, row in schedules.items() if name != "recurrent_blocks"]
    result["long_gates"] = dict(
        sdk_single_call_65=evidence["sdk_single_call_65"]["status"],
        native_kernel="PASS" if all(row["status"] == "PASS" for row in schedules.values()) else "FAIL",
        schedule_parity="PASS" if all(c.get("logits", {}).get("pass_") and c.get("cache", {}).get("pass_")
                                     for c in comparisons) else "FAIL")


@torch.inference_mode()
def experiment(mc, result, out):
    from fla.modules import convolution
    from mom.layers import mom_gated_deltanet as ops

    with interventions(ops, convolution, packing=True, history=True), expert_padding_policy(ops, convolution):
        short_checks(mc, ops, result, out)
        write(out, "short_summary.json", dict(gates=result["short_gates"], counts=result["short"]["counts"]))
        long_checks(mc._backbone, ops, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True)
    result = dict(format=FORMAT, status="INVALID", engineering_status="FAIL",
                  model_revision=REVISION_13B, split_tolerance=SPLIT, parity_tolerance=PARITY,
                  max_new_tokens=OUTPUT_TOKENS, minimum_bos_continuations=MIN_MATCHES,
                  environment=environment(), a32_results_commit=A32_RESULTS)
    started = time.monotonic()
    try:
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected one visible CUDA device at cuda:0")
        torch.cuda.set_device(0)
        gpu = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        if (gpu, capability) != ("Tesla T4", (7, 5)) and not native_bf16:
            raise RuntimeError("Expected Tesla T4 or a device with native BF16 support")
        torch.set_num_threads(2)
        result["environment"].update(cuda_runtime=torch.version.cuda, torch=str(torch.__version__),
                                     fla=version("flash-linear-attention"), triton=version("triton"),
                                     selected_gpu=gpu, selected_compute_capability=list(capability),
                                     native_bf16=native_bf16,
                                     accelerator_profile="NATIVE_BF16" if native_bf16 else "TESLA_T4")
        with MiniCells.open(MODEL_13B, out / "work" / "empty", device="cuda:0", dtype="fp16") as mc:
            result["identity"] = mc.model_info
            if mc.cells.list() or mc.model_info["model_revision"] != REVISION_13B:
                raise RuntimeError("Expected pinned weights and an empty Cell registry")
            experiment(mc, result, out)
        gates = {**result["short_gates"], **result["long_gates"]}
        passed = all(value == "PASS" for value in gates.values())
        result.update(gates=gates, status="PASS" if passed else "FAIL", engineering_status="PASS",
                      phase_b_ready=passed,
                      diagnosis="A.4 acceptance passed." if passed else "A.4 acceptance gates remain unmet.")
    except Exception as exc:
        result["diagnosis"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write(out, "phase_a4.json", result)
        keys = ("format", "status", "engineering_status", "diagnosis", "phase_b_ready", "gates")
        write(out, "decision.json", {key: result[key] for key in keys if key in result})
        lines = ["# Phase A.4 base runtime acceptance", "", result["diagnosis"], ""]
        lines.extend(f"- {name} — {value}" for name, value in result.get("gates", {}).items())
        lines += ["", "See phase_a4.json for generations, cache comparisons and observed kernel dispatch."]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write(out, "manifest.json", dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            artifacts_sha256={f.name: sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}))


if __name__ == "__main__":
    main()
