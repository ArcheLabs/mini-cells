"""Compare two isolated interventions on the pinned MoM 1.3B runtime."""
import argparse
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from importlib.metadata import version
import inspect
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import torch

from run import (MODEL_13B, REVISION_13B, ROOT, SPLIT, MiniCells, compare_caches,
                 differences, encode, environment, prompts, write)
from run_a1 import SCHEDULES, compare, packing_cases, replay
from run_a2 import compare_routing, compare_trace, first_difference, trace

FORMAT = "minicells.root-cause.001.phase-a3.v1"
A2_RESULTS = "eb3acfe54f9cbe59dc23ac640683e173d94f8a3d"
ARMS = {"original": (False, False), "packing_only": (True, False),
        "convolution_only": (False, True), "both": (True, True)}


def stable_transform(original):
    source = inspect.getsource(original)
    needle = "sorted_indices = combined.argsort()"
    if source.count(needle) != 1:
        raise RuntimeError("Expected the pinned MoM sort statement")
    namespace = dict(original.__globals__)
    code = source.replace(needle, "sorted_indices = combined.argsort(stable=True)")
    exec(compile(code, inspect.getfile(original), "exec"), namespace)
    return namespace[original.__name__]


def convolution_history(original):
    def forward(self, x, residual=None, mask=None, cache=None, output_final_state=False,
                cu_seqlens=None, **kwargs):
        if self.use_fast_conv1d or cu_seqlens is not None or kwargs.get("seq_idx") is not None:
            raise RuntimeError("Phase A.3 requires dense FLA Triton convolution")
        if cache is None or x.shape[1] == 1:
            return original(self, x, residual, mask, cache, output_final_state, cu_seqlens, **kwargs)
        # The last W-1 inputs supply the causal context at this chunk boundary.
        history = cache.transpose(1, 2)[:, 1:]
        prefix = history.shape[1]
        joined = torch.cat((history, x), dim=1)
        if residual is not None:
            residual = torch.nn.functional.pad(residual, (0, 0, prefix, 0))
        if mask is not None:
            mask = torch.nn.functional.pad(mask, (prefix, 0), value=1)
        output, updated = original(self, joined, residual, mask, cache, output_final_state, cu_seqlens, **kwargs)
        return output[:, prefix:].contiguous(), updated
    return forward


@contextmanager
def interventions(ops, convolution, packing, history):
    with ExitStack() as stack:
        if packing:
            stack.enter_context(patch.object(ops, "transform", stable_transform(ops.transform)))
        if history:
            short_conv = convolution.ShortConvolution

            def step(self, x, residual, cache, cu_seqlens=None):
                return convolution.causal_conv1d_update(
                    x=x, cache=cache, residual=residual, weight=self.weight.squeeze(1),
                    bias=self.bias, activation=self.activation)

            stack.enter_context(patch.object(short_conv, "step", step))
            stack.enter_context(patch.object(short_conv, "forward", convolution_history(short_conv.forward)))
        yield


@torch.inference_mode()
def convolution_probe(convolution, device):
    """Exercise the actual FLA entry points against an independent causal reference."""
    rng = torch.Generator(device=device).manual_seed(407)
    rows = []
    for lanes, channels, activation in ((1, 16, None), (4, 1024, "silu")):
        conv = convolution.ShortConvolution(channels, 4, activation=activation, use_fast_conv1d=False,
                                            device=device, dtype=torch.float16)
        conv.weight.copy_(torch.tensor([.25, .5, .75, 1.], device=device))
        x = torch.randn(lanes, 12, channels, generator=rng, device=device).half()
        expected = torch.nn.functional.conv1d(x.transpose(1, 2).float(), conv.weight.float(),
                                               groups=channels, padding=3)[..., :x.shape[1]]
        if activation == "silu":
            expected = torch.nn.functional.silu(expected)
        expected = expected.transpose(1, 2).half()
        for name, sizes in SCHEDULES.items():
            y, cache = replay(lambda part, state: conv(x[:, part].contiguous(), cache=state,
                                                        output_final_state=True), sizes)
            rows.append(dict(lanes=lanes, channels=channels, activation=activation, schedule=name,
                             output=compare(expected, y),
                             final_cache_exact=bool(torch.equal(x[:, -4:].transpose(1, 2), cache))))
    return rows


def packing_summary(evidence):
    return dict(calls=len(evidence["routing"]),
                nonchronological=sum(not r["chronological"] for r in evidence["routing"]),
                convolution_backends=sorted({b for call in evidence["convolution_calls"]
                                             for b in call["backends"]}))


def run_arm(backbone, schedules, original_logits=None):
    logit_key = (len(backbone.model.model.layers), "logits")
    full, experts, evidence, cache = trace(backbone, schedules["whole"])
    logits = {"whole": full[logit_key]}
    arm = dict(whole=packing_summary(evidence), splits={})
    for name, chunks in schedules.items():
        if name == "whole":
            continue
        observed, split_experts, split, state = trace(backbone, chunks)
        logits[name] = observed[logit_key]
        rows = compare_trace(full, observed, experts, split_experts)
        arm["splits"][name] = dict(
            chunks=list(map(len, chunks)), packing=packing_summary(split),
            logits=differences(logits["whole"], logits[name], SPLIT),
            first_exact_difference=first_difference(rows, "first_changed_token"),
            first_outside_tolerance=first_difference(rows, "first_outside_tolerance_token"),
            routing=compare_routing(evidence["routing"], split["routing"]),
            cache=compare_caches(cache, state), cache_cursor=state.get_seq_length(),
            whole_cache_cursor=cache.get_seq_length())
        del state
    if original_logits is not None:
        arm["same_schedule_vs_original"] = {name: differences(original_logits[name], value, SPLIT)
                                             for name, value in logits.items()}
    return arm, logits


def paired_effects(arms):
    effects = {}
    for name in arms["original"]["splits"]:
        errors = {arm: data["splits"][name]["logits"]["mean_abs"] for arm, data in arms.items()}
        original = errors["original"]
        effects[name] = dict(
            mean_abs_errors=errors,
            reduction_from_original={arm: original - value for arm, value in errors.items() if arm != "original"},
            interaction=errors["both"] - errors["packing_only"] - errors["convolution_only"] + original)
    return effects


def experiment(backbone, result, out):
    from fla.modules import convolution
    from mom.layers import mom_gated_deltanet as ops

    device = backbone.device
    original_probe = convolution_probe(convolution, device)
    with interventions(ops, convolution, packing=True, history=True):
        packing = packing_cases(ops.transform, device)
        fixed_probe = convolution_probe(convolution, device)
    gates = dict(
        stable_packing=all(row["chronological"] for row in packing),
        fixed_convolution=all(row["output"]["pass_"] and row["final_cache_exact"] for row in fixed_probe),
    )
    result["preflight"] = dict(gates=gates, stable_packing=packing, original_convolution=original_probe,
                               fixed_convolution=fixed_probe)
    write(out, "preflight.json", result["preflight"])
    if not all(gates.values()):
        raise RuntimeError("A.3 intervention preflight failed")
    print("A3_INTERVENTION_PREFLIGHT_PASSED", flush=True)

    fixture = next(row for row in prompts() if row["id"] == "france" and row["style"] == "continuation")
    ids = encode(backbone.tokenizer, fixture["text"], bos=True)
    if not 6 <= len(ids) <= 64:
        raise RuntimeError("Expected the short A.2 fused recurrent fixture")
    schedules = {"whole": [ids], "prefill_decode": [ids[:-4], *[[i] for i in ids[-4:]]],
                 "two_chunks": [ids[:-4], ids[-4:]], "tokenwise": [[i] for i in ids]}
    result.update(fixture=fixture, tokens=ids, arms={})
    original_logits = None
    for name, (packing, history) in ARMS.items():
        with interventions(ops, convolution, packing, history):
            metrics, logits = run_arm(backbone, schedules, original_logits)
        result["arms"][name] = dict(metrics, stable_packing=packing, fixed_convolution=history)
        if name == "original":
            original_logits = logits
        print(f"A3_ARM_COMPLETED {name}", flush=True)
    result["paired_effects"] = paired_effects(result["arms"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True)
    result = dict(format=FORMAT, status="INVALID", engineering_status="FAIL",
                  model_revision=REVISION_13B, tolerance=SPLIT, environment=environment(),
                  a2_results_commit=A2_RESULTS)
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
                      diagnosis="A.3 interventions measured. Base runtime acceptance remains a separate A.4 gate.")
    except Exception as exc:
        result["diagnosis"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write(out, "phase_a3.json", result)
        write(out, "decision.json", {key: result[key] for key in ("format", "status", "engineering_status", "diagnosis")})
        lines = ["# Phase A.3 controlled interventions", "", result["diagnosis"], "",
                 "| Arm | Schedule | Logits max absolute difference | Expert set agreement |",
                 "| --- | --- | ---: | ---: |"]
        for name, arm in result.get("arms", {}).items():
            for schedule, split in arm["splits"].items():
                lines.append(f"| {name} | {schedule} | {split['logits']['max_abs']:.5g} | "
                             f"{split['routing']['set_agreement']:.2%} |")
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write(out, "manifest.json", dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            artifacts_sha256={f.name: sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}))


if __name__ == "__main__":
    main()
