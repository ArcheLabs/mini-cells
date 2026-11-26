"""Isolate chunk-boundary behavior of the pinned FLA kernels and MoM packing."""
import argparse
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import subprocess
import time

import torch

from run import ROOT, environment, write
from minicells_native.loader import UPSTREAM
from minicells_native.model_assets import MOM

FORMAT = "minicells.root-cause.001.phase-a1.v1"
ATOL, RTOL = 0.002, 0.002
SCHEDULES = {"whole": [12], "prefill_decode": [5, 1, 1, 1, 1, 1, 1, 1],
             "two_chunks": [5, 7], "tokenwise": [1] * 12}


def compare(expected, actual):
    if (expected.shape != actual.shape or not torch.isfinite(expected).all() or not torch.isfinite(actual).all()):
        raise RuntimeError("Kernel produced an unexpected shape or nonfinite values")
    expected, actual = expected.float(), actual.float()
    difference = (expected - actual).abs()
    return dict(pass_=torch.allclose(expected, actual, atol=ATOL, rtol=RTOL),
                max_abs=float(difference.max()), mean_abs=float(difference.mean()))


def replay(step, schedule):
    state = None
    parts = []
    start = 0
    for length in schedule:
        output, state = step(slice(start, start + length), state)
        parts.append(output)
        start += length
    return torch.cat(parts, dim=1), state


def compare_schedules(step):
    output, cache = replay(step, SCHEDULES["whole"])
    return {
        name: dict(outputs=compare(output, out), final_state=compare(cache, state))
        for name, (out, state) in ((n, replay(step, sizes)) for n, sizes in SCHEDULES.items())
    }


def packing_cases(transform, device):
    cases = []
    for length in (5, 13, 20, 21, 64):
        for pattern in ("same_experts", "rotating_second"):
            positions = torch.arange(length, device=device)
            second = torch.ones_like(positions) if pattern == "same_experts" else positions % 3 + 1
            assignments = torch.stack((torch.zeros_like(positions), second), dim=-1).unsqueeze(0)
            mask = torch.zeros((1, length, 4), device=device, dtype=torch.int64).scatter_(-1, assignments, 1)
            ids = (positions + 1).to(torch.float32)[None, :, None]
            packed, _, _, _, _ = transform(ids, mask, 4, assignments, 1.0)
            observed = [[int(i) for i in expert[0, :, 0].tolist() if i] for expert in packed]
            expected = [(positions[mask[0, :, expert] != 0] + 1).tolist() for expert in range(4)]
            cases.append(dict(length=length, pattern=pattern, observed=observed, expected=expected,
                              chronological=observed == expected,
                              padding_per_expert=[length - len(tokens) for tokens in observed]))
    return cases


@torch.inference_mode()
def kernels(device):
    from fla.modules import ShortConvolution
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    conv = ShortConvolution(16, 4, activation=None, use_fast_conv1d=False,
                            device=device, dtype=torch.float16)
    conv.weight.copy_(torch.tensor([.25, .5, .75, 1.], device=device))
    inputs = (torch.arange(12 * 16, device=device, dtype=torch.float32).reshape(1, 12, 16) / 128).half()
    conv_results = compare_schedules(lambda part, state: conv(
        inputs[:, part].contiguous(), cache=state, output_final_state=True))
    reference = torch.nn.functional.conv1d(inputs.transpose(1, 2), conv.weight, groups=16, padding=3)
    full, _ = replay(lambda part, state: conv(
        inputs[:, part].contiguous(), cache=state, output_final_state=True), SCHEDULES["whole"])
    conv_reference = compare(reference[:, :, :12].transpose(1, 2), full)

    rng = torch.Generator(device=device).manual_seed(407)
    q, k, v = (torch.randn((1, 12, 4, 32), device=device, generator=rng).mul(.15).half() for _ in range(3))
    g = torch.full((1, 12, 4), -.03, device=device, dtype=torch.float32)
    beta = torch.full((1, 12, 4), .5, device=device, dtype=torch.float16)

    def recurrent(part, state):
        return fused_recurrent_gated_delta_rule(
            q=q[:, part].contiguous(), k=k[:, part].contiguous(), v=v[:, part].contiguous(),
            g=g[:, part].contiguous(), beta=beta[:, part].contiguous(),
            initial_state=state, output_final_state=True)

    return conv_results, conv_reference, compare_schedules(recurrent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True)
    work = out / "work"
    work.mkdir()
    result = dict(format=FORMAT, status="INVALID", engineering_status="FAIL",
                  environment=environment(), mom_revision=MOM,
                  tolerance=dict(atol=ATOL, rtol=RTOL), schedules=SCHEDULES)
    start = time.monotonic()
    try:
        if (torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "Tesla T4"
                or torch.cuda.get_device_capability(0) != (7, 5)):
            raise RuntimeError("Expected one visible Tesla T4 at cuda:0")
        torch.cuda.set_device(0)
        result["environment"].update(cuda_runtime=torch.version.cuda, triton=version("triton"),
                                     fla=version("flash-linear-attention"), torch=str(torch.__version__))
        source = work / "upstream"
        source.mkdir()
        for command in (("init", "-q"), ("fetch", "--depth", "1", UPSTREAM, MOM),
                        ("checkout", "--detach", "FETCH_HEAD")):
            subprocess.run(["git", *command], cwd=source, check=True, timeout=180)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        if actual != MOM:
            raise RuntimeError("Upstream MoM revision mismatch")
        import sys
        sys.path.insert(0, str(source))
        from mom.layers.mom_gated_deltanet import transform

        device = torch.device("cuda:0")
        result.update(cpu_packing=packing_cases(transform, "cpu"), cuda_packing=packing_cases(transform, device))
        print("MoM expert-packing probes complete", flush=True)
        conv, ref, delta = kernels(device)
        if not ref["pass_"]:
            raise RuntimeError("Whole short convolution disagrees with the independent causal convolution reference")
        result.update(short_convolution=conv, convolution_whole_reference=ref, delta_recurrence=delta)
        mismatches = [f"{section}/{case}/{field}" for section, values in (("conv", conv), ("delta", delta))
                      for case, fields in values.items() for field, detail in fields.items() if not detail["pass_"]]
        result["mismatches"] = mismatches
        bad_cuda_pack = sum(not case["chronological"] for case in result["cuda_packing"])
        status = "FAIL" if mismatches or bad_cuda_pack else "INCONCLUSIVE"
        result.update(status=status, engineering_status="PASS",
                      diagnosis="Component probes completed. Phase A remains closed.")
    except Exception as exc:
        result["diagnosis"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        write(out, "phase_a1.json", result)
        write(out, "decision.json", {k: result[k] for k in ("format", "status", "engineering_status", "diagnosis")})
        lines = ["# MiniCells Root Cause · Phase A.1", "", f"**{result['status']}**", "", result["diagnosis"]]
        for issue in result.get("mismatches", []):
            lines.append(f"- Outside component tolerance: {issue}")
        lines.extend(("", "See phase_a1.json for every component result and packed expert token order.", ""))
        (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
        write(out, "manifest.json", dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            artifacts_sha256={f.name: sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}))


if __name__ == "__main__":
    main()
