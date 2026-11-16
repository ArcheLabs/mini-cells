"""Measure training capacity without entering the screening loop."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from episodes import encode, generate
from execution import answer_loss, forward_episode
from integration import LAYERS, MODEL, REVISION, build_memory, install


def attempt(model, upstream, layer, chunk, segments, accelerated):
    torch.manual_seed(42)
    wrapper = None
    result = {"layer": layer, "chunk_size": chunk, "accelerated_requested": accelerated,
              "detach_inner_grads": True, "detach_mem_state": False}
    phase = "setup"
    try:
        wrapper = install(model, layer, build_memory(upstream, chunk_size=chunk, accelerated=accelerated))
        params = [p for p in wrapper.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=.001, betas=(.9, .95), weight_decay=.01)
        torch.cuda.synchronize()
        started = time.monotonic()
        phase = "forward"
        torch.cuda.reset_peak_memory_stats()
        logits, targets = forward_episode(model, segments, wrapper=wrapper)
        loss = answer_loss(logits, targets)
        torch.cuda.synchronize()
        result["forward_peak_bytes"] = torch.cuda.max_memory_allocated()
        phase = "backward"
        torch.cuda.reset_peak_memory_stats()
        loss.backward()
        torch.cuda.synchronize()
        result["backward_peak_bytes"] = torch.cuda.max_memory_allocated()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite answer loss")
        bank = wrapper.memory.memory_model_parameters
        if any(p.grad is None for p in bank):
            raise RuntimeError("An active Titans weight did not receive an outer gradient")
        result["memory_gradient_norm"] = torch.stack([p.grad.norm() for p in bank]).norm().item()
        if not result["memory_gradient_norm"] > 0:
            raise RuntimeError("Active Titans weights received no training signal")
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        phase = "optimizer"
        torch.cuda.reset_peak_memory_stats()
        optimizer.step()
        torch.cuda.synchronize()
        result.update(status="PASS", seconds=time.monotonic() - started,
                      optimizer_peak_bytes=torch.cuda.max_memory_allocated())
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="OOM", phase=phase, error=str(error),
                      peak_bytes=torch.cuda.max_memory_allocated())
    except RuntimeError as error:
        result.update(status="ERROR", phase=phase, error=str(error))
    finally:
        if wrapper is not None:
            wrapper.reset_episode()
            model.model.layers[layer].self_attn = wrapper.attention
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float16, attn_implementation="eager",
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    episode = encode(generate("train", 1)[0], tokenizer)
    print({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
           "episode_segment_tokens": [len(ids) for ids, _ in episode]}, flush=True)
    raw = tokenizer.encode("The quick brown fox crossed the field. " * 32,
                           add_special_tokens=False)[:128]
    assert len(raw) == 128
    segment = [(raw, [-100] + raw[1:])]
    rows = []
    report = args.output / "preflight.json"

    def record(row):
        rows.append(row)
        report.write_text(json.dumps(rows, indent=2))
        print(row, flush=True)

    accelerated = False
    try:
        scan = build_memory(args.upstream, accelerated=True).assoc_scan
        reference = build_memory(args.upstream, accelerated=False).assoc_scan
        gates = torch.rand(1, 128, 32, device="cuda")
        inputs = torch.randn_like(gates, requires_grad=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            actual = scan(gates, inputs)
            actual_grad = torch.autograd.grad(actual.sum(), inputs)[0]
            torch.cuda.synchronize()
        expected = reference(gates, inputs)
        expected_grad = torch.autograd.grad(expected.sum(), inputs)[0]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=1e-4)
        events = sorted({event.key for event in profile.key_averages()})
        accelerated = True
        record({"scope": "scan_probe", "status": "PASS", "scan_events": events})
    except (RuntimeError, AssertionError) as error:
        record({"scope": "scan_probe", "status": "ERROR", "error": str(error),
                "fallback": "python"})

    for backend in ((True, False) if accelerated else (False,)):
        for chunk in (1, 4, 8, 16):
            passed = True
            backend_error = False
            for scope, segments in (("segment_128", segment), ("episode", episode)):
                for layer in LAYERS:
                    row = attempt(model, args.upstream, layer, chunk, segments, backend)
                    row["scope"] = scope
                    record(row)
                    passed &= row["status"] == "PASS"
                    backend_error |= row["status"] == "ERROR"
                    gc.collect()
                    torch.cuda.empty_cache()
                if not passed:
                    break
            if passed:
                record({"status": "PREFLIGHT_PASS", "chunk_size": chunk,
                        "accelerated_requested": backend})
                return
            if backend_error:
                break
    raise SystemExit("Neither placement pair fit the training preflight")


if __name__ == "__main__":
    main()
