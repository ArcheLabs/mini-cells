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


def attempt(model, upstream, layer, chunk, segments):
    torch.manual_seed(42)
    wrapper = None
    result = {"layer": layer, "chunk_size": chunk, "accelerated_requested": True,
              "detach_inner_grads": True, "detach_mem_state": False}
    phase = "setup"
    try:
        wrapper = install(model, layer, build_memory(upstream, chunk_size=chunk))
        params = [p for p in wrapper.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=.001, betas=(.9, .95), weight_decay=.01)
        scan = wrapper.memory.assoc_scan
        gates = torch.rand(2, 32, 8, device="cuda")
        inputs = torch.randn_like(gates)
        reference = build_memory(upstream, chunk_size=chunk, accelerated=False).assoc_scan
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            actual = scan(gates, inputs)
        result["scan_events"] = sorted({event.key for event in profile.key_averages()})
        torch.testing.assert_close(actual, reference(gates, inputs), atol=1e-5, rtol=1e-4)
        result["scan_forward_parity"] = True
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
        if not torch.isfinite(loss) or not any(p.grad is not None for p in params):
            raise RuntimeError("Missing training signal or nonfinite loss")
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
    raw = tokenizer.encode("The quick brown fox crossed the field. " * 32,
                           add_special_tokens=False)[:128]
    assert len(raw) == 128
    segment = [(raw, [-100] + raw[1:])]
    rows = []
    for chunk in (1, 4, 8, 16):
        passed = True
        for scope, segments in (("segment_128", segment), ("episode", episode)):
            for layer in LAYERS:
                row = attempt(model, args.upstream, layer, chunk, segments)
                row["scope"] = scope
                rows.append(row)
                (args.output / "preflight.json").write_text(json.dumps(rows, indent=2))
                print({k: v for k, v in row.items() if k != "scan_events"}, flush=True)
                passed &= row["status"] == "PASS"
                gc.collect()
                torch.cuda.empty_cache()
            if not passed:
                break
        if passed:
            print({"status": "PREFLIGHT_PASS", "chunk_size": chunk}, flush=True)
            return
    raise SystemExit("Neither placement pair fit the training preflight")


if __name__ == "__main__":
    main()
