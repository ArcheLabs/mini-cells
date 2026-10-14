"""Cold-model and warm-cache prefill/decode measurements; no free repeated work."""

import os
import resource
import time
import numpy as np
import torch
from .accounting import Counter


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def cache_bytes(memory):
    if hasattr(memory, "key"):
        return (
            memory.key.numel() * memory.key.element_size()
            + memory.value.numel() * memory.value.element_size()
        )
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in memory)


@torch.no_grad()
def benchmark(model, prefix, decode_token, *, repeats=10, warmup=3):
    device = prefix.device
    model.eval()
    torch.set_num_threads(1)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    def timed(fn):
        sync(device)
        start = time.perf_counter()
        out = fn()
        sync(device)
        return out, (time.perf_counter() - start) * 1000

    cold, cold_ms = timed(lambda: model.prefill(prefix))
    for _ in range(warmup):
        model.prefill(prefix)
        model.decode_step(decode_token, cold.memory)
    prefill, decode = [], []
    for _ in range(repeats):
        _, t = timed(lambda: model.prefill(prefix))
        prefill.append(t)
        _, t = timed(lambda: model.decode_step(decode_token, cold.memory))
        decode.append(t)
    with Counter(model) as counter:
        out = model.prefill(prefix)
        model.decode_step(decode_token, out.memory)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(device).startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, with_flops=True) as prof:
        model.decode_step(decode_token, cold.memory)

    def summary(times, tokens):
        return {
            "p50_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "tokens_per_second": tokens / (float(np.mean(times)) / 1000),
            "repetitions": repeats,
        }

    return {
        "prefill": summary(prefill, prefix.numel()),
        "decode": summary(decode, prefix.shape[0]),
        "cold_model_prefill_ms": cold_ms,
        "cold_scope": "first invocation; not an OS/CPU hardware-cache flush",
        "hot_cache_scope": "same frozen prefix K/V cache reused; one decode token",
        "cache_bytes": cache_bytes(cold.memory),
        "cost": counter.report(prefix.numel() + prefix.shape[0]),
        "observed_profiler_gemm_flops": sum(
            e.flops or 0 for e in prof.key_averages() if e.key in ("aten::mm", "aten::addmm", "aten::bmm")
        ),
        "operators": [
            {"operator": e.key, "calls": e.count, "cpu_us": e.cpu_time_total, "flops": e.flops or None}
            for e in prof.key_averages()
        ],
        "cpu_rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "cpu_rss_scope": "Linux lifetime peak RSS, not incremental model memory",
        "gpu_allocated_peak_bytes": torch.cuda.max_memory_allocated(device)
        if str(device).startswith("cuda")
        else None,
        "gpu_reserved_peak_bytes": torch.cuda.max_memory_reserved(device)
        if str(device).startswith("cuda")
        else None,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(device),
        "threads": 1,
        "pid": os.getpid(),
    }
