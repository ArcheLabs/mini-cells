"""Incremental KV-cache correctness and end-to-end decode profiling."""
from __future__ import annotations

import statistics
import time
from typing import Any

import torch
from torch import Tensor, nn

from .accounting import ObservedGemmFlops, active_weight_summary, execution_summary


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty measurement")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@torch.inference_mode()
def check_decode_parity(model: nn.Module, tokens: Tensor, *, atol: float = 1e-5,
                        rtol: float = 1e-4) -> dict[str, float | bool]:
    if tokens.ndim != 2 or tokens.shape[1] < 2:
        raise ValueError("decode parity needs a batch of at least two tokens")
    was_training = model.training
    model.eval()
    full, _, _ = model(tokens)
    prefix_logits, cache = model.prefill(tokens[:, :-1])
    del prefix_logits
    decoded, _ = model.decode_step(tokens[:, -1], cache)
    expected = full[:, -1]
    max_error = float((decoded - expected).abs().max())
    passed = torch.allclose(decoded, expected, atol=atol, rtol=rtol)
    model.train(was_training)
    return {"passed": bool(passed), "max_abs_logit_error": max_error,
            "atol": atol, "rtol": rtol,
            "scope": "one appended token; full causal forward compared with KV-cache decode"}


@torch.inference_mode()
def benchmark_decode(model: nn.Module, tokens: Tensor, *, prefix_length: int = 16,
                     decode_tokens: int = 16, warmup_steps: int = 3,
                     profile_kernels: bool = False) -> dict[str, Any]:
    if tokens.ndim != 2 or prefix_length < 1 or tokens.shape[1] <= prefix_length:
        raise ValueError("tokens must contain the requested prefix and at least one decode token")
    if decode_tokens < 1 or warmup_steps < 0:
        raise ValueError("invalid decode measurement count")
    device = next(model.parameters()).device
    if tokens.device != device:
        tokens = tokens.to(device)
    model.eval()
    batch = tokens.shape[0]
    end = min(tokens.shape[1], prefix_length + decode_tokens)
    prefix = tokens[:, :prefix_length]
    future = tokens[:, prefix_length:end]

    # Warmup is separate from timed evidence and uses real incremental steps.
    for _ in range(warmup_steps):
        _, cache = model.prefill(prefix)
        for index in range(min(future.shape[1], 2)):
            _, cache = model.decode_step(future[:, index], cache)

    with ObservedGemmFlops(model) as counter:
        _, cache = model.prefill(prefix)
        host_ms: list[float] = []
        cuda_ms: list[float] = []
        for index in range(future.shape[1]):
            token = future[:, index]
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                start_event.record()
            started = time.perf_counter()
            _, cache = model.decode_step(token, cache)
            if device.type == "cuda":
                end_event.record()
                torch.cuda.synchronize(device)
                cuda_ms.append(float(start_event.elapsed_time(end_event)))
            host_ms.append((time.perf_counter() - started) * 1000.0)
        torch.cuda.synchronize(device) if device.type == "cuda" else None

    kernel_report: dict[str, Any] = {"kernel_count": None,
                                     "profiler_scope": "not collected"}
    if profile_kernels:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, profile_memory=True) as profiler:
            _, cache = model.prefill(prefix)
            model.decode_step(future[:, 0], cache)
        events = profiler.events()
        kernel_report = {
            "kernel_count": sum(1 for event in events
                                 if "kernel" in event.name.lower() or "gemm" in event.name.lower()),
            "profiled_memory_bytes_proxy": sum(max(0, int(event.self_cuda_memory_usage))
                                                for event in events) if device.type == "cuda" else None,
            "profiler_scope": "profiler event/memory proxy; not physical DRAM traffic",
        }

    model.set_collect_trace(True)
    try:
        with ObservedGemmFlops(model) as trace_counter:
            _, _, _ = model(prefix)
        weights = active_weight_summary(model)
        execution = execution_summary(model)
        prefill_flops = trace_counter.report(batch * prefix_length)
    finally:
        model.set_collect_trace(False)
    return {
        "batch_size": batch,
        "prefix_tokens": prefix_length,
        "decode_tokens": int(future.shape[1]),
        "weights_resident_on_device": True,
        "dtype": str(next(model.parameters()).dtype),
        "cold_first_decode_host_ms": host_ms[0],
        "warm_decode_host_ms": host_ms[1:],
        "warm_decode_host_p50_ms": statistics.median(host_ms[1:] or host_ms),
        "warm_decode_host_p95_ms": _percentile(host_ms[1:] or host_ms, 0.95),
        "decode_cuda_kernel_p50_ms": statistics.median(cuda_ms[1:] or cuda_ms) if cuda_ms else None,
        "decode_cuda_kernel_p95_ms": _percentile(cuda_ms[1:] or cuda_ms, 0.95) if cuda_ms else None,
        "decode_tokens_per_second_host": 1000.0 * batch / statistics.mean(host_ms),
        "peak_device_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "prefill_observed_gemm": prefill_flops,
        "prefill_active_weight_proxy": weights,
        "prefill_execution": execution,
        **kernel_report,
        "memory_traffic_bytes": None,
        "memory_traffic_note": "not inferred from touched weights or allocator counters",
        "timing_scope": "host elapsed includes route/indexing/kernel launch/synchronization; CUDA event is device elapsed",
    }
