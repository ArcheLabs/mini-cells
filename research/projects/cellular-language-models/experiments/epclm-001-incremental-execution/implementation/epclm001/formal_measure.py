"""Uniform synchronized inference timing; profiler runs are separate."""

import statistics
import time

import torch

from .common import full_logits, native
from .incremental import IncrementalLM, PreallocatedIncrementalLM


def timed(call, device):
    events = None
    if torch.device(device).type == "cuda":
        events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        torch.cuda.synchronize(device)
    begin = time.perf_counter()
    if events:
        events[0].record()
    value = call()
    if events:
        events[1].record()
        torch.cuda.synchronize(device)
    host = (time.perf_counter() - begin) * 1000
    return value, host, events[0].elapsed_time(events[1]) if events else None


def cache_sizes(cache):
    if cache is None:
        return {
            "kv_visible_bytes": 0,
            "kv_storage_bytes": 0,
            "cache_tensor_storage_bytes": 0,
            "per_layer": {},
        }
    storages = {}
    kv_storages = {}
    visible = 0
    layers = {}
    for key, entry in cache["layers"].items():
        sizes = {}
        for field in ("k", "v"):
            t = entry[field]
            visible += t.numel() * t.element_size()
            storage = t.untyped_storage()
            kv_storages[storage.data_ptr()] = storage.nbytes()
            sizes[field] = {
                "shape": list(t.shape),
                "visible_bytes": t.numel() * t.element_size(),
                "storage_bytes": storage.nbytes(),
            }
        layers[key] = sizes
    storages.update(kv_storages)
    for t in cache["states"].values():
        storage = t.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
    return {
        "kv_visible_bytes": visible,
        "kv_storage_bytes": sum(kv_storages.values()),
        "cache_tensor_storage_bytes": sum(storages.values()),
        "per_layer": layers,
    }


@torch.inference_mode()
def request(model, ids, prefill, mode):
    device = ids.device
    adapter = None
    cache = None
    if mode != "full":
        cls = PreallocatedIncrementalLM if mode == "A1" else IncrementalLM
        adapter = cls(model)
        cache = adapter.init_cache(
            ids.shape[0], device, next(model.parameters()).dtype, max_context=ids.shape[1]
        )
    gpu = device.type == "cuda"
    if gpu:
        torch.cuda.reset_peak_memory_stats(device)
    resident = torch.cuda.memory_allocated(device) if gpu else None
    _, pre_host, pre_device = timed(
        lambda: (
            full_logits(model, ids[:, :prefill])
            if mode == "full"
            else adapter.prefill(ids[:, :prefill], cache)
        ),
        device,
    )
    del _
    pre_peak = torch.cuda.max_memory_allocated(device) if gpu else None
    post_prefill = torch.cuda.memory_allocated(device) if gpu else None
    if gpu:
        torch.cuda.reset_peak_memory_stats(device)
    steps = []
    begin = time.perf_counter()
    for position in range(prefill, ids.shape[1]):
        _, host, stream = timed(
            lambda position=position: (
                full_logits(model, ids[:, : position + 1])
                if mode == "full"
                else adapter.decode_one(
                    ids[:, position : position + 1], cache, absolute_position=position
                )
            ),
            device,
        )
        del _
        steps.append({"position": position, "host_ms": host, "device_stream_ms": stream})
    decode = (time.perf_counter() - begin) * 1000
    peak = torch.cuda.max_memory_allocated(device) if gpu else None
    values = [r["host_ms"] for r in steps]
    return {
        "mode": mode,
        "prefill_host_ms": pre_host,
        "prefill_device_stream_ms": pre_device,
        "decode_host_ms": decode,
        "decode_device_stream_sum_ms": sum(r["device_stream_ms"] for r in steps) if gpu else None,
        "total_host_ms": pre_host + decode,
        "token_samples": steps,
        "tpot_p50_ms": statistics.median(values),
        "tpot_p95_ms": sorted(values)[max(0, __import__("math").ceil(0.95 * len(values)) - 1)],
        "tpot_p99_ms": sorted(values)[max(0, __import__("math").ceil(0.99 * len(values)) - 1)],
        "weights_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "resident_allocated_bytes": resident,
        "post_prefill_allocated_bytes": post_prefill,
        "prefill_peak_above_resident_bytes": pre_peak - resident if gpu else None,
        "decode_peak_above_post_prefill_bytes": max(0, peak - post_prefill) if gpu else None,
        "prefill_peak_allocated_bytes": pre_peak,
        "decode_peak_allocated_bytes": peak,
        "request_peak_allocated_bytes": max(pre_peak, peak) if gpu else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if gpu else None,
        "cache": cache_sizes(cache),
        "cache_allocation_overhead_bytes": None,
        "cache_allocation_overhead_status": "NOT_SEPARATELY_ATTRIBUTED",
        "ttft_ms": None,
        "ttft_status": "NOT_MEASURED_NO_SAMPLER",
        "e2e_ms": None,
        "e2e_status": "NOT_MEASURED",
        "timing_scope": "Model-only teacher forced; synchronized token completion; excludes load/tokenize/sample/return; device stream intervals are not kernel-only time",
        "prefill_strategy": "parallel"
        if hasattr(model, "blocks")
        else "sequential_incremental"
        if mode != "full"
        else "original_full",
        "flops_scope": "Matmul-only analytic estimates; norm/softmax/RoPE/activation/cache traffic omitted",
    }


def analytic_work(model_key, mode, prefill, decode):
    base, _, phase_c = native()
    name = (
        base.T1_NAME
        if model_key == "T1"
        else phase_c.X3_NAME
        if model_key == "X3"
        else phase_c.X4_NAME
    )
    cost = (
        (lambda n: base.estimate_forward_flops(name, n))
        if model_key == "T1"
        else (lambda n: phase_c.estimate_forward_flops(name, n))
    )
    if mode == "full":
        return {
            "prefill": cost(prefill),
            "decode": sum(cost(n) for n in range(prefill + 1, prefill + decode + 1)),
            "coverage": "PARTIAL_MATMUL_ONLY",
        }
    # Analytic linear projections are one-token; attention history is position dependent.
    if model_key == "T1":
        dim = base.T1_DIM
        one = cost(1)
        decode_cost = sum(
            one + 4 * dim * base.T1_LAYERS * p for p in range(prefill, prefill + decode)
        )
        pre = cost(prefill)
    else:
        dim = 188
        one = cost(1)

        def incremental(p):
            return one + sum(4 * dim * 4 * (min(p + 1, w) - 1) for w in base.C0_WINDOWS)

        pre = sum(incremental(p) for p in range(prefill))
        decode_cost = sum(incremental(p) for p in range(prefill, prefill + decode))
    return {"prefill": pre, "decode": decode_cost, "coverage": "PARTIAL_MATMUL_ONLY_NOT_MEASURED"}
