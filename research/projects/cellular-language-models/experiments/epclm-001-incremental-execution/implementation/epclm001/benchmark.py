"""Paired, interleaved timing. No scientific support from engineering samples."""

import math
import statistics
import time
from dataclasses import asdict

import torch

from .common import sha
from .events import IMPLEMENTATIONS, Runtime, trace
from .incremental import IncrementalLM, PreallocatedIncrementalLM


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def quantile(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)]


@torch.inference_mode()
def incremental_timings(model, tokens, prefill_length, *, repeats=5, warmup=2):
    if not 0 < prefill_length < tokens.shape[1] <= 128 or repeats < 2:
        raise ValueError("INVALID_DECODE_WORKLOAD")
    adapter = IncrementalLM(model)
    records = []
    for repeat in range(warmup + repeats):
        for mode in (
            ("full", "incremental", "preallocated")
            if repeat % 2 == 0
            else ("preallocated", "incremental", "full")
        ):
            adapter = (PreallocatedIncrementalLM if mode == "preallocated" else IncrementalLM)(
                model
            )
            cache = adapter.init_cache(
                tokens.shape[0],
                tokens.device,
                next(model.parameters()).dtype,
                max_context=tokens.shape[1],
            )
            sync(tokens.device)
            begin = time.perf_counter()
            if mode != "full":
                adapter.prefill(tokens[:, :prefill_length], cache)
            else:
                model(tokens[:, :prefill_length])
            sync(tokens.device)
            prefill_ms = (time.perf_counter() - begin) * 1000
            host, device_ms = [], []
            for pos in range(prefill_length, tokens.shape[1]):
                sync(tokens.device)
                if tokens.is_cuda:
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                begin = time.perf_counter()
                if mode != "full":
                    adapter.decode_one(tokens[:, pos : pos + 1], cache, absolute_position=pos)
                else:
                    model(tokens[:, : pos + 1])
                if tokens.is_cuda:
                    end.record()
                sync(tokens.device)
                host.append((time.perf_counter() - begin) * 1000)
                if tokens.is_cuda:
                    device_ms.append(start.elapsed_time(end))
            if repeat >= warmup:
                records.append(
                    {
                        "repeat": repeat - warmup,
                        "implementation": mode,
                        "prefill_ms": prefill_ms,
                        "host_decode_ms": host,
                        "device_decode_ms": device_ms or None,
                        "tpot_p50_ms": statistics.median(host),
                        "tpot_p95_ms": quantile(host, 0.95),
                        "tpot_p99_ms": quantile(host, 0.99),
                        "cache_bytes": adapter.cache_bytes(cache),
                        "ttft_ms": None,
                        "user_e2e_ms": None,
                        "peak_vram_bytes": None,
                        "limitations": [
                            "Teacher-forced timing excludes tokenizer/sampling/return",
                            "Sequential prefill; A1 preallocates KV only; no kernel fusion",
                        ],
                    }
                )
    return records


@torch.inference_mode()
def event_matrix(matrix, *, device="cpu"):
    raw = []
    for point in matrix["event_points"]:
        n, d, steps = point["cells"], point["dim"], point["steps"]
        rng = torch.Generator().manual_seed(matrix["seed"])
        state = torch.randn(n, d, generator=rng).to(device)
        weight = (torch.randn(d, d, generator=rng) / math.sqrt(d)).to(device)
        events = trace(n, steps, point["ratio"], point["layout"], matrix["seed"])
        expected = None
        for repeat in range(matrix["warmup"] + matrix["repeats"]):
            order = IMPLEMENTATIONS if repeat % 2 == 0 else tuple(reversed(IMPLEMENTATIONS))
            for impl in order:
                runtime = Runtime(
                    state, weight, implementation=impl, tile=point["tile"], max_steps=steps
                )
                for e in events:
                    runtime.enqueue(e)
                sync(device)
                begin = time.perf_counter()
                rows = [runtime.microstep() for _ in range(steps)]
                sync(device)
                elapsed = (time.perf_counter() - begin) * 1000
                if expected is None:
                    expected = (
                        runtime.state.clone(),
                        [r["event_hash"] for r in rows],
                        [r["consumed"] for r in rows],
                    )
                torch.testing.assert_close(runtime.state, expected[0], atol=2e-6, rtol=2e-5)
                if [r["event_hash"] for r in rows] != expected[1] or [
                    r["consumed"] for r in rows
                ] != expected[2]:
                    raise RuntimeError("EVENT_PARITY_FAILED")
                if repeat >= matrix["warmup"]:
                    raw.append(
                        {
                            "workload": point,
                            "repeat": repeat - matrix["warmup"],
                            "implementation": impl,
                            "trace_hash": sha([asdict(e) for e in events]),
                            "elapsed_ms": elapsed,
                            "logical_updates": runtime.total_logical,
                            "physical_updates": runtime.total_physical,
                            "unique_cells_touched": len(
                                {i for row in rows for i in row["active_ids"]}
                            ),
                            "repeated_activations": runtime.total_logical
                            - len({i for row in rows for i in row["active_ids"]}),
                            "microsteps": rows,
                            "parity_passed": True,
                            "limitations": [
                                "Diagnostic state hashes and synchronized phase timers included",
                                "No profiler DRAM traffic or energy measurement",
                                "External trace generation excluded; enqueue excluded",
                            ],
                        }
                    )
    return raw
