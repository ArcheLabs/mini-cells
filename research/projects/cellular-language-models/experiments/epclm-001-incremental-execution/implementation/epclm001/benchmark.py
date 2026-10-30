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
                            "T1 parallel prefill; CLM sequential A0/A1 prefill; no fused kernels",
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
        trace_hash = sha([asdict(e) for e in events])

        def new_runtime(
            impl, mode, state=state, weight=weight, point=point, steps=steps, events=events
        ):
            r = Runtime(
                state, weight, implementation=impl, tile=point["tile"], max_steps=steps, mode=mode
            )
            for event in events:
                r.enqueue(event)
            return r

        # Entire correctness pass is outside every measured sample.
        checks = {impl: new_runtime(impl, "correctness") for impl in IMPLEMENTATIONS}
        for _ in range(steps):
            rows = {impl: r.microstep() for impl, r in checks.items()}
            reference = checks[IMPLEMENTATIONS[0]]
            for impl, r in checks.items():
                torch.testing.assert_close(r.state, reference.state, atol=2e-6, rtol=2e-5)
                for key in ("consumed", "event_hash", "active_ids", "dropped", "reads"):
                    if rows[impl][key] != rows[IMPLEMENTATIONS[0]][key]:
                        raise RuntimeError("EVENT_PARITY_FAILED")
        for repeat in range(matrix["warmup"] + matrix["repeats"]):
            order = IMPLEMENTATIONS if repeat % 2 == 0 else tuple(reversed(IMPLEMENTATIONS))
            for impl in order:
                runtime = new_runtime(impl, "performance")
                start_event = end_event = None
                if torch.device(device).type == "cuda":
                    start_event, end_event = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                sync(device)
                begin = time.perf_counter()
                if start_event is not None:
                    start_event.record()
                for _ in range(steps):
                    runtime.microstep()
                if end_event is not None:
                    end_event.record()
                sync(device)
                elapsed = (time.perf_counter() - begin) * 1000
                device_ms = start_event.elapsed_time(end_event) if start_event is not None else None
                # Validation and serialization occur only after the ending sync.
                reference = checks[impl]
                torch.testing.assert_close(runtime.state, reference.state, atol=2e-6, rtol=2e-5)
                if runtime.consumed != reference.consumed or runtime.queue != reference.queue:
                    raise RuntimeError("PERFORMANCE_EVENT_SEMANTICS_MISMATCH")
                if (runtime.total_logical, runtime.total_physical) != (
                    reference.total_logical,
                    reference.total_physical,
                ):
                    raise RuntimeError("PERFORMANCE_COST_MISMATCH")
                if repeat >= matrix["warmup"]:
                    rows = reference.history
                    unique = len({i for row in rows for i in row["active_ids"]})
                    raw.append(
                        {
                            "workload": point,
                            "repeat": repeat - matrix["warmup"],
                            "implementation": impl,
                            "trace_hash": trace_hash,
                            "elapsed_ms": elapsed,
                            "device_stream_elapsed_ms": device_ms,
                            "logical_updates": runtime.total_logical,
                            "physical_updates": runtime.total_physical,
                            "unique_cells_touched": unique,
                            "repeated_activations": runtime.total_logical - unique,
                            "microsteps": rows,
                            "microsteps_origin": "SEPARATE_UNTIMED_CORRECTNESS_PASS",
                            "parity_passed": True,
                            "measurement_mode": "performance",
                            "timing_scope": "host end-to-end microsteps; queue/packing/transfers included; no hashes or phase syncs",
                            "limitations": [
                                "Event interval is GPU stream elapsed time, not kernel-only profiler sum",
                                "Phase timings belong to correctness pass, not performance decomposition",
                                "No measured DRAM traffic or energy; trace generation/initial enqueue excluded",
                            ],
                        }
                    )
    return raw
