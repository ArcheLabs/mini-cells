"""Fixed, separate diagnostic passes: never mix profiler samples into latency distributions."""

from pathlib import Path

import torch

from .common import digest
from .events import IMPLEMENTATIONS
from .formal_events import runtime, setup
from .formal_measure import request


def capture(call, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as prof:
        call()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    events = prof.events()
    kernels = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
    return {
        "trace_sha256": digest(path),
        "kernel_event_count": len(kernels) if kernels else None,
        "kernel_count_status": "MEASURED" if kernels else "UNAVAILABLE_NO_CUDA_PROFILER_EVENTS",
        "summed_cuda_event_us": sum(
            getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) for e in kernels
        )
        if kernels
        else None,
        "cpu_operator_event_count": sum(
            e.device_type == torch.autograd.DeviceType.CPU for e in events
        ),
        "measured_dram_bytes": None,
        "dram_status": "UNAVAILABLE_NO_HARDWARE_COUNTERS",
        "energy_joules": None,
        "energy_status": "UNAVAILABLE",
        "scope": "Separate profiler pass; summed CUDA event durations may overlap; not wall time or latency samples",
    }


@torch.inference_mode()
def models(runner):
    work = next(w for w in runner.c["workloads"] if w["id"] == runner.c["profiler_model_workload"])
    assets = runner.assets()
    rows = []
    for key, mode in [("T1", "A0"), ("X3", "A1"), ("X4", "A1")]:
        runner.check()
        model = runner.model(key)
        ids = torch.tensor(
            [assets["inputs"][str(work["offset"])][: work["prefill"] + work["decode"]]],
            device="cuda:0",
        )
        path = runner.root / f"profiling/model-{key}-{mode}.json"
        row = capture(
            lambda model=model, ids=ids, mode=mode: request(model, ids, work["prefill"], mode), path
        )
        rows.append(dict(model=key, mode=mode, workload=work["id"], **row))
        del model, ids
        torch.cuda.empty_cache()
    runner.output("MODEL_PROFILER.json", {"records": rows})


@torch.inference_mode()
def events(runner):
    rows = []
    for point in runner.c["event_points"]:
        if point["id"] not in runner.c["profiler_event_points"]:
            continue
        for kind in runner.c["event_kernels"]:
            state, weight, inputs, kernel = setup(point, runner.c, "cuda:0", kind)
            for impl in IMPLEMENTATIONS:
                runner.check()
                value = runtime(state, weight, inputs, point, impl, "profile", kernel)

                def execute(value=value, steps=point["steps"]):
                    for _ in range(steps):
                        value.microstep()

                path = runner.root / f"profiling/event-{point['id']}-{kind}-{impl}.json"
                row = capture(execute, path)
                marks = value.profile_marks
                phases = ("queue", "compact", "gather", "compute", "scatter", "message")
                intervals = []
                for step in range(point["steps"]):
                    events = [e for s, e in marks if s == step]
                    # Seven boundary marks per nonempty step. Stream intervals include CPU submission gaps.
                    if len(events) == 7:
                        intervals.append(
                            {
                                phase: events[i].elapsed_time(events[i + 1])
                                for i, phase in enumerate(phases)
                            }
                        )
                rows.append(
                    dict(
                        point=point["id"],
                        kernel=kind,
                        implementation=impl,
                        host_phase_ms=value.history,
                        device_stream_phase_ms=intervals,
                        **row,
                    )
                )
    runner.output("EVENT_PROFILER.json", {"records": rows})
