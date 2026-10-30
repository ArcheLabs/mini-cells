"""B0/B1 share the same synchronous runtime, not a replacement language model."""

import math
from dataclasses import asdict

import torch

from .common import model_hash, native, sha
from .events import IMPLEMENTATIONS, Runtime, trace
from .formal_measure import timed


class NativeLikeKernel(torch.nn.Module):
    def __init__(self, kind, seed):
        super().__init__()
        base, pb, _ = native()
        hidden = 638 if kind == "B1_X3" else 639
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.norm = torch.nn.LayerNorm(188) if kind == "B1_X3" else base.RMSNorm(188)
            self.ff = base.SwiGLU(188, hidden)
        self.alpha = pb.N1_ALPHA
        self.eval().requires_grad_(False)
        self.identity = model_hash(self)
        self.matmul_flops_per_cell = 6 * 188 * hidden

    def forward(self, state, message):
        return state + self.alpha * self.ff(self.norm(state + message))


def kernel(kind, config, device):
    return None if kind == "B0" else NativeLikeKernel(kind, config["event_seed"]).to(device)


def setup(point, config, device, kind):
    rng = torch.Generator().manual_seed(config["event_seed"])
    state = torch.randn(point["cells"], 188, generator=rng).to(device)
    weight = (torch.randn(188, 188, generator=rng) / math.sqrt(188)).to(device)
    events = trace(
        point["cells"], point["steps"], point["ratio"], point["layout"], config["event_seed"]
    )
    return state, weight, events, kernel(kind, config, device)


def runtime(state, weight, events, point, impl, mode, update):
    value = Runtime(
        state,
        weight,
        implementation=impl,
        tile=point["tile"],
        max_steps=point["steps"],
        mode=mode,
        kernel=update,
    )
    for event in events:
        value.enqueue(event)
    return value


@torch.inference_mode()
def semantics(point, kind, config, device):
    state, weight, events, update = setup(point, config, device, kind)
    runtimes = {
        impl: runtime(state, weight, events, point, impl, "correctness", update)
        for impl in IMPLEMENTATIONS
    }
    errors = []
    microsteps = []
    for _ in range(point["steps"]):
        rows = {impl: r.microstep() for impl, r in runtimes.items()}
        ref = runtimes[IMPLEMENTATIONS[0]]
        for impl, value in runtimes.items():
            difference = (value.state - ref.state).abs().max().item()
            errors.append(difference)
            torch.testing.assert_close(value.state, ref.state, **config["event_tolerances"])
            if value.versions != ref.versions or value.last_sequence != ref.last_sequence:
                raise RuntimeError("EVENT_VERSION_OR_ORDER_MISMATCH")
            for field in ("consumed", "dropped", "event_hash", "active_ids"):
                if rows[impl][field] != rows[IMPLEMENTATIONS[0]][field]:
                    raise RuntimeError("EVENT_SEMANTICS_MISMATCH")
            if set(value.last_reads) != set(ref.last_reads):
                raise RuntimeError("READ_IDS_MISMATCH")
            for key in ref.last_reads:
                torch.testing.assert_close(
                    value.last_reads[key], ref.last_reads[key], **config["event_tolerances"]
                )
        microsteps.append(
            {
                "step": ref.step - 1,
                "consumed_sha256": sha(rows[IMPLEMENTATIONS[0]]["consumed"]),
                "emitted_sha256": rows[IMPLEMENTATIONS[0]]["event_hash"],
                "versions_sha256": sha(ref.versions),
                "state_sha256_by_impl": {key: row["state_hash"] for key, row in rows.items()},
            }
        )
    return {
        "passed": True,
        "max_abs_state_error": max(errors),
        "microsteps": microsteps,
        "trace_hash": sha([asdict(e) for e in events]),
        "counters": {
            k: {"logical": r.total_logical, "physical": r.total_physical}
            for k, r in runtimes.items()
        },
        "scope": "External synthetic events; B1 has no attention or learned event policy",
    }


@torch.inference_mode()
def samples(point, kind, config, device, budget, repeats, warmup, existing=None, progress=None):
    state, weight, events, update = setup(point, config, device, kind)
    expected = runtime(state, weight, events, point, IMPLEMENTATIONS[0], "correctness", update)
    for _ in range(point["steps"]):
        expected.microstep()
    rows = list(existing or [])
    done = {(r["implementation"], r["repeat"]) for r in rows}
    for repeat in range(warmup + repeats):
        order = IMPLEMENTATIONS if repeat % 2 == 0 else tuple(reversed(IMPLEMENTATIONS))
        for order_index, impl in enumerate(order):
            if repeat >= warmup and (impl, repeat - warmup) in done:
                continue
            budget.check()
            value = runtime(state, weight, events, point, impl, "performance", update)
            if torch.device(device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            def execute(value=value, steps=point["steps"]):
                for _ in range(steps):
                    value.microstep()

            _, host, stream = timed(execute, device)
            torch.testing.assert_close(value.state, expected.state, **config["event_tolerances"])
            if (
                value.consumed != expected.consumed
                or value.queue != expected.queue
                or value.versions != expected.versions
            ):
                raise RuntimeError("EVENT_PERFORMANCE_VALIDATION_FAILED")
            if repeat >= warmup:
                rows.append(
                    {
                        "point": point["id"],
                        "kernel": kind,
                        "implementation": impl,
                        "repeat": repeat - warmup,
                        "order_index": order_index,
                        "host_ms": host,
                        "device_stream_ms": stream,
                        "logical_updates": value.total_logical,
                        "physical_updates": value.total_physical,
                        "logical_ratio": value.total_logical / (point["cells"] * point["steps"]),
                        "physical_ratio": value.total_physical / (point["cells"] * point["steps"]),
                        "region_overcompute": value.total_physical - value.total_logical,
                        "partial_matmul_flops": value.total_physical
                        * getattr(update, "matmul_flops_per_cell", 2 * 188**2),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                        if torch.device(device).type == "cuda"
                        else None,
                        "trace_hash": sha([asdict(e) for e in events]),
                        "parity_passed": True,
                    }
                )
                if progress:
                    progress(rows)
    return rows


def special_semantics(config, device):
    from .events import Event

    output = []
    for kind in config["event_kernels"]:
        state = torch.zeros(8, 188, device=device)
        weight = torch.eye(188, device=device)
        update = kernel(kind, config, device)
        runs = [
            Runtime(state, weight, implementation=impl, tile=4, max_steps=8, kernel=update)
            for impl in IMPLEMENTATIONS
        ]
        for r in runs:
            r.enqueue(Event("input", 0, 0, ttl=1))
        for phase in range(5):
            for r in runs:
                if phase == 2:
                    r.enqueue(
                        Event("read", 2, 0, version=r.versions[0], sequence=2, kind="READ_QUERY")
                    )
                if phase == 3:
                    r.enqueue(Event("stale", 3, 0, version=0, sequence=3))
                if phase == 4:
                    r.enqueue(Event("reset", 4, 0, version=r.versions[0], sequence=4, kind="RESET"))
            rows = [r.microstep() for r in runs]
            for r, row in zip(runs, rows):
                torch.testing.assert_close(r.state, runs[0].state, **config["event_tolerances"])
                if (
                    set(r.last_reads) != set(runs[0].last_reads)
                    or row["event_hash"] != rows[0]["event_hash"]
                    or row["consumed"] != rows[0]["consumed"]
                    or row["dropped"] != rows[0]["dropped"]
                ):
                    raise RuntimeError("SPECIAL_EVENT_SEMANTICS_FAILED")
                for key in runs[0].last_reads:
                    torch.testing.assert_close(
                        r.last_reads[key], runs[0].last_reads[key], **config["event_tolerances"]
                    )
            if phase == 0 and runs[0].state[1].ne(0).any():
                raise RuntimeError("SAME_STEP_MESSAGE_VISIBILITY")
            if phase == 3 and rows[0]["dropped"] != ["stale"]:
                raise RuntimeError("STALE_EVENT_NOT_DROPPED")
            if phase == 4 and runs[0].state.ne(0).any():
                raise RuntimeError("RESET_STATE_MISMATCH")
        output.append(
            {
                "kernel": kind,
                "passed": True,
                "covers": [
                    "READ_QUERY",
                    "RESET",
                    "next_step_message",
                    "stale_version",
                    "event_order",
                ],
            }
        )
    return output
