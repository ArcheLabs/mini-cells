"""Synchronous event semantics with four physical executors, no learned routing."""

import time
from dataclasses import asdict, dataclass

import torch

from .common import sha, tensor_hash

IMPLEMENTATIONS = ("D0_dense_reference", "D1_masked_dense", "S0_sparse_cell", "S1_sparse_region")
KINDS = ("INPUT", "STATE_MESSAGE", "SELF_TICK", "READ_QUERY", "RESET")


@dataclass(frozen=True)
class Event:
    event_id: str
    due: int
    target: int
    source: int = -1
    version: int = 0
    sequence: int = 0
    kind: str = "INPUT"
    value: float = 1.0
    ttl: int = 0


class Runtime:
    def __init__(self, state, weight, *, implementation, tile=16, max_steps=32):
        if implementation not in IMPLEMENTATIONS or tile < 1 or max_steps < 1:
            raise ValueError("INVALID_EXECUTOR_OR_BUDGET")
        self.state = state.clone()
        self.initial = state.clone()
        self.weight = weight.clone()
        self.impl, self.tile, self.max_steps = implementation, tile, max_steps
        self.queue, self.consumed, self.last_sequence = [], set(), {}
        self.versions = [0] * len(state)
        self.step = 0
        self.history = []
        self.total_physical = 0
        self.total_logical = 0

    def enqueue(self, event):
        if (
            event.kind not in KINDS
            or event.due < self.step
            or not 0 <= event.target < len(self.state)
        ):
            raise ValueError("INVALID_EVENT")
        if event.event_id in self.consumed or any(e.event_id == event.event_id for e in self.queue):
            raise ValueError("DUPLICATE_EVENT")
        if event.ttl < 0 or event.ttl > self.max_steps:
            raise ValueError("UNBOUNDED_SELF_TICK_OR_MESSAGE")
        self.queue.append(event)

    def _sync(self):
        if self.state.is_cuda:
            torch.cuda.synchronize(self.state.device)

    @torch.inference_mode()
    def microstep(self):
        if self.step >= self.max_steps:
            raise ValueError("MICROSTEP_BUDGET_EXHAUSTED")
        self._sync()
        start = time.perf_counter()
        ready = sorted(
            (e for e in self.queue if e.due == self.step),
            key=lambda e: (e.source, e.target, e.sequence, e.event_id),
        )
        self.queue = [e for e in self.queue if e.due != self.step]
        resets = [e for e in ready if e.kind == "RESET"]
        if resets and len(ready) != 1:
            raise ValueError("RESET_MUST_BE_EXCLUSIVE")
        dropped, valid = [], []
        for e in ready:
            self.consumed.add(e.event_id)
            key = (e.source, e.target)
            if e.version != self.versions[e.target] or (
                e.kind != "RESET" and e.sequence <= self.last_sequence.get(key, -1)
            ):
                dropped.append(e.event_id)
            else:
                self.last_sequence[key] = e.sequence
                valid.append(e)
        if resets and valid:
            self.state.copy_(self.initial)
            self.queue.clear()
            self.versions = [0] * len(self.state)
            self.last_sequence.clear()
            self.consumed.clear()
        reads = {e.event_id: self.state[e.target].clone() for e in valid if e.kind == "READ_QUERY"}
        updating = [e for e in valid if e.kind in ("INPUT", "STATE_MESSAGE", "SELF_TICK")]
        active = sorted({e.target for e in updating})
        queue_end = time.perf_counter()
        messages = {}
        for e in updating:
            messages[e.target] = messages.get(e.target, 0.0) + e.value
        if not active:
            physical = []  # Empty events must not launch a full-state update.
        elif self.impl.startswith("D"):
            physical = list(range(len(self.state)))
        elif self.impl == "S0_sparse_cell":
            physical = active
        else:
            regions = sorted({i // self.tile for i in active})
            physical = [
                i
                for r in regions
                for i in range(r * self.tile, min((r + 1) * self.tile, len(self.state)))
            ]
        ids = torch.tensor(physical, device=self.state.device, dtype=torch.long)
        self._sync()
        compact_end = time.perf_counter()
        if physical:
            if self.impl.startswith("D"):
                before = self.state
            else:
                before = self.state.index_select(0, ids)
            aggregation = self.state.new_tensor([messages.get(i, 0.0) for i in physical])[:, None]
            self._sync()
            gather_end = time.perf_counter()
            # Shared deterministic kernel; all reads precede every write.
            proposed = torch.tanh(before @ self.weight + aggregation)
            self._sync()
            compute_end = time.perf_counter()
            mask = torch.tensor([i in messages for i in physical], device=ids.device)
            if self.impl == "D1_masked_dense":
                self.state = torch.where(mask[:, None], proposed, self.state)
            elif self.impl == "D0_dense_reference":
                selected = torch.tensor(active, device=ids.device)
                self.state.index_copy_(0, selected, proposed.index_select(0, selected))
            else:
                self.state.index_copy_(0, ids[mask], proposed[mask])
            for i in active:
                self.versions[i] += 1
        else:
            gather_end = compute_end = compact_end
        self._sync()
        scatter_end = time.perf_counter()
        outgoing = []
        # Send rule independent of floating-point differences: absolute event payload.
        for e in updating:
            if e.ttl > 0 and self.step + 1 < self.max_steps:
                target = e.target if e.kind == "SELF_TICK" else (e.target + 1) % len(self.state)
                outgoing.append(
                    Event(
                        "send:" + e.event_id,
                        self.step + 1,
                        target,
                        e.target,
                        self.versions[target],
                        self.step + 1,
                        "STATE_MESSAGE",
                        e.value,
                        e.ttl - 1,
                    )
                )
        for e in outgoing:
            self.enqueue(e)
        self._sync()
        end = time.perf_counter()
        row = {
            "step": self.step,
            "consumed": [e.event_id for e in ready],
            "dropped": dropped,
            "active_ids": active,
            "physical_updates": len(physical),
            "logical_updates": len(active),
            "region_overcompute": len(physical) - len(active),
            "outgoing": [asdict(e) for e in outgoing],
            "state_hash": tensor_hash(self.state),
            "event_hash": sha([asdict(e) for e in outgoing]),
            "reads": {k: tensor_hash(v) for k, v in reads.items()},
            "message_bytes": len(outgoing) * 8,
            "message_bytes_scope": "float64 payload only; metadata excluded",
            "kernel_matmul_flops": 2 * len(physical) * self.state.shape[1] ** 2,
            "timing_ms": {
                "queue": (queue_end - start) * 1000,
                "compact": (compact_end - queue_end) * 1000,
                "gather": (gather_end - compact_end) * 1000,
                "compute": (compute_end - gather_end) * 1000,
                "scatter": (scatter_end - compute_end) * 1000,
                "message": (end - scatter_end) * 1000,
                "total": (end - start) * 1000,
            },
            "timing_scope": "host synchronized phases; instrumentation overhead included",
        }
        self.total_physical += len(physical)
        self.total_logical += len(active)
        self.history.append(row)
        self.step += 1
        return row

    def snapshot(self):
        import copy

        return copy.deepcopy(self.__dict__)

    def restore(self, snapshot):
        import copy

        if snapshot["impl"] != self.impl or tensor_hash(snapshot["weight"]) != tensor_hash(
            self.weight
        ):
            raise ValueError("EVENT_SNAPSHOT_IDENTITY_MISMATCH")
        self.__dict__ = copy.deepcopy(snapshot)


def trace(n, steps, ratio, layout, seed):
    import random

    rng = random.Random(seed)
    result = []
    for k in range(steps):
        count = round(n * ratio)
        if layout == "all-active":
            count = n
        elif layout == "zero-active":
            count = 0
        elif layout == "burst":
            count = n if k % 4 == 0 else count
        if layout in ("clustered", "wavefront", "hotspot"):
            offset = 0 if layout == "hotspot" else (k * max(1, count)) % n
            ids = [(offset + i) % n for i in range(count)]
        elif layout in ("uniform-random", "all-active", "zero-active", "burst"):
            ids = rng.sample(range(n), count)
        else:
            raise ValueError("UNKNOWN_TRACE_LAYOUT")
        result.extend(
            Event(
                f"input:{k}:{i}",
                k,
                i,
                version=sum(e.target == i for e in result),
                sequence=k,
                value=rng.uniform(-1, 1),
            )
            for i in ids
        )
    return result
