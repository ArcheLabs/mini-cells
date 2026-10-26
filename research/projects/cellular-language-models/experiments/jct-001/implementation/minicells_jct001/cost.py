"""Append-only attempted-work ledger; never roll failed computation back with model state."""

import copy
import os
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from .common import canonical, sha


class WorkMeter:
    def __init__(self):
        self.counts = Counter()

    def __deepcopy__(self, memo):
        return self  # Counterfactual candidate copies must count actual extra executions too.

    def add(self, kind, rows):
        self.counts[kind] += int(rows)


class Journal:
    def __init__(self, path, identity):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.meter = WorkMeter()
        self.events = []
        if self.path.exists():
            import json

            for line in self.path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except Exception:
                    raise RuntimeError("JCT_TRUNCATED_LEDGER_QUARANTINE") from None
                expected = dict(row)
                expected.pop("sha256", None)
                previous = self.events[-1]["sha256"] if self.events else None
                if (
                    row.get("seq") != len(self.events)
                    or row.get("previous") != previous
                    or row.get("sha256") != sha(expected)
                ):
                    raise RuntimeError("JCT_LEDGER_CHAIN_INVALID")
                self.events.append(row)
            if not self.events or self.events[0]["data"] != identity:
                raise RuntimeError("JCT_LEDGER_IDENTITY_MISMATCH")
        else:
            self.append("HEADER", identity)

    def append(self, kind, data):
        row = dict(
            seq=len(self.events),
            previous=self.events[-1]["sha256"] if self.events else None,
            kind=kind,
            data=data,
        )
        row["sha256"] = sha(row)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.events.append(row)
        return row

    def watermark(self):
        return dict(events=len(self.events), sha256=self.events[-1]["sha256"])

    def verify_prefix(self, watermark):
        index = watermark["events"] - 1
        if (
            index < 0
            or index >= len(self.events)
            or self.events[index]["sha256"] != watermark["sha256"]
        ):
            raise RuntimeError("JCT_CHECKPOINT_LEDGER_PREFIX_MISMATCH")

    @contextmanager
    def phase(self, kind):
        key = "phase-" + str(len(self.events))
        before = copy.deepcopy(self.meter.counts)
        start = time.perf_counter()
        self.append("PHASE_BEGIN", dict(id=key, phase=kind))
        try:
            yield
        except Exception:
            self.append(
                "PHASE_ERROR",
                dict(
                    id=key,
                    phase=kind,
                    cpu_wall_seconds=time.perf_counter() - start,
                    work=dict(self.meter.counts - before),
                ),
            )
            raise
        else:
            self.append(
                "PHASE_END",
                dict(
                    id=key,
                    phase=kind,
                    cpu_wall_seconds=time.perf_counter() - start,
                    work=dict(self.meter.counts - before),
                ),
            )

    def optimizer_call(self, kind, call, parameters):
        import torch

        key = kind + "-" + str(len(self.events))
        grads = [p.grad.detach() for p in parameters if p.grad is not None]
        norm = float(torch.stack([g.square().sum() for g in grads]).sum().sqrt()) if grads else 0.0
        if (
            not grads
            or not all(torch.isfinite(g).all() for g in grads)
            or (norm == 0 and not getattr(self, "allow_zero_signal", False))
        ):
            raise RuntimeError("JCT_INTENDED_OPTIMIZER_SIGNAL_MISSING_" + kind)
        self.append("OPT_BEGIN", dict(id=key, optimizer=kind, gradient_norm=norm))
        try:
            result = call()
        except Exception:
            self.append(
                "OPT_ERROR",
                dict(id=key, optimizer=kind, completion="UNKNOWN_PARTIAL_MUTATION_POSSIBLE"),
            )
            raise
        self.append("OPT_END", dict(id=key, optimizer=kind))
        return result

    def summary(self):
        begun = [r["data"] for r in self.events if r["kind"] == "OPT_BEGIN"]
        ended = {r["data"]["id"] for r in self.events if r["kind"] == "OPT_END"}
        failures = [
            r for r in self.events if r["kind"] in ("OPT_ERROR", "ATTEMPT_ERROR", "PHASE_ERROR")
        ]
        committed = {
            r["data"]["step"]
            for r in self.events
            if r["kind"] in ("LOGICAL_COMMIT", "LOGICAL_RECOVERED")
        }
        attempts = [r for r in self.events if r["kind"] == "ATTEMPT_BEGIN"]
        done_attempts = {
            r["data"]["id"] for r in self.events if r["kind"] in ("ATTEMPT_END", "ATTEMPT_ERROR")
        }
        unknown = [r for r in begun if r["id"] not in ended]
        pending_attempts = [r for r in attempts if r["data"]["id"] not in done_attempts]
        phases = [r["data"] for r in self.events if r["kind"] in ("PHASE_END", "PHASE_ERROR")]
        work = Counter()
        for p in phases:
            work.update(p["work"])
        return dict(
            logical_updates=len(committed),
            logical_steps=sorted(committed),
            actual_optimizer_attempts=len(begun),
            completed_optimizer_calls=len(ended),
            optimizer_calls_by_group=dict(
                Counter(r["optimizer"] for r in begun if r["id"] in ended)
            ),
            uncertain_optimizer_calls=len(unknown),
            pending_attempts=len(pending_attempts),
            failed_or_retried_attempt_records=len(failures),
            input_examples_attempted=sum(r["data"]["examples"] for r in attempts),
            counterfactual_target_evaluations=sum(
                r["data"]["target_evaluations"]
                for r in self.events
                if r["kind"] == "AUXILIARY_LABEL_WORK"
            ),
            distinct_extra_counterfactual_examples=0,
            training_tokens=None,
            input_unit="SYNTHETIC_FEATURE_EXAMPLES_NOT_LANGUAGE_TOKENS",
            gpu_seconds=None,
            gpu_used=False,
            cpu_seconds_by_phase=dict(
                (name, sum(p["cpu_wall_seconds"] for p in phases if p["phase"] == name))
                for name in {p["phase"] for p in phases}
            ),
            executed_operator_work=dict(work),
            analytic_router_forward_mac=work["policy_rows"] * 108,
            analytic_cell_forward_mac=work["cell_rows"],
            observed_flops=None,
            accounting_status="UNCERTAIN_INFLIGHT_WORK"
            if unknown or pending_attempts
            else "RECONCILED_CPU_FIXTURE",
            claim_scope="ACTUAL_CPU_FIXTURE_WORK_NOT_GPU_RUNTIME_PROOF",
        )
