"""Reuse R3e Cell AdamW, schedule, CE/backward and full RNG restart state."""

import copy
import time
from minicells_r3e.training import Trainer as ParentTrainer
from .parent_import import fingerprint, canonical
from .instrumentation import Exposure


class Trainer(ParentTrainer):
    def __init__(self, m, c, arm):
        super().__init__(m, c, arm)
        self.mode = m.routing_mode
        self.temperature = m.temperature
        self.alignment = []
        self.connection_sha = fingerprint(m.connection.state_dict())
        self.opt_identity = canonical(
            {
                "parent": self.opt_identity,
                "router": self.mode,
                "temperature": self.temperature,
                "random_protocol": c["alignment"]["random_protocol"],
            }
        )
        self.parent_identity = None
        self.monitor_seconds = 0.0

    def epsilon(self):
        q = self.c["training"]
        fraction = min(self.step, q["steps"] - 1) / max(1, q["steps"] - 1)
        return q["epsilon_initial"] + (q["epsilon_final"] - q["epsilon_initial"]) * fraction

    def update(self, data, batches=None):
        m = self.model
        obs = Exposure(m)
        m.observer = obs.observe
        m.active_step = self.step
        m.microbatch_index = 0
        try:
            row = super().update(data, batches)
        finally:
            m.observer = None
            m.active_step = None
            m.training_context = None
        begin = time.perf_counter()
        row["exposure"] = obs.result()
        row["epsilon"] = self.c["training"]["epsilon_initial"] + (
            self.c["training"]["epsilon_final"] - self.c["training"]["epsilon_initial"]
        ) * (self.step - 1) / max(1, self.c["training"]["steps"] - 1)
        self.monitor_seconds += time.perf_counter() - begin
        return row

    def verify_frozen(self):
        super().verify_frozen()
        if fingerprint(self.model.connection.state_dict()) != self.connection_sha:
            raise RuntimeError("FROZEN_CONNECTION_CHANGED")
        if any(p.requires_grad or p.grad is not None for p in self.model.connection.parameters()):
            raise RuntimeError("CONNECTION_GRADIENT_FORBIDDEN")

    def state(self):
        s = super().state()
        s["r3h"] = {
            "routing_mode": self.mode,
            "temperature": self.temperature,
            "rho": self.model.rho,
            "random_protocol": self.c["alignment"]["random_protocol"],
            "connection_sha": self.connection_sha,
            "alignment": copy.deepcopy(self.alignment),
            "parent_identity": copy.deepcopy(self.parent_identity),
            "monitor_seconds": self.monitor_seconds,
            "sampling_enqueue_seconds": self.model.sampling_enqueue_seconds,
        }
        return s

    def restore(self, s):
        r = s["r3h"]
        if any(
            r[k] != v
            for k, v in [
                ("routing_mode", self.mode),
                ("temperature", self.temperature),
                ("rho", self.model.rho),
                ("random_protocol", self.c["alignment"]["random_protocol"]),
                ("connection_sha", self.connection_sha),
            ]
        ):
            raise RuntimeError("RESTORE_ROUTER_IDENTITY_CHANGED")
        if r["parent_identity"] != self.parent_identity:
            raise RuntimeError("RESTORE_PARENT_CHANGED")
        super().restore(s)
        self.alignment = copy.deepcopy(r["alignment"])
        self.monitor_seconds = r["monitor_seconds"]
        self.model.sampling_enqueue_seconds = r["sampling_enqueue_seconds"]
