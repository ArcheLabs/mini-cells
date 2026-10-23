"""Reuse all R3f operators/forward and baseline-preserving stable hard routing."""

import time
import torch
from minicells_r3f.stable_connection import Model as ParentModel, hard_action, baseline_action
from .sampler import random_inputs, sample, probabilities


class Model(ParentModel):
    def __init__(self, c, seed, arm):
        # C names select the stable connection semantics; correction remains frozen zero.
        super().__init__(c, seed, arm[0] + "-C")
        self.routing_mode = arm[-1]
        self.temperature = 1.0
        self.rho = c["alignment"]["rho"]
        self.training_context = None
        self.observer = None
        self.sampling_enqueue_seconds = 0.0

    def select(self, h, t, previous, *, epsilon=0, rng=None, forced=None, random_ids=None):
        with torch.no_grad():
            q, _, z = self.scores(h, t, previous)
            hard = hard_action(q, baseline_action(self, z), self.route_switch_margin)
            action = hard
            if self.training_context is not None:
                step, micro = self.training_context
                started = time.perf_counter()
                u = random_inputs(self.seed, step, micro, t, len(h), q.shape[-1], h.device)
                action = sample(q, hard, self.routing_mode, epsilon, self.temperature, self.rho, u)
                self.sampling_enqueue_seconds += time.perf_counter() - started
            if forced is not None:
                action = forced
            if self.observer is not None:
                p = probabilities(q, hard, self.routing_mode, epsilon, self.temperature, self.rho)
                self.observer(t, previous, action, hard, p)
        return action, z

    def forward(self, tokens, **kwargs):
        if getattr(self, "active_step", None) is not None:
            self.training_context = (self.active_step, self.microbatch_index)
            self.microbatch_index += 1
        try:
            return super().forward(tokens, **kwargs)
        finally:
            self.training_context = None
