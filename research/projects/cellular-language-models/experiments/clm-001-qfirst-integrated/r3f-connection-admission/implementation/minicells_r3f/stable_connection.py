"""Explicit baseline/raw-score/stable-hard-action semantics. Original weights and sampler unchanged."""

import torch
from minicells_r3e.stable_connection import Model as OriginalModel


def baseline_action(model, z):
    if model.graph:
        return torch.argsort(z, descending=True, stable=True, dim=-1)[:, 0]
    return model.connection.base_action(z)


def hard_action(raw, base, margin):
    if not torch.isfinite(raw).all():
        raise FloatingPointError("NONFINITE_ROUTING_LOGITS")
    candidate = raw.argmax(-1)
    gain = raw.gather(-1, candidate[:, None]).squeeze(-1) - raw.gather(-1, base[:, None]).squeeze(
        -1
    )
    return torch.where((candidate != base) & (gain > margin), candidate, base)


class Model(OriginalModel):
    def __init__(self, c, seed, arm):
        super().__init__(c, seed, arm)
        self.register_buffer("route_switch_margin", torch.tensor(float("inf")))
        self.register_buffer("route_numeric_tolerance", torch.tensor(float("inf")))
        self.margin_evidence = None

    def scores(self, h, t, previous):
        raw, delta, z = super().scores(h, t, previous)
        if not torch.isfinite(raw).all():
            raise FloatingPointError("NONFINITE_ROUTING_LOGITS")
        return raw, delta, z

    def select(self, h, t, previous, *, epsilon=0, rng=None, forced=None, random_ids=None):
        # Stochastic training delegates the unchanged R3e Gumbel/epsilon rule.
        if epsilon and self.arm_name != "B-R":
            return super().select(
                h, t, previous, epsilon=epsilon, rng=rng, forced=forced, random_ids=random_ids
            )
        with torch.no_grad():
            raw, _, z = self.scores(h, t, previous)
            base = baseline_action(self, z)
            action = (
                base
                if self.arm_name in ("A-F", "B-F")
                else hard_action(raw, base, self.route_switch_margin)
            )
            if self.arm_name == "B-R":
                action = random_ids
            if forced is not None:
                action = forced
        return action, z


def build(c, parent, seed, arm, device, data):
    from .diagnostics import calibrate

    m = Model(c, seed, arm)
    current = m.state_dict()
    expected = {
        k
        for k in current
        if not k.startswith("connection.")
        and k not in ("route_switch_margin", "route_numeric_tolerance")
    }
    if set(parent["model"]) != expected:
        raise RuntimeError("PARENT_MODEL_KEYS_DIFFER")
    current.update(parent["model"])
    m.load_state_dict(current, strict=True)
    m = m.to(device)
    calibrate(m, c, data)
    return m
