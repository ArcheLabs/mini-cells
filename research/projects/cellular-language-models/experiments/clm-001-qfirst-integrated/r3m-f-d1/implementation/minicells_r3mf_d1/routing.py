"""R3h H-routing adapter that aligns keyed random inputs to packed source offsets."""


def source_random_inputs(
    seed, step, microbatch, stage, source_start, rows, actions, device, factory
):
    if source_start < 0 or rows < 0 or source_start + rows > 64:
        raise ValueError("D1_SOURCE_ROUTE_OFFSET_OUT_OF_RANGE")
    full = factory(seed, step, microbatch, stage, 64, actions, device)
    return {name: values[source_start : source_start + rows] for name, values in full.items()}


def make_model_class():
    """Build lazily so CPU-gate and synthetic tests do not import Torch/models."""
    import torch
    from minicells_r3h.routing import Model as ParentModel
    from minicells_r3h.sampler import probabilities, random_inputs, sample
    from minicells_r3f.stable_connection import baseline_action, hard_action

    class D1Model(ParentModel):
        def __init__(self, c, seed, arm="B-H"):
            super().__init__(c, seed, arm)
            self.d1_source_context = None

        def forward_fragment(self, tokens, *, source_context, **kwargs):
            if self.active_step is None:
                raise RuntimeError("D1_FRAGMENT_FORWARD_REQUIRES_TRAINING_STEP")
            self.d1_source_context = source_context
            try:
                return self(tokens, **kwargs)
            finally:
                self.d1_source_context = None

        def select(self, h, t, previous, *, epsilon=0, rng=None, forced=None, random_ids=None):
            if self.d1_source_context is None:
                return super().select(
                    h, t, previous, epsilon=epsilon, rng=rng, forced=forced, random_ids=random_ids
                )
            with torch.no_grad():
                q, _, z = self.scores(h, t, previous)
                hard = hard_action(q, baseline_action(self, z), self.route_switch_margin)
                step = self.d1_source_context["absolute_step"]
                microbatch = self.d1_source_context["source_microbatch"]
                start = self.d1_source_context["source_start"]
                u = source_random_inputs(
                    self.seed,
                    step,
                    microbatch,
                    t,
                    start,
                    len(h),
                    q.shape[-1],
                    h.device,
                    random_inputs,
                )
                action = sample(q, hard, self.routing_mode, epsilon, self.temperature, self.rho, u)
                if forced is not None:
                    action = forced
                if self.observer is not None:
                    p = probabilities(
                        q, hard, self.routing_mode, epsilon, self.temperature, self.rho
                    )
                    self.observer(t, previous, action, hard, p)
            return action, z

    return D1Model
