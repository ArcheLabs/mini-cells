"""Device adapter over the CPU-audited trainer; no historical source mutation."""

import torch
from minicells_r3k.reuse import bind
from .training import Trainer
from .data import batches


class DeviceTrainer(Trainer):
    def __init__(self, config, arm, identity, journal, base_state=None, device="cuda:0"):
        self.device = torch.device(device)
        super().__init__(config, arm, identity, journal, base_state)
        self.model.to(self.device)
        self.journal.allow_zero_signal = True

    def state(self):
        state = super().state()
        state["execution_device_type"] = self.device.type
        state["cuda_rng"] = (
            torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        )
        return state

    def _restore_values(self, state):
        if state.get("execution_device_type") != self.device.type:
            raise RuntimeError("JCT_DEVICE_BASELINE_IDENTITY_CHANGED")
        super()._restore_values(state)
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)

    def advance(self, data, checkpoint, failure_hook=None):
        def device_batches(dataset, engine, rng, ignored_device):
            return batches(dataset, engine, rng, self.device)

        execute = bind(Trainer.advance, batches=device_batches)
        return execute(self, data, checkpoint, failure_hook)
