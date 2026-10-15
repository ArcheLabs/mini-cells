"""Byte-level recursive witnesses for the accepted optimizer, registry and random streams."""

from dataclasses import asdict
import numpy as np
import torch
from .common import canonical_sha256, tensor_hash, hash_accepted_state


def normalized(value):
    if isinstance(value, torch.Tensor):
        return {"tensor": tensor_hash(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): normalized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    return value


class Witness:
    def __init__(self, model, trainer=None):
        self.model = model
        self.trainer = trainer
        self.sha = self.digest()

    def digest(self):
        value = {
            "model": hash_accepted_state(self.model),
            "registry": asdict(self.model.registry),
            "version": self.model.version,
        }
        if self.trainer:
            t = self.trainer
            value.update(
                optimizer=normalized(t.opt.state_dict()),
                rng={k: tensor_hash(g.get_state()) for k, g in t.rng.items()},
                step=t.step,
                data_position=t.data_position,
                journal=normalized(t.journal),
                cache=normalized(t.cache),
            )
        return canonical_sha256(value)

    def assert_unchanged(self, model=None):
        if self.digest() != self.sha:
            raise RuntimeError(
                "INTEGRITY_FAILURE: accepted bundle mutated during candidate transaction"
            )
