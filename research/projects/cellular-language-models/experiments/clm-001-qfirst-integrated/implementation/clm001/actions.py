"""Accepted registry has stable Cell IDs, one reserved slot, CTX and real STOP."""

from dataclasses import dataclass, field, asdict
import torch
from .common import canonical_sha256


@dataclass
class Registry:
    cells: int
    reserved: int = 1
    version: int = 0
    activated: list = field(default_factory=list)
    plan: list = field(default_factory=list)
    model_version: int = 0
    policy_version: int = 0

    @property
    def slots(self):
        return self.cells + self.reserved

    @property
    def ctx(self):
        return self.slots

    @property
    def stop(self):
        return self.slots + 1

    @property
    def start(self):
        return self.slots + 2

    @property
    def actions(self):
        return self.slots + 2

    @property
    def fingerprint(self):
        return canonical_sha256(asdict(self))

    def legal(self, depth, budget, minimum, device):
        mask = torch.zeros(self.actions, dtype=torch.bool, device=device)
        if depth >= budget:
            return mask
        mask[: self.cells] = True
        for slot in self.activated:
            mask[slot] = True
        mask[self.ctx] = True
        mask[self.stop] = depth >= minimum
        return mask

    def activate(self, slot):
        if slot < self.cells or slot >= self.slots or slot in self.activated:
            raise ValueError("not an unused reserved Cell")
        self.activated.append(slot)
        self.version += 1
        self.policy_version += 1
        self.model_version += 1
