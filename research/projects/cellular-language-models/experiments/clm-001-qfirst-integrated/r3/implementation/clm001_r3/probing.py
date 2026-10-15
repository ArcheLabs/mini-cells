"""Bounded detached event reservoir and hooks on actual post-operation tensors."""

from dataclasses import dataclass
import torch
from exp006.memory import Memory
from clm001_r2.rollout import Probe


@dataclass
class Event:
    state: object
    post: torch.Tensor
    gradient: object
    ctx_seen: bool
    stratum: tuple
    inclusion: float = 0.0


class EventSampler:
    def __init__(self, capacity, generator, bucket_width=4, features=None):
        self.features = features
        self.exposure = {0: {}, 1: {}}
        self.capacity = capacity
        self.generator = generator
        self.width = bucket_width
        self.pools = {}
        self.counts = {}
        self.hooks = []
        self.events = []

    def observe(
        self,
        hidden,
        update,
        selected,
        action,
        previous,
        batches,
        positions,
        depth,
        memory,
        n,
        registry,
        ctx_seen,
    ):
        if not len(selected) or self.capacity == 0:
            return
        features = self.features if self.features is not None else positions % 2
        counts = torch.bincount(features[selected].long(), minlength=2).tolist()
        for group, count in enumerate(counts):
            self.exposure[group][int(action)] = self.exposure[group].get(int(action), 0) + count
        # One bounded reservoir per depth/context/action stratum; detached tensors only.
        for seen in (False, True):
            offsets = torch.where(ctx_seen[selected] == seen)[0]
            count = len(offsets)
            if not count:
                continue
            key = (depth // self.width, bool(seen), int(action))
            old = self.counts.get(key, 0)
            self.counts[key] = old + count
            # Vectorized random-priority reservoir. Never synchronize once per visited event.
            priorities = torch.rand(count, generator=self.generator, device=hidden.device)
            take = min(self.capacity, count)
            values, indices = torch.topk(priorities, take, largest=False)
            pool = self.pools.setdefault(key, [])
            offered = list(pool)
            detached = Memory(memory.key.detach(), memory.value.detach())
            for priority, offset in zip(values.tolist(), offsets[indices].tolist()):
                token = int(selected[offset])
                state = Probe(
                    hidden[token].detach().clone(),
                    detached,
                    int(batches[token]),
                    int(positions[token]),
                    int(previous[token]),
                    depth,
                    int(action),
                    registry.fingerprint,
                    (token // n, token % n),
                )
                event = Event(state, update[offset].detach().clone(), None, bool(seen), key)
                event.group_feature = int(features[token])
                event.priority = priority
                event.handle = None
                if update.requires_grad:

                    def receive(gradient, event=event, offset=offset):
                        event.gradient = gradient[offset].detach().clone()

                    event.handle = update.register_hook(receive)
                offered.append(event)
            offered.sort(key=lambda e: e.priority)
            self.pools[key] = offered[: self.capacity]
            for event in offered[self.capacity :]:
                if event.handle is not None:
                    event.handle.remove()

    def finish(self):
        # Randomized, target-free order; equal rotating allocation to nonempty strata.
        keys = sorted(self.pools)
        if keys:
            order = torch.randperm(
                len(keys), generator=self.generator, device=self.generator.device
            ).tolist()
            keys = [keys[i] for i in order]
        chosen = []
        quota = {k: 0 for k in keys}
        for rank in range(self.capacity):
            for key in keys:
                if rank < len(self.pools[key]) and len(chosen) < self.capacity:
                    chosen.append(self.pools[key][rank])
                    quota[key] += 1
        expected = {key: 0.0 for key in keys}
        remaining = self.capacity
        for rank in range(self.capacity):
            active = [key for key in keys if rank < len(self.pools[key])]
            if not active or remaining <= 0:
                break
            fraction = min(1.0, remaining / len(active))
            for key in active:
                expected[key] += fraction
            remaining -= min(remaining, len(active))
        for event in chosen:
            event.inclusion = expected[event.stratum] / self.counts[event.stratum]
        # Unchosen hooks are harmless for values; remove all after the caller's backward.
        self.events = chosen
        return chosen

    def close(self):
        for pool in self.pools.values():
            for event in pool:
                if event.handle is not None:
                    event.handle.remove()
        self.hooks = []
        self.pools = {}
