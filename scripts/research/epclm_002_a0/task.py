"""Deterministic synthetic episodes for Persistent State Recall.

The writer-facing view deliberately contains no query or evaluator labels.
Queries are sampled only after the Writer subprocess has exited.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Episode:
    episode_id: str
    events: tuple[dict[str, Any], ...]
    initial_facts: dict[int, int]
    final_facts: dict[int, int]
    overwritten_values: dict[int, int]

    def writer_payload(self) -> dict[str, Any]:
        """Return only write-time events; this object has no query field."""
        return {"episode_id": self.episode_id, "events": list(self.events)}


def generate_episode(
    seed: int,
    *,
    fact_count: int,
    distractor_count: int = 0,
    overwrite_count: int = 0,
    key_bits: int = 16,
    value_bits: int = 16,
) -> Episode:
    """Create an independent random mapping and an interleaved event stream."""
    if fact_count < 1 or distractor_count < 0 or overwrite_count < 0:
        raise ValueError("fact_count must be positive; other counts cannot be negative")
    if overwrite_count > fact_count:
        raise ValueError("overwrite_count cannot exceed fact_count")
    if not 1 <= key_bits <= 32 or not 1 <= value_bits <= 32:
        raise ValueError("key_bits and value_bits must be in [1, 32]")
    key_space, value_space = 1 << key_bits, 1 << value_bits
    if fact_count > key_space:
        raise ValueError("fact_count exceeds the key space")

    rng = random.Random(int(seed))
    keys = rng.sample(range(key_space), fact_count)
    initial = {key: rng.randrange(value_space) for key in keys}
    final = dict(initial)

    facts = [{"kind": "fact", "key": key, "value": value} for key, value in initial.items()]
    distractors = [
        {"kind": "distractor", "payload": rng.getrandbits(64)}
        for _ in range(distractor_count)
    ]
    events = facts + distractors
    rng.shuffle(events)

    overwritten: dict[int, int] = {}
    if overwrite_count:
        for key in rng.sample(keys, overwrite_count):
            old_value = final[key]
            new_value = rng.randrange(value_space)
            while new_value == old_value:
                new_value = rng.randrange(value_space)
            overwritten[key] = old_value
            final[key] = new_value
            events.append({"kind": "fact", "key": key, "value": new_value, "is_update": True})

    digest = hashlib.sha256(f"epclm-002-a0:{seed}:{fact_count}:{distractor_count}:{overwrite_count}".encode()).hexdigest()[:16]
    return Episode(
        episode_id=f"A0-{digest}",
        events=tuple(events),
        initial_facts=initial,
        final_facts=final,
        overwritten_values=overwritten,
    )


def generate_queries(
    episode: Episode,
    *,
    seed: int,
    query_count: int | None = None,
) -> list[int]:
    """Sample only after writing; values and query labels stay evaluator-side."""
    keys = sorted(episode.final_facts)
    count = len(keys) if query_count is None else int(query_count)
    if count < 1:
        raise ValueError("query_count must be positive")
    if count > len(keys):
        raise ValueError("query_count cannot exceed the number of distinct facts")
    rng = random.Random(int(seed))
    return rng.sample(keys, count)


def key_bits(key: int, width: int = 16) -> list[int]:
    if not 0 <= key < (1 << width):
        raise ValueError("key does not fit the requested bit width")
    return [(key >> bit) & 1 for bit in range(width - 1, -1, -1)]


def value_bits(value: int, width: int = 16) -> list[int]:
    if not 0 <= value < (1 << width):
        raise ValueError("value does not fit the requested bit width")
    return [(value >> bit) & 1 for bit in range(width - 1, -1, -1)]
