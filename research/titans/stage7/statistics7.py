"""Episode-clustered summaries for the paired retrieval controls."""
import random
from statistics import mean


def paired(values, *, seed=42):
    if not values:
        raise ValueError('Cannot aggregate an empty evaluation')
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(mean(rng.choices(values, k=n)) for _ in range(10000))
    return dict(mean=mean(values), ci95=[draws[249], draws[9749]], episodes=n)
