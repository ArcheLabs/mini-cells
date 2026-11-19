"""Frozen P1.1 pairing protocol and paired hierarchical bootstrap."""
import random
from statistics import mean

SEEDS = (42, 43, 44)
DONOR_SEED = 20261117
BOOTSTRAP_SEED = 20261118


def donor_assignments(size=64, count=3):
    if not 0 < count < size:
        raise ValueError("Each query needs distinct non-self donors")
    rng = random.Random(DONOR_SEED)
    order = rng.sample(range(size), size)
    shifts = rng.sample(range(1, size), count)
    return {query: [order[(index + shift) % size] for shift in shifts]
            for index, query in enumerate(order)}


def paired_summary(seeds, replicates=10000):
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(replicates):
        sample = rng.choices(seeds, k=len(seeds))
        draws.append(mean(mean(rng.choices(rows, k=len(rows))) for rows in sample))
    draws.sort()
    return {
        "mean": mean(map(mean, seeds)),
        "ci95": [draws[int(.025 * replicates)], draws[int(.975 * replicates)]],
        "per_seed": [mean(rows) for rows in seeds],
        "win_rate": mean(mean(x > 0 for x in rows) for rows in seeds),
        "pairs": sum(map(len, seeds)),
    }
