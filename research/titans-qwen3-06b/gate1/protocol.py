"""Shared Gate 1 budget and paired statistics."""
import random
from statistics import mean, stdev

SEEDS = (42, 43, 44)
STEPS = 500
EVAL_STEPS = (0, 100, 200, 300, 400, 500)


def mixed(synthetic, web):
    if len(synthetic) != len(web):
        raise ValueError("Gate 1 requires a 50/50 episode mix")
    return [row for pair in zip(synthetic, web) for row in pair]


def summarize(rows, comparison, metric="nll"):
    """Treat the episode as the paired unit; resample seeds hierarchically."""
    sign = 1 if metric == "nll" else -1
    values = [[sign * (r[comparison][metric] - r["correct"][metric]) for r in rows[s]]
              for s in SEEDS]
    rng = random.Random(20261119)
    draws = sorted(mean(mean(rng.choices(v, k=len(v))) for v in
                        rng.choices(values, k=len(values))) for _ in range(10000))
    averages = list(map(mean, values))
    return dict(mean=mean(averages), seed_sd=stdev(averages),
                ci95=[draws[250], draws[9749]],
                per_seed=dict(zip(SEEDS, averages)),
                win_rate={s:mean(x > 0 for x in v) for s,v in zip(SEEDS,values)})
