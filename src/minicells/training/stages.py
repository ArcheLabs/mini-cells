"""Token/step based, frozen 25/25/50 schedule, independent of device speed."""
import math


def stage(step, total_steps, kind):
    if kind != 'hard-cell':
        return 'soft' if kind == 'soft-cell' else 'native', 1.0
    warmup, anneal = total_steps // 4, total_steps // 2
    if step <= warmup:
        return 'soft', 2.0
    if step <= anneal:
        fraction = (step - warmup - 1) / max(1, anneal - warmup - 1)
        return 'anneal', math.exp(math.log(2.0) + fraction * math.log(0.1 / 2.0))
    return 'hard', 0.1
