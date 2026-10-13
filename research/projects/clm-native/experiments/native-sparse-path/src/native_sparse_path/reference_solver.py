"""Independent NumPy executor. Never imported by the learned model/router."""
import numpy as np


def solve(instance, *, k=2, active_limit=8, return_trace=False):
    x = instance.x.cpu().numpy().astype(np.float64)
    addresses = instance.candidates.cpu().numpy()
    values = x[:, 7:9].copy()
    active = list(instance.roots.cpu().numpy())
    trace = []
    for _ in range(instance.depth):
        packets = []
        width = k if instance.task == 'B' else 1
        for source in active:
            message = np.tanh(x[source, :2] * values[source] + x[source, 2:4])
            candidates = addresses[source]
            scores = (-x[candidates, 6] if instance.task == 'C'
                      else x[candidates, 4:6] @ message)
            # Priorities resolve ties without using physical node numbers.
            order = np.lexsort((x[candidates, 6], -scores))[:width]
            for slot in order:
                packets.append((source, int(candidates[slot]), message / width))
        targets = sorted({j for _, j, _ in packets}, key=lambda j: x[j, 6])[:active_limit]
        allowed = set(targets)
        packets = [p for p in packets if p[1] in allowed]
        old = values.copy()
        for target in targets:
            incoming = sum((m for _, j, m in packets if j == target), np.zeros(2))
            values[target] = np.tanh(0.3 * old[target] + incoming)
        trace.append({'sources': list(active), 'targets': targets,
                      'edges': [(int(i), int(j)) for i, j, _ in packets],
                      'values': values[targets].tolist()})
        active = targets
    answer = ((values[active] * x[active, 4:6]).sum(0) / np.sqrt(len(active))
              if instance.task == 'B' else values[active].mean(0))
    return (answer, trace) if return_trace else answer
