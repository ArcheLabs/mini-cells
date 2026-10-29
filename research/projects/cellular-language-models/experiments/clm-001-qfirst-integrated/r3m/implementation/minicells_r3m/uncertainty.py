"""Descriptive paired-group bootstrap, explicitly separate from independent-seed inference."""

import numpy as np


def paired(old, new, task, seed=153301, replicates=500):
    metric = "nll" if task == "story" else "answer_nll"

    def groups(rows):
        result = {}
        for r in rows:
            if r["task"] == task:
                result.setdefault(r["group"], []).append(r[metric])
        return {k: float(np.mean(v)) for k, v in result.items()}

    a, b = groups(old["rows"]), groups(new["rows"])
    if set(a) != set(b) or not a:
        raise RuntimeError("PAIRED_REPORT_GROUPS_CHANGED")
    delta = np.array([a[k] - b[k] for k in sorted(a)])
    rng = np.random.default_rng(seed)
    boot = [
        float(rng.choice(delta, size=len(delta), replace=True).mean()) for _ in range(replicates)
    ]
    return dict(
        mean=float(delta.mean()),
        interval95=np.quantile(boot, [0.025, 0.975]).tolist(),
        groups=len(delta),
        replicates=replicates,
        unit="PAIRED_REPORT_GROUP_WITHIN_SEED_NOT_INDEPENDENT_SEED_REPLICATION",
    )
