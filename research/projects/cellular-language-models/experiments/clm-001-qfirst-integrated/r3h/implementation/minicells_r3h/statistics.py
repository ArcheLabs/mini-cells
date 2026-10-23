"""Story-cluster paired bootstrap inside Seed; Holm across four primary comparisons."""

import numpy as np
from minicells_r3c_moe_bridge.statistics import story_interval


def compare(c, results):
    out = {}
    for candidate, control in (
        c["statistics"]["primary_comparisons"] + c["statistics"]["secondary_comparisons"]
    ):
        rows = []
        deltas = []
        draws = []
        for seed in c["seeds"]:
            a = results[str(seed)][candidate]["quality"]["stories"]
            b = results[str(seed)][control]["quality"]["stories"]
            if set(a) != set(b):
                raise RuntimeError("STORY_PAIRING_CHANGED")
            keys = sorted(a)
            if any(a[k]["tokens"] != b[k]["tokens"] for k in keys):
                raise RuntimeError("STORY_TOKEN_WEIGHTS_CHANGED")
            x = np.array([a[k]["mean_nll"] - b[k]["mean_nll"] for k in keys])
            weights = np.array([a[k]["tokens"] for k in keys], dtype=float)
            delta = (
                results[str(seed)][candidate]["quality"]["nll"]
                - results[str(seed)][control]["quality"]["nll"]
            )
            interval = story_interval(
                x, c["statistics"]["bootstrap_seed"] + seed, c["statistics"]["bootstrap_replicates"]
            )
            interval["delta_token_weighted_nll"] = delta
            rows.append(dict(seed=seed, story_paired=interval))
            deltas.append(delta)
            rng = np.random.default_rng(c["statistics"]["bootstrap_seed"] + seed)
            ix = rng.integers(len(x), size=(c["statistics"]["bootstrap_replicates"], len(x)))
            draws.append((x[ix] * weights[ix]).sum(1) / weights[ix].sum(1))
        samples = np.stack(draws).mean(0)
        mean = float(np.mean(deltas))
        p = float((1 + np.sum(np.abs(samples - mean) >= abs(mean))) / (len(samples) + 1))
        supported = all(d < 0 for d in deltas) and mean <= -c["statistics"]["minimum_gain"]
        out[candidate + " vs " + control] = dict(
            per_seed=rows,
            deltas=deltas,
            mean_delta=mean,
            ci95_story_bootstrap_conditional_on_three_seeds=np.quantile(
                samples, [0.025, 0.975]
            ).tolist(),
            exploratory_bootstrap_p=p,
            status="SUPPORTED_EXPLORATORY" if supported else "NOT_SUPPORTED",
            statistical_confirmation=False,
            independent_training_seeds=3,
            caveat="Story bootstrap does not replace training Seed replication; Development previously used.",
        )
    family = [a + " vs " + b for a, b in c["statistics"]["primary_comparisons"]]
    previous = 0.0
    for i, key in enumerate(sorted(family, key=lambda k: out[k]["exploratory_bootstrap_p"])):
        previous = max(previous, min(1.0, (len(family) - i) * out[key]["exploratory_bootstrap_p"]))
        out[key]["holm_exploratory_p"] = previous
    return out
