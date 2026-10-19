"""Frozen hash-stratified sampling with exact within-stratum inclusion weights."""

from collections import defaultdict
from minicells_runtime.store import canonical, ContinuationError


def stratum(row):
    p = row["previous_action"]
    group = str(p) if p in (0, 3, 5) else "CTX" if p == row["ctx_action"] else "OTHER"
    return (row["depth_zero_based"], group, min(2, row["n_ctx_so_far"]), row["forced_end_path"])


def sample(records, count, seed):
    bins = defaultdict(list)
    for r in records:
        if r["stop_legal"] and not r["actual_stop"]:
            bins[stratum(r)].append(r)
    if not bins:
        raise ContinuationError("CF_NO_ELIGIBLE_STATES")
    target = min(count, sum(map(len, bins.values())))
    if target < len(bins):
        raise ContinuationError("CF_SAMPLE_BUDGET_BELOW_STRATA")
    quotas = {k: 1 for k in bins}
    left = target - len(bins)
    while left:
        progress = False
        for k in sorted(bins, key=lambda k: canonical([seed, k])):
            if quotas[k] < len(bins[k]):
                quotas[k] += 1
                left -= 1
                progress = True
            if not left:
                break
        if not progress:
            break
    chosen = []
    report = []
    for k, rows in sorted(bins.items()):
        ordered = sorted(
            rows,
            key=lambda r: canonical(
                [
                    seed,
                    r["window_id"],
                    r["story_id"],
                    r["token_position"],
                    r["depth_zero_based"],
                    r["state_fingerprint"],
                ]
            ),
        )
        n = quotas[k]
        prob = n / len(rows)
        for r in ordered[:n]:
            chosen.append(
                {
                    **r,
                    "inclusion_probability": prob,
                    "design_weight": 1 / prob,
                    "sampling_stratum": list(k),
                }
            )
        report.append(
            {
                "stratum": list(k),
                "population": len(rows),
                "sampled": n,
                "inclusion_probability": prob,
            }
        )
    manifest = {
        "seed": seed,
        "requested": count,
        "selected": len(chosen),
        "eligible_population": sum(map(len, bins.values())),
        "strata": report,
        "story_counts": {
            str(s): sum(r["story_id"] == s for r in chosen)
            for s in sorted({r["story_id"] for r in records})
        },
        "sampling_rule": "HASH_ORDER_WITHIN_FIXED_JOINT_STRATA",
        "weights": "N_h/n_h",
    }
    manifest["sample_sha256"] = canonical(chosen)
    return chosen, manifest
