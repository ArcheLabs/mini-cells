"""Deterministic SEARCH/SELECT separation and quality-independent random control."""

import math


def shortlist(registry, search_results):
    valid = {r["candidate_id"]: r for r in registry if r["family"] != "BASELINE" and r["valid"]}
    if set(search_results) != set(valid):
        raise ValueError("SEARCH_INVENTORY_MISMATCH")
    if any(not math.isfinite(x) for x in search_results.values()):
        raise ValueError("NONFINITE_SEARCH_SCORE")
    return sorted(valid, key=lambda key: (search_results[key], key))[:4]


def finalists(registry, search_results, select_results):
    chosen = shortlist(registry, search_results)
    if set(select_results) != set(chosen):
        raise ValueError("SELECT_INVENTORY_MISMATCH")
    if any(not math.isfinite(x) for x in select_results.values()):
        raise ValueError("NONFINITE_SELECT_SCORE")
    random = sorted(r["candidate_id"] for r in registry if r["valid"] and r["family"] == "RANDOM")
    baseline = [r["candidate_id"] for r in registry if r["family"] == "BASELINE"]
    historical = [r["candidate_id"] for r in registry if r["valid"] and r["family"] == "HISTORICAL"]
    if len(baseline) != 1 or len(historical) > 1:
        raise ValueError("BASELINE_OR_HISTORY_IDENTITY_INVALID")
    roles = dict(
        BASELINE=baseline[0],
        HISTORICAL=historical[0] if historical else None,
        SEARCH_BEST=min(chosen, key=lambda key: (select_results[key], key)) if chosen else None,
        RANDOM_CONTROL=random[0] if random else None,
    )
    physical = {}
    by_id = {r["candidate_id"]: r for r in registry}
    for role, key in roles.items():
        if key is not None:
            physical.setdefault(by_id[key]["parameter_sha256"], []).append(role)
    return dict(
        roles=roles,
        physical_groups=physical,
        random_rule="LOWEST_CANDIDATE_HASH_NOT_QUALITY",
        scope="FINITE_CANDIDATE_SET_NOT_GLOBAL_OPTIMUM",
    )
