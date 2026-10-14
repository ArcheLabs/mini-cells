"""No inference from missing baselines, unmatched assets or an engineering gate."""

import json
from pathlib import Path
from .analysis import noninferiority, seed_summary, internal_comparison


def report(manifest, c):
    manifest = Path(manifest)
    m = json.loads(manifest.read_text())
    if m.get("experiment_id") != "006-PD1":
        raise ValueError("wrong report identity")
    records = []
    failures = []
    for path in m.get("metrics", []):
        p = manifest.parent / path
        if not p.exists():
            failures.append({"path": path, "status": "MISSING"})
            continue
        row = json.loads(p.read_text())
        if row.get("status") == "INCOMPLETE":
            failures.append(row)
        else:
            records.append(row)
    paired = []
    for row in records:
        if row["identity"]["arm"] == "M0":
            continue
        baseline = next(
            (
                r
                for r in records
                if r["identity"]["arm"] == "M0" and r["identity"]["seed"] == row["identity"]["seed"]
            ),
            None,
        )
        if baseline is None:
            failures.append({"arm": row["identity"]["arm"], "status": "MISSING_MOE"})
            continue
        comparison = noninferiority(
            baseline,
            row,
            c["evaluation"]["noninferiority_margin"],
            c["evaluation"]["bootstrap_replicates"],
            c["evaluation"]["bootstrap_seed"],
        )
        comparison.update(arm=row["identity"]["arm"], seed=row["identity"]["seed"])
        paired.append(comparison)
    grouped = {
        arm: seed_summary([r for r in paired if r["arm"] == arm and "upper95_one_sided" in r])
        for arm in {r["arm"] for r in paired}
    }
    internal = {}
    contrasts = {
        "H2": ("P50-F16", "P50-F32"),
        "H3": ("P50-F32", "P50-A32"),
        "H4": ("P50-A32", "P50-A32-J1"),
        "H5": ("P50-STATE32", "P50-A32"),
    }
    for label, (first, second) in contrasts.items():
        internal[label] = []
        for a in records:
            if a["identity"]["arm"] != first:
                continue
            b = next(
                (
                    r
                    for r in records
                    if r["identity"]["arm"] == second and r["identity"]["seed"] == a["identity"]["seed"]
                ),
                None,
            )
            if b is not None:
                internal[label].append(
                    internal_comparison(
                        a, b, c["evaluation"]["bootstrap_seed"], c["evaluation"]["bootstrap_replicates"]
                    )
                )
    return {
        "internal_hypothesis_comparisons": internal,
        "experiment_id": "006-PD1",
        "status": "INCOMPLETE" if failures or not paired else "INCONCLUSIVE",
        "scientific_decision": False,
        "paired_comparisons": paired,
        "per_arm_seed_summary": grouped,
        "failures": failures,
        "H1": "NOT_ESTABLISHED",
        "H2": "NOT_ESTABLISHED",
        "H3": "NOT_ESTABLISHED",
        "H4": "NOT_ESTABLISHED",
        "H5": "NOT_ESTABLISHED",
        "note": "evidence states remain separate; final registered thresholds and review are required",
    }
