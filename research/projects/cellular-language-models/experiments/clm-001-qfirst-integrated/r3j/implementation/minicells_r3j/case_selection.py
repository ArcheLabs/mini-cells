"""Predeclared selection; no Development inputs, no replacement for missing cases."""

from minicells_r3e.common import canonical


def select(records):
    if (
        len(records) != 48
        or len({(r["seed"], r["arm"], r["proposal_at_step"]) for r in records}) != 48
    ):
        raise RuntimeError("HISTORICAL_48_PROPOSAL_MATRIX_INCOMPLETE")
    accepted = [r for r in records if r["historical_accepted"]]
    if len(accepted) != 8:
        raise RuntimeError("HISTORICAL_ACCEPTED_COUNT_MISMATCH")
    selected = [dict(r, selection_reason="ALL_EIGHT_ACCEPTED") for r in accepted]
    missing = []
    for arm in ("A-CH", "A-CT", "B-CH", "B-CT"):
        choices = sorted(
            [
                r
                for r in records
                if r["arm"] == arm and not r["historical_accepted"] and r["trust_region_passed"]
            ],
            key=lambda r: (r["seed"], r["proposal_at_step"]),
        )
        if choices:
            selected.append(
                dict(choices[0], selection_reason="EARLIEST_TRUST_PASSED_REJECTED_IN_ARM")
            )
        else:
            missing.append(arm)
    manifest = {
        "rule_version": "R3J_V1_ALL8_ACCEPTED_PLUS_FIRST_REJECTED_PER_ARM",
        "cases": sorted(selected, key=lambda r: (r["seed"], r["arm"], r["proposal_at_step"])),
        "missing_rejected_arms": missing,
        "uses_long_horizon_metrics": False,
    }
    return {**manifest, "sha256": canonical(manifest)}
