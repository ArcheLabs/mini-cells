"""Immutable pre-outcome prediction contract. No model fitting or outcome reader."""

import hashlib
import json
import math
from pathlib import Path

SCORES = ("S_Cal", "S_Direct", "S_K10", "S_Cell", "S_Slope")
FORMULAS = dict(
    S_Cal="historical_cal_OO10-historical_cal_NN10",
    S_Direct="probe_OO0-probe_ON0",
    S_K10="probe_OO10-probe_NN10",
    S_Cell="probe_ON10-probe_NN10",
    S_Slope="S_K10-S_Direct",
)


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def scores(historical_cal, direct, k10):
    result = dict(
        S_Cal=historical_cal["OO"] - historical_cal["NN"],
        S_Direct=direct["OO"] - direct["ON"],
        S_K10=k10["OO"] - k10["NN"],
        S_Cell=k10["ON"] - k10["NN"],
    )
    result["S_Slope"] = result["S_K10"] - result["S_Direct"]
    if not all(math.isfinite(x) for x in result.values()):
        raise ValueError("NONFINITE_PREDICTION")
    return result


def freeze(path, rows, data_identity, source_identity, protocol=None):
    ids = [row["case_id"] for row in rows]
    matrix = {(r["seed"], r["arm"], r["proposal_at_step"]) for r in rows}
    expected = {
        (s, a, t)
        for s in (151001301, 151001302, 151001303)
        for a in ("A-CH", "A-CT", "B-CH", "B-CT")
        for t in (100, 200, 300, 400)
    }
    if len(ids) != 48 or len(set(ids)) != 48 or matrix != expected:
        raise ValueError("COMPLETE_HISTORICAL_48_REQUIRED")
    for row in rows:
        if set(row["scores"]) != set(SCORES) or not all(
            math.isfinite(x) for x in row["scores"].values()
        ):
            raise ValueError("INVALID_PREDICTION")
    manifest = dict(
        scope="PREDECLARED_POST_R3J_EXPLORATORY_PREDICTION",
        formulas=FORMULAS,
        protocol=protocol,
        data_identity=data_identity,
        source_identity=source_identity,
        rows=sorted(rows, key=lambda x: x["case_id"]),
    )
    manifest["sha256"] = canonical(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError("FROZEN_PREDICTIONS_CHANGED") from None
    return manifest


def require_persisted(manifest, readback):
    if canonical(readback) != canonical(manifest):
        raise ValueError("PREDICTIONS_REMOTE_READBACK_MISMATCH")
    check = dict(manifest)
    claimed = check.pop("sha256")
    if canonical(check) != claimed:
        raise ValueError("PREDICTION_MANIFEST_CORRUPT")
    return claimed
