from pathlib import Path
from minicells_r3f.common import (
    ROOT as ROOT,
    read as read,
    canonical as canonical,
    digest as digest,
    fingerprint as fingerprint,
    atomic as atomic,
    source_sha as source_sha,
)

BASE = Path(__file__).resolve().parents[2]
ID = "CLM-001-R3f-v2-Local-Connection-001"


def configuration(path=None):
    c = read(path or BASE / "config/frozen.json")
    if (
        not c.get("synthetic")
        and canonical(c) != read(BASE / "config/releases.lock")["config_sha256"]
    ):
        raise RuntimeError("FROZEN_CONFIG_CHANGED")
    if (
        c["experiment_id"] != ID
        or c["confirmation"] != "SEALED"
        or c["arms"] != ["A-C", "B-C"]
        or c["seeds"] != [151001301]
    ):
        raise RuntimeError("EXPERIMENT_BOUNDARY_CHANGED")
    if c["proposal"]["backtrack"] != [2.0**-k for k in range(21)]:
        raise RuntimeError("FROZEN_SCALES_CHANGED")
    if (
        c["proposal"]["kl_max"],
        c["proposal"]["action_change_max"],
        c["proposal"]["path_change_max"],
    ) != (0.01, 0.1, 0.2):
        raise RuntimeError("THRESHOLDS_CHANGED")
    if not c.get("synthetic"):
        for path, sha in c["source_file_sha256"].items():
            if digest(ROOT / path) != sha:
                raise RuntimeError("HISTORICAL_SOURCE_CHANGED")
        if (
            c["training"]["states"] != 12
            or c["training"]["policy_steps"] != 8
            or c["training"]["adapt_steps"] != 10
        ):
            raise RuntimeError("BUDGET_CHANGED")
    return c
