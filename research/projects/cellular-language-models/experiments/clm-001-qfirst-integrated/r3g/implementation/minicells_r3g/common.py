from pathlib import Path
from minicells_r3f.common import (
    ROOT as ROOT,
    atomic as atomic,
    canonical as canonical,
    digest as digest,
    fingerprint as fingerprint,
    read as read,
    source_sha as source_sha,
)

BASE = Path(__file__).resolve().parents[2]
ID = "CLM-001-R3g-CoAdaptation-Attribution-001"


def configuration(path=None):
    c = read(path or BASE / "config/frozen.json")
    if (
        not c.get("synthetic")
        and canonical(c) != read(BASE / "config/releases.lock")["config_sha256"]
    ):
        raise RuntimeError("R3G_CONFIG_CHANGED")
    if (
        c["experiment_id"] != ID
        or c["seed"] != 151001301
        or c["arms"] != ["A-C", "B-C"]
        or c["optimizer_updates"] != 0
        or c["confirmation"] != "SEALED"
        or c["data_roles"] != ["calibration"]
        or c["development_access"]
    ):
        raise RuntimeError("READONLY_AUDIT_BOUNDARY_CHANGED")
    return c


def parent_config(c):
    old = read(BASE.parent / "r3f-v2/config/frozen.json")
    if canonical(old) != c["parent"]["config_sha256"]:
        raise RuntimeError("PARENT_CONFIG_MISMATCH")
    return old


def parent_identity(c):
    old = parent_config(c)
    return {
        **{
            k: c["parent"][k]
            for k in ("experiment_id", "source_sha", "config_sha256", "run_id", "execution_id")
        },
        "parent_source_sha": old["local_experiment"]["parent_source_sha"],
        "parent_run_id": old["local_experiment"]["parent_run_id"],
    }
