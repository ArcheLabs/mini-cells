from pathlib import Path
import subprocess
from minicells_r3c_moe_bridge.common import (
    canonical,
    atomic as atomic,
    digest,
    read,
    fingerprint as fingerprint,
    ContinuationError,
)

BASE = Path(__file__).resolve().parents[2]
ROOT = next(p for p in BASE.parents if (p / "pyproject.toml").exists())
ID = "CLM-001-R3d-LHCJL-001"
ARMS = ("F0", "L", "M", "F", "P", "J")


def configuration(path=None):
    c = read(path or BASE / "config/frozen.json")
    if c["experiment_id"] != ID or c["confirmation"] != "SEALED" or tuple(c["arms"]) != ARMS:
        raise ContinuationError("R3D_CONTRACT_MISMATCH")
    if (
        c["engine"]["input"]["confirmation_materialized"]
        or c["engine"]["release"]["confirmation"] != "SEALED"
    ):
        raise ContinuationError("CONFIRMATION_SEALED")
    if not c.get("synthetic"):
        lock = read(BASE / "config/releases.lock")
        if lock["config_sha256"] != canonical(c):
            raise ContinuationError("FROZEN_CONFIG_CHANGED")
        if canonical(c["engine"]) != c["parent"]["config_sha256"]:
            raise ContinuationError("PARENT_CONFIG_MISMATCH")
        for name, sha in c["engine"]["development"]["parent_pilot"]["core_file_sha256"].items():
            if digest(ROOT / name) != sha:
                raise ContinuationError("HISTORICAL_CORE_CHANGED", name)
    return c


def source_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def identity(c, source, seed, arm):
    return {
        "experiment_id": ID,
        "source_sha": source,
        "config_sha256": canonical(c),
        "seed": seed,
        "arm": arm,
        "parent_checkpoint_sha256": c["parent"]["checkpoint_sha256"].get(str(seed), "synthetic"),
        "assets_sha256": c["engine"]["input"]["assets_manifest_sha256"],
        "confirmation": "SEALED",
    }
