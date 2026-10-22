from pathlib import Path
from minicells_r3e.common import ROOT, atomic as atomic, source_sha as source_sha
from minicells_r3c_moe_bridge.common import read, canonical, digest, fingerprint as fingerprint

BASE = Path(__file__).resolve().parents[2]
ID = "r3f-connection-admission"


def configuration(path=None):
    c = read(path or BASE / "config/frozen.json")
    if c["experiment_id"] != ID or c["confirmation"] != "SEALED":
        raise RuntimeError("R3F_IDENTITY_OR_CONFIRMATION_CHANGED")
    if not c.get("synthetic"):
        if canonical(c) != read(BASE / "config/releases.lock")["config_sha256"]:
            raise RuntimeError("R3F_CONFIG_CHANGED")
        for f, sha in c["source_file_sha256"].items():
            if digest(ROOT / f) != sha:
                raise RuntimeError("FROZEN_SOURCE_CHANGED")
        original = read(ROOT / c["origin_r3e"]["config_path"])
        if canonical(original) != c["origin_r3e"]["config_sha256"]:
            raise RuntimeError("R3E_ORIGINAL_CONFIG_CHANGED")
    back = c["proposal"]["backtrack"]
    if any(a <= b for a, b in zip(back, back[1:])) or back[0] != 1 or back[-1] > 1 / 1024:
        raise RuntimeError("BACKTRACK_RESOLUTION_CHANGED")
    if not c.get("synthetic") and any(
        c["proposal"][k] != v
        for k, v in [("kl_max", 0.01), ("action_change_max", 0.1), ("path_change_max", 0.2)]
    ):
        raise RuntimeError("TRUST_THRESHOLDS_NOT_RELAXABLE")
    return c


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
