from pathlib import Path
import json
import os
import tempfile
from minicells_r3d.common import ROOT, source_sha as source_sha
from minicells_r3c_moe_bridge.common import (
    canonical,
    digest,
    read,
    fingerprint as fingerprint,
    ContinuationError,
)

BASE = Path(__file__).resolve().parents[2]
ID = "CLM-001-R3e-SCAGL-001"
ARMS = ("A-F", "A-T", "A-C", "B-F", "B-R", "B-C")


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as f:
            tmp = Path(f.name)
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp:
            tmp.unlink(missing_ok=True)


def configuration(path=None):
    c = read(path or BASE / "config/frozen.json")
    if c["experiment_id"] != ID or c["confirmation"] != "SEALED" or tuple(c["arms"]) != ARMS:
        raise ContinuationError("R3E_IDENTITY_MISMATCH")
    if not c.get("synthetic"):
        if canonical(c) != read(BASE / "config/releases.lock")["config_sha256"]:
            raise ContinuationError("R3E_CONFIG_CHANGED")
        if canonical(c["engine"]) != c["parent"]["config_sha256"]:
            raise ContinuationError("R3E_PARENT_CONFIG_CHANGED")
        for name, sha in c["source_file_sha256"].items():
            if digest(ROOT / name) != sha:
                raise ContinuationError("FROZEN_SOURCE_CHANGED", name)
    if c["engine"]["input"]["confirmation_materialized"]:
        raise ContinuationError("CONFIRMATION_SEALED")
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


def seed_all(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
