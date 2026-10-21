from pathlib import Path
from minicells_r3d.common import ROOT, identity as training_identity
from minicells_r3c_moe_bridge.common import (
    atomic as atomic,
    canonical,
    digest,
    read,
    fingerprint as fingerprint,
    ContinuationError as ContinuationError,
)

BASE = Path(__file__).resolve().parents[2]
ID = "CLM-001-R3d-AUDIT-001"


def configuration(path=None):
    c = read(path or BASE / "config/frozen.json")
    if (
        c["experiment_id"] != ID
        or c["confirmation"] != "SEALED"
        or c["runtime"]["training_updates"]
        or c["runtime"]["optimizer_instances"]
    ):
        raise RuntimeError("READ_ONLY_AUDIT_CONTRACT_REQUIRED")
    if canonical(c) != read(BASE / "config/releases.lock")["audit_config_sha256"]:
        raise RuntimeError("FROZEN_AUDIT_CONFIGURATION_CHANGED")
    for name, sha in c["source_file_sha256"].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError("TRAINING_SOURCE_BYTES_DIFFER: " + name)
    if digest(ROOT / c["historical_results"]["git_path"]) != c["historical_results"]["sha256"]:
        raise RuntimeError("HISTORICAL_RESULT_HASH_MISMATCH")
    original = read(ROOT / c["training_config_path"])
    if canonical(original) != c["training_config_sha256"]:
        raise RuntimeError("TRAINING_CONFIG_IDENTITY_MISMATCH")
    historical = read(ROOT / c["historical_results"]["git_path"])
    if historical["source_sha"] != c["training_source_sha"] or historical[
        "config_sha256"
    ] != canonical(original):
        raise RuntimeError("ACTUAL_TRAINING_IDENTITY_MISMATCH")
    return c, original, historical


def run_identity(c, original):
    # Exact recorded R3d automatic() derivation, not a guessed run_id.
    source = c["training_source_sha"]
    cfg = canonical(original)
    return {
        "experiment_id": original["experiment_id"],
        "source_sha": source,
        "config_sha256": cfg,
        "run_id": canonical({"source": source, "config": cfg}),
    }


def checkpoint_identity(c, original, seed, arm):
    return training_identity(original, c["training_source_sha"], seed, arm)


def failure_code(error):
    import re

    code = getattr(error, "code", None)
    if code:
        return code
    value = str(error).split(":", 1)[0]
    return value if re.fullmatch(r"[A-Z][A-Z0-9_]{0,110}", value) else type(error).__name__
