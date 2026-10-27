"""Verify immutable historical Output bytes before loading any model; never train replacements."""

from pathlib import Path

import torch
from minicells_r3e.common import fingerprint

from .common import digest, read


def locate(c, reports, roots):
    expected = {
        f"{seed}-{arm}": read(reports / "jobs" / f"{seed}-{arm}.json")
        for seed in c["seeds"]
        for arm in c["arms"]
        if arm.startswith("A-") or arm[:3] in ("B00", "B10")
    }
    candidates = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.glob("**/latest.pt"):
            key = path.parent.name
            if key not in expected or key in candidates or path.is_symlink():
                continue
            if digest(path) == expected[key]["checkpoint_sha256"]:
                candidates[key] = path
    missing = sorted(set(expected) - set(candidates))
    if missing:
        raise RuntimeError(
            "FAST002_ATTACH_ORIGINAL_FAST001_SAVE_VERSION_OUTPUT_MISSING:" + ",".join(missing)
        )
    return candidates


def load(path, result):
    if digest(path) != result["checkpoint_sha256"]:
        raise RuntimeError("FAST002_HISTORICAL_CHECKPOINT_HASH_CHANGED")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state"]
    if (
        payload["identity"] != result["identity"]
        or state["step"] != 500
        or payload["full_state_sha256"] != fingerprint((state, payload["stream"]))
    ):
        raise RuntimeError("FAST002_HISTORICAL_CHECKPOINT_IDENTITY_OR_STATE_CHANGED")
    return state
