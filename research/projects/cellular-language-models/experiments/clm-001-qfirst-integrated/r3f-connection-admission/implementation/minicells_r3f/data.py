"""Reuse QFIRST identity/contract; materialize only the roles permitted in the current phase."""

from pathlib import Path
import shutil
import numpy as np
from clm001_r3.data import contract, FILES
from minicells_r3c_moe_bridge.data import legacy_configuration
from minicells_artifacts.hf_storage import relative
from .common import read, digest


def load(c, folder, roles):
    if not set(roles) <= set(("train", "calibration", "development")):
        raise RuntimeError("FORBIDDEN_DATA_ROLE")
    folder = Path(folder)
    expected = c["engine"]["input"]
    old = legacy_configuration()
    if (
        digest(folder / "manifest.json") != expected["assets_manifest_sha256"]
        or digest(folder / "tokenizer.json") != expected["tokenizer_sha256"]
    ):
        raise RuntimeError("FROZEN_ASSET_IDENTITY_CHANGED")
    meta = read(folder / "manifest.json")
    windows = read(folder / "windows.json")
    if (
        meta.get("confirmation_materialized") is not False
        or meta["contract_sha256"] != contract(old)
        or meta["experiment_id"] != old["experiment_id"]
        or set(meta["files"]) != set(FILES)
    ):
        raise RuntimeError("FROZEN_ASSET_CONTRACT_CHANGED")
    for name in ["tokenizer.json", "windows.json"] + [r + ".npy" for r in roles]:
        if (folder / name).is_symlink() or digest(folder / name) != meta["files"][name]:
            raise RuntimeError("SELECTED_ASSET_HASH_MISMATCH")
    seen = set()
    seen_ids = set()
    for role in ("train", "calibration", "admission", "development"):
        w = windows[role]
        lo, hi = old["data"][role + "_range"]
        ids = w["raw_story_ids"]
        hashes = w["story_hashes"]
        if (
            len(ids) != len(hashes)
            or len(set(ids)) != len(ids)
            or any(not lo <= v < hi for v in ids)
            or seen.intersection(hashes)
            or seen_ids.intersection(ids)
        ):
            raise RuntimeError("STORY_IDENTITY_OR_DISJOINTNESS_FAILED")
        seen.update(hashes)
        seen_ids.update(ids)
    result = {"windows": {r: windows[r] for r in roles}, "manifest": meta}
    n = c["engine"]["training"]["sequence_length"]
    for role in roles:
        a = np.load(folder / (role + ".npy"), mmap_mode="r")
        w = windows[role]
        offsets = w["story_offsets"]
        if (
            a.ndim != 1
            or not len(a)
            or a.min() < 0
            or a.max() >= c["engine"]["model"]["vocab_size"]
        ):
            raise RuntimeError("INVALID_SELECTED_TOKENS")
        if (
            len(offsets) != len(w["story_hashes"]) + 1
            or offsets[0] != 0
            or offsets[-1] != len(a)
            or len(w["starts"]) != len(w["groups"])
        ):
            raise RuntimeError("INVALID_STORY_EXTENTS")
        for start, group in zip(w["starts"], w["groups"]):
            if (
                not 0 <= group < len(offsets) - 1
                or not offsets[group] <= start < start + n < offsets[group + 1]
            ):
                raise RuntimeError("TARGET_CROSSES_STORY")
        if role != "train":
            ev = w["evaluation"]
            pairs = set(zip(w["starts"], w["groups"]))
            if (
                len(ev["starts"]) != old["evaluation"][role + "_windows"]
                or len(ev["starts"]) != len(ev["groups"])
                or any(pair not in pairs for pair in zip(ev["starts"], ev["groups"]))
            ):
                raise RuntimeError("FROZEN_EVALUATION_WINDOWS_CHANGED")
        result[role] = a
    return result


def ensure(c, backend, folder, roles):
    folder = Path(folder)
    names = ["manifest.json", "tokenizer.json", "windows.json"] + [r + ".npy" for r in roles]
    if not all((folder / name).exists() for name in names):
        ref = c["engine"]["input"]["asset_bundle_ref"]
        path = backend.get(ref["bundle_path"], ref["hf_revision"])
        if digest(path) != ref["bundle_sha256"]:
            raise RuntimeError("FROZEN_BUNDLE_HASH_MISMATCH")
        bundle = read(path)
        if (
            bundle["identity"]["assets_manifest_sha256"]
            != c["engine"]["input"]["assets_manifest_sha256"]
            or bundle["identity"]["tokenizer_sha256"] != c["engine"]["input"]["tokenizer_sha256"]
        ):
            raise RuntimeError("FROZEN_BUNDLE_IDENTITY_CHANGED")
        folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            row = bundle["files"]["assets/" + relative(name)]
            source = backend.get(row["object"], ref["hf_revision"])
            if digest(source) != row["sha256"]:
                raise RuntimeError("ASSET_READBACK_MISMATCH")
            target = folder / name
            if any(p.is_symlink() for p in (target, *target.parents)):
                raise RuntimeError("UNSAFE_PATH")
            if target.exists() and digest(target) != row["sha256"]:
                raise RuntimeError("ASSET_CACHE_COLLISION")
            if not target.exists():
                shutil.copyfile(source, target)
    return load(c, folder, roles)
