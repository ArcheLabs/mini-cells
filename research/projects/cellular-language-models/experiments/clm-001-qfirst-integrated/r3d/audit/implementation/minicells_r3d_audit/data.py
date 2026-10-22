"""Read only the two permitted evaluation arrays; reuse the frozen QFIRST contract."""

from pathlib import Path
import os
import shutil
import numpy as np
from clm001_r3.data import contract
from minicells_r3c_moe_bridge.data import legacy_configuration
from minicells_artifacts.hf_storage import relative
from .common import read, digest

FILES = {"manifest.json", "tokenizer.json", "windows.json", "calibration.npy", "development.npy"}


def load(original, folder):
    folder = Path(folder)
    engine = original["engine"]
    expected = engine["input"]
    old = legacy_configuration()
    if (
        digest(folder / "manifest.json") != expected["assets_manifest_sha256"]
        or digest(folder / "tokenizer.json") != expected["tokenizer_sha256"]
    ):
        raise RuntimeError("HISTORICAL_ASSET_OR_TOKENIZER_HASH_MISMATCH")
    manifest = read(folder / "manifest.json")
    if (
        manifest.get("confirmation_materialized") is not False
        or manifest["contract_sha256"] != contract(old)
        or manifest["experiment_id"] != old["experiment_id"]
    ):
        raise RuntimeError("FROZEN_DATA_CONTRACT_MISMATCH")
    if expected["confirmation_materialized"] or engine["release"]["confirmation"] != "SEALED":
        raise RuntimeError("CONFIRMATION_MUST_REMAIN_SEALED")
    for name in FILES - {"manifest.json"}:
        path = folder / name
        if path.is_symlink() or digest(path) != manifest["files"][name]:
            raise RuntimeError("CORRUPT_SELECTED_ASSET: " + name)
    windows = read(folder / "windows.json")
    data = {}
    seen_ids = set()
    seen_hashes = set()
    for role in ("calibration", "development"):
        array = np.load(folder / (role + ".npy"), mmap_mode="r")
        meta = windows[role]
        if (
            array.ndim != 1
            or not len(array)
            or array.min() < 0
            or array.max() >= engine["model"]["vocab_size"]
        ):
            raise RuntimeError("INVALID_SELECTED_TOKEN_ARRAY")
        ids = meta["raw_story_ids"]
        hashes = meta["story_hashes"]
        lo, hi = old["data"][role + "_range"]
        if (
            len(ids) != len(hashes)
            or len(ids) != len(set(ids))
            or any(not lo <= v < hi for v in ids)
        ):
            raise RuntimeError("INVALID_STORY_IDENTITIES")
        if seen_ids.intersection(ids) or seen_hashes.intersection(hashes):
            raise RuntimeError("CROSS_ROLE_LEAKAGE")
        seen_ids.update(ids)
        seen_hashes.update(hashes)
        offsets = meta["story_offsets"]
        n = engine["training"]["sequence_length"]
        if len(offsets) != len(ids) + 1 or offsets[0] != 0 or offsets[-1] != len(array):
            raise RuntimeError("STORY_EXTENTS_DIFFER")
        if len(meta["starts"]) != len(meta["groups"]):
            raise RuntimeError("WINDOW_GROUP_COUNT_DIFFERS")
        pairs = set(zip(meta["starts"], meta["groups"]))
        ev = meta["evaluation"]
        if len(ev["starts"]) != old["evaluation"][role + "_windows"] or len(ev["starts"]) != len(
            ev["groups"]
        ):
            raise RuntimeError("FROZEN_EVALUATION_WINDOW_COUNT_DIFFERS")
        for start, group in zip(ev["starts"], ev["groups"]):
            if (
                (start, group) not in pairs
                or not 0 <= group < len(ids)
                or not offsets[group] <= start < start + n < offsets[group + 1]
            ):
                raise RuntimeError("TARGET_CROSSES_STORY_BOUNDARY")
        data[role] = array
    data.update(
        windows={role: windows[role] for role in ("calibration", "development")}, manifest=manifest
    )
    return data


def restore(backend, reference, destination, original):
    bundle = backend.get(reference["bundle_path"], reference["hf_revision"])
    if digest(bundle) != reference["bundle_sha256"]:
        raise RuntimeError("ASSET_BUNDLE_HASH_MISMATCH")
    meta = read(bundle)
    expected = original["engine"]["input"]
    if (
        meta["identity"]["assets_manifest_sha256"] != expected["assets_manifest_sha256"]
        or meta["identity"]["tokenizer_sha256"] != expected["tokenizer_sha256"]
    ):
        raise RuntimeError("ASSET_BUNDLE_IDENTITY_MISMATCH")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for leaf in sorted(FILES):
        name = "assets/" + leaf
        row = meta["files"][name]
        relative(name)
        source = backend.get(row["object"], reference["hf_revision"])
        if digest(source) != row["sha256"]:
            raise RuntimeError("ASSET_READBACK_HASH_MISMATCH")
        path = destination / leaf
        if path.is_symlink() or path.exists() and digest(path) != row["sha256"]:
            raise RuntimeError("ASSET_CACHE_COLLISION")
        if not path.exists():
            try:
                os.link(source, path)
            except OSError:
                shutil.copyfile(source, path)
    return load(original, destination)
