"""Admission is routing-only. Calibration is quality-only; never alias the splits."""

from pathlib import Path
from itertools import islice
import shutil
import numpy as np
import torch
from minicells_r3f.data import ensure as existing_ensure, load as existing_load
from minicells_r3e.evaluation import calibration
from .common import digest, read


def data(c, backend, folder):
    folder = Path(folder)
    result = existing_ensure(c, backend, folder, ("train", "calibration"))
    path = folder / "admission.npy"
    if not path.exists():
        ref = c["engine"]["input"]["asset_bundle_ref"]
        bundle_path = backend.get(ref["bundle_path"], ref["hf_revision"])
        if digest(bundle_path) != ref["bundle_sha256"]:
            raise RuntimeError("ASSET_BUNDLE_HASH_MISMATCH")
        bundle = read(bundle_path)
        row = bundle["files"]["assets/admission.npy"]
        source = backend.get(row["object"], ref["hf_revision"])
        if digest(source) != row["sha256"]:
            raise RuntimeError("ADMISSION_HASH_MISMATCH")
        shutil.copyfile(source, path)
    return load(c, folder, result)


def load(c, folder, result=None):
    folder = Path(folder)
    result = existing_load(c, folder, ("train", "calibration")) if result is None else result
    meta = read(folder / "manifest.json")
    windows = read(folder / "windows.json")
    path = folder / "admission.npy"
    if path.is_symlink() or digest(path) != meta["files"]["admission.npy"]:
        raise RuntimeError("ADMISSION_HASH_MISMATCH")
    a = np.load(path, mmap_mode="r")
    w = windows["admission"]
    n = c["engine"]["training"]["sequence_length"]
    if a.ndim != 1 or not len(a) or a.min() < 0 or a.max() >= c["engine"]["model"]["vocab_size"]:
        raise RuntimeError("INVALID_ADMISSION_TOKENS")
    offsets = w["story_offsets"]
    if offsets[0] != 0 or offsets[-1] != len(a):
        raise RuntimeError("INVALID_ADMISSION_EXTENTS")
    starts = w["evaluation"]["starts"]
    groups = w["evaluation"]["groups"]
    if len(starts) != 32 or len(groups) != len(starts):
        raise RuntimeError("ADMISSION_WINDOWS_CHANGED")
    for start, group in zip(starts, groups):
        if (
            not 0 <= group < len(offsets) - 1
            or not offsets[group] <= start < start + n < offsets[group + 1]
        ):
            raise RuntimeError("ADMISSION_WINDOW_CROSSES_STORY")
    result["admission"] = a
    result["windows"]["admission"] = w
    return result


def admission_windows(model, c, data):
    role = data["windows"]["admission"]["evaluation"]
    n = c["engine"]["training"]["sequence_length"]
    device = next(model.parameters()).device
    for start in islice(role["starts"], c["proposal"]["reference_admission_windows"]):
        yield torch.tensor(
            np.array(data["admission"][start : start + n], dtype=np.int64)[None], device=device
        )


def quality(model, c, data):
    return calibration(model, c, data)
