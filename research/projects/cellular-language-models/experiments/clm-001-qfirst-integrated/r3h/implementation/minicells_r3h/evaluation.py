"""Selective historical assets; no Development mapping before complete training barrier."""

from pathlib import Path
import shutil
import time
import numpy as np
import torch
from minicells_r3f.data import ensure as prior_ensure
from minicells_r3e.common import digest, read, fingerprint
from minicells_r3c_moe_bridge.evaluation import evaluate
from minicells_r3d.evaluation import mean_nll, sync


def assets(c, backend, folder, roles):
    roles = tuple(roles)
    if not set(roles) <= {"train", "calibration", "admission", "development"}:
        raise RuntimeError("CONFIRMATION_SEALED")
    core = tuple(r for r in roles if r != "admission")
    result = prior_ensure(c, backend, folder, core)
    if "admission" not in roles:
        return result
    folder = Path(folder)
    ref = c["engine"]["input"]["asset_bundle_ref"]
    bundle = backend.get(ref["bundle_path"], ref["hf_revision"])
    if digest(bundle) != ref["bundle_sha256"]:
        raise RuntimeError("ASSET_BUNDLE_HASH_CHANGED")
    row = read(bundle)["files"]["assets/admission.npy"]
    dest = folder / "admission.npy"
    if not dest.exists():
        src = backend.get(row["object"], ref["hf_revision"])
        if digest(src) != row["sha256"]:
            raise RuntimeError("ADMISSION_OBJECT_HASH_CHANGED")
        shutil.copyfile(src, dest)
    if (
        dest.is_symlink()
        or digest(dest) != row["sha256"]
        or digest(dest) != result["manifest"]["files"]["admission.npy"]
    ):
        raise RuntimeError("ADMISSION_HASH_CHANGED")
    w = read(folder / "windows.json")["admission"]
    a = np.load(dest, mmap_mode="r")
    offsets = w["story_offsets"]
    n = c["engine"]["training"]["sequence_length"]
    if (
        a.ndim != 1
        or a.min() < 0
        or a.max() >= c["engine"]["model"]["vocab_size"]
        or offsets[0] != 0
        or offsets[-1] != len(a)
    ):
        raise RuntimeError("ADMISSION_EXTENTS_INVALID")
    for s, g in zip(w["starts"], w["groups"]):
        if not 0 <= g < len(offsets) - 1 or not offsets[g] <= s < s + n < offsets[g + 1]:
            raise RuntimeError("ADMISSION_STORY_BOUNDARY_INVALID")
    if len(w["starts"]) != len(w["groups"]) or len(w["evaluation"]["starts"]) != len(
        w["evaluation"]["groups"]
    ):
        raise RuntimeError("ADMISSION_WINDOWS_INVALID")
    pairs = set(zip(w["starts"], w["groups"]))
    if any(p not in pairs for p in zip(w["evaluation"]["starts"], w["evaluation"]["groups"])):
        raise RuntimeError("ADMISSION_EVALUATION_WINDOWS_INVALID")
    result.update(admission=a)
    result["windows"]["admission"] = w
    return result


def windows(data, c, role, device, limit=None, labels=True):
    if role not in ("calibration", "admission", "development") or role not in data:
        raise RuntimeError("UNAVAILABLE_OR_FORBIDDEN_ROLE")
    w = data["windows"][role]["evaluation"]
    n = c["engine"]["training"]["sequence_length"]
    for i, (start, group) in enumerate(zip(w["starts"], w["groups"])):
        if limit is not None and i >= limit:
            break
        x = torch.tensor(
            np.array(data[role][start : start + n], dtype="int64")[None], device=device
        )
        y = (
            torch.tensor(
                np.array(data[role][start + 1 : start + n + 1], dtype="int64")[None], device=device
            )
            if labels
            else None
        )
        yield str(group), x, y


@torch.no_grad()
def quality(m, c, data, role):
    if role not in ("calibration", "development"):
        raise RuntimeError("QUALITY_ROLE_FORBIDDEN")
    device = next(m.parameters()).device
    before = fingerprint(m.state_dict())
    m.training_context = None
    m.observer = None
    m.eval()
    sync(device)
    begin = time.perf_counter()
    rows = evaluate(m, windows(data, c, role, device), m.seed)
    sync(device)
    elapsed = time.perf_counter() - begin
    if fingerprint(m.state_dict()) != before:
        raise RuntimeError("EVALUATION_MUTATED_MODEL")
    tokens = sum(r["tokens"] for r in rows.values())
    return dict(
        nll=mean_nll(rows),
        stories=rows,
        role=role,
        seconds=elapsed,
        tokens=tokens,
        tokens_per_second=tokens / max(elapsed, 1e-12),
        throughput_scope="TEACHER_FORCED_DETERMINISTIC_FORWARD_NOT_AUTOREGRESSIVE_DECODE",
    )
