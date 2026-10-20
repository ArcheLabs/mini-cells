"""Read existing QFIRST assets; no new tokenizer, new split or Confirmation reader."""

from pathlib import Path
import numpy as np
import torch
from clm001_r3.data import load_assets as original_load, batches as original_batches
from minicells_runtime.store import digest, read, ContinuationError
from .common import ROOT


def legacy_configuration():
    return read(
        ROOT
        / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b/config/proposed.json"
    )["engine"]


def load_assets(c, folder):
    folder = Path(folder)
    old = legacy_configuration()
    if c["input"]["confirmation_materialized"] or c["release"]["confirmation"] != "SEALED":
        raise ContinuationError("R3C_CONFIRMATION_SEALED")
    if (
        digest(folder / "manifest.json") != c["input"]["assets_manifest_sha256"]
        or digest(folder / "tokenizer.json") != c["input"]["tokenizer_sha256"]
    ):
        raise ContinuationError("R3C_HISTORICAL_ASSET_IDENTITY_MISMATCH")
    if (
        c["input"]["data_config"] != old["data"]
        or c["training"]["sequence_length"] != old["training"]["sequence_length"]
        or c["model"]["vocab_size"] != old["model"]["vocab_size"]
    ):
        raise ContinuationError("R3C_DATA_CONTRACT_CHANGED")
    return original_load(old, folder)


def train_batches(data, c, rng, device):
    return original_batches(data, {"training": c["training"]}, rng, device)


def fixed_windows(data, c, role, device):
    if role not in ("calibration", "development"):
        raise ValueError("evaluation role must be Calibration or previously-used Development")
    meta = data["windows"][role]["evaluation"]
    n = c["training"]["sequence_length"]
    for start, story in zip(meta["starts"], meta["groups"]):
        x = torch.tensor(
            np.array(data[role][start : start + n], dtype="int64")[None], device=device
        )
        y = torch.tensor(
            np.array(data[role][start + 1 : start + n + 1], dtype="int64")[None], device=device
        )
        yield str(story), x, y


def synthetic_batches(c, rng, device="cpu"):
    length = c["training"]["sequence_length"]
    result = []
    for _ in range(c["training"]["effective_batch"] // c["training"]["microbatch"]):
        rows = []
        targets = []
        for _ in range(c["training"]["microbatch"]):
            family = int(rng.integers(0, 2))
            offset = int(rng.integers(0, 4))
            pattern = [2, 3, 4, 5] if family == 0 else [10, 9, 8, 7]
            sequence = [pattern[(offset + j) % 4] for j in range(length + 1)]
            rows.append(sequence[:-1])
            targets.append(sequence[1:])
        result.append((torch.tensor(rows, device=device), torch.tensor(targets, device=device)))
    return result


def private_asset_subset(backend, reference, destination, c):
    # Reuse the existing pinned HF bundle/HubBackend; download assets only, never historical weights.
    if not backend.info(reference["hf_revision"]).private:
        raise ContinuationError("HF_PRIVATE_REPO_REQUIRED")
    bundle = backend.get(reference["bundle_path"], reference["hf_revision"])
    if digest(bundle) != reference["bundle_sha256"]:
        raise ContinuationError("HF_READBACK_MISMATCH")
    metadata = read(bundle)
    if (
        metadata["identity"]["assets_manifest_sha256"] != c["input"]["assets_manifest_sha256"]
        or metadata["identity"]["tokenizer_sha256"] != c["input"]["tokenizer_sha256"]
    ):
        raise ContinuationError("R3C_HISTORICAL_ASSET_IDENTITY_MISMATCH")
    import shutil
    from minicells_artifacts.hf_storage import relative

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, row in metadata["files"].items():
        if not name.startswith("assets/"):
            continue
        relative(name)
        path = destination / Path(name).relative_to("assets")
        source = backend.get(row["object"], reference["hf_revision"])
        if digest(source) != row["sha256"]:
            raise ContinuationError("HF_READBACK_MISMATCH")
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ContinuationError("UNSAFE_PATH")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and digest(path) != row["sha256"]:
            raise ContinuationError("R3C_ASSET_COLLISION")
        if not path.exists():
            shutil.copyfile(source, path)
    return load_assets(c, destination)
