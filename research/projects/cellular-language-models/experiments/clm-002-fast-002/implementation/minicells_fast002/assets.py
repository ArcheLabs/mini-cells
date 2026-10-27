"""Reproduce old training bytes; open new Confirm only after train-only calibration passes."""

from minicells_fast001 import data
from minicells_fast001.common import BASE as OLD_BASE
from minicells_fast001.common import configuration

from .common import BASE, atomic, digest, read, sha


def training(folder):
    c = configuration()
    registered = read(OLD_BASE / "registration/DATASET_MANIFEST.json")
    data.generate(c, folder)
    for task in ("Story", "Arithmetic"):
        name = task + "-train.npy"
        if digest(folder / name) != registered["files"][name]["sha256"]:
            raise RuntimeError("FAST002_ORIGINAL_TRAINING_BYTES_CHANGED")
    return c


def fresh(c, folder, old_folder):
    manifest = data.generate(c, folder)
    if sha(manifest) != sha(read(BASE / "registration/FRESH_DATASET_MANIFEST.json")):
        raise RuntimeError("FAST002_FRESH_DATASET_NOT_REGISTERED_BYTES")
    for task in ("Story", "Arithmetic"):
        if digest(folder / (task + "-train.npy")) != digest(old_folder / (task + "-train.npy")):
            raise RuntimeError("FAST002_TRAINING_POOL_CHANGED")
    # Whole-record/normalized-text/exact 65-token window exclusion. Shared templates remain intentional.
    fields = ("raw_text_sha256", "normalized_text_sha256", "window_sha256")
    historical = {k: set() for k in fields}
    for task in ("Story", "Arithmetic"):
        for role in ("train", "confirm"):
            for row in read(old_folder / f"{task}-{role}.json"):
                for key in fields:
                    historical[key].add(row[key])
    for task in ("Story", "Arithmetic"):
        for row in read(folder / (task + "-confirm.json")):
            for key in fields:
                if row[key] in historical[key]:
                    raise RuntimeError(
                        "FAST002_FRESH_CONFIRM_EXACT_CONTENT_CONFLICT_NO_RESELECTION"
                    )
    proof = {
        "status": "FRESH_CONFIRM_EXACT_CONTENT_AND_WINDOW_PASS",
        "dataset_sha256": sha(manifest),
        "original_train_unchanged": True,
        "historical_exact_overlaps": 0,
        "selection_used_model_quality": False,
        "partial_template_overlap_excluded": False,
        "scope": "CONTROLLED_GENERATOR_NEW_CONFIRM_NOT_EXTERNAL_INDEPENDENT_REPLICATION",
    }
    atomic(folder / "FRESH_CONFIRM_PROOF.json", proof)
    return manifest, proof
