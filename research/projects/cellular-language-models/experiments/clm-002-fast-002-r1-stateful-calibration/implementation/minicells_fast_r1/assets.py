"""Role-separated generation. Runtime diagnostic never opens any Confirm array."""

import hashlib
import unicodedata
from pathlib import Path

import numpy as np
import torch
from minicells_fast001 import data as frozen
from minicells_fast001.common import BASE as OLD_BASE
from minicells_fast001.common import configuration
from tokenizers import Tokenizer

from .common import BASE, ROOT, atomic, digest, read, sha


def produce(c, role, count, folder, namespace):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if digest(ROOT / c["tokenizer"]["path"]) != c["tokenizer"]["sha256"]:
        raise RuntimeError("R1_ORIGINAL_TOKENIZER_CHANGED")
    tokenizer = Tokenizer.from_file(str(ROOT / c["tokenizer"]["path"]))
    seen = {key: set() for key in ("raw", "normalized", "window", "input", "target")}
    files = {}
    for task in ("Story", "Arithmetic"):
        rows, windows = [], []
        for index in range(count):
            for attempt in range(c["data"]["duplicate_attempt_cap"]):
                text, _facts, rng = frozen.record(c, task, role, index, attempt)
                tokens = tokenizer.encode(text).ids + [tokenizer.token_to_id("<eos>")]
                offset = rng.integer(len(tokens) - c["sequence_length"])
                window = tokens[offset : offset + c["sequence_length"] + 1]
                hashes = {
                    "raw": hashlib.sha256(text.encode()).hexdigest(),
                    "normalized": hashlib.sha256(
                        " ".join(unicodedata.normalize("NFKC", text).casefold().split()).encode()
                    ).hexdigest(),
                    "window": sha(window),
                    "input": sha(window[:-1]),
                    "target": sha(window[1:]),
                }
                if all(value not in seen[key] for key, value in hashes.items()):
                    break
            else:
                raise RuntimeError("R1_DEDUPLICATION_FAILED_NO_RESELECTION")
            for key, value in hashes.items():
                seen[key].add(value)
            rows.append(
                {
                    "sample_id": sha([namespace, task, role, index]),
                    "hashes": hashes,
                    "offset": offset,
                    "attempt": attempt,
                }
            )
            windows.append(window)
        name = task + "-" + role
        np.save(folder / (name + ".npy"), np.asarray(windows, dtype=np.int32))
        atomic(folder / (name + ".json"), rows, compact=True)
        for suffix in (".json", ".npy"):
            path = folder / (name + suffix)
            files[path.name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return {
        "role": role,
        "files": files,
        "records_per_task": count,
        "generator_sha256": digest(Path(__file__)),
        "frozen_record_generator_sha256": digest(Path(frozen.__file__)),
        "tokenizer_sha256": c["tokenizer"]["sha256"],
        "no_model_quality_used": True,
    }


def admission_config(c):
    working = configuration()
    working["data"]["generation_seeds"]["admission"] = c["admission"]["generation_seed"]
    working["data"]["numeric_ranges"]["admission"] = c["admission"]["range"]
    working["data"]["names"]["admission"] = c["admission"]["names"]
    return working


def admission(c, folder):
    expected = read(BASE / "registration/ADMISSION_MANIFEST.json")
    if (folder / "ADMISSION_MANIFEST.json").exists():
        actual = read(folder / "ADMISSION_MANIFEST.json")
    else:
        actual = produce(
            admission_config(c),
            "admission",
            c["admission"]["records_per_task"],
            folder,
            c["admission"]["namespace"],
        )
        atomic(folder / "ADMISSION_MANIFEST.json", actual)
    if actual != expected:
        raise RuntimeError("R1_ADMISSION_MANIFEST_CHANGED")
    for name, row in actual["files"].items():
        if digest(folder / name) != row["sha256"]:
            raise RuntimeError("R1_ADMISSION_FILE_CHANGED")
    return actual


def training(folder):
    old = configuration()
    registered = read(OLD_BASE / "registration/DATASET_MANIFEST.json")
    if not (folder / "DATASET_MANIFEST.json").exists():
        produce(
            old,
            "train",
            old["data"]["train_windows"],
            folder,
            "FAST001_ORIGINAL_TRAIN_RECONSTRUCTION",
        )
        # Preserve the historical stream pool identity; only Train bytes are materialized.
        atomic(folder / "DATASET_MANIFEST.json", registered)
    if read(folder / "DATASET_MANIFEST.json") != registered:
        raise RuntimeError("R1_ORIGINAL_TRAIN_POOL_IDENTITY_CHANGED")
    for task in ("Story", "Arithmetic"):
        name = task + "-train.npy"
        if digest(folder / name) != registered["files"][name]["sha256"]:
            raise RuntimeError("R1_ORIGINAL_TRAIN_ARRAY_CHANGED")
    return old


def refs(c, folder, device):
    manifest = admission(c, folder)
    inputs = []
    for task in ("Story", "Arithmetic"):
        values = np.load(folder / (task + "-admission.npy"), mmap_mode="r")
        for row in values:
            inputs.append(
                torch.tensor(np.array(row[:-1], copy=True), dtype=torch.long, device=device)[None]
            )
    return {"role": "admission", "manifest_sha256": sha(manifest), "inputs": inputs}
