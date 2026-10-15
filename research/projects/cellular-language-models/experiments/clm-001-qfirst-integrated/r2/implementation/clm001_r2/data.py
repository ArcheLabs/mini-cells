"""Reuse FAST dedup/BPE and story-contained windows with four independent QFIRST roles."""

import itertools
import json
from pathlib import Path
import numpy as np
from fast001.data import Deduplicator, grouped_windows, SHARD
from exp006.data import encode_stories
from .common import validate, write_json, canonical_sha256, file_sha256

ROLES = ("train", "calibration", "admission", "development")
FILES = tuple(role + ".npy" for role in ROLES) + ("windows.json", "tokenizer.json")


def contract(c):
    return canonical_sha256(
        {
            "id": c["experiment_id"],
            "data": c["data"],
            "length": c["training"]["sequence_length"],
            "vocab": c["model"]["vocab_size"],
            "evaluation": c["evaluation"],
        }
    )


def select(rows, c):
    validate(c)
    high = max(c["data"][role + "_range"][1] for role in ROLES)
    seen = Deduplicator()
    selected = {role: [] for role in ROLES}
    ids = {role: [] for role in ROLES}
    removed = {role: 0 for role in ROLES}
    scanned = 0
    for index, row in enumerate(itertools.islice(iter(rows), high)):
        scanned = index + 1
        for role in ROLES:
            lo, hi = c["data"][role + "_range"]
            if lo <= index < hi:
                if seen.accept(row["text"]):
                    selected[role].append(row["text"])
                    ids[role].append(index)
                else:
                    removed[role] += 1
    if scanned != high:
        raise ValueError("source shorter than specified ranges")
    return selected, {
        "raw_story_ids": ids,
        "duplicates_removed": removed,
        "raw_rows_scanned": scanned,
        "near_duplicate_rule": "normalized word5gram SimHash64 Hamming<=3, global across selected roles",
        "confirmation_materialized": False,
    }


def build(c, selected, audit, destination, source_hash, tokenizer=None):
    validate(c)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("QFIRST assets are immutable")
    if set(selected) != set(ROLES):
        raise ValueError("exact four roles required")
    check = Deduplicator()
    for role in ROLES:
        for text in selected[role]:
            if not check.accept(text):
                raise ValueError("cross-role exact/near leakage")
    if tokenizer is None:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(
            selected["train"],
            trainers.BpeTrainer(
                vocab_size=c["model"]["vocab_size"],
                special_tokens=["<unk>", "<eos>"],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            ),
        )
    if tokenizer.get_vocab_size() > c["model"]["vocab_size"]:
        raise ValueError("tokenizer exceeds model vocabulary")
    metadata = {}
    arrays = {}
    for role, texts in selected.items():
        tokens, starts, groups, hashes = encode_stories(
            texts, tokenizer, c["training"]["sequence_length"]
        )
        if not starts:
            raise ValueError("no complete windows in " + role)
        lengths = [len(tokenizer.encode(text).ids) + 1 for text in texts]
        offsets = np.cumsum([0] + lengths).tolist()
        metadata[role] = {
            "starts": starts,
            "groups": groups,
            "story_offsets": offsets,
            "story_hashes": hashes,
            "raw_story_ids": audit["raw_story_ids"][role],
        }
        if role != "train":
            metadata[role]["evaluation"] = grouped_windows(
                starts, groups, c["evaluation"][role + "_windows"], c["evaluation"]["seed"]
            )
        arrays[role] = tokens
    destination.mkdir(parents=True)
    tokenizer.save(str(destination / "tokenizer.json"))
    for role, tokens in arrays.items():
        np.save(destination / (role + ".npy"), tokens)
    write_json(destination / "windows.json", metadata)
    manifest = {
        "experiment_id": c["experiment_id"],
        "contract_sha256": contract(c),
        "source_sha256": source_hash,
        "dataset": c["data"]["dataset"],
        "revision": c["data"]["revision"],
        "split": "train",
        "confirmation_materialized": False,
        "tokenizer_training": "train only",
        "tokenizer_actual_vocab": tokenizer.get_vocab_size(),
        "selection_audit": audit,
        "files": {name: file_sha256(destination / name) for name in FILES},
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def prepare(c, destination):
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset

    validate(c)
    source = hf_hub_download(
        repo_id=c["data"]["dataset"],
        repo_type="dataset",
        revision=c["data"]["revision"],
        filename=SHARD,
    )
    rows = load_dataset("parquet", data_files=source, split="train", streaming=True)
    selected, audit = select(rows, c)
    return build(c, selected, audit, destination, file_sha256(source))


def load_assets(c, folder):
    validate(c)
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        manifest.get("experiment_id") != c["experiment_id"]
        or manifest.get("contract_sha256") != contract(c)
        or manifest.get("confirmation_materialized") is not False
    ):
        raise ValueError("QFIRST-only asset identity required")
    if set(manifest["files"]) != set(FILES):
        raise ValueError("asset file inventory differs")
    for name, digest in manifest["files"].items():
        if (folder / name).is_symlink() or file_sha256(folder / name) != digest:
            raise ValueError("corrupt asset " + name)
    windows = json.loads((folder / "windows.json").read_text())
    data = {}
    all_hashes = set()
    all_ids = set()
    for role in ROLES:
        a = np.load(folder / (role + ".npy"), mmap_mode="r")
        meta = windows[role]
        if a.ndim != 1 or not len(a) or a.min() < 0 or a.max() >= c["model"]["vocab_size"]:
            raise ValueError("invalid tokens")
        hashes = meta["story_hashes"]
        ids = meta["raw_story_ids"]
        lo, hi = c["data"][role + "_range"]
        if (
            len(ids) != len(hashes)
            or len(set(ids)) != len(ids)
            or any(not lo <= v < hi for v in ids)
        ):
            raise ValueError("invalid raw story identity")
        if all_hashes.intersection(hashes) or all_ids.intersection(ids):
            raise ValueError("cross-role leakage")
        all_hashes.update(hashes)
        all_ids.update(ids)
        offsets = meta["story_offsets"]
        if len(offsets) != len(hashes) + 1 or offsets[0] != 0 or offsets[-1] != len(a):
            raise ValueError("story extents differ")
        if len(meta["starts"]) != len(meta["groups"]):
            raise ValueError("window/group count differs")
        for start, group in zip(meta["starts"], meta["groups"]):
            if (
                not 0 <= group < len(hashes)
                or not offsets[group]
                <= start
                < start + c["training"]["sequence_length"]
                < offsets[group + 1]
            ):
                raise ValueError("shifted target crosses story boundary")
        if role != "train":
            pairs = set(zip(meta["starts"], meta["groups"]))
            ev = meta["evaluation"]
            if len(ev["starts"]) != c["evaluation"][role + "_windows"] or any(
                v not in pairs for v in zip(ev["starts"], ev["groups"])
            ):
                raise ValueError("invalid evaluation windows")
        data[role] = a
    data.update(
        windows=windows,
        manifest=manifest,
        asset_sha256=file_sha256(folder / "manifest.json"),
        tokenizer_sha256=manifest["files"]["tokenizer.json"],
    )
    return data


def batches(data, c, rng, device):
    import torch

    meta = data["windows"]["train"]
    starts = meta["starts"]
    length = c["training"]["sequence_length"]
    result = []
    for _ in range(c["training"]["effective_batch"] // c["training"]["microbatch"]):
        chosen = [starts[int(v)] for v in rng.integers(0, len(starts), c["training"]["microbatch"])]
        x = torch.tensor(
            np.stack([data["train"][s : s + length] for s in chosen]).astype("int64"), device=device
        )
        y = torch.tensor(
            np.stack([data["train"][s + 1 : s + length + 1] for s in chosen]).astype("int64"),
            device=device,
        )
        result.append((x, y))
    return result
