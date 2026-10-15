"""FAST-only BPE, raw-row split guard, deterministic exact/near dedup and windows."""

import hashlib
import itertools
import json
from pathlib import Path
import re
import numpy as np
from exp006.data import encode_stories
from exp006.reuse import canonical_sha256, file_sha256
from .io import write_json

SHARD = "data/train-00000-of-00004-2d5a1467fff1081b.parquet"
FILES = ("train.npy", "development.npy", "tokenizer.json", "windows.json")


def normalized(text):
    return re.findall(r"\w+", text.casefold())


def signature(words):
    grams = {" ".join(words[i : i + 5]) for i in range(max(1, len(words) - 4))}
    sums = [0] * 64
    for gram in grams:
        bits = int.from_bytes(hashlib.sha256(gram.encode()).digest()[:8], "little")
        for bit in range(64):
            sums[bit] += 1 if bits >> bit & 1 else -1
    return sum((v >= 0) << i for i, v in enumerate(sums))


class Deduplicator:
    def __init__(self):
        self.exact = set()
        self.bands = {}
        self.kept = 0

    def accept(self, text):
        words = normalized(text)
        if not words:
            return False
        exact = hashlib.sha256(" ".join(words).encode()).hexdigest()
        sig = signature(words)
        candidates = set()
        for band in range(4):
            candidates.update(self.bands.get((band, sig >> (band * 16) & 65535), ()))
        if exact in self.exact or any(bin(sig ^ old).count("1") <= 3 for old in candidates):
            return False
        self.exact.add(exact)
        for band in range(4):
            self.bands.setdefault((band, sig >> (band * 16) & 65535), set()).add(sig)
        self.kept += 1
        return True


def select_rows(rows, c):
    d = c["data"]
    high = d["development_range"][1]
    if high > min(60000, d["reserved_train_boundary"]):
        raise ValueError("reserved rows blocked")
    seen = Deduplicator()
    selected = {"train": [], "development": []}
    ids = {k: [] for k in selected}
    removed = {k: 0 for k in selected}
    scanned = 0
    # islice never fetches even one row at/after high, unlike enumerate then break.
    for index, row in enumerate(itertools.islice(iter(rows), high)):
        scanned = index + 1
        for role, key in (("train", "train_range"), ("development", "development_range")):
            low, upper = d[key]
            if low <= index < upper:
                text = row["text"]
                if seen.accept(text):
                    selected[role].append(text)
                    ids[role].append(index)
                else:
                    removed[role] += 1
    if scanned < high:
        raise ValueError("source shorter than declared raw range")
    return selected, {
        "raw_story_ids": ids,
        "duplicates_removed": removed,
        "raw_rows_scanned": scanned,
        "near_duplicate_rule": d["near_duplicate"],
        "scope": "selected training and development stories only",
    }


def grouped_windows(starts, groups, count, seed):
    if len(starts) < count:
        raise ValueError("insufficient story-contained development windows")
    rng = np.random.default_rng(seed)
    buckets = {}
    for start, group in zip(starts, groups):
        buckets.setdefault(int(group), []).append(int(start))
    keys = list(buckets)
    rng.shuffle(keys)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    level = 0
    while len(selected) < count:
        for group in keys:
            if level < len(buckets[group]):
                selected.append((buckets[group][level], group))
            if len(selected) == count:
                break
        level += 1
    return {"starts": [r[0] for r in selected], "groups": [r[1] for r in selected]}


def contract(c):
    return canonical_sha256(
        {
            "data": c["data"],
            "vocab_size": c["model"]["vocab_size"],
            "length": c["training"]["sequence_length"],
            "evaluation": c["evaluation"],
        }
    )


def build(c, selected, audit, destination, source_hash, *, tokenizer=None):
    if tokenizer is None:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(
            selected["train"][: c["data"]["tokenizer_stories"]],
            trainers.BpeTrainer(
                vocab_size=c["model"]["vocab_size"],
                special_tokens=["<unk>", "<eos>"],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            ),
        )
    vocab = tokenizer.get_vocab_size()
    if vocab > c["model"]["vocab_size"] or vocab < 2:
        raise ValueError("tokenizer vocabulary incompatible")
    # All written stories are rechecked; callers cannot bypass cross-split near isolation.
    check = Deduplicator()
    for role in ("train", "development"):
        for text in selected[role]:
            if not check.accept(text):
                raise ValueError("exact/near story leakage")
    output = Path(destination)
    if output.exists():
        raise FileExistsError("FAST assets are immutable")
    metadata = {}
    arrays = {}
    for role, texts in selected.items():
        tokens, starts, groups, hashes = encode_stories(texts, tokenizer, c["training"]["sequence_length"])
        if not starts:
            raise ValueError("no complete story-contained windows")
        # Story extents let loaders validate shifted targets remain inside each story.
        lengths = [len(tokenizer.encode(t).ids) + 1 for t in texts]
        offsets = np.cumsum([0] + lengths).tolist()
        metadata[role] = {
            "starts": starts,
            "groups": groups,
            "story_hashes": hashes,
            "story_offsets": offsets,
            "raw_story_ids": audit["raw_story_ids"][role],
        }
        arrays[role] = tokens
    evaluation = grouped_windows(
        metadata["development"]["starts"],
        metadata["development"]["groups"],
        c["evaluation"]["final_windows"],
        c["evaluation"]["window_seed"],
    )
    metadata["evaluation"] = evaluation
    output.mkdir(parents=True)
    tokenizer.save(str(output / "tokenizer.json"))
    for role, tokens in arrays.items():
        np.save(output / (role + ".npy"), tokens)
    write_json(output / "windows.json", metadata)
    manifest = {
        "experiment_id": c["experiment_id"],
        "contract_sha256": contract(c),
        "dataset": c["data"]["dataset"],
        "revision": c["data"]["revision"],
        "split": "train",
        "source_sha256": source_hash,
        "tokenizer_training": "train only",
        "tokenizer_actual_vocab": vocab,
        "model_vocab_size": c["model"]["vocab_size"],
        "sequence_length": c["training"]["sequence_length"],
        "confirmation_materialized": False,
        "selection_audit": audit,
        "evaluation_sha256": canonical_sha256(evaluation),
        "files": {n: file_sha256(output / n) for n in FILES},
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def prepare(c, destination, source_file=None):
    from .config import validate

    validate(c)
    if source_file is not None and c["data"].get("source_sha256") is None:
        raise ValueError(
            "explicit local source requires its frozen SHA256; default pinned Hub download records it automatically"
        )
    if source_file is None:
        from huggingface_hub import hf_hub_download

        source_file = hf_hub_download(
            repo_id=c["data"]["dataset"], repo_type="dataset", revision=c["data"]["revision"], filename=SHARD
        )
    digest = file_sha256(source_file)
    if c["data"].get("source_sha256") not in (None, digest):
        raise ValueError("source digest mismatch")
    from datasets import load_dataset

    rows = load_dataset("parquet", data_files=str(source_file), split="train", streaming=True)
    selected, audit = select_rows(rows, c)
    return build(c, selected, audit, destination, digest)


def load_assets(c, folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        manifest.get("experiment_id") != c["experiment_id"]
        or manifest.get("confirmation_materialized") is not False
        or manifest.get("contract_sha256") != contract(c)
        or set(manifest["files"]) != set(FILES)
    ):
        raise ValueError("FAST-only asset contract required; historical/confirmation bundles forbidden")
    for name, digest in manifest["files"].items():
        if (folder / name).is_symlink() or file_sha256(folder / name) != digest:
            raise ValueError("corrupt asset " + name)
    windows = json.loads((folder / "windows.json").read_text())
    data = {}
    for role in ("train", "development"):
        data[role] = np.load(folder / (role + ".npy"), mmap_mode="r")
        if (
            not len(data[role])
            or int(data[role].min()) < 0
            or int(data[role].max()) >= c["model"]["vocab_size"]
        ):
            raise ValueError("asset token IDs outside frozen vocabulary")
        meta = windows[role]
        if meta["story_offsets"][-1] != len(data[role]) or len(meta["story_hashes"]) != len(
            meta["raw_story_ids"]
        ):
            raise ValueError("story extent or raw identity mismatch")
        if len(meta["starts"]) != len(meta["groups"]):
            raise ValueError("window groups mismatch")
        for start, group in zip(meta["starts"], meta["groups"]):
            if (
                not meta["story_offsets"][group]
                <= start
                < start + c["training"]["sequence_length"]
                < meta["story_offsets"][group + 1]
            ):
                raise ValueError("window or shifted target crosses story boundary")
    if set(windows["train"]["story_hashes"]) & set(windows["development"]["story_hashes"]):
        raise ValueError("cross-split exact leakage")
    if canonical_sha256(windows["evaluation"]) != manifest["evaluation_sha256"]:
        raise ValueError("frozen evaluation windows differ")
    data.update(
        windows=windows,
        manifest=manifest,
        asset_sha256=file_sha256(folder / "manifest.json"),
        tokenizer_sha256=manifest["files"]["tokenizer.json"],
    )
    return data
