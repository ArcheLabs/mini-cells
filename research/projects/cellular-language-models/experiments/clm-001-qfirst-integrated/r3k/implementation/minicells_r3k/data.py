"""Read only the frozen source indices. Build research roles with the existing encoder."""

import itertools
from pathlib import Path
import numpy as np
from minicells_r3e.common import read, atomic, digest, canonical
from minicells_r3h.evaluation import assets
from .data_boundaries import boundary_proof, RANGES, REVISION


def prepare(c, runtime, backend, folder):
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from fast001.data import SHARD, grouped_windows
    from exp006.data import encode_stories

    folder = Path(folder)
    historical = assets(runtime, backend, folder / "historical", ("calibration", "admission"))
    dest = folder / "research"
    if (dest / "manifest.json").exists():
        load(c, folder, "predictor_probe")
        return read(dest / "manifest.json")
    config = runtime["engine"]["input"]["data_config"]
    ranges = {r: config[r + "_range"] for r in ("train", "calibration", "admission", "development")}
    if config["revision"] != REVISION or config["reserved_boundary"] != 60000:
        raise RuntimeError("RESEARCH_HOLDOUT_BOUNDARY_CONFLICT")
    source = hf_hub_download(
        repo_id=config["dataset"], repo_type="dataset", revision=REVISION, filename=SHARD
    )
    # Verify the exact historical shard before using the same raw indices.
    if digest(source) != historical["manifest"]["source_sha256"]:
        raise RuntimeError("HISTORICAL_SOURCE_HASH_CHANGED")
    rows = load_dataset("parquet", data_files=source, split="train", streaming=True)
    texts = {}
    hashes = []
    import hashlib

    for i, row in enumerate(itertools.islice(rows, max(x[1] for x in ranges.values()))):
        if any(a <= i < b for a, b in ranges.values()):
            hashes.append(hashlib.sha256(row["text"].encode("utf-8")).hexdigest())
        if 8192 <= i < 8960:
            texts[i] = row["text"]
    tokenizer_path = folder / "historical/tokenizer.json"
    proof = boundary_proof(REVISION, digest(tokenizer_path), ranges, hashes, texts)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    n = runtime["engine"]["training"]["sequence_length"]
    dest.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for role in RANGES:
        ids = [x["story_index"] for x in proof["roles"][role]["accepted"]]
        encoded, starts, groups, story_hashes = encode_stories(
            [texts[i] for i in ids], tokenizer, n
        )
        if not starts:
            raise RuntimeError("EMPTY_RESEARCH_EVALUATION")
        np.save(dest / (role + ".npy"), encoded)
        lengths = [len(tokenizer.encode(texts[i]).ids) + 1 for i in ids]
        metadata[role] = dict(
            starts=starts,
            groups=groups,
            story_offsets=np.cumsum([0] + lengths).tolist(),
            raw_story_ids=ids,
            story_hashes=story_hashes,
            evaluation=grouped_windows(
                starts,
                groups,
                c["predictor_windows" if role == "predictor_probe" else "value_windows"],
                c["evaluation_seed"],
            ),
        )
    atomic(dest / "windows.json", metadata)
    atomic(dest / "DATA_BOUNDARY_PROOF.json", proof)
    manifest = dict(
        config_sha256=canonical(c),
        source_sha256=digest(source),
        tokenizer_sha256=digest(tokenizer_path),
        confirmation="SEALED",
        files={x.name: digest(x) for x in dest.iterdir() if x.is_file()},
    )
    atomic(dest / "manifest.json", manifest)
    return manifest


def load(c, folder, role, freeze_verified=False):
    if role not in RANGES or (role == "value_holdout" and not freeze_verified):
        raise RuntimeError("RESEARCH_ROLE_PHASE_FORBIDDEN")
    folder = Path(folder) / "research"
    manifest = read(folder / "manifest.json")
    if manifest["config_sha256"] != canonical(c) or manifest["confirmation"] != "SEALED":
        raise RuntimeError("RESEARCH_ASSET_IDENTITY_CHANGED")
    # Hash checks do not map or evaluate the holdout before the prediction barrier.
    for name, sha in manifest["files"].items():
        if (folder / name).is_symlink() or digest(folder / name) != sha:
            raise RuntimeError("RESEARCH_ASSET_HASH_CHANGED")
    w = read(folder / "windows.json")[role]
    a = np.load(folder / (role + ".npy"), mmap_mode="r")
    return {"development": a, "windows": {"development": w}, "manifest": manifest}
