"""Frozen tokenizer and original stream writer; never calls prepare_corpus."""

import json
import shutil
from pathlib import Path

from .common import ROOT, digest, native, write


def prepare(cache, config):
    base, _, _ = native()
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer_path = cache / "tokenizer.json"
    original = ROOT / config["tokenizer_path"]
    if digest(original) != config["tokenizer_sha256"]:
        raise RuntimeError("ORIGINAL_TOKENIZER_HASH_MISMATCH")
    if not tokenizer_path.exists():
        shutil.copyfile(original, tokenizer_path)
    if digest(tokenizer_path) != config["tokenizer_sha256"]:
        raise RuntimeError("CACHED_TOKENIZER_HASH_MISMATCH_NO_RETRAIN")
    tokenizer = base.load_tokenizer(tokenizer_path)
    if tokenizer.get_vocab_size() != 2048:
        raise RuntimeError("TOKENIZER_VOCAB_MISMATCH")
    base.DATASET_REVISION = config["dataset_revision"]
    manifest = dict(config["historical_corpus_manifest"])
    observed = {}
    for split, count in (("train", 12000000), ("validation", 1000000)):
        path = cache / (split + ".u16")
        if not path.exists():
            pending = cache / (split + ".u16.pending")
            stories = base._write_token_stream(
                tokenizer, split=split, output_path=pending, target_tokens=count
            )
            pending.replace(path)
            observed[split + "_stories_consumed"] = stories
        actual = digest(path)
        observed[split + "_raw_sha256"] = actual
        if (
            path.stat().st_size != count * 2
            or actual != config["historical_corpus_manifest"][split + "_raw_sha256"]
        ):
            write(
                cache / "DATA_MISMATCH.json",
                {"status": "HISTORICAL_DATA_IDENTITY_MISMATCH", "actual": observed},
            )
            raise RuntimeError("DATA_HASH_MISMATCH_TRAINING_FORBIDDEN")
    for field in ("train_stories_consumed", "validation_stories_consumed"):
        if field in observed and observed[field] != manifest[field]:
            raise RuntimeError("STORY_COUNT_MISMATCH")
    write(cache / "manifest.json", manifest)
    write(
        cache / "DATA_GATE.json",
        {
            "status": "HISTORICAL_DATA_IDENTITY_MATCHED",
            "actual": observed,
            "tokenizer_sha256": digest(tokenizer_path),
        },
    )
    return base.Corpus(cache / "train.u16", cache / "validation.u16", tokenizer_path, manifest)


def load_verified(cache, config):
    base, _, _ = native()
    cache = Path(cache)
    for name, key in (
        ("tokenizer.json", "tokenizer_sha256"),
        ("train.u16", "train_raw_sha256"),
        ("validation.u16", "validation_raw_sha256"),
    ):
        expected = (
            config["tokenizer_sha256"]
            if name == "tokenizer.json"
            else config["historical_corpus_manifest"][key]
        )
        if digest(cache / name) != expected:
            raise RuntimeError("DATA_CHANGED_AFTER_R0")
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest != config["historical_corpus_manifest"]:
        raise RuntimeError("CORPUS_MANIFEST_CHANGED_AFTER_R0")
    return base.Corpus(
        cache / "train.u16", cache / "validation.u16", cache / "tokenizer.json", manifest
    )
