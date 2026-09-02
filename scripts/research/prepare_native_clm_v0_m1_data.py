#!/usr/bin/env python3
"""Prepare a deterministic text cache for Native CLM v0 M1.

Default source is TinyStories. The trainer itself only consumes local UTF-8 files, so
a dataset snapshot is separated from model code and can be replaced without changing
the Native CLM architecture.

Kaggle currently runs Python 3.12. Some pyarrow / Hugging Face streaming combinations
can abort *after* a successful early-stop read while CPython is finalizing native Arrow
threads. This CLI therefore makes the cache idempotent and, on Python 3.12+, uses a
hard process exit only after every output has been flushed/fsynced successfully. The
hard exit is deliberately confined to this one-shot data-preparation subprocess; the
training process does not use it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_docs(path: Path, docs: Iterable[str], limit: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for text in docs:
            if count >= limit:
                break
            clean = str(text).strip()
            if not clean:
                continue
            handle.write(clean)
            handle.write("\n\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def _write_json_fsynced(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _reuse_complete_cache(output: Path, args: argparse.Namespace) -> dict | None:
    """Return a verified existing manifest without importing datasets/pyarrow."""
    manifest_path = output / "manifest.json"
    train_path = output / "train.txt"
    validation_path = output / "validation.txt"
    if not (manifest_path.exists() and train_path.exists() and validation_path.exists()):
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    expected = {
        "dataset_id": args.dataset_id,
        "text_column": args.text_column,
        "train_split": args.train_split,
        "validation_split": args.validation_split,
        "train_docs": args.train_docs,
        "validation_docs": args.validation_docs,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    if manifest.get("train_bytes") != train_path.stat().st_size:
        return None
    if manifest.get("validation_bytes") != validation_path.stat().st_size:
        return None
    if manifest.get("train_sha256") != _sha256(train_path):
        return None
    if manifest.get("validation_sha256") != _sha256(validation_path):
        return None
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="roneneldan/TinyStories")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--train-docs", type=int, default=50_000)
    parser.add_argument("--validation-docs", type=int, default=2_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/native-clm-m1-data"),
    )
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    # A previous Arrow-backed run may have completed all writes and then aborted only
    # during interpreter teardown. Verify and reuse that cache before importing any
    # Arrow-backed package so rerunning the notebook is cheap and safe.
    cached = _reuse_complete_cache(output, args)
    if cached is not None:
        print("Reusing verified Native CLM M1 data cache.")
        print(json.dumps(cached, indent=2))
        return 0

    # Keep the heavy native dependency out of module import and cache-reuse paths.
    from datasets import load_dataset

    train = load_dataset(args.dataset_id, split=args.train_split, streaming=True)
    validation = load_dataset(args.dataset_id, split=args.validation_split, streaming=True)

    def texts(stream):
        for row in stream:
            yield row[args.text_column]

    train_path = output / "train.txt"
    validation_path = output / "validation.txt"
    train_count = _write_docs(train_path, texts(train), args.train_docs)
    validation_count = _write_docs(validation_path, texts(validation), args.validation_docs)

    if train_count < args.train_docs:
        raise RuntimeError(f"dataset yielded only {train_count} usable train documents")
    if validation_count < args.validation_docs:
        raise RuntimeError(f"dataset yielded only {validation_count} usable validation documents")

    manifest = {
        "format": "minicells.native-clm-v0.m1-data-manifest.v1",
        "dataset_id": args.dataset_id,
        "text_column": args.text_column,
        "train_split": args.train_split,
        "validation_split": args.validation_split,
        "train_docs": train_count,
        "validation_docs": validation_count,
        "train_bytes": train_path.stat().st_size,
        "validation_bytes": validation_path.stat().st_size,
        "train_sha256": _sha256(train_path),
        "validation_sha256": _sha256(validation_path),
    }
    _write_json_fsynced(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def _main_cli() -> None:
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()

    # Apache Arrow has a known Python-3.12 finalization failure after an early-stop
    # dataset scan. At this point all registered outputs are already durable. Bypass
    # only native interpreter teardown in this dedicated subprocess.
    if code == 0 and sys.version_info >= (3, 12):
        os._exit(0)
    raise SystemExit(code)


if __name__ == "__main__":
    _main_cli()
