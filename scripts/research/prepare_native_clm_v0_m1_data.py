#!/usr/bin/env python3
"""Prepare a deterministic text cache for Native CLM v0 M1.

Default source is TinyStories. The trainer itself only consumes local UTF-8 files, so
a dataset snapshot is separated from model code and can be replaced without changing
the Native CLM architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    return count


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

    from datasets import load_dataset

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

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
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
