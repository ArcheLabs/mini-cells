#!/usr/bin/env python3
"""Prepare CLM-0.4 Release assets for the 1M smoke or 30M release profile."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time
from typing import Iterator
from urllib.request import Request, urlopen

from minicells.clm04mini.preview import (
    PREVIEW_ROUTING_SALT,
    prepare_preview_data_assets,
    preview_math_stream,
    preview_story_stream,
)
from minicells.clm04mini.release import expected_profile_tokens, release_pipeline_sha256


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TINYSTORIES_TRAIN_SHARDS = (
    "train-00000-of-00004-2d5a1467fff1081b.parquet",
    "train-00001-of-00004-5852b56a2bd28fd9.parquet",
    "train-00002-of-00004-a26307300439e943.parquet",
    "train-00003-of-00004-d243063613e5a057.parquet",
)
CARRIER_TRANSPORT_VERSION = "tinystories-pinned-parquet-cache-v1"


def _download_pinned_shard(*, revision: str, filename: str, cache_dir: Path) -> Path:
    """Download one pinned parquet shard serially and atomically, without HF worker threads."""
    cache_root = cache_dir / revision
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / filename
    if target.is_file() and target.stat().st_size > 0:
        return target

    url = (
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
        f"{revision}/data/{filename}"
    )
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, 6):
        partial.unlink(missing_ok=True)
        try:
            request = Request(url, headers={"User-Agent": "mini-cells-release/0.4"})
            with urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            if partial.stat().st_size <= 0:
                raise RuntimeError(f"downloaded empty TinyStories shard: {filename}")
            partial.replace(target)
            return target
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt >= 5:
                raise
            time.sleep(float(attempt))
    raise AssertionError("unreachable")


def iter_cached_tinystories(
    *,
    revision: str,
    cache_dir: Path,
    max_examples: int | None = None,
) -> Iterator[str]:
    """Yield pinned TinyStories rows from a local parquet cache in canonical shard order."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for cached TinyStories release preparation") from exc

    yielded = 0
    for filename in TINYSTORIES_TRAIN_SHARDS:
        shard = _download_pinned_shard(
            revision=revision,
            filename=filename,
            cache_dir=cache_dir,
        )
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=2048, columns=["text"]):
            for value in batch.column(0).to_pylist():
                text = str(value or "").strip()
                if not text:
                    continue
                yield text
                yielded += 1
                if max_examples is not None and yielded >= int(max_examples):
                    return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke-1m", "release-30m"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-revision", default=DEFAULT_REVISION)
    parser.add_argument("--carrier-cache-dir", type=Path)
    parser.add_argument("--tokenizer-carrier-examples", type=int, default=50_000)
    parser.add_argument("--tokenizer-math-examples", type=int, default=4096)
    parser.add_argument("--tokenizer-story-examples", type=int, default=2048)
    args = parser.parse_args()

    target_tokens = expected_profile_tokens(args.profile)
    cache_dir = args.carrier_cache_dir or (args.out.parent / "tinystories-pinned-cache")
    source = {
        "dataset": "roneneldan/TinyStories",
        "revision": str(args.dataset_revision),
        "split": "train",
        "tokenizer_carrier_examples": int(args.tokenizer_carrier_examples),
        "routing_salt": PREVIEW_ROUTING_SALT,
        "release_profile": args.profile,
        "carrier_transport": CARRIER_TRANSPORT_VERSION,
        "carrier_cache_dir": str(cache_dir),
    }
    carrier_tokenizer = list(
        iter_cached_tinystories(
            revision=args.dataset_revision,
            cache_dir=cache_dir,
            max_examples=args.tokenizer_carrier_examples,
        )
    )
    if not carrier_tokenizer:
        raise RuntimeError("pinned TinyStories revision produced no tokenizer text")
    tokenizer_texts = [
        *carrier_tokenizer,
        *itertools.islice(preview_math_stream(), int(args.tokenizer_math_examples)),
        *itertools.islice(preview_story_stream(), int(args.tokenizer_story_examples)),
    ]
    summary = prepare_preview_data_assets(
        out_dir=args.out,
        tokenizer_training_texts=tokenizer_texts,
        carrier_texts=iter_cached_tinystories(
            revision=args.dataset_revision,
            cache_dir=cache_dir,
        ),
        carrier_source=source,
        target_tokens=target_tokens,
    )
    release_manifest = {
        "format": "minicells.clm-0.4-release-asset-profile.v1",
        "profile": args.profile,
        "target_tokens": target_tokens,
        "pipeline_sha256": release_pipeline_sha256(),
        "carrier_transport": CARRIER_TRANSPORT_VERSION,
        "asset_summary": summary,
    }
    (args.out / "release-profile.json").write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(release_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
