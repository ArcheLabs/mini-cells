#!/usr/bin/env python3
"""Prepare CLM-0.4 Release assets for the 1M smoke or 30M release profile."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from minicells.clm04mini.data import iter_hf_tinystories
from minicells.clm04mini.preview import (
    PREVIEW_ROUTING_SALT,
    prepare_preview_data_assets,
    preview_math_stream,
    preview_story_stream,
)
from minicells.clm04mini.release import expected_profile_tokens, release_pipeline_sha256


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke-1m", "release-30m"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-revision", default=DEFAULT_REVISION)
    parser.add_argument("--tokenizer-carrier-examples", type=int, default=50_000)
    parser.add_argument("--tokenizer-math-examples", type=int, default=4096)
    parser.add_argument("--tokenizer-story-examples", type=int, default=2048)
    args = parser.parse_args()

    target_tokens = expected_profile_tokens(args.profile)
    source = {
        "dataset": "roneneldan/TinyStories",
        "revision": str(args.dataset_revision),
        "split": "train",
        "tokenizer_carrier_examples": int(args.tokenizer_carrier_examples),
        "routing_salt": PREVIEW_ROUTING_SALT,
        "release_profile": args.profile,
    }
    carrier_tokenizer = list(
        iter_hf_tinystories(
            revision=args.dataset_revision,
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
        carrier_texts=iter_hf_tinystories(revision=args.dataset_revision),
        carrier_source=source,
        target_tokens=target_tokens,
    )
    release_manifest = {
        "format": "minicells.clm-0.4-release-asset-profile.v1",
        "profile": args.profile,
        "target_tokens": target_tokens,
        "pipeline_sha256": release_pipeline_sha256(),
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
