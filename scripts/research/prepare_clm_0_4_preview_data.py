#!/usr/bin/env python3
"""Prepare CLM-0.4 Preview data: 60/30/10 mix + digit-aware tokenizer."""

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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "clm-0.4-preview-data"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--target-tokens", type=int, default=30_000_000)
    parser.add_argument("--tokenizer-carrier-examples", type=int, default=50_000)
    parser.add_argument("--tokenizer-math-examples", type=int, default=4096)
    parser.add_argument("--tokenizer-story-examples", type=int, default=2048)
    args = parser.parse_args()

    source = {
        "dataset": "roneneldan/TinyStories",
        "revision": str(args.dataset_revision),
        "split": "train",
        "tokenizer_carrier_examples": int(args.tokenizer_carrier_examples),
        "routing_salt": PREVIEW_ROUTING_SALT,
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
        target_tokens=int(args.target_tokens),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
