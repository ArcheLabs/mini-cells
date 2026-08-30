#!/usr/bin/env python3
"""Prepare seed-independent CLM-0.4-mini M1-v2 data assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minicells.clm04mini.data import iter_hf_tinystories
from minicells.clm04mini.v2 import V2_ROUTING_SALT, prepare_v2_data_assets


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = (
    ROOT
    / "research"
    / "validations"
    / "clm-0.4-mini-m1-v2-language-validation"
)
DEFAULT_PROTOCOL = VALIDATION / "protocol.json"
DEFAULT_OUT = ROOT / "results" / "clm-0.4-mini-m1-v2-data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--routing-salt", default=V2_ROUTING_SALT)
    parser.add_argument("--tokenizer-carrier-examples", type=int, default=50000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = {
        "dataset": "roneneldan/TinyStories",
        "revision": str(args.dataset_revision),
        "split": "train",
        "tokenizer_carrier_examples": int(args.tokenizer_carrier_examples),
    }
    tokenizer_texts = list(
        iter_hf_tinystories(
            revision=args.dataset_revision,
            max_examples=args.tokenizer_carrier_examples,
        )
    )
    if not tokenizer_texts:
        raise RuntimeError("pinned TinyStories revision produced no tokenizer text")
    assets = prepare_v2_data_assets(
        protocol_path=args.protocol,
        out_dir=args.out,
        routing_salt=args.routing_salt,
        tokenizer_training_texts=tokenizer_texts,
        carrier_texts=iter_hf_tinystories(revision=args.dataset_revision),
        carrier_source=source,
    )
    print(json.dumps(assets["summary"], indent=2, sort_keys=True))
    print(
        "\nDATA LOCK REQUIRED: commit these hashes into "
        "asset-lock.json before opening development seed 90402."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
