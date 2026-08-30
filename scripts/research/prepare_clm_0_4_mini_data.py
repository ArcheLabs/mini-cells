#!/usr/bin/env python3
"""Prepare formal CLM-0.4-mini tokenizer/base-corpus/curriculum assets.

This command is seed-independent. It requires an explicitly pinned TinyStories
revision and never runs calibration or formal model seeds.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from minicells.clm04mini.data import iter_hf_tinystories
from minicells.clm04mini.m1 import prepare_formal_data_assets


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    ROOT / "research" / "validations" / "clm-0.4-mini-language-validation" / "protocol.json"
)
DEFAULT_OUT = ROOT / "results" / "clm-0.4-mini-data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--routing-salt", default="clm-0.4-mini-v1")
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
    tokenizer_carrier = iter_hf_tinystories(
        revision=args.dataset_revision,
        max_examples=args.tokenizer_carrier_examples,
    )
    # The controlled generators are added later by the tokenizer/curriculum code;
    # here the pinned carrier ordering is supplied unchanged.
    tokenizer_texts = list(tokenizer_carrier)
    if not tokenizer_texts:
        raise RuntimeError("pinned TinyStories revision produced no tokenizer text")
    carrier = iter_hf_tinystories(revision=args.dataset_revision)
    assets = prepare_formal_data_assets(
        protocol_path=args.protocol,
        out_dir=args.out,
        routing_salt=args.routing_salt,
        tokenizer_training_texts=tokenizer_texts,
        carrier_texts=carrier,
        carrier_source=source,
    )
    summary = {
        "scientific_decision": False,
        "formal_seeds_observed": False,
        "development_seed_observed": False,
        "routing_salt": args.routing_salt,
        "tokenizer_manifest_hash": assets["tokenizer_manifest"]["manifest_sha256"],
        "tokenizer_hash": assets["tokenizer_manifest"]["tokenizer_sha256"],
        "base_corpus_manifest_hash": assets["base_corpus_manifest"]["manifest_sha256"],
        "curriculum_manifest_hash": assets["curriculum_manifest"]["manifest_sha256"],
        "base_tokens": assets["base_corpus_manifest"]["actual_tokens"],
    }
    (args.out / "asset-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
