"""Inspect a pinned Granite checkpoint without selecting a scientific placement."""

from __future__ import annotations

import argparse

from minicells import HybridCLM


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="ibm-granite/granite-3.1-1b-a400m-base")
    parser.add_argument("--revision", required=True, help="immutable Hugging Face commit")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(HybridCLM.inspect_pretrained(args.model, revision=args.revision, device=args.device).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

