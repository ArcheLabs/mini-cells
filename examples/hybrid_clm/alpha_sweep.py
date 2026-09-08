"""Demonstrate reversible expression control over measured alpha values."""

from __future__ import annotations

import argparse

from minicells import CellMutation, CellPlacement, HybridCLM


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mutation")
    parser.add_argument("--model", default="ibm-granite/granite-3.1-1b-a400m-base")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 0.75, 1.0])
    args = parser.parse_args()
    hybrid = HybridCLM.from_pretrained(args.model, revision=args.revision)
    hybrid.cellularize(CellPlacement(layer=args.layer, experts="all"))
    mutation = CellMutation.from_pretrained(args.mutation)
    hybrid.attach(mutation)
    for alpha in args.alphas:
        hybrid.set_alpha(mutation, alpha)
        print({"alpha": alpha, "note": "evaluate with your fixed inputs"})
    hybrid.detach(mutation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

